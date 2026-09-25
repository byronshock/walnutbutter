//! The wave loop of `propagation.Schedule.run` (AUTHORITY.md §4.4), in Rust.
//!
//! The engine owns the network's state for the life of a run, so an epoch costs one
//! crossing of the Python boundary rather than one per event. Everything here mirrors
//! `neuron.py` and `propagation.py` operation for operation and in the same order, so
//! that the floating point agrees: the point of a second engine in this project has
//! always been that two independent implementations landing on the same bits is evidence
//! the specification is unambiguous.
//!
//! Under exploration at the synapse (AUTHORITY.md §7.5-§7.9, §8.16-§8.17) the loop mirrors
//! `Network._decide_synapses` and the gain of `neuron.py` the same way: the synapses' E + O
//! uniforms are taken after the floor, every source that did not spike decides after the fire
//! phase, and its escapes are pushed as ventured signals after the wave's spikes. The charged
//! drive (5.4b) is the EXTERNAL event, taken before the wave's signals.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyLong};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Event kinds, in the order they are taken within a wave (propagation.py).
const EXTERNAL: u8 = 0;
const STIMULUS: u8 = 1;
const SIGNAL: u8 = 2;

/// clock.py's TOLERANCE, mirrored: two moments closer than this are the same moment. A.0 allows no
/// second literal of a register value, so this one is exported and held equal to Python's by a test.
const TOLERANCE: f64 = 1e-12;

/// exploration.py: the Box-Muller transform's constant.
const TWO_PI: f64 = std::f64::consts::TAU;

// --- the exploration stream ---------------------------------------------------------
// AUTHORITY.md §6.1 says both engines draw from the same Box-Muller stream, so a seed
// gives the same noise whichever engine runs. That is only true if this engine draws the
// same uniforms in the same order as Python's `random.Random`, which is MT19937. So it is
// MT19937, seeded by handing over Python's own 625-word state and handed back at the end,
// rather than a generator of our own: `fast.py` round-trips the state through
// `getstate`/`setstate`, and Python's stream carries on exactly where Rust left it.

const MT_N: usize = 624;
const MT_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

struct MersenneTwister {
    mt: [u32; MT_N],
    index: usize,
}

impl MersenneTwister {
    /// From Python's `rng.getstate()[1]`: 624 state words followed by the index.
    fn from_state(state: &[u32]) -> Result<Self, String> {
        if state.len() != MT_N + 1 {
            return Err(format!("the stream's state needs {} words, got {}", MT_N + 1, state.len()));
        }
        let index = state[MT_N] as usize;
        if index > MT_N {
            return Err(format!("the stream's index must be at most {MT_N}, got {index}"));
        }
        let mut mt = [0u32; MT_N];
        mt.copy_from_slice(&state[..MT_N]);
        Ok(Self { mt, index })
    }

    /// The state in Python's shape, so `setstate` can take it back.
    fn state(&self) -> Vec<u32> {
        let mut out = self.mt.to_vec();
        out.push(self.index as u32);
        out
    }

    fn genrand(&mut self) -> u32 {
        if self.index >= MT_N {
            for i in 0..MT_N {
                let y = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % MT_N] & LOWER_MASK);
                let mut next = self.mt[(i + MT_M) % MT_N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= MATRIX_A;
                }
                self.mt[i] = next;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// CPython's `random_random`: 53 bits of randomness from two 32-bit draws.
    fn random(&mut self) -> f64 {
        let a = (self.genrand() >> 5) as f64;
        let b = (self.genrand() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }
}

#[inline]
fn slack(time: f64) -> f64 {
    TOLERANCE * f64::max(1.0, time.abs())
}

/// True if `time` is before `until` by more than the slack.
#[inline]
fn before(time: f64, until: f64) -> bool {
    time < until - slack(until)
}

/// One scheduled event. `payload` is an edge index for a signal, a neuron index otherwise. `ventured` marks a
/// signal an escape of its synapse sent rather than its source's spike (AUTHORITY.md §7.9); it rides on the event
/// and is no part of its order, as propagation.py carries it in the payload and never in the kind or the sort key.
#[derive(Debug, Clone, Copy)]
struct Event {
    time: f64,
    kind: u8,
    seq: u64,
    payload: u32,
    ventured: bool,
}

// A min-heap by (time, kind, seq), matching heapq on the Python side: earliest first,
// then externals before stimuli before signals, then insertion order. BinaryHeap is a
// max-heap, so the ordering is reversed here rather than at every call site. The
// ventured mark is left out, so a ventured signal sums in push order with the relayed
// ones (§3.7).
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.kind.cmp(&self.kind))
            .then_with(|| other.seq.cmp(&self.seq))
    }
}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Event {}

/// The network's state and its schedule, owned across a run.
#[pyclass]
pub struct Engine {
    // --- topology, as compressed adjacency -------------------------------------------
    neurons: usize,
    out_start: Vec<u32>,  // neuron i owns out_edges[out_start[i]..out_start[i + 1]]
    out_edges: Vec<u32>,
    in_start: Vec<u32>,   // and in_edges[in_start[i]..in_start[i + 1]]
    in_edges: Vec<u32>,
    edge_target: Vec<u32>,
    weight: Vec<f64>,
    active: Vec<bool>,
    last_signal: Vec<f64>, // NEG_INFINITY when this synapse has never carried one

    // --- neuron state ----------------------------------------------------------------
    potential: Vec<f64>,
    threshold: Vec<f64>,
    floor: Vec<f64>,
    fired_at: Vec<f64>, // NEG_INFINITY when never
    previous_fired_at: Vec<f64>,
    last_update: Vec<f64>,
    spikes: Vec<u64>,
    spikes_at_reset: Vec<u64>, // and at the epoch's start: the difference is the count read (§4.3)
    rate_level: Vec<f64>,
    rate_at: Vec<f64>,
    forced: Vec<bool>,
    fired_wave: Vec<i64>, // -1 when it has not fired this epoch
    delivered_wave: Vec<i64>, // per edge: the wave it delivered in this epoch, landed or not; -1 when it did not
    trace: Vec<f64>,    // per edge: the charge it still has in its target's potential, brought up to trace_at (§6.7)
    trace_at: Vec<f64>,
    score: Vec<f64>,    // per edge: the hazard eligibility accumulated this epoch (§6.7)
    delta: Vec<f64>,    // per neuron: the width of its firing decision; 0 is the deterministic threshold (§5.2)
    escape_scale: Vec<f64>, // per neuron: sqrt(N0 / N), the count's scaling of every hazard (§5.2, September 16, 2026)
    exposed_since: Vec<f64>, // per neuron: since when its hazard has run
    draw: Vec<f64>,     // per neuron: this wave's uniform for the decision
    hazard: bool,       // any delta > 0: the decision is a draw
    centred: bool,      // the hebb eligibility charges every decision against the neuron's own expectation (§6.7)
    traced: bool,       // hazard || centred || synaptic: the traces and scores are kept
    decision_memory: f64,  // DECISION_MEMORY: the per-decision move of each neuron's expectation of its own spike
    expectation: Vec<f64>, // per neuron: p_hat_j, NaN until its first decision
    decisions: Vec<u64>,   // per neuron: decisions to date, for the expectation's warm start
    expected: Vec<f64>,    // per neuron: E_j, the spikes expected over its decisions since its last spike (§6.7)
    credit: Vec<f64>,      // per neuron: what its firing decision credits each open arrival with, for fire() to settle
    noted: Vec<f64>,       // per edge: B_ij, the sum over its open arrivals of the target's E_j as each arrived

    // --- exploration at the synapse (§7.5-§7.9, §8.16-§8.17) ---------------------------
    synaptic: bool,         // EXPLORATION synapse (§7.1): every width 0, the synapses take the chance (§6.13)
    rest: f64,              // h0, the rest hazard (§7.6)
    rest_power: f64,        // h0 ** 1.0, the loglinear h at u = 0, where most sources sit: the one pow call, made once
    linear: bool,          // the linear family, h0 + (1 - h0) u; else the loglinear, h0 ** (1 - u) (§7.6)
    trace_ventured: bool,   // TRACE ventured: a trace counts the ventured arrivals alone (§8.17)
    synapse_scale: Vec<f64>, // per neuron: kappa_i (§7.7), computed once in Python and handed over (§7.5)
    read_slot: Vec<i64>,    // per neuron: its read synapse's draw, E + k in output order, or -1 for none (§7.9)
    read_count: Vec<u64>,   // per neuron: its read synapse's escapes this epoch, added to its spikes in the count read (§5.10)
    gain: Vec<f64>,         // per neuron: G_j, the net credit its synapses' decisions posted since its last settle (§8.16)
    synapse_draw: Vec<f64>, // this wave's E + O uniforms, E by edge id then O by read slot (§3.8); allocated once
    drive_steps: u64,       // DRIVE_STEPS under the charged drive (5.4b): an EXTERNAL event delivers theta / this; 0 for none

    // --- the clock and the rules ------------------------------------------------------
    tau: f64,
    refractory: f64,
    hop: f64,
    bored_after: f64,
    rate_tau: f64,
    quash_rate: f64,
    quash_k: f64,
    weight_low: f64,
    weight_high: f64,
    explore: Option<MersenneTwister>, // Python's own stream, handed over for the run

    // --- the schedule -----------------------------------------------------------------
    heap: BinaryHeap<Event>,
    seq: u64,
    stamp: Vec<u64>, // per neuron: the mark of the wave it was last touched in
    wave_no: u64,    // this epoch's wave count, for fired_wave and delivered_wave; reset each epoch
    marks: u64,      // every wave ever, never reset: propagation.py's _wave_stamps, a stamp that never repeats
}

#[pymethods]
impl Engine {
    /// Build from the flattened network. `edge_source`/`edge_target` are parallel to `weight`.
    #[allow(clippy::too_many_arguments)]
    #[new]
    fn new(
        neurons: usize,
        edge_source: Vec<u32>,
        edge_target: Vec<u32>,
        weight: Vec<f64>,
        active: Vec<bool>,
        threshold: Vec<f64>,
        floor: Vec<f64>,
        tau: f64,
        refractory: f64,
        hop: f64,
        bored_after: f64,
        rate_tau: f64,
    ) -> PyResult<Self> {
        let edges = weight.len();
        if edge_source.len() != edges || edge_target.len() != edges || active.len() != edges {
            return Err(PyValueError::new_err(
                "edge_source, edge_target, weight and active must be the same length",
            ));
        }
        if threshold.len() != neurons || floor.len() != neurons {
            return Err(PyValueError::new_err(
                "threshold and floor must have one entry per neuron",
            ));
        }
        let out_start = counts_to_starts(&edge_source, neurons);
        let in_start = counts_to_starts(&edge_target, neurons);
        let out_edges = bucket(&edge_source, &out_start, neurons);
        let in_edges = bucket(&edge_target, &in_start, neurons);
        Ok(Engine {
            neurons,
            out_start,
            out_edges,
            in_start,
            in_edges,
            edge_target,
            weight,
            active,
            last_signal: vec![f64::NEG_INFINITY; edges],
            potential: vec![0.0; neurons],
            threshold,
            floor,
            fired_at: vec![f64::NEG_INFINITY; neurons],
            previous_fired_at: vec![f64::NEG_INFINITY; neurons],
            last_update: vec![0.0; neurons],
            spikes: vec![0; neurons],
            spikes_at_reset: vec![0; neurons],
            rate_level: vec![0.0; neurons],
            rate_at: vec![0.0; neurons],
            forced: vec![false; neurons],
            fired_wave: vec![-1; neurons],
            delivered_wave: vec![-1; edges],
            trace: vec![0.0; edges],
            trace_at: vec![0.0; edges],
            score: vec![0.0; edges],
            delta: vec![0.0; neurons],
            escape_scale: vec![1.0; neurons],
            exposed_since: vec![0.0; neurons],
            draw: vec![1.0; neurons],
            hazard: false,
            centred: false,
            traced: false,
            decision_memory: 1e-4,
            expectation: vec![f64::NAN; neurons],
            decisions: vec![0; neurons],
            expected: vec![0.0; neurons],
            credit: vec![0.0; neurons],
            noted: vec![0.0; edges],
            synaptic: false,
            rest: 0.0,
            rest_power: 0.0,
            linear: false,
            trace_ventured: false,
            synapse_scale: vec![1.0; neurons],
            read_slot: vec![-1; neurons],
            read_count: vec![0; neurons],
            gain: vec![0.0; neurons],
            synapse_draw: Vec::new(),
            drive_steps: 0,
            tau,
            refractory,
            hop,
            bored_after,
            rate_tau,
            quash_rate: 0.0,
            quash_k: 0.2,
            weight_low: -1.0,
            weight_high: 1.0,
            explore: None,
            heap: BinaryHeap::new(),
            seq: 0,
            stamp: vec![0; neurons],
            wave_no: 0,
            marks: 0,
        })
    }

    /// The local rules that must run inside the loop, and the weight range they clip to.
    #[allow(clippy::too_many_arguments)]
    fn set_rules(
        &mut self,
        quash_rate: f64,
        quash_k: f64,
        weight_low: f64,
        weight_high: f64,
    ) -> PyResult<()> {
        self.quash_rate = quash_rate;
        self.quash_k = quash_k;
        self.weight_low = weight_low;
        self.weight_high = weight_high;
        Ok(())
    }

    /// Take over Python's exploration stream: `rng.getstate()[1]`, 624 words then the index.
    ///
    /// The engine draws from it for the life of the run and `explore_state()` hands it back,
    /// so `rng.setstate` can carry Python's stream on from exactly where this left it (§6.1).
    fn set_explore_state(&mut self, state: Vec<u32>) -> PyResult<()> {
        self.explore = Some(MersenneTwister::from_state(&state).map_err(PyValueError::new_err)?);
        Ok(())
    }

    /// The stream's state as Python's `setstate` wants it, or None if none was handed over.
    fn explore_state(&self) -> Option<Vec<u32>> {
        self.explore.as_ref().map(|rng| rng.state())
    }

    /// Start a new epoch: clear the fired-this-epoch state, exactly as `Network.reset` does.
    /// Potentials, spike times and the signals in flight are kept (AUTHORITY.md §4.2).
    fn reset(&mut self, discharge: bool) {
        if discharge {
            self.potential.iter_mut().for_each(|p| *p = 0.0);
        }
        self.fired_wave.iter_mut().for_each(|w| *w = -1);
        self.delivered_wave.iter_mut().for_each(|w| *w = -1);
        self.forced.iter_mut().for_each(|f| *f = false);
        self.spikes_at_reset.copy_from_slice(&self.spikes); // Neuron.reset snapshots the count
        self.read_count.iter_mut().for_each(|r| *r = 0); // the read synapse's escapes are the epoch's (§7.9)
        if self.traced {
            if discharge {
                for i in 0..self.neurons {
                    self.clear_arrivals(i); // nothing is left in a zeroed potential (§6.7)
                }
            }
            self.score.iter_mut().for_each(|s| *s = 0.0); // the score is the epoch's (§6.7)
            if self.tau.is_infinite() {
                // an open arrival's debit counts from here: what it accrued was paid, or dropped, with the score --
                // B = x E, or under exploration at the synapse B = x G (§8.16)
                let owed = if self.synaptic { &self.gain } else { &self.expected };
                for edge in 0..self.weight.len() {
                    self.noted[edge] = self.trace[edge] * owed[self.edge_target[edge] as usize];
                }
            }
        }
        self.wave_no = 0;
    }

    /// `neuron` is forced to fire at `time`, refractory period permitting.
    fn stimulus(&mut self, neuron: u32, time: f64) {
        self.push(time, STIMULUS, neuron);
    }

    /// A whole epoch's drive in one crossing: `neurons[k]` forced at `times[k]`, in the order given.
    fn stimulate_many(&mut self, neurons: Vec<u32>, times: Vec<f64>) -> PyResult<()> {
        if neurons.len() != times.len() {
            return Err(PyValueError::new_err("neurons and times must be the same length"));
        }
        for (&neuron, &time) in neurons.iter().zip(times.iter()) {
            self.push(time, STIMULUS, neuron);
        }
        Ok(())
    }

    /// The charged drive (AUTHORITY.md 5.4b): each arrival an EXTERNAL event delivering theta / `steps` to its input,
    /// a division on the threshold the input holds when it lands. Under exploration at the synapse only, and `steps`
    /// a whole number of at least 1 -- any integral type but a bool, which Python counts an int, as
    /// Network._drive_steps takes it; set it before `charge_many` or `push_events` hand the engine a charge. Refused
    /// (§12.2), with the clause and not a conversion error: anything else, and a count past the 64 bits this engine
    /// carries it in, rather than wrapped.
    fn set_charged_drive(&mut self, steps: &Bound<'_, PyAny>) -> PyResult<()> {
        let refused = || {
            let named = steps.repr().map(|r| r.to_string()).unwrap_or_else(|_| "a value with no repr".to_string());
            PyValueError::new_err(format!(
                "DRIVE_STEPS is a count of deliveries, a whole number of at least 1; got {named} (5.4b)"
            ))
        };
        if steps.is_instance_of::<PyBool>() {
            return Err(refused());
        }
        let steps: u64 = match steps.extract::<i64>() {
            Ok(n) if n >= 1 => n as u64,
            Ok(_) => return Err(refused()),
            Err(e) if e.is_instance_of::<PyOverflowError>(steps.py()) && steps.gt(0i64)? => match steps.extract::<u64>() {
                Ok(n) => n,
                Err(_) => {
                    return Err(PyValueError::new_err(format!(
                        "DRIVE_STEPS {steps} is past the 64-bit count this engine carries it in: refused rather than \
                         wrapped (5.4b, §12.2)"
                    )))
                }
            },
            Err(_) => return Err(refused()), // not integral, or a negative count past 64 bits
        };
        if !self.synaptic {
            return Err(PyValueError::new_err(
                "the charged drive runs under exploration at the synapse only: under the neuron rule a charged input \
                 would fire by its own hazard, not by the comparison its arithmetic is written on (5.4b)",
            ));
        }
        self.drive_steps = steps;
        Ok(())
    }

    /// A whole epoch's charged drive in one crossing, beside `stimulate_many`: `neurons[k]` delivered to at `times[k]`,
    /// in the order given -- each an EXTERNAL event, taken before its wave's signals (§3.5, 5.4b).
    fn charge_many(&mut self, neurons: Vec<u32>, times: Vec<f64>) -> PyResult<()> {
        if neurons.len() != times.len() {
            return Err(PyValueError::new_err("neurons and times must be the same length"));
        }
        if self.drive_steps == 0 {
            return Err(PyValueError::new_err(
                "a charge delivers theta / DRIVE_STEPS: call set_charged_drive(steps) first (5.4b)",
            ));
        }
        if neurons.iter().any(|&i| i as usize >= self.neurons) {
            return Err(PyValueError::new_err("a charge names a neuron the engine does not have"));
        }
        for (&neuron, &time) in neurons.iter().zip(times.iter()) {
            self.push(time, EXTERNAL, neuron);
        }
        Ok(())
    }

    /// Escape noise (§5.2): each neuron's decision width, `Network.set_delta`'s per-neuron values; all 0 turns it off.
    fn set_deltas(&mut self, deltas: Vec<f64>) -> PyResult<()> {
        if deltas.len() != self.neurons {
            return Err(PyValueError::new_err("one delta per neuron"));
        }
        if self.synaptic && deltas.iter().any(|&d| d != 0.0) {
            // a width that is not a number included, as Network.set_delta refuses it
            return Err(PyValueError::new_err(
                "a neuron width and exploration at the synapse together are refused: the system explores by one \
                 thing, and under exploration at the synapse every width is 0 (§6.13, §7.1)",
            ));
        }
        let on = deltas.iter().any(|&d| d > 0.0);
        if on && self.explore.is_none() {
            return Err(PyValueError::new_err(
                "escape noise needs Python's stream: call set_explore_state(rng.getstate()[1]) first (§5.2)",
            ));
        }
        self.delta = deltas;
        self.hazard = on;
        self.traced = on || self.centred || self.synaptic;
        Ok(())
    }

    /// What explores (AUTHORITY.md §7.1): "neuron", or "synapse" -- every synapse deciding at every wave on its
    /// source's potential (§7.5), at the rest hazard `rest` in [0, 1) -- an int or a float, never a bool, as the object
    /// engine takes it -- and in the `family` of §7.6, "loglinear" or "linear"; `trace` is §8.17's, "all" or
    /// "ventured"; `scales` is each neuron's kappa_i (§7.7), computed once in Python as §7.5's engine note says;
    /// `outputs` the output zone by place, as `Network.output_row` lays it out -- one read synapse per distinct output
    /// neuron, in the order it first appears, its draw after every synapse's (§3.8, §7.9). Every neuron keeps its trace.
    ///
    /// Refused (§12.2), the engine left as it was: a setting the file does not name; a positive width (§6.13); hebb
    /// (§8.3); no exploration stream, since the synapses draw at every wave whatever the rest hazard (§7.3); an edge
    /// layout that is not source-major, the draws being indexed by edge id in the order the object engine lays them
    /// out (§3.8); an inactive synapse (§3.8), a threshold at or below zero (§7.5) and a bored threshold (§7.5), as the
    /// object engine refuses them; and a switch of exploration on an engine carrying the other's bookkeeping (§12.9).
    #[allow(clippy::too_many_arguments)]
    fn set_exploration(
        &mut self,
        mode: &str,
        rest: &Bound<'_, PyAny>,
        family: &str,
        trace: &str,
        scales: Vec<f64>,
        outputs: Vec<u32>,
    ) -> PyResult<()> {
        let synaptic = match mode {
            "neuron" => false,
            "synapse" => true,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown exploration {mode:?}; choose from neuron, synapse (§7.1)"
                )))
            }
        };
        // h0 as Network._refuse_settings takes it: an int or a float, never a bool, which Python counts an int
        let number = !rest.is_instance_of::<PyBool>() && (rest.is_instance_of::<PyFloat>() || rest.is_instance_of::<PyLong>());
        let rest = match if number { rest.extract::<f64>().ok() } else { None } {
            Some(h0) if (0.0..1.0).contains(&h0) => h0,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "the rest hazard h0 must lie in [0, 1), got {}: at 1 the hazard is flat and scores nothing, above \
                     1 the family turns over, below 0 it is undefined (§7.6)",
                    rest.repr().map(|r| r.to_string()).unwrap_or_else(|_| "a value with no repr".to_string())
                )))
            }
        };
        let linear = match family {
            "loglinear" => false,
            "linear" => true,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown synapse hazard family {family:?}; choose from loglinear, linear (§7.6)"
                )))
            }
        };
        let trace_ventured = match trace {
            "all" => false,
            "ventured" => true,
            _ => return Err(PyValueError::new_err(format!("unknown trace {trace:?}; choose from all, ventured (§8.17)"))),
        };
        if scales.len() != self.neurons {
            return Err(PyValueError::new_err("one kappa_i per neuron"));
        }
        if outputs.iter().any(|&i| i as usize >= self.neurons) {
            return Err(PyValueError::new_err("an output names a neuron the engine does not have"));
        }
        if synaptic != self.synaptic {
            // E_j and the credit, or the gain, and the notes taken on them: nothing maps one onto the other (§12.9)
            let carried = if self.synaptic {
                self.gain.iter().any(|&g| g != 0.0)
            } else {
                self.expected.iter().chain(self.credit.iter()).any(|&e| e != 0.0)
            };
            if carried || self.noted.iter().any(|&b| b != 0.0) {
                return Err(PyValueError::new_err(
                    "this engine carries the other exploration's open bookkeeping, and nothing maps it across: a run is \
                     not switched between the two explorations (§12.9)",
                ));
            }
            if self.fired_at.iter().any(|&t| t > f64::NEG_INFINITY) {
                return Err(PyValueError::new_err(
                    "a neuron of this engine has spiked, and the exposure clock it holds runs from the refractory \
                     period's end under the neuron rule (§6.12) and from the spike under exploration at the synapse \
                     (§7.8): a run is not switched between the two explorations (§12.9)",
                ));
            }
            if !synaptic && self.drive_steps != 0 {
                return Err(PyValueError::new_err(
                    "the charged drive runs under exploration at the synapse only (5.4b)",
                ));
            }
        }
        if synaptic {
            if self.delta.iter().any(|&d| d != 0.0) {
                return Err(PyValueError::new_err(
                    "a neuron width and exploration at the synapse together are refused: the system explores by one \
                     thing, and under exploration at the synapse every width is 0 (§6.13, §7.1)",
                ));
            }
            if self.centred {
                return Err(PyValueError::new_err(
                    "hebb is refused under exploration at the synapse: the decisions are the synapses', and it has no \
                     neuron decision to centre (§8.3)",
                ));
            }
            if self.explore.is_none() {
                return Err(PyValueError::new_err(
                    "exploration at the synapse draws for every synapse at every wave, whatever its rest hazard, and \
                     needs Python's stream: call set_explore_state(rng.getstate()[1]) first (§7.3)",
                ));
            }
            if self.out_edges.iter().enumerate().any(|(k, &edge)| edge as usize != k) {
                return Err(PyValueError::new_err(
                    "the synapses' draws are taken in edge order and indexed by edge id (§3.8), which needs the edges \
                     numbered source-major, as fast.flatten numbers them; this layout is not",
                ));
            }
            if self.active.iter().any(|&a| !a) {
                return Err(PyValueError::new_err(
                    "an inactive connection keeps its draw under exploration at the synapse, and what it does with it \
                     is not specified: a network holding one is refused (§3.8)",
                ));
            }
            if self.threshold.iter().any(|&t| t <= 0.0) {
                return Err(PyValueError::new_err(
                    "a threshold at or below zero leaves u = clip(V, 0, theta) / theta undefined, and under exploration \
                     at the synapse a network holding one is refused (§7.5)",
                ));
            }
            if self.bored_after > 0.0 {
                return Err(PyValueError::new_err(
                    "a bored threshold moves the theta the synapses' u is read on, and no clause gives it a place under \
                     exploration at the synapse: refused (§7.5)",
                ));
            }
        }
        let edges = self.weight.len();
        let mut slot = vec![-1i64; self.neurons];
        let mut reads = 0usize;
        if synaptic {
            for &i in &outputs {
                // a neuron at two places of the output zone has one read synapse (§7.9)
                if slot[i as usize] < 0 {
                    slot[i as usize] = (edges + reads) as i64;
                    reads += 1;
                }
            }
        }
        self.synaptic = synaptic;
        self.rest = rest;
        // pow(h0, 1.0), the call synapse_expected makes wherever 1 - u is 1, made here once -- the same call on the same
        // arguments, kept from being folded to h0 so it stays the platform's pow, as CPython's float ** reaches it
        self.rest_power = rest.powf(std::hint::black_box(1.0));
        self.linear = linear;
        self.trace_ventured = synaptic && trace_ventured;
        self.synapse_scale = scales;
        self.read_slot = slot;
        self.synapse_draw = vec![1.0; if synaptic { edges + reads } else { 0 }];
        self.traced = self.hazard || self.centred || synaptic;
        Ok(())
    }

    /// "neuron" or "synapse" (§7.1).
    fn exploration(&self) -> &'static str {
        if self.synaptic {
            "synapse"
        } else {
            "neuron"
        }
    }

    /// The settings the engine holds, as §12.9 has a checkpoint carry them, for a harness to hold against the network's
    /// with ==: (exploration, h0, family, trace, each neuron's kappa_i, each neuron's read slot -- its draw's index,
    /// E + k in output order, or -1 for none (§3.8, §7.9) -- and DRIVE_STEPS, 0 with no charged drive set). The engine
    /// takes them when they are set and keeps them; the object engine re-reads h0 and the family at every wave and
    /// DRIVE_STEPS at every epoch's scheduling, so a setting moved on the network after the build parts the two here.
    #[allow(clippy::type_complexity)]
    fn exploration_settings(&self) -> (&'static str, f64, &'static str, &'static str, Vec<f64>, Vec<i64>, u64) {
        (
            self.exploration(),
            self.rest,
            if self.linear { "linear" } else { "loglinear" },
            if self.trace_ventured { "ventured" } else { "all" },
            self.synapse_scale.clone(),
            self.read_slot.clone(),
            self.drive_steps,
        )
    }

    /// Whether the traces, notes and scores are kept: a width, the centred rule, or exploration at the synapse.
    fn traced(&self) -> bool {
        self.traced
    }

    /// Each neuron's gain, G_j (§8.16), as a checkpoint carries it; nothing but 0 under the neuron rule (§12.9).
    fn set_gains(&mut self, gains: Vec<f64>) -> PyResult<()> {
        if gains.len() != self.neurons {
            return Err(PyValueError::new_err("one gain per neuron"));
        }
        if !self.synaptic && gains.iter().any(|&g| g != 0.0) {
            return Err(PyValueError::new_err(
                "a gain is exploration at the synapse's bookkeeping (§8.16), and nothing maps it onto the neuron rule's \
                 (§12.9)",
            ));
        }
        self.gain = gains;
        Ok(())
    }
    fn gains(&self) -> Vec<f64> {
        self.gain.clone()
    }

    /// Each neuron's read-synapse escapes this epoch (§7.9), 0 for any neuron without one; zeroed at the reset.
    fn set_read_counts(&mut self, counts: Vec<u64>) -> PyResult<()> {
        if counts.len() != self.neurons {
            return Err(PyValueError::new_err("one read count per neuron"));
        }
        if counts.iter().zip(&self.read_slot).any(|(&c, &slot)| c != 0 && slot < 0) {
            return Err(PyValueError::new_err(
                "only an output's read synapse counts escapes, under exploration at the synapse (§7.9)",
            ));
        }
        self.read_count = counts;
        Ok(())
    }
    fn read_counts(&self) -> Vec<u64> {
        self.read_count.clone()
    }

    /// §5.2: the count's scaling of every hazard, sqrt(ESCAPE_REFERENCE_COUNT / N), as `Network.set_delta` gave each neuron.
    fn set_escape_scales(&mut self, scales: Vec<f64>) -> PyResult<()> {
        if scales.len() != self.neurons {
            return Err(PyValueError::new_err("one escape scale per neuron"));
        }
        self.escape_scale = scales;
        Ok(())
    }

    /// The hebb eligibility (§6.7, the single-spike rule): every neuron charges its decisions against its own
    /// expectation, moved by `memory` a decision. Keeps the traces whether or not there is escape noise. Refused under
    /// exploration at the synapse, whose decisions are the synapses' (§8.3).
    fn set_centred(&mut self, on: bool, memory: f64) -> PyResult<()> {
        if on && self.synaptic {
            return Err(PyValueError::new_err(
                "hebb is refused under exploration at the synapse: the decisions are the synapses', and it has no \
                 neuron decision to centre (§8.3)",
            ));
        }
        self.centred = on;
        self.decision_memory = memory;
        self.traced = self.hazard || on || self.synaptic;
        Ok(())
    }

    /// Each neuron's p_hat_j (NaN for none yet), decisions to date and E_j, as a checkpoint carries them (§6.7).
    fn set_centred_state(&mut self, expectation: Vec<f64>, decisions: Vec<u64>, expected: Vec<f64>) -> PyResult<()> {
        if expectation.len() != self.neurons || decisions.len() != self.neurons || expected.len() != self.neurons {
            return Err(PyValueError::new_err("one expectation, decision count and expected count per neuron"));
        }
        self.expectation = expectation;
        self.decisions = decisions;
        self.expected = expected;
        Ok(())
    }
    fn expectations(&self) -> Vec<f64> {
        self.expectation.clone()
    }
    fn decision_counts(&self) -> Vec<u64> {
        self.decisions.clone()
    }
    fn expected_since_spike(&self) -> Vec<f64> {
        self.expected.clone()
    }
    /// Each synapse's note, B_ij (§6.7).
    fn set_notes(&mut self, notes: Vec<f64>) -> PyResult<()> {
        if notes.len() != self.weight.len() {
            return Err(PyValueError::new_err("one note per edge"));
        }
        self.noted = notes;
        Ok(())
    }
    fn notes(&self) -> Vec<f64> {
        self.noted.clone()
    }

    /// Under the evidence accumulator (§5.1): every open arrival's debit into its synapse's score (§6.7); the pay
    /// calls it first, and the arrivals stay open, their debit counting from now.
    fn settle_scores(&mut self) {
        if !self.tau.is_infinite() || !self.traced {
            return;
        }
        if self.synaptic {
            // §8.16, §1.9: the read posts x G - B, the net credit, and re-bases B = x G, leaving x and G as they are
            for edge in 0..self.weight.len() {
                let x = self.trace[edge];
                if x != 0.0 {
                    let owed = x * self.gain[self.edge_target[edge] as usize];
                    self.score[edge] += owed - self.noted[edge];
                    self.noted[edge] = owed;
                }
            }
            return;
        }
        for edge in 0..self.weight.len() {
            let x = self.trace[edge];
            if x != 0.0 {
                let e = x * self.expected[self.edge_target[edge] as usize];
                self.score[edge] -= e - self.noted[edge];
                self.noted[edge] = e;
            }
        }
    }

    /// §6.7 with the hazard eligibility: `reinforce_scores`, which needs something that draws here -- escape noise, or
    /// exploration at the synapse, whose decisions post the hazard's row (§8.3, §8.16).
    fn reinforce_hazard(&mut self, advantage: f64, lr: f64) -> PyResult<usize> {
        if !self.hazard && !self.synaptic {
            return Err(PyValueError::new_err(
                "the hazard eligibility needs a decision that is a draw: set_deltas with a positive width, or \
                 set_exploration(\"synapse\", ...), first (§8.3)",
            ));
        }
        self.reinforce_scores(advantage, lr)
    }

    /// §6.7, the single-spike rule under the hazard and the hebb eligibility alike: every synapse into an unforced
    /// neuron moves by `lr * advantage * score`, the score being what the neuron's decisions charged on that synapse's
    /// trace this epoch, settled first under the evidence accumulator.
    fn reinforce_scores(&mut self, advantage: f64, lr: f64) -> PyResult<usize> {
        if !self.traced {
            return Err(PyValueError::new_err(
                "the scores need a trace: escape noise (set_deltas), the centred rule (set_centred) or exploration at \
                 the synapse (set_exploration) first (§6.7)",
            ));
        }
        self.settle_scores();
        if advantage == 0.0 {
            return Ok(0);
        }
        let step = lr * advantage;
        let mut changed = 0;
        for edge in 0..self.weight.len() {
            if self.score[edge] == 0.0 {
                continue;
            }
            let target = self.edge_target[edge] as usize;
            if self.forced[target] {
                continue; // a forced input: its firing was not the network's doing
            }
            let w = self.weight[edge] + step * self.score[edge];
            self.weight[edge] = w.clamp(self.weight_low, self.weight_high);
            changed += 1;
        }
        Ok(changed)
    }

    fn scores(&self) -> Vec<f64> {
        self.score.clone()
    }
    fn traces(&self) -> Vec<f64> {
        self.trace.clone()
    }
    fn deltas(&self) -> Vec<f64> {
        self.delta.clone()
    }

    /// An external input of a given `amount` is not supported yet; stimuli, signals and the charged drive's
    /// deliveries (`charge_many`) are.
    fn external(&mut self, _neuron: u32, _amount: f64, _time: f64) -> PyResult<()> {
        Err(PyValueError::new_err(
            "external amounts are not implemented here; use the object engine",
        ))
    }

    /// Process every wave due before `until`. Returns (time, fired neuron indices) per wave.
    fn run(&mut self, until: f64) -> Vec<(f64, Vec<u32>)> {
        let mut waves: Vec<(f64, Vec<u32>)> = Vec::new();
        let mut batch: Vec<Event> = Vec::new();
        let mut touched: Vec<u32> = Vec::new();
        let mut forced: Vec<u32> = Vec::new();
        // the run's settings, which nothing inside it moves, read once
        let (traced, accumulating, synaptic) = (self.traced, self.tau.is_infinite(), self.synaptic);
        let relayed_traced = traced && !self.trace_ventured; // under TRACE ventured a relayed arrival moves no trace (§8.17)

        while let Some(first) = self.heap.peek().map(|e| e.time) {
            if !before(first, until) {
                break;
            }
            let limit = first + slack(first);
            batch.clear();
            let mut anchored: Option<f64> = None;
            let mut charges = false;
            while let Some(next) = self.heap.peek().map(|e| e.time) {
                if next > limit {
                    break;
                }
                let event = self.heap.pop().expect("peeked");
                // an input's exact time anchors the wave it joins, so chains never drift
                if event.kind != SIGNAL && anchored.is_none() {
                    anchored = Some(event.time);
                }
                charges |= event.kind == EXTERNAL;
                batch.push(event);
            }
            let time = anchored.unwrap_or(first);

            self.wave_no += 1;
            // The mark must never repeat across epochs: reset() zeroes wave_no, and a neuron last touched in
            // wave k of an earlier epoch would otherwise pass for touched in wave k of this one and never be
            // asked whether it fired (found September 14, 2026, on goo with no direct projection, where an
            // output can go a whole epoch untouched).
            self.marks += 1;
            let mark = self.marks;
            touched.clear();
            forced.clear();

            // the charged drive's deliveries first, in the order the wave holds them (§3.5, 5.4b): theta / DRIVE_STEPS,
            // a division on the threshold the input holds now; each sets 5.8's driven mark, taken or dropped at a
            // refractory input. They come along no synapse, so no trace, note, synapse stamp (last_signal, §1.7) or
            // score moves; a delivery taken marks the input touched this wave, as a signal marks its target
            if charges {
                let steps = self.drive_steps as f64;
                for k in 0..batch.len() {
                    if batch[k].kind == EXTERNAL {
                        let i = batch[k].payload as usize;
                        let amount = self.threshold[i] / steps;
                        self.forced[i] = true;
                        if self.receive(i, amount, time) && self.stamp[i] != mark {
                            self.stamp[i] = mark;
                            touched.push(i as u32);
                        }
                    }
                }
            }

            // every signal of the wave is delivered before any neuron of it decides to fire
            for event in &batch {
                match event.kind {
                    SIGNAL => {
                        let edge = event.payload as usize;
                        let target = self.edge_target[edge] as usize;
                        // recorded whether or not it lands: §6.7 judges a connection by its last delivery
                        self.delivered_wave[edge] = self.wave_no as i64;
                        if self.receive(target, self.weight[edge], time) {
                            self.last_signal[edge] = time;
                            // under TRACE ventured a relayed arrival is, for the synapse's learning, not there: no
                            // count, no note, no decay, no moment moved (§8.17)
                            if relayed_traced || (traced && event.ventured) {
                                // what this synapse now has in its target's potential (§6.7). Under the evidence
                                // accumulator (§5.1, TAU infinite) nothing leaks, the trace is the count of arrivals
                                // -- the decay, exp(-0) = 1, is not evaluated -- and the arrival notes the target's
                                // expected spikes so far, so its debit counts from here; under exploration at the
                                // synapse it notes the gain as it stands before this wave's decisions post (§8.16)
                                if accumulating {
                                    self.trace[edge] += 1.0;
                                    self.noted[edge] += if synaptic { self.gain[target] } else { self.expected[target] };
                                } else {
                                    self.trace[edge] =
                                        self.trace[edge] * (-(time - self.trace_at[edge]) / self.tau).exp() + 1.0;
                                }
                                self.trace_at[edge] = time;
                            }
                            if self.stamp[target] != mark {
                                self.stamp[target] = mark;
                                touched.push(target as u32);
                            }
                        }
                    }
                    STIMULUS => forced.push(event.payload),
                    _ => {} // EXTERNAL: the charges, taken above
                }
            }

            // the floor applies to the wave's total, whatever order it arrived in
            for &i in &touched {
                let i = i as usize;
                if self.potential[i] < self.floor[i] {
                    self.potential[i] = self.floor[i];
                    if traced {
                        self.clear_arrivals(i); // the floor bit: the weights are not in it (§6.7)
                    }
                }
            }

            if self.hazard {
                self.hazard_draws(); // §7.3: one uniform per neuron, in neuron order
            } else if synaptic {
                self.synapse_draws(); // §3.8: one per synapse in edge order, then one per read synapse, whatever fires
            }

            let mut fired: Vec<u32> = Vec::new();
            // a stimulus listed twice fires once: the first spike makes it refractory
            for k in 0..forced.len() {
                let i = forced[k] as usize;
                if !self.refractory_at(i, time) {
                    self.fire(i, time, &mut fired);
                    self.forced[i] = true;
                }
            }
            for k in 0..touched.len() {
                let i = touched[k] as usize;
                if self.decide(i, time) {
                    self.fire(i, time, &mut fired);
                }
            }
            // everyone else: a threshold that has fallen with its silence (§5.4), or the hazard (§5.2)
            for i in 0..self.neurons {
                if self.stamp[i] != mark && self.decide(i, time) {
                    self.fire(i, time, &mut fired);
                }
            }
            if synaptic {
                self.decide_synapses(time); // §7.5: after the spikes, and its escapes pushed after theirs (§3.6)
            }

            if self.quash_rate != 0.0 {
                self.quash(time, &fired);
            }
            waves.push((time, fired));
        }
        waves
    }

    /// How many events are still in flight (they outlive the epoch).
    fn pending(&self) -> usize {
        self.heap.len()
    }

    // --- reading the state back -----------------------------------------------------
    fn weights(&self) -> Vec<f64> {
        self.weight.clone()
    }
    fn potentials(&self) -> Vec<f64> {
        self.potential.clone()
    }
    fn spike_counts(&self) -> Vec<u64> {
        self.spikes.clone()
    }
    /// How many times each neuron has fired since the epoch began: what the count read thresholds (§4.3).
    fn epoch_spike_counts(&self) -> Vec<u64> {
        self.spikes.iter().zip(&self.spikes_at_reset).map(|(s, r)| s - r).collect()
    }
    fn fired_times(&self) -> Vec<f64> {
        self.fired_at.clone()
    }
    fn previous_fired_times(&self) -> Vec<f64> {
        self.previous_fired_at.clone()
    }
    fn firing_rates(&self, now: f64) -> Vec<f64> {
        (0..self.neurons)
            .map(|i| {
                if self.rate_level[i] == 0.0 {
                    0.0
                } else {
                    self.rate_level[i] * (-(now - self.rate_at[i]) / self.rate_tau).exp()
                }
            })
            .collect()
    }
    fn forced_flags(&self) -> Vec<bool> {
        self.forced.clone()
    }
    fn fired_this_epoch(&self) -> Vec<bool> {
        self.fired_wave.iter().map(|&w| w >= 0).collect()
    }

    // --- writing it, so a teacher can pay at the read -------------------------------
    fn set_weights(&mut self, weights: Vec<f64>) -> PyResult<()> {
        if weights.len() != self.weight.len() {
            return Err(PyValueError::new_err("wrong number of weights"));
        }
        self.weight = weights;
        Ok(())
    }
    fn set_potentials(&mut self, potentials: Vec<f64>) -> PyResult<()> {
        // §12.11: a resumed engine takes the potentials the run had reached; reset(false) keeps them across epochs
        if potentials.len() != self.neurons {
            return Err(PyValueError::new_err("wrong number of potentials"));
        }
        self.potential = potentials;
        Ok(())
    }
    fn set_traces(&mut self, traces: Vec<f64>) -> PyResult<()> {
        // §12.11: and the per-synapse traces of §6.7, in edge order; noted is rebuilt from them at the next reset
        if traces.len() != self.trace.len() {
            return Err(PyValueError::new_err("wrong number of traces"));
        }
        self.trace = traces;
        Ok(())
    }
    fn exposed_since(&self) -> Vec<f64> {
        self.exposed_since.clone()
    }
    fn set_fired_at(&mut self, fired_at: Vec<f64>) -> PyResult<()> {
        // §12.11: the absolute times the refractory test and the hazard's elapsed run from persist across epochs
        if fired_at.len() != self.neurons {
            return Err(PyValueError::new_err("wrong number of fired_at"));
        }
        self.fired_at = fired_at;
        Ok(())
    }
    fn set_previous_fired_at(&mut self, previous: Vec<f64>) -> PyResult<()> {
        if previous.len() != self.neurons {
            return Err(PyValueError::new_err("wrong number of previous_fired_at"));
        }
        self.previous_fired_at = previous;
        Ok(())
    }
    fn set_exposed_since(&mut self, since: Vec<f64>) -> PyResult<()> {
        if since.len() != self.neurons {
            return Err(PyValueError::new_err("wrong number of exposed_since"));
        }
        self.exposed_since = since;
        Ok(())
    }
    /// §12.11: the events still in the queue, in the order the heap would deliver them -- signals in flight across an
    /// epoch boundary, which reset(false) keeps. (time, kind, payload); a resume pushes them back in this order. With
    /// `marks`, (time, kind, payload, ventured), the mark a checkpoint carries with each signal (§7.9, §12.9); without,
    /// a queue holding a ventured signal is refused rather than handed over as relayed.
    #[pyo3(signature = (marks=false))]
    fn pending_events(&self, py: Python<'_>, marks: bool) -> PyResult<PyObject> {
        let mut events: Vec<(f64, u8, u64, u32, bool)> =
            self.heap.iter().map(|e| (e.time, e.kind, e.seq, e.payload, e.ventured)).collect();
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        if marks {
            let out: Vec<(f64, u8, u32, bool)> = events.into_iter().map(|(t, k, _, p, v)| (t, k, p, v)).collect();
            return Ok(out.into_py(py));
        }
        if events.iter().any(|e| e.4) {
            return Err(PyValueError::new_err(
                "a signal in flight carries the ventured mark (§7.9), which (time, kind, payload) would drop: ask for \
                 pending_events(marks=True) (§12.9)",
            ));
        }
        let out: Vec<(f64, u8, u32)> = events.into_iter().map(|(t, k, _, p, _)| (t, k, p)).collect();
        Ok(out.into_py(py))
    }
    /// The queue back, in the order given, with fresh seqs: `ventured` one mark per event, or none for all relayed.
    /// Refused (§12.2): a kind the engine does not know; an index it does not have; a charge (EXTERNAL) with no
    /// charged drive set (5.4b); a ventured mark on anything but a signal, or on an engine under the neuron rule,
    /// which has no escapes and resumes no run saved under the other exploration (§7.9, §12.9).
    #[pyo3(signature = (times, kinds, payloads, ventured=None))]
    fn push_events(&mut self, times: Vec<f64>, kinds: Vec<u8>, payloads: Vec<u32>, ventured: Option<Vec<bool>>) -> PyResult<()> {
        if times.len() != kinds.len() || times.len() != payloads.len() {
            return Err(PyValueError::new_err("one time, kind and payload per event"));
        }
        let ventured = ventured.unwrap_or_else(|| vec![false; times.len()]);
        if ventured.len() != times.len() {
            return Err(PyValueError::new_err("one ventured mark per event"));
        }
        for k in 0..times.len() {
            let (kind, payload) = (kinds[k], payloads[k] as usize);
            let known = match kind {
                SIGNAL => payload < self.weight.len(),
                STIMULUS | EXTERNAL => payload < self.neurons,
                _ => return Err(PyValueError::new_err(format!("unknown event kind {kind}; the engine knows 0, 1 and 2"))),
            };
            if !known {
                return Err(PyValueError::new_err(format!("event {k} names an edge or neuron the engine does not have")));
            }
            if kind == EXTERNAL && self.drive_steps == 0 {
                return Err(PyValueError::new_err(
                    "an EXTERNAL event is a charge of theta / DRIVE_STEPS: call set_charged_drive(steps) first (5.4b)",
                ));
            }
            if ventured[k] && (kind != SIGNAL || !self.synaptic) {
                return Err(PyValueError::new_err(
                    "a ventured mark rides on a signal an escape sent (§7.9), under exploration at the synapse only; a \
                     run is not resumed under the other exploration (§12.9)",
                ));
            }
        }
        for k in 0..times.len() {
            self.push_marked(times[k], kinds[k], payloads[k], ventured[k]); // fresh seqs, same relative order
        }
        Ok(())
    }
    fn last_signals(&self) -> Vec<f64> {
        self.last_signal.clone()
    }
    fn set_last_signals(&mut self, last: Vec<f64>) -> PyResult<()> {
        if last.len() != self.last_signal.len() {
            return Err(PyValueError::new_err("wrong number of last_signal"));
        }
        self.last_signal = last;
        Ok(())
    }
    /// §12.11 under the leak (§2.3): the times the lazy decays run from -- the potential's and each trace's.
    fn last_updates(&self) -> Vec<f64> {
        self.last_update.clone()
    }
    fn set_last_updates(&mut self, at: Vec<f64>) -> PyResult<()> {
        if at.len() != self.neurons {
            return Err(PyValueError::new_err("wrong number of last_update"));
        }
        self.last_update = at;
        Ok(())
    }
    fn trace_ats(&self) -> Vec<f64> {
        self.trace_at.clone()
    }
    fn set_trace_ats(&mut self, at: Vec<f64>) -> PyResult<()> {
        if at.len() != self.trace_at.len() {
            return Err(PyValueError::new_err("wrong number of trace_at"));
        }
        self.trace_at = at;
        Ok(())
    }
    /// The spike counts to date, so a resumed engine's cumulative counts are the run's; the epoch's count, which the
    /// read thresholds, is the difference from the epoch's start either way.
    fn set_spike_counts(&mut self, spikes: Vec<u64>) -> PyResult<()> {
        if spikes.len() != self.neurons {
            return Err(PyValueError::new_err("wrong number of spike counts"));
        }
        self.spikes_at_reset.copy_from_slice(&spikes);
        self.spikes = spikes;
        Ok(())
    }
    fn set_thresholds(&mut self, thresholds: Vec<f64>) -> PyResult<()> {
        if thresholds.len() != self.neurons {
            return Err(PyValueError::new_err("wrong number of thresholds"));
        }
        if self.synaptic && thresholds.iter().any(|&t| t <= 0.0) {
            return Err(PyValueError::new_err(
                "a threshold moved to or below zero leaves u = clip(V, 0, theta) / theta undefined, and under \
                 exploration at the synapse it is refused (§7.5)",
            ));
        }
        self.threshold = thresholds;
        Ok(())
    }
    /// The thresholds the engine reads -- the charge's theta (5.4b), u's (§7.5), the comparison's -- as the Teacher's
    /// moves last set them.
    fn thresholds(&self) -> Vec<f64> {
        self.threshold.clone()
    }
}

impl Engine {
    #[inline]
    fn push(&mut self, time: f64, kind: u8, payload: u32) {
        self.push_marked(time, kind, payload, false);
    }

    /// Push an event with its ventured mark (§7.9): true only for a signal an escape sent.
    #[inline]
    fn push_marked(&mut self, time: f64, kind: u8, payload: u32, ventured: bool) {
        self.seq += 1;
        let seq = self.seq;
        self.heap.push(Event { time, kind, seq, payload, ventured });
    }

    /// neuron.py: the potential decayed to `now`, without applying it.
    #[inline]
    fn potential_at(&self, i: usize, now: f64) -> f64 {
        let elapsed = now - self.last_update[i];
        if elapsed <= 0.0 || self.tau.is_infinite() {
            return self.potential[i];
        }
        self.potential[i] * (-elapsed / self.tau).exp()
    }


    /// The traces into neuron `i` are zero: its potential was reset by a spike or is the floor (§6.7).
    #[inline]
    fn clear_traces(&mut self, i: usize) {
        let (lo, hi) = (self.in_start[i] as usize, self.in_start[i + 1] as usize);
        for k in lo..hi {
            self.trace[self.in_edges[k] as usize] = 0.0;
        }
    }

    /// Under the evidence accumulator (§5.1): close every open arrival on neuron `i`'s incoming synapses (§6.7). Each
    /// score takes the credit for its count and the debit accrued since each arrival, the spikes expected of `i` from
    /// then until now; then its count and its note are cleared. A spike settles with its decision's credit; the floor,
    /// a forced spike and a discharge with none. Under exploration at the synapse the credit is inside the gain, a net
    /// credit where E_j is a debit, so each open arrival settles x G - B, and G restarts whether or not one was open
    /// (§8.16).
    fn settle_arrivals(&mut self, i: usize, credit: f64) {
        if self.synaptic {
            self.settle_gain(i);
            return;
        }
        let (lo, hi) = (self.in_start[i] as usize, self.in_start[i + 1] as usize);
        let expected = self.expected[i];
        for k in lo..hi {
            let edge = self.in_edges[k] as usize;
            let x = self.trace[edge];
            if x != 0.0 {
                self.score[edge] += credit * x - (x * expected - self.noted[edge]);
                self.trace[edge] = 0.0;
                self.noted[edge] = 0.0;
            }
        }
    }

    /// settle_arrivals under exploration at the synapse (§8.16): each open arrival on neuron `i`'s incoming synapses
    /// settles x G - B and is cleared, and G restarts at 0, whether or not an arrival was open.
    #[inline(never)]
    fn settle_gain(&mut self, i: usize) {
        let (lo, hi) = (self.in_start[i] as usize, self.in_start[i + 1] as usize);
        let gain = self.gain[i];
        for k in lo..hi {
            let edge = self.in_edges[k] as usize;
            let x = self.trace[edge];
            if x != 0.0 {
                self.score[edge] += x * gain - self.noted[edge];
                self.trace[edge] = 0.0;
                self.noted[edge] = 0.0;
            }
        }
        self.gain[i] = 0.0;
    }

    /// Nothing any synapse delivered is in the potential: the floor bit, or it was discharged (§6.7). Under the
    /// evidence accumulator the open arrivals close without credit, their debit so far settled; under the leak the
    /// traces are zeroed. Under exploration at the synapse the gain restarts either way (§8.16).
    fn clear_arrivals(&mut self, i: usize) {
        if self.tau.is_infinite() {
            self.settle_arrivals(i, 0.0);
        } else {
            self.clear_traces(i);
            self.gain[i] = 0.0;
        }
    }

    /// exploration.hazard_draws: one uniform per neuron from Python's stream, in neuron order (§5.2).
    fn hazard_draws(&mut self) {
        let n = self.neurons;
        let mut draws = Vec::with_capacity(n);
        match self.explore.as_mut() {
            Some(rng) => {
                for _ in 0..n {
                    draws.push(rng.random());
                }
            }
            None => return, // set_deltas refuses a width without a stream, so this cannot happen
        }
        self.draw.copy_from_slice(&draws);
    }

    /// Network._explore_synapses: this wave's E + O uniforms from Python's stream, every synapse's in edge order and
    /// then every read synapse's in output order, all of them whatever fires (§3.8), into the buffer set_exploration
    /// sized once.
    #[inline(never)] // kept out of run(): the neuron rule's wave loop is laid out as it was
    fn synapse_draws(&mut self) {
        if let Some(rng) = self.explore.as_mut() {
            for draw in self.synapse_draw.iter_mut() {
                *draw = rng.random();
            }
        } // set_exploration refuses the mode without a stream, so there is always one here
    }

    /// Network._decide_synapses, after the fire phase: every synapse of every source that did not spike this wave
    /// decides (AUTHORITY.md §7.5), refractory or not (§7.8), touched or not. A source that spiked transmitted on every
    /// synapse and its draws go unused (§3.8). The others decide together on one m and one P = -expm1(-m), each
    /// escaping iff its uniform is strictly below P, an output's read synapse with them, its escape counted toward the
    /// output's read at this wave (§7.9). An escape is pushed as a ventured signal one hop later, in edge order and
    /// after the wave's spike signals (§3.6), and leaves the source's potential where it was; the source's one
    /// exposure clock is brought to now, F_i = 0 included.
    ///
    /// Where the source's potential is above zero and m > 0 the wave posts §8.16's entry for it, a c - (F - a) m with
    /// c = m e^-m / (1 - e^-m), times rho~ = (1 - h0) / h(u) under the linear family: into the gain under the
    /// evidence accumulator, and under the leak by one walk over the source's fan-in (§8.12).
    #[inline(never)]
    fn decide_synapses(&mut self, now: f64) {
        let wave = self.wave_no as i64;
        let (rest, rest_power, linear, hop) = (self.rest, self.rest_power, self.linear, self.hop);
        for i in 0..self.neurons {
            if self.fired_wave[i] == wave {
                continue; // it spiked: its synapses transmitted, and decide nothing (§6.13)
            }
            let (lo, hi) = (self.out_start[i] as usize, self.out_start[i + 1] as usize);
            let slot = self.read_slot[i];
            let synapses = (hi - lo) as u64 + (slot >= 0) as u64; // F_i, the read synapse among them
            if synapses == 0 {
                self.exposed_since[i] = now; // one clock per source, brought to now though nothing decides on it
                continue;
            }
            let potential = self.potential_at(i, now);
            let (m, h) = synapse_expected(potential, self.threshold[i], now - self.exposed_since[i], hop,
                                          self.synapse_scale[i], rest, rest_power, linear);
            let chance = -(-m).exp_m1();
            let mut escapes: u64 = 0;
            for k in lo..hi {
                let edge = self.out_edges[k]; // the draw is the edge's, by id (§3.8)
                if self.synapse_draw[edge as usize] < chance {
                    self.push_marked(now + hop, SIGNAL, edge, true);
                    escapes += 1;
                }
            }
            if slot >= 0 && self.synapse_draw[slot as usize] < chance {
                self.read_count[i] += 1; // it delivers nothing, so nothing of it is in flight (§7.9)
                escapes += 1;
            }
            self.exposed_since[i] = now;
            if m > 0.0 && potential > 0.0 {
                // §8.16: posted only while V_i > 0, and c only where m > 0
                let mut entry = escapes as f64 * (m * (-m).exp() / -(-m).exp_m1()) - (synapses - escapes) as f64 * m;
                if linear {
                    entry = (1.0 - rest) / h * entry; // rho~, the log-derivative the linear family leaves unfolded
                }
                if self.tau.is_infinite() {
                    self.gain[i] += entry; // after this wave's arrivals have noted (§8.16)
                } else {
                    // the leak (§8.12): the walk, once a wave per posting source
                    let (lo, hi) = (self.in_start[i] as usize, self.in_start[i + 1] as usize);
                    for k in lo..hi {
                        let edge = self.in_edges[k] as usize;
                        if self.trace[edge] != 0.0 {
                            self.score[edge] += entry * self.trace[edge] * (-(now - self.trace_at[edge]) / self.tau).exp();
                        }
                    }
                }
            }
        }
    }

    /// neuron.py's decide: the threshold when the width is 0, else the escape-noise draw (§5.2). The decision charges
    /// the eligibility of every incoming synapse (§6.7, the single-spike rule): each score moves by (c - q) * trace,
    /// the credit c and the expectation q of the decision -- the hazard's m e^-m / (1 - e^-m) and 0 when it fires,
    /// 0 and m when it does not; the centred rule's outcome, 1 or 0, and the neuron's own expectation of it, moved
    /// after the charge. Under the evidence accumulator the charge is lazy: q goes onto the neuron's expected count
    /// and a spike's credit is left for fire() to settle per arrival.
    fn decide(&mut self, i: usize, now: f64) -> bool {
        if self.refractory_at(i, now) {
            return false;
        }
        let mut m = 0.0;
        let fired = if self.delta[i] <= 0.0 {
            self.can_fire(i, now)
        } else {
            let s = self.potential_at(i, now) - self.threshold_at(i, now);
            let mut elapsed = now - self.exposed_since[i];
            if elapsed < 0.0 {
                elapsed = 0.0;
            }
            m = (elapsed / self.hop * self.escape_scale[i] * (s / self.delta[i]).exp()).min(1e3);
            self.exposed_since[i] = now;
            self.draw[i] < -(-m).exp_m1()
        };
        let credit;
        let q;
        if self.centred {
            let y = if fired { 1.0 } else { 0.0 };
            let p = self.expectation[i];
            self.decisions[i] += 1;
            if p.is_nan() {
                self.expectation[i] = y; // the first decision sets the expectation and charges nothing
                return fired;
            }
            credit = y;
            q = p;
            self.expectation[i] = p + f64::max(self.decision_memory, 1.0 / self.decisions[i] as f64) * (y - p);
        } else if m > 0.0 {
            if fired {
                credit = m * (-m).exp() / -(-m).exp_m1(); // m e^-m / (1 - e^-m), finite at any m
                q = 0.0;
            } else {
                credit = 0.0;
                q = m;
            }
        } else {
            return fired;
        }
        if self.tau.is_infinite() {
            // the evidence accumulator (§5.1): the debit settles per arrival, the credit at the spike
            self.expected[i] += q;
            if fired {
                self.credit[i] = credit;
            }
        } else {
            let e = credit - q;
            let (lo, hi) = (self.in_start[i] as usize, self.in_start[i + 1] as usize);
            for k in lo..hi {
                let edge = self.in_edges[k] as usize;
                if self.trace[edge] != 0.0 {
                    self.score[edge] += e * self.trace[edge] * (-(now - self.trace_at[edge]) / self.tau).exp();
                }
            }
        }
        fired
    }

    /// neuron.py: bring the potential up to `now`. Lazy, so call it on arrival.
    #[inline]
    fn leak(&mut self, i: usize, now: f64) {
        let elapsed = now - self.last_update[i];
        if elapsed > 0.0 {
            if !self.tau.is_infinite() {
                self.potential[i] *= (-elapsed / self.tau).exp();
            }
            self.last_update[i] = now;
        }
    }

    /// neuron.py: fired within the refractory period before `now`, the clock's slack allowed.
    #[inline]
    fn refractory_at(&self, i: usize, now: f64) -> bool {
        self.fired_at[i] > f64::NEG_INFINITY
            && now + slack(now) < self.fired_at[i] + self.refractory
    }

    /// neuron.py: the threshold a bored neuron faces at `now` (§5.4).
    #[inline]
    fn threshold_at(&self, i: usize, now: f64) -> f64 {
        if self.bored_after <= 0.0 {
            return self.threshold[i];
        }
        let since = if self.fired_at[i] > f64::NEG_INFINITY { self.fired_at[i] } else { 0.0 };
        self.threshold[i] - self.threshold[i] * (now - since) / self.bored_after
    }

    #[inline]
    fn can_fire(&self, i: usize, now: f64) -> bool {
        if self.refractory_at(i, now) {
            return false;
        }
        self.potential_at(i, now) >= self.threshold_at(i, now)
    }

    /// neuron.py: leak first, then integrate, unless refractory. Returns whether it was taken in.
    #[inline]
    fn receive(&mut self, i: usize, amount: f64, now: f64) -> bool {
        if self.refractory_at(i, now) {
            return false;
        }
        self.leak(i, now);
        self.potential[i] += amount;
        true
    }

    /// Spike, and schedule this neuron's outgoing signals one hop later.
    fn fire(&mut self, i: usize, now: f64, fired: &mut Vec<u32>) {
        self.previous_fired_at[i] = self.fired_at[i];
        self.fired_at[i] = now;
        self.last_update[i] = now;
        self.potential[i] = 0.0; // the spike resets the potential
        self.exposed_since[i] = if self.synaptic {
            now // the synapses' exposure runs from the spike, and nothing suspends it (§7.5, §7.8)
        } else {
            now + self.refractory // the hazard resumes when the refractory period ends (§5.2)
        };
        if self.traced {
            if self.tau.is_infinite() {
                let credit = self.credit[i];
                self.settle_arrivals(i, credit); // the evidence accumulator: the spike credits and closes every open arrival (§6.7)
            } else {
                self.clear_traces(i); // nothing any synapse delivered is still in the potential (§6.7)
            }
            self.expected[i] = 0.0;
            self.credit[i] = 0.0;
            self.gain[i] = 0.0; // G restarts at the spike (§8.16)
        }
        self.spikes[i] += 1;
        self.fired_wave[i] = self.wave_no as i64;
        // the rate trace: one spike's worth on, decaying with rate_tau (§4.3)
        let level = if self.rate_level[i] == 0.0 {
            0.0
        } else {
            self.rate_level[i] * (-(now - self.rate_at[i]) / self.rate_tau).exp()
        };
        self.rate_level[i] = level + 1000.0 / self.rate_tau;
        self.rate_at[i] = now;

        let (lo, hi) = (self.out_start[i] as usize, self.out_start[i + 1] as usize);
        for k in lo..hi {
            let edge = self.out_edges[k];
            if self.active[edge as usize] {
                self.push(now + self.hop, SIGNAL, edge);
            }
        }
        fired.push(i as u32);
    }

    /// §6.11: a refire is a cycle, so weaken the synapses that carried it.
    fn quash(&mut self, time: f64, fired: &[u32]) {
        for &n in fired {
            let i = n as usize;
            let previous = self.previous_fired_at[i];
            if previous == f64::NEG_INFINITY {
                continue; // its first spike: no cycle to quash
            }
            let factor = self.quash_rate * (-self.quash_k * (time - previous)).exp();
            if factor == 0.0 {
                continue;
            }
            let (lo, hi) = (self.in_start[i] as usize, self.in_start[i + 1] as usize);
            for k in lo..hi {
                let edge = self.in_edges[k] as usize;
                if self.active[edge] && self.last_signal[edge] > previous {
                    let w = self.weight[edge] * (1.0 - factor);
                    self.weight[edge] = w.clamp(self.weight_low, self.weight_high);
                }
            }
        }
    }
}

/// neuron.py's synapse_expected, in §7.5's engine note's forms: u = min(max(V, 0), theta) / theta -- Python's max and
/// min, each keeping its first argument unless the second is strictly past it -- then h = h0 ** (1 - u) under the
/// loglinear family (the platform's pow, as CPython's float ** calls it for a positive h0 and a nonzero exponent) or
/// h0 + (1 - h0) * u under the linear; m = dt / hop, times kappa_i, times h, left to right as §6.5's is, then capped at
/// 1e3, dt never negative. `rest_power` is pow(h0, 1.0), taken where 1 - u is exactly 1 -- the same call made once,
/// since a source at or below zero potential, most of them, asks for it at every wave. Returns (m, h).
#[inline]
#[allow(clippy::too_many_arguments)]
fn synapse_expected(potential: f64, threshold: f64, elapsed: f64, hop: f64, scale: f64, rest: f64, rest_power: f64,
                    linear: bool) -> (f64, f64) {
    let clipped = if 0.0 > potential { 0.0 } else { potential };
    let clipped = if threshold < clipped { threshold } else { clipped };
    let u = clipped / threshold;
    let h = if linear {
        rest + (1.0 - rest) * u
    } else {
        let exponent = 1.0 - u;
        if exponent == 1.0 { rest_power } else { rest.powf(exponent) }
    };
    let elapsed = if elapsed < 0.0 { 0.0 } else { elapsed };
    let m = elapsed / hop * scale * h;
    (if 1e3 < m { 1e3 } else { m }, h)
}

/// The engine's synapse hazard, for the test that holds it to neuron.py's (§7.5's engine note): (m, h) for a source at
/// `potential` and `threshold`, `elapsed` since its synapses last decided, one `hop`, kappa_i `scale`, rest hazard
/// `rest`, in the `family` named.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn synapse_hazard(potential: f64, threshold: f64, elapsed: f64, hop: f64, scale: f64, rest: f64, family: &str) -> PyResult<(f64, f64)> {
    let linear = match family {
        "loglinear" => false,
        "linear" => true,
        _ => return Err(PyValueError::new_err(format!("unknown synapse hazard family {family:?}; choose from loglinear, linear (§7.6)"))),
    };
    // pow(h0, 1.0) made here as set_exploration makes it, so the test holds the loop's u = 0 case to neuron.py's too
    Ok(synapse_expected(potential, threshold, elapsed, hop, scale, rest, rest.powf(std::hint::black_box(1.0)), linear))
}

/// Offsets into a bucketed adjacency: `starts[i]..starts[i + 1]` is neuron i's slice.
fn counts_to_starts(owner: &[u32], neurons: usize) -> Vec<u32> {
    let mut starts = vec![0u32; neurons + 1];
    for &o in owner {
        starts[o as usize + 1] += 1;
    }
    for i in 0..neurons {
        starts[i + 1] += starts[i];
    }
    starts
}

/// The edge indices owned by each neuron, in edge order within a neuron.
fn bucket(owner: &[u32], starts: &[u32], neurons: usize) -> Vec<u32> {
    let mut cursor: Vec<u32> = starts[..neurons].to_vec();
    let mut edges = vec![0u32; owner.len()];
    for (edge, &o) in owner.iter().enumerate() {
        let at = cursor[o as usize] as usize;
        edges[at] = edge as u32;
        cursor[o as usize] += 1;
    }
    edges
}

#[pymodule]
fn walnutbutter_schedule(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(synapse_hazard, m)?)?;
    m.add("TOLERANCE", TOLERANCE)?;
    Ok(())
}
