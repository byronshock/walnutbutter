//! The wave loop of `propagation.Schedule.run` (AUTHORITY.md §4.4), in Rust.
//!
//! The engine owns the network's state for the life of a run, so an epoch costs one
//! crossing of the Python boundary rather than one per event. Everything here mirrors
//! `neuron.py` and `propagation.py` operation for operation and in the same order, so
//! that the floating point agrees: the point of a second engine in this project has
//! always been that two independent implementations landing on the same bits is evidence
//! the specification is unambiguous.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Event kinds, in the order they are taken within a wave (propagation.py).
const EXTERNAL: u8 = 0;
const STIMULUS: u8 = 1;
const SIGNAL: u8 = 2;

/// clock.py: two moments closer than this are the same moment.
const TOLERANCE: f64 = 1e-9;

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

/// One scheduled event. `payload` is an edge index for a signal, a neuron index otherwise.
#[derive(Debug, Clone, Copy)]
struct Event {
    time: f64,
    kind: u8,
    seq: u64,
    payload: u32,
}

// A min-heap by (time, kind, seq), matching heapq on the Python side: earliest first,
// then externals before stimuli before signals, then insertion order. BinaryHeap is a
// max-heap, so the ordering is reversed here rather than at every call site.
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
    eligibility: Vec<f64>,

    // --- neuron state ----------------------------------------------------------------
    potential: Vec<f64>,
    noise: Vec<f64>, // the exploration draw each neuron decided under this epoch (§6.1)
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
    hazard: bool,       // any delta > 0: the decision is a draw and the traces and scores are kept

    // --- the clock and the rules ------------------------------------------------------
    tau: f64,
    refractory: f64,
    hop: f64,
    bored_after: f64,
    rate_tau: f64,
    quash_rate: f64,
    quash_k: f64,
    hebb_rate: f64,
    synapse_tau: f64,
    weight_low: f64,
    weight_high: f64,
    earn: bool, // accumulate the eligibility a teacher pays at the read (§6.9)
    sigma: f64, // exploration noise, 0 = off (§6.1)
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
            eligibility: vec![0.0; edges],
            potential: vec![0.0; neurons],
            noise: vec![0.0; neurons],
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
            tau,
            refractory,
            hop,
            bored_after,
            rate_tau,
            quash_rate: 0.0,
            quash_k: 0.2,
            hebb_rate: 0.0,
            synapse_tau: 10.0,
            weight_low: -1.0,
            weight_high: 1.0,
            earn: false,
            sigma: 0.0,
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
        hebb_rate: f64,
        synapse_tau: f64,
        weight_low: f64,
        weight_high: f64,
        earn: bool,
        sigma: f64,
    ) -> PyResult<()> {
        if sigma > 0.0 && self.explore.is_none() {
            return Err(PyValueError::new_err(
                "exploration noise needs Python's stream: call set_explore_state(rng.getstate()[1]) \
                 first, so the draws come in the same order as the other two engines (§6.1)",
            ));
        }
        self.sigma = sigma;
        self.quash_rate = quash_rate;
        self.quash_k = quash_k;
        self.hebb_rate = hebb_rate;
        self.synapse_tau = synapse_tau;
        self.weight_low = weight_low;
        self.weight_high = weight_high;
        self.earn = earn;
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
    fn reset(&mut self, discharge: bool, clear_eligibility: bool) {
        if discharge {
            self.potential.iter_mut().for_each(|p| *p = 0.0);
        }
        self.fired_wave.iter_mut().for_each(|w| *w = -1);
        self.delivered_wave.iter_mut().for_each(|w| *w = -1);
        self.forced.iter_mut().for_each(|f| *f = false);
        self.noise.iter_mut().for_each(|x| *x = 0.0); // Neuron.reset clears it too
        self.spikes_at_reset.copy_from_slice(&self.spikes); // Neuron.reset snapshots the count
        if self.hazard {
            self.score.iter_mut().for_each(|s| *s = 0.0); // the hazard eligibility is the epoch's (§6.7)
            if discharge {
                self.trace.iter_mut().for_each(|x| *x = 0.0); // nothing is left in a zeroed potential
            }
        }
        if clear_eligibility {
            self.eligibility.iter_mut().for_each(|e| *e = 0.0);
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

    /// §6.7 with the wrong_hebb eligibility (the ±1 rule, so named September 16, 2026): every connection that
    /// delivered into an unforced neuron moves by `lr * advantage * (+1 if that neuron fired, else -1)`.
    /// Returns connections changed.
    fn reinforce_wrong_hebb(&mut self, advantage: f64, lr: f64) -> usize {
        if advantage == 0.0 {
            return 0;
        }
        let step = lr * advantage;
        let mut changed = 0;
        for edge in 0..self.weight.len() {
            if self.delivered_wave[edge] < 0 {
                continue;
            }
            let target = self.edge_target[edge] as usize;
            if self.forced[target] {
                continue; // a forced input: its firing was not the network's doing
            }
            let e = if self.fired_wave[target] >= 0 { 1.0 } else { -1.0 };
            let w = self.weight[edge] + step * e;
            self.weight[edge] = w.clamp(self.weight_low, self.weight_high);
            changed += 1;
        }
        changed
    }

    /// §6.7 with the hebb eligibility: every synapse into an unforced neuron moves by `lr * advantage * x_ij * c_j`,
    /// x_ij the signals it delivered this epoch that its target integrated (the tally, kept when `earn` is set) and
    /// c_j the target's spike count minus its expected count, which the Teacher keeps in Python (fast.py) as it keeps
    /// the rate memory, and hands over per epoch. Returns connections changed.
    fn reinforce_hebb(&mut self, advantage: f64, lr: f64, centred: Vec<f64>) -> PyResult<usize> {
        if centred.len() != self.neurons {
            return Err(PyValueError::new_err("one centred count per neuron"));
        }
        if !self.earn {
            return Err(PyValueError::new_err(
                "the hebb eligibility needs the tally: build the engine with earn = True, so every synapse counts \
                 what it delivered (§6.7)",
            ));
        }
        if advantage == 0.0 {
            return Ok(0);
        }
        let step = lr * advantage;
        let mut changed = 0;
        for edge in 0..self.weight.len() {
            let x = self.eligibility[edge];
            if x == 0.0 {
                continue; // delivered nothing this epoch
            }
            let target = self.edge_target[edge] as usize;
            if self.forced[target] {
                continue; // a forced input: its firing was not the network's doing
            }
            let e = x * centred[target];
            if e == 0.0 {
                continue; // exactly as many spikes as expected, or a first epoch: nothing to credit or blame
            }
            let w = self.weight[edge] + step * e;
            self.weight[edge] = w.clamp(self.weight_low, self.weight_high);
            changed += 1;
        }
        Ok(changed)
    }

    /// §6.7 with the perturb eligibility: every connection that delivered into an unforced
    /// neuron moves by `lr * advantage * (xi_j / sigma)`, the draw that neuron decided under.
    /// Returns connections changed. The late-signal rule is "count", the default.
    fn reinforce_perturb(&mut self, advantage: f64, lr: f64, sigma: f64) -> PyResult<usize> {
        if sigma <= 0.0 {
            return Err(PyValueError::new_err(
                "the perturb eligibility needs a positive sigma: e_j is xi_j / sigma (§6.7)",
            ));
        }
        if advantage == 0.0 {
            return Ok(0);
        }
        let step = lr * advantage;
        let mut changed = 0;
        for edge in 0..self.weight.len() {
            if self.delivered_wave[edge] < 0 {
                continue;
            }
            let target = self.edge_target[edge] as usize;
            if self.forced[target] {
                continue; // a forced input: its firing was not the network's doing
            }
            let e = self.noise[target] / sigma;
            if e == 0.0 {
                continue;
            }
            let w = self.weight[edge] + step * e;
            self.weight[edge] = w.clamp(self.weight_low, self.weight_high);
            changed += 1;
        }
        Ok(changed)
    }

    /// Escape noise (§5.2): each neuron's decision width, `Network.set_delta`'s per-neuron values; all 0 turns it off.
    fn set_deltas(&mut self, deltas: Vec<f64>) -> PyResult<()> {
        if deltas.len() != self.neurons {
            return Err(PyValueError::new_err("one delta per neuron"));
        }
        let on = deltas.iter().any(|&d| d > 0.0);
        if on && self.explore.is_none() {
            return Err(PyValueError::new_err(
                "escape noise needs Python's stream: call set_explore_state(rng.getstate()[1]) first (§5.2)",
            ));
        }
        self.delta = deltas;
        self.hazard = on;
        Ok(())
    }

    /// §5.2: the count's scaling of every hazard, sqrt(ESCAPE_REFERENCE_COUNT / N), as `Network.set_delta` gave each neuron.
    fn set_escape_scales(&mut self, scales: Vec<f64>) -> PyResult<()> {
        if scales.len() != self.neurons {
            return Err(PyValueError::new_err("one escape scale per neuron"));
        }
        self.escape_scale = scales;
        Ok(())
    }

    /// §6.7 with the hazard eligibility: every synapse into an unforced neuron moves by `lr * advantage * score`,
    /// the score being what the neuron's decisions summed to on that synapse's trace this epoch.
    fn reinforce_hazard(&mut self, advantage: f64, lr: f64) -> PyResult<usize> {
        if !self.hazard {
            return Err(PyValueError::new_err(
                "the hazard eligibility needs escape noise: set_deltas with a positive width first (§5.2)",
            ));
        }
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

    /// An external input of `amount` is not supported yet; stimuli and signals are.
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

        while let Some(first) = self.heap.peek().map(|e| e.time) {
            if !before(first, until) {
                break;
            }
            let limit = first + slack(first);
            batch.clear();
            let mut anchored: Option<f64> = None;
            while let Some(next) = self.heap.peek().map(|e| e.time) {
                if next > limit {
                    break;
                }
                let event = self.heap.pop().expect("peeked");
                // an input's exact time anchors the wave it joins, so chains never drift
                if event.kind != SIGNAL && anchored.is_none() {
                    anchored = Some(event.time);
                }
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
                            if self.hazard {
                                // escape noise (§6.7): what this synapse now has in its target's potential
                                self.trace[edge] =
                                    self.trace[edge] * (-(time - self.trace_at[edge]) / self.tau).exp() + 1.0;
                                self.trace_at[edge] = time;
                            }
                            if self.earn {
                                self.eligibility[edge] += 1.0;
                            }
                            if self.stamp[target] != mark {
                                self.stamp[target] = mark;
                                touched.push(target as u32);
                            }
                        }
                    }
                    STIMULUS => forced.push(event.payload),
                    EXTERNAL => {}
                    _ => {}
                }
            }

            // the floor applies to the wave's total, whatever order it arrived in
            for &i in &touched {
                let i = i as usize;
                if self.potential[i] < self.floor[i] {
                    self.potential[i] = self.floor[i];
                    if self.hazard {
                        self.clear_traces(i); // the floor bit: the weights are not in it (§6.7)
                    }
                }
            }

            // §6.1: the exploration draw comes before the decision it is meant to explain,
            // in propagation.py's position exactly -- after settle(), before anything fires
            if self.sigma > 0.0 {
                self.explore_wave(time);
            }
            if self.hazard {
                self.hazard_draws(); // §5.2: one uniform per neuron, in neuron order, after the additive draw
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

            if self.quash_rate != 0.0 {
                self.quash(time, &fired);
            }
            if self.hebb_rate != 0.0 {
                self.leaky_hebb(time, &fired);
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
    fn eligibilities(&self) -> Vec<f64> {
        self.eligibility.clone()
    }
    fn noises(&self) -> Vec<f64> {
        self.noise.clone()
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
    fn clear_eligibility(&mut self) {
        self.eligibility.iter_mut().for_each(|e| *e = 0.0);
    }
    fn set_thresholds(&mut self, thresholds: Vec<f64>) -> PyResult<()> {
        if thresholds.len() != self.neurons {
            return Err(PyValueError::new_err("wrong number of thresholds"));
        }
        self.threshold = thresholds;
        Ok(())
    }
    /// Every weight moves toward zero by `decay` (§6.8), once an epoch.
    fn forget(&mut self, decay: f64) {
        if decay > 0.0 {
            let keep = 1.0 - decay;
            self.weight.iter_mut().for_each(|w| *w *= keep);
        }
    }
}

impl Engine {
    #[inline]
    fn push(&mut self, time: f64, kind: u8, payload: u32) {
        self.seq += 1;
        let seq = self.seq;
        self.heap.push(Event { time, kind, seq, payload });
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

    /// `Network.perturb(sigma, rng, time, hold_fired=True)`, operation for operation (§6.1).
    ///
    /// Every neuron leaks to `time`, then takes a draw on top of its potential, floored.
    /// A neuron that has already fired this epoch **keeps** the draw it decided under as
    /// its record -- §6.7 credits one e_j per epoch and it must refer to the perturbation
    /// that produced the spike -- but its potential still takes the new one: the hold is on
    /// the record, not on the dynamics.
    ///
    /// The uniforms are drawn in one block of `count + (count & 1)`, as `exploration.uniforms`
    /// does, and paired by Box-Muller in the same order, so neuron `j` gets the cosine of pair
    /// `j / 2` when `j` is even and the sine when it is odd.
    fn explore_wave(&mut self, time: f64) {
        let n = self.neurons;
        if n == 0 {
            return;
        }
        let count = n + (n & 1); // rounded up so the pairs are whole
        let mut draws = Vec::with_capacity(count);
        match self.explore.as_mut() {
            Some(rng) => {
                for _ in 0..count {
                    draws.push(rng.random());
                }
            }
            None => return, // set_rules refuses sigma > 0 without a stream, so this cannot happen
        }
        for i in 0..n {
            let pair = i & !1;
            let angle = TWO_PI * draws[pair];
            let radius = self.sigma * (-2.0 * (1.0 - draws[pair + 1]).ln()).sqrt();
            let draw = if i & 1 == 0 { angle.cos() * radius } else { angle.sin() * radius };
            self.leak(i, time);
            if self.fired_wave[i] < 0 {
                self.noise[i] = draw;
            }
            let p = self.potential[i] + draw;
            if p < self.floor[i] {
                self.potential[i] = self.floor[i];
                if self.hazard {
                    self.clear_traces(i); // the floor bit: the weights are not in it (§6.7)
                }
            } else {
                self.potential[i] = p;
            }
        }
    }

    /// The traces into neuron `i` are zero: its potential was reset by a spike or is the floor (§6.7).
    #[inline]
    fn clear_traces(&mut self, i: usize) {
        let (lo, hi) = (self.in_start[i] as usize, self.in_start[i + 1] as usize);
        for k in lo..hi {
            self.trace[self.in_edges[k] as usize] = 0.0;
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

    /// neuron.py's decide: the threshold when the width is 0, else the escape-noise draw (§5.2), and under it the
    /// hazard eligibility of every incoming synapse is settled for the decision (§6.7): each score moves by
    /// e_j * trace, e_j = m e^-m / (1 - e^-m) when the neuron fires and -m when it does not.
    fn decide(&mut self, i: usize, now: f64) -> bool {
        if self.delta[i] <= 0.0 {
            return self.can_fire(i, now);
        }
        if self.refractory_at(i, now) {
            return false;
        }
        let s = self.potential_at(i, now) - self.threshold_at(i, now);
        let mut elapsed = now - self.exposed_since[i];
        if elapsed < 0.0 {
            elapsed = 0.0;
        }
        let m = (elapsed / self.hop * self.escape_scale[i] * (s / self.delta[i]).exp()).min(1e3);
        let fired = self.draw[i] < -(-m).exp_m1();
        self.exposed_since[i] = now;
        if m > 0.0 {
            let e = if fired { m * (-m).exp() / -(-m).exp_m1() } else { -m }; // m e^-m / (1 - e^-m), finite at any m
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
        self.exposed_since[i] = now + self.refractory; // the hazard resumes when the refractory period ends (§5.2)
        if self.hazard {
            self.clear_traces(i); // nothing any synapse delivered is still in the potential (§6.7)
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

    /// §6.12: a neuron that fires potentiates the synapses that still had charge in it.
    fn leaky_hebb(&mut self, time: f64, fired: &[u32]) {
        for &n in fired {
            let i = n as usize;
            let previous = self.previous_fired_at[i];
            let (lo, hi) = (self.in_start[i] as usize, self.in_start[i + 1] as usize);
            for k in lo..hi {
                let edge = self.in_edges[k] as usize;
                if !self.active[edge] || self.last_signal[edge] == f64::NEG_INFINITY {
                    continue;
                }
                if previous > f64::NEG_INFINITY && self.last_signal[edge] <= previous {
                    continue; // nothing carried since its previous spike
                }
                let step = self.hebb_rate
                    * (-(time - self.last_signal[edge] + self.hop) / self.synapse_tau).exp();
                if step == 0.0 {
                    continue;
                }
                let w = self.weight[edge] + step;
                self.weight[edge] = w.clamp(self.weight_low, self.weight_high);
            }
        }
    }
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
    m.add("TOLERANCE", TOLERANCE)?;
    Ok(())
}
