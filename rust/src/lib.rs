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
    threshold: Vec<f64>,
    floor: Vec<f64>,
    fired_at: Vec<f64>, // NEG_INFINITY when never
    previous_fired_at: Vec<f64>,
    last_update: Vec<f64>,
    spikes: Vec<u64>,
    rate_level: Vec<f64>,
    rate_at: Vec<f64>,
    forced: Vec<bool>,
    fired_wave: Vec<i64>, // -1 when it has not fired this epoch

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

    // --- the schedule -----------------------------------------------------------------
    heap: BinaryHeap<Event>,
    seq: u64,
    stamp: Vec<u64>, // per neuron: the wave it was last touched in
    wave_no: u64,
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
            threshold,
            floor,
            fired_at: vec![f64::NEG_INFINITY; neurons],
            previous_fired_at: vec![f64::NEG_INFINITY; neurons],
            last_update: vec![0.0; neurons],
            spikes: vec![0; neurons],
            rate_level: vec![0.0; neurons],
            rate_at: vec![0.0; neurons],
            forced: vec![false; neurons],
            fired_wave: vec![-1; neurons],
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
            heap: BinaryHeap::new(),
            seq: 0,
            stamp: vec![0; neurons],
            wave_no: 0,
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
        if sigma > 0.0 {
            return Err(PyValueError::new_err(
                "exploration noise is not implemented here: the draws must come from Python's \
                 stream in the same order for the engines to agree (see rust/README.md)",
            ));
        }
        self.quash_rate = quash_rate;
        self.quash_k = quash_k;
        self.hebb_rate = hebb_rate;
        self.synapse_tau = synapse_tau;
        self.weight_low = weight_low;
        self.weight_high = weight_high;
        self.earn = earn;
        Ok(())
    }

    /// Start a new epoch: clear the fired-this-epoch state, exactly as `Network.reset` does.
    /// Potentials, spike times and the signals in flight are kept (AUTHORITY.md §4.2).
    fn reset(&mut self, discharge: bool, clear_eligibility: bool) {
        if discharge {
            self.potential.iter_mut().for_each(|p| *p = 0.0);
        }
        self.fired_wave.iter_mut().for_each(|w| *w = -1);
        self.forced.iter_mut().for_each(|f| *f = false);
        if clear_eligibility {
            self.eligibility.iter_mut().for_each(|e| *e = 0.0);
        }
        self.wave_no = 0;
    }

    /// `neuron` is forced to fire at `time`, refractory period permitting.
    fn stimulus(&mut self, neuron: u32, time: f64) {
        self.push(time, STIMULUS, neuron);
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
            let mark = self.wave_no;
            touched.clear();
            forced.clear();

            // every signal of the wave is delivered before any neuron of it decides to fire
            for event in &batch {
                match event.kind {
                    SIGNAL => {
                        let edge = event.payload as usize;
                        let target = self.edge_target[edge] as usize;
                        if self.receive(target, self.weight[edge], time) {
                            self.last_signal[edge] = time;
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
                }
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
                if self.can_fire(i, time) {
                    self.fire(i, time, &mut fired);
                }
            }
            // everyone else: a threshold that has fallen with its silence (§5.4)
            for i in 0..self.neurons {
                if self.stamp[i] != mark && self.can_fire(i, time) {
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
    fn potentials(&self) -> Vec<f64> {
        self.potential.clone()
    }
    fn spike_counts(&self) -> Vec<u64> {
        self.spikes.clone()
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
