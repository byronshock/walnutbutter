"""Instrument Network._decide_synapses on the object engine (read only: it draws nothing and changes nothing).

At every wave, before the engine's own _decide_synapses runs, it recomputes for every source that decides (did not
spike this wave, F_i > 0): m_i and P_i = -expm1(-m_i) by the engine's own Neuron.synapse_expected, the number of escapes
a_i from the wave's laid-out draws (strictly below P, edge order then the read slots), the gate V_i > 0 and m_i > 0, and
the §8.16 entry a c - (F - a) m in the engine note's form. After the engine has run it checks, bit for bit:
  - the escapes the engine returned are exactly the edges whose draw < P (the draw uses the credit's m);
  - an output's read count moved by exactly its read slot's escape;
  - every gated source's gain moved by exactly the recomputed entry (g + entry, the engine's own addition), every
    ungated source's gain not at all.
It posts the entry directly, per wave, to every synapse k -> i into a gated source: direct[k->i] += entry * x_ki, x_ki the
trace as it stands at the decision (after this wave's arrivals). Per wave it keeps the direct sum D_t = sum_i entry_i X_i
(X_i = sum_k x_ki), its exact conditional variance given the state before the draw, sum_i X_i^2 F_i m^2 e^-m / P, and
the escape and silent halves against their conditional expectations. Per state bin (X_i, m_i) it keeps the same sums.
"""
from __future__ import annotations

import math

import numpy as np

XBINS = (0, 1, 2, 3, 4, 6, 10, 1e18)  # X_i in [lo, hi)
MBINS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1e18)


class Instrument:
    def __init__(self, g):
        self.g = g
        self.orig = g._decide_synapses
        g._decide_synapses = self.wrapped
        self.conns = [c for n in g.all_neurons() for c in n.outgoing]  # edge order
        self.cidx = {id(c): k for k, c in enumerate(self.conns)}
        self.direct = np.zeros(len(self.conns))
        self.waves = []  # per wave: epoch, D, var, esc, esc_exp, sil, sil_exp, n_gated, n_gated_x
        self.cal = np.zeros(4)  # gated: sum(a - F P), sum F P (1 - P); ungated: the same
        self.cal_u = np.zeros(2)
        self.xb = np.zeros((len(XBINS) - 1, 3))  # sum entry X, sum var, count (decisions with X > 0)
        self.mb = np.zeros((len(MBINS) - 1, 3))
        self.epoch = 0
        self.bad = []
        self.layout_key = None
        self.wave_chunks = []
        self.rows = []  # per epoch: epoch, settled sum, direct sum, max |settled - direct| per synapse, max |score|
        self.orig_reset = g.reset
        g.reset = self.reset

    def reset(self, discharge: bool = False):
        # the previous epoch's read has settled and paid; its scores stand until the original reset clears them
        self.close_epoch()
        self.epoch += 1
        self.direct[:] = 0.0
        return self.orig_reset(discharge)

    def close_epoch(self):
        if self.epoch > 0:
            self.rows.append((self.epoch,) + self.identity())
            if self.waves:
                self.wave_chunks.append(np.array(self.waves, dtype=np.float64))
                self.waves = []

    def wave_array(self):
        self.close_epoch_waves()
        return np.concatenate(self.wave_chunks) if self.wave_chunks else np.zeros((0, 9))

    def close_epoch_waves(self):
        if self.waves:
            self.wave_chunks.append(np.array(self.waves, dtype=np.float64))
            self.waves = []

    def finish(self):
        self.close_epoch()
        self.epoch += 1  # nothing more is recorded under the closed epoch

    def _layout_arrays(self):
        g = self.g
        key = id(g._layout)
        if key == self.layout_key:
            return
        self.layout_key = key
        src_of = []
        self.ranges = []
        k = 0
        for s, (neuron, slot) in enumerate(g._layout):
            n = len(neuron.outgoing)
            self.ranges.append((k, k + n, slot))
            src_of += [s] * n
            k += n
        slots = np.full(g._draw_count - k, -1)
        for s, (neuron, slot) in enumerate(g._layout):
            if slot >= 0:
                slots[slot - k] = s
        self.src_of = np.concatenate([np.array(src_of, dtype=np.int64), slots])
        self.E = k

    def wrapped(self, wave):
        g = self.g
        self._layout_arrays()
        draws = np.asarray(g._draws)
        number, time = wave.number, wave.time
        rest = g.synapse_hazard_rest
        linear = g.synapse_hazard_family == "linear"
        layout = g._layout
        S = len(layout)
        chance = np.full(S, -1.0)  # -1: decides nothing (spiked, or F = 0)
        rec = []
        for s, (neuron, slot) in enumerate(layout):
            lo, hi, _ = self.ranges[s]
            if neuron.fired_in_wave == number:
                continue
            F = (hi - lo) + (slot >= 0)
            if not F:
                continue
            m, h = neuron.synapse_expected(time, rest, linear)
            P = -math.expm1(-m)
            chance[s] = P
            rec.append((s, neuron, slot, F, m, h, P))
        esc = draws < chance[self.src_of]
        a_of = np.bincount(self.src_of, weights=esc.astype(float), minlength=S)
        before = [(neuron.gain, neuron.read_count) for (_, neuron, _, _, _, _, _) in rec]
        # the decision-time traces, before the engine runs (it moves none of them, but read them here all the same)
        pre = []
        for (s, neuron, slot, F, m, h, P) in rec:
            V = neuron.potential_at(time)
            gate = m > 0.0 and V > 0.0
            if gate:
                xs = [(self.cidx[id(c)], c.trace) for c in neuron.incoming if c.trace != 0.0]
            else:
                xs = None
            pre.append((gate, xs))
        escaped = self.orig(wave)
        # 1. the engine's escapes are exactly draw < P on the credit's m
        mine = sorted(int(k) for k in np.nonzero(esc[: self.E])[0])
        theirs = sorted(self.cidx[id(c)] for c in escaped)
        if mine != theirs:
            self.bad.append((self.epoch, number, "escapes differ"))
        D = var = e_esc = e_esc_exp = e_sil = e_sil_exp = 0.0
        ng = ngx = 0
        for (s, neuron, slot, F, m, h, P), (g0, r0), (gate, xs) in zip(rec, before, pre):
            a = int(a_of[s])
            if slot >= 0:
                ra = int(esc[slot])
                if neuron.read_count - r0 != ra:
                    self.bad.append((self.epoch, number, "read count"))
            if gate:
                c = m * math.exp(-m) / -math.expm1(-m)
                entry = a * c - (F - a) * m
                if linear:
                    entry = (1.0 - rest) / h * entry
                if neuron.gain != g0 + entry:
                    self.bad.append((self.epoch, number, f"gain {neuron.gain!r} != {g0!r} + {entry!r}"))
                self.cal[0] += a - F * P
                self.cal[1] += F * P * (1 - P)
                ng += 1
                X = 0.0
                for k, x in xs:
                    self.direct[k] += entry * x
                    X += x
                if X != 0.0:
                    ngx += 1
                    v1 = F * m * m * math.exp(-m) / P  # Var(entry | m), F independent synapses
                    if linear:
                        v1 *= ((1.0 - rest) / h) ** 2
                    D += entry * X
                    var += X * X * v1
                    e_esc += a * c * X
                    e_esc_exp += F * P * c * X
                    e_sil += (F - a) * m * X
                    e_sil_exp += F * math.exp(-m) * m * X
                    xi = next(b for b in range(len(XBINS) - 1) if X < XBINS[b + 1])
                    mi = next(b for b in range(len(MBINS) - 1) if m < MBINS[b + 1])
                    for arr, b in ((self.xb, xi), (self.mb, mi)):
                        arr[b, 0] += entry * X
                        arr[b, 1] += X * X * v1
                        arr[b, 2] += 1
            else:
                if neuron.gain != g0:
                    self.bad.append((self.epoch, number, "ungated gain moved"))
                self.cal_u[0] += a - F * P
                self.cal_u[1] += F * P * (1 - P)
        self.waves.append((self.epoch, D, var, e_esc, e_esc_exp, e_sil, e_sil_exp, ng, ngx))
        return escaped

    def identity(self):
        """The settled scores against the per-wave direct posting, per synapse: (sum settled, sum direct, max |diff|,
        max |score|)."""
        s = np.array([c.score for c in self.conns])
        return float(s.sum()), float(self.direct.sum()), float(np.abs(s - self.direct).max()), float(np.abs(s).max())
