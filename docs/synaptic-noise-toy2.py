#!/usr/bin/env python3
"""The second toy: Byron's mechanism, as he stated it on September 23, 2026.

  The synapse escapes with a hazard that rises monotonically with its source's
  potential as a fraction of the threshold, u = clip(V_pre, 0, V_threshold) /
  V_threshold, and is certain at threshold, where the source spikes
  deterministically and every synapse it has transmits (the action potential).
  A speculative transmission delivers the synapse's weight to its target and
  leaves the source's potential alone. V_threshold is the neuron's. The neuron
  keeps its threshold and its deterministic spike.

Here the hazard is h(u) = h0 ** (1 - u) spikes per hop, scaled by sqrt(60/N)
as the file scales every hazard (6.6): h0 is the hazard at rest, the quantity
Byron's third answer leaves "to be determined through rigorous investigation",
and it is what this toy sweeps. (Today's exponential hazard is this same
family with h0 = exp(-1/Delta), clipped at zero potential and cut off at
threshold, where the neuron now fires for certain.)

The network is the first toy's: two layers, complement-coded, K classes, one
output each, every output hearing every input; accumulator outputs with the
September axis, floor, refractory period, mnist's evidence critic and running
baseline. The inputs are now neurons too, since their potentials set their
synapses' hazards:

  --drive forced     the file's drive: an on-input is forced to spike, its
                     potential resets, and it sits at zero between spikes, so
                     every input's synapses -- on or off -- whisper at h0.
  --drive potential  an on-input is charged instead: nine deliveries of a
                     third of its threshold at random waves, so it spikes
                     about three times and sits between at a potential its
                     synapses can read. An off-input sits at zero.

Rules:

  local   Seung's R1 and R2 laid on the weight: at every escape decision the
          synapse posts (escaped - P) * sign(w) from its own outcome and its
          own expectation, nothing of any neuron. Not the gradient (note 7.9).
  exact   Williams's rule. A synapse is credited by the draws its weight
          shaped: the escapes of its TARGET's outgoing synapses. In two layers
          an output projects nowhere, so this posts nothing -- the control.
          With --read transmissions each output has one outgoing synapse to a
          read neuron that counts everything it receives, spikes and
          speculation alike; the output's escapes then reach the reward and
          the exact rule credits its incoming synapses by the hazard row of
          8.4 -- (c - q) times the deliveries standing in the potential --
          at every wave the output is between zero and threshold.
  none    no learning: the mechanism's own dynamics.

Reported per checkpoint: accuracy, output spikes an epoch, outputs stuck
off, the correlation of the weight change with the supervised direction
d_ij = P(input i on | class j) - P(input i on) (11.16's instrument), whispers
an epoch per synapse from on-inputs and from off-inputs, the fraction of
weights on the rails, their spread, and the fraction whose sign has flipped.

Not an engine and not held to the file: a scratch model.
"""
import argparse
import json
import sys
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

THETA0 = 0.6
FLOOR0 = -0.2
FAN_IN_QUOTE = 18.0
ESCAPE_REFERENCE_COUNT = 60
TEMPERATURE = 2.0
BASELINE_RATE = 0.05
FLIP = 0.1
DRIVE_STEPS = 3          # a charged input takes this many drive deliveries to reach its threshold
SPIKES_PER_ON = 3        # about this many spikes an epoch from an on-input, either drive


def task(B, K, rng):
    protos = rng.random((K, B)) < 0.5
    p_on_given = np.where(protos, 1 - FLIP, FLIP)
    p_on_given = np.concatenate([p_on_given, 1 - p_on_given], axis=1)
    d_sup = (p_on_given - p_on_given.mean(0, keepdims=True)).T
    return protos, d_sup


def run(rule, B, K, waves, drive, h0, lr, epochs, seed, read="spikes",
        refractory=2, checkpoints=20):
    rng = np.random.default_rng(seed)
    I, O = 2 * B, K
    N = I + O
    d = I
    theta_out = THETA0 * d / FAN_IN_QUOTE
    floor_out = FLOOR0 * d / FAN_IN_QUOTE
    theta_in = THETA0                           # an input hears nothing: the quoted pair at scale 1 (4.11)
    kappa = np.sqrt(ESCAPE_REFERENCE_COUNT / N)
    protos, d_sup = task(B, K, rng)
    W = rng.uniform(-1.0, 1.0, size=(I, O))
    W0 = W.copy()
    e = np.zeros((I, O))
    x = np.zeros((I, O))                        # deliveries standing in the target since its last spike
    p_out = np.zeros(O)
    p_in = np.zeros(I)
    last_out = np.full(O, -10 ** 9)
    last_in = np.full(I, -10 ** 9)
    b = None
    per = max(1, epochs // checkpoints)
    hist = []
    right = spikes = 0
    esc_on = esc_off = 0.0
    fired_any = np.zeros(O, dtype=bool)
    for ep in range(1, epochs + 1):
        label = int(rng.integers(K))
        bits = protos[label] ^ (rng.random(B) < FLIP)
        onmask = np.concatenate([bits, ~bits])
        on = np.flatnonzero(onmask)
        if drive == "forced":
            F = np.zeros((waves, I), dtype=bool)
            F[rng.integers(waves, size=on.size * SPIKES_PER_ON), np.repeat(on, SPIKES_PER_ON)] = True
        else:
            D = np.zeros((waves, I))
            n = SPIKES_PER_ON * DRIVE_STEPS
            np.add.at(D, (rng.integers(waves, size=on.size * n), np.repeat(on, n)), theta_in / DRIVE_STEPS)
        counts = np.zeros(O, dtype=np.int64)
        extra = np.zeros(O, dtype=np.int64)     # the read neuron's count of speculative transmissions (toy 3)
        p_out[:] = 0.0
        p_in[:] = 0.0
        x[:] = 0.0
        e[:] = 0.0
        last_out[:] = -10 ** 9
        last_in[:] = -10 ** 9
        for w in range(waves):
            act_in = (w - last_in) >= refractory
            act_out = (w - last_out) >= refractory
            # the inputs: the drive, then the deterministic spike
            if drive == "forced":
                spk_in = F[w] & act_in
            else:
                p_in += D[w] * act_in
                spk_in = act_in & (p_in >= theta_in)
            # every synapse of a source that is awake and below threshold decides, on its source's potential
            u = np.clip(p_in, 0.0, theta_in) / theta_in
            m_in = kappa * h0 ** (1.0 - u)
            P_in = -np.expm1(-m_in)
            deciding = act_in & ~spk_in
            esc = deciding[:, None] & (rng.random((I, O)) < P_in[:, None])
            deliver = spk_in[:, None] | esc
            contrib = (deliver * W).sum(0)
            p_out += np.where(act_out, contrib, 0.0)
            x += deliver * act_out[None, :]
            if rule == "local":
                e += np.where(deciding[:, None], (esc - P_in[:, None]) * np.sign(W), 0.0)
            esc_on += esc[onmask].sum()
            esc_off += esc[~onmask].sum()
            p_in[spk_in] = 0.0
            last_in[spk_in] = w
            # the outputs: the floor once a wave, then the comparison
            bit = p_out < floor_out
            if bit.any():
                p_out[bit] = floor_out
                x[:, bit] = 0.0
            spk_out = act_out & (p_out >= theta_out)
            if read == "transmissions":
                # each output's one outgoing synapse, to its read neuron, decides on the output's potential
                dec_o = act_out & ~spk_out
                u_o = np.clip(p_out, 0.0, theta_out) / theta_out
                m_o = kappa * h0 ** (1.0 - u_o)
                P_o = -np.expm1(-m_o)
                y_o = dec_o & (rng.random(O) < P_o)
                extra += y_o
                if rule == "exact":
                    # the hazard row of 8.4; the derivative of the hazard by the potential is zero below zero
                    entry = np.where(y_o, m_o * np.exp(-m_o) / np.where(P_o > 0, P_o, 1.0), -m_o)
                    entry = np.where(dec_o & (p_out > 0.0), entry, 0.0)
                    e += x * entry[None, :]
            if spk_out.any():
                p_out[spk_out] = 0.0
                x[:, spk_out] = 0.0
                last_out[spk_out] = w
                counts[spk_out] += 1
        total = counts + extra
        z = total / TEMPERATURE
        zmax = z.max()
        r = z[label] - zmax - np.log(np.exp(z - zmax).sum())
        if b is None:
            b = r
        adv = r - b
        b += BASELINE_RATE * (r - b)
        if rule != "none":
            W = np.clip(W + lr * adv * e, -1.0, 1.0)
        others = np.delete(total, label)
        right += int(total[label] > others.max())
        spikes += int(counts.sum())
        fired_any |= total > 0
        if ep % per == 0:
            dw = (W - W0).ravel()
            corr = float(np.corrcoef(dw, d_sup.ravel())[0, 1]) if dw.std() > 0 else 0.0
            hist.append(dict(epoch=ep, acc=right / per, spikes=spikes / per,
                             stuck_off=int((~fired_any).sum()), corr=corr,
                             whisper_on=esc_on / per / (B * O), whisper_off=esc_off / per / (B * O),
                             rails=float((np.abs(W) > 0.99).mean()), wstd=float(W.std()),
                             flipped=float((np.sign(W) != np.sign(W0)).mean())))
            right = spikes = 0
            esc_on = esc_off = 0.0
            fired_any[:] = False
    return dict(rule=rule, B=B, K=K, fan_in=d, waves=waves, drive=drive, h0=h0, lr=lr, read=read,
                seed=seed, epochs=epochs, hist=hist)


def _job(kw):
    t0 = time.time()
    out = run(**kw)
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", nargs="+", default=["local"], choices=["local", "exact", "none"])
    ap.add_argument("--drive", nargs="+", default=["forced"], choices=["forced", "potential"])
    ap.add_argument("--read", default="spikes", choices=["spikes", "transmissions"])
    ap.add_argument("--B", type=int, nargs="+", default=[32])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--waves", type=int, default=20)
    ap.add_argument("--h0", type=float, nargs="+", default=[0.03])
    ap.add_argument("--lr", type=float, nargs="+", default=[0.01])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    jobs = [dict(rule=rule, B=B, K=args.K, waves=args.waves, drive=drive, h0=h0, lr=lr,
                 epochs=args.epochs, seed=seed, read=args.read)
            for rule, drive, B, h0, lr, seed
            in product(args.rule, args.drive, args.B, args.h0, args.lr, args.seeds)]
    print(f"{len(jobs)} runs, {args.workers} workers", file=sys.stderr)
    t0 = time.time()
    if args.workers > 1 and len(jobs) > 1:
        with Pool(args.workers) as pool:
            results = pool.map(_job, jobs, chunksize=1)
    else:
        results = [_job(j) for j in jobs]
    print(f"done in {time.time() - t0:.0f} s", file=sys.stderr)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f)
    for r in results:
        h = r["hist"][-1]
        print(f"{r['rule']:6s} {r['drive']:9s} read {r['read']:13s} fan-in {r['fan_in']:3d} h0 {r['h0']:<6g} "
              f"lr {r['lr']:<6g} seed {r['seed']} acc {h['acc']:.3f} spikes/ep {h['spikes']:5.2f} "
              f"stuck {h['stuck_off']} corr {h['corr']:+.3f} whisper on/off {h['whisper_on']:.2f}/{h['whisper_off']:.2f} "
              f"flipped {h['flipped']:.2f} ({r['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
