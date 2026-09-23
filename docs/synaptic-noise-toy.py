#!/usr/bin/env python3
"""A toy for one question: if the exploration noise is generated at the synapse
instead of at the neuron, the reinforce rule becomes local to the synapse --
what does that cost, in a network shaped like walnutbutter's?

Two rules on the same two-layer, complement-coded, accumulator network:

  neuron  -- AUTHORITY.md sections 6 and 8 as they stand. Each output makes a
             stochastic firing decision every wave on its margin (escape
             noise, width DELTA * theta_j, hazard scaled by sqrt(60/N)), and
             every synapse into it posts (credit - expectation) * trace at every
             decision: the hazard row of the single-spike rule (8.4).
  synapse -- the noise moves to the synapse. Every arrival delivers
             w_ij + sigma_j * xi with xi a unit normal drawn for that arrival,
             the output fires by the plain comparison p >= theta_j, and the
             synapse posts its own xi and nothing else. No trace, no note, no
             expectation of the neuron: the synapse needs only its own draw.

Both are paid once an epoch by LR * (r - b) * score, r the evidence critic's
log score at TEMPERATURE 2 on the output counts, b a running baseline.

Two instruments:

  snr    -- no learning. From one starting network, accumulate the update the
            rule would have made over E epochs and report its correlation with
            the supervised direction d_ij = P(input i on | class j) - P(i on)
            (AUTHORITY.md 11.16's instrument). How fast the correlation climbs
            is the estimator's signal-to-noise; the ratio of epochs the two
            rules need for the same correlation is the price of locality.
  sweep  -- learning, over a grid of (noise, lr) for each rule, several seeds.
            Reports accuracy over time, spikes per epoch, outputs stuck off.

Not an engine and not held to the file: a scratch model of the two rules.
"""
import argparse
import json
import sys
import time
from itertools import product
from multiprocessing import Pool

import numpy as np

THETA0 = 0.6        # threshold quoted at fan-in 18, as the September 2026 mnist point runs it
FLOOR0 = -0.2       # floor quoted at fan-in 18 (minimum potential -0.2)
FAN_IN_QUOTE = 18.0
ESCAPE_REFERENCE_COUNT = 60
TEMPERATURE = 2.0
BASELINE_RATE = 0.05
FLIP = 0.1          # each bit of a class prototype flips with this probability at presentation


def task(B, K, rng):
    protos = rng.random((K, B)) < 0.5
    # supervised direction for input i (complement-coded, I = 2B) onto output k
    p_on_given = np.where(protos, 1 - FLIP, FLIP)            # K x B
    p_on_given = np.concatenate([p_on_given, 1 - p_on_given], axis=1)  # K x 2B
    d_sup = (p_on_given - p_on_given.mean(0, keepdims=True)).T    # I x K
    return protos, d_sup


def run(rule, B, K, waves, arrivals, delta, sigma, lr, epochs, seed,
        learn=True, refractory=2, checkpoints=20):
    rng = np.random.default_rng(seed)
    I, O = 2 * B, K
    N = I + O
    d = I                                    # every output hears every input
    theta = THETA0 * d / FAN_IN_QUOTE
    floor = FLOOR0 * d / FAN_IN_QUOTE
    Delta = delta * theta                    # width, quoted in units of the threshold (6.4)
    sig = sigma * theta                      # the synapse's jitter, quoted the same way
    kappa = np.sqrt(ESCAPE_REFERENCE_COUNT / N)
    protos, d_sup = task(B, K, rng)
    W = rng.uniform(-1.0, 1.0, size=(I, O))
    U = np.zeros((I, O))                     # accumulated update direction (snr instrument)
    S2 = np.zeros((I, O))                    # accumulated squared update, for the per-epoch signal-to-noise
    e = np.zeros((I, O))
    x = np.zeros((I, O))
    p = np.zeros(O)
    last = np.full(O, -10 ** 9)
    b = None
    per = max(1, epochs // checkpoints)
    hist = []                                # per checkpoint: accuracy, spikes/epoch, stuck off, corr(U, d)
    right = 0
    spikes = 0
    fired_any = np.zeros(O, dtype=bool)
    for ep in range(1, epochs + 1):
        label = int(rng.integers(K))
        bits = protos[label] ^ (rng.random(B) < FLIP)
        on = np.flatnonzero(np.concatenate([bits, ~bits]))
        A = np.zeros((waves, I), dtype=np.int64)
        np.add.at(A, (rng.integers(waves, size=on.size * arrivals), np.repeat(on, arrivals)), 1)
        counts = np.zeros(O, dtype=np.int64)
        p[:] = 0.0
        x[:] = 0.0
        e[:] = 0.0
        last[:] = -10 ** 9
        for w in range(waves):
            active = (w - last) >= refractory          # a refractory output ignores arrivals and decides nothing
            a = A[w]
            firing = np.flatnonzero(a)
            if firing.size:
                n = a[firing][:, None].astype(float)
                if rule == "neuron":
                    contrib = (n * W[firing]).sum(0)
                    p += np.where(active, contrib, 0.0)
                    x[firing] += n * active[None, :]
                else:
                    xi = rng.standard_normal((firing.size, O)) * np.sqrt(n)   # the sum of n unit normals
                    contrib = (n * W[firing] + sig * xi).sum(0)
                    p += np.where(active, contrib, 0.0)
                    e[firing] += xi * active[None, :]                        # its own draw, and nothing else
            bit = p < floor
            if bit.any():
                p[bit] = floor
                x[:, bit] = 0.0
            if rule == "neuron":
                m = np.minimum(kappa * np.exp((p - theta) / Delta), 1e3)
                P = -np.expm1(-m)
                fired = active & (rng.random(O) < P)
                entry = np.where(fired, m * np.exp(-m) / np.where(P > 0, P, 1.0), -m)
                e += x * np.where(active, entry, 0.0)[None, :]
            else:
                fired = active & (p >= theta)
            if fired.any():
                p[fired] = 0.0
                x[:, fired] = 0.0
                last[fired] = w
                counts[fired] += 1
        z = counts / TEMPERATURE
        zmax = z.max()
        r = z[label] - zmax - np.log(np.exp(z - zmax).sum())
        if b is None:
            b = r
        adv = r - b
        b += BASELINE_RATE * (r - b)
        upd = adv * e
        U += upd
        S2 += upd * upd
        if learn:
            W = np.clip(W + lr * upd, -1.0, 1.0)
        others = np.delete(counts, label)
        right += int(counts[label] > others.max())
        spikes += int(counts.sum())
        fired_any |= counts > 0
        if ep % per == 0:
            uf, df = U.ravel(), d_sup.ravel()
            corr = float(np.corrcoef(uf, df)[0, 1]) if uf.std() > 0 else 0.0
            mean = U / ep
            var = np.maximum(S2 / ep - mean * mean, 0.0)
            signal = float((mean * mean - var / ep).sum())       # unbiased for the sum of squared means
            noise = float(var.sum())
            hist.append(dict(epoch=ep, acc=right / per, spikes=spikes / per,
                             stuck_off=int((~fired_any).sum()), corr=corr,
                             snr=signal / noise if noise > 0 else 0.0,
                             rails=float((np.abs(W) > 0.99).mean()),   # weights on the rails of WEIGHT_RANGE
                             wstd=float(W.std())))                     # the weights' spread (0.577 at the start)
            right = 0
            spikes = 0
            fired_any[:] = False
    return dict(rule=rule, B=B, K=K, fan_in=d, delta=delta, sigma=sigma, lr=lr, seed=seed,
                epochs=epochs, learn=learn, hist=hist)


def _job(kw):
    t0 = time.time()
    out = run(**kw)
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "snr", "sweep"])
    ap.add_argument("--B", type=int, nargs="+", default=[32])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--waves", type=int, default=20)
    ap.add_argument("--arrivals", type=int, default=3)
    ap.add_argument("--delta", type=float, nargs="+", default=[0.3125])
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.3])
    ap.add_argument("--lr", type=float, nargs="+", default=[0.01])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jobs = []
    for B, seed in product(args.B, args.seeds):
        common = dict(B=B, K=args.K, waves=args.waves, arrivals=args.arrivals,
                      epochs=args.epochs, seed=seed, learn=(args.mode != "snr"))
        if args.mode == "smoke":
            jobs.append(dict(rule="neuron", delta=args.delta[0], sigma=0.0, lr=args.lr[0], **common))
            jobs.append(dict(rule="synapse", delta=0.0, sigma=args.sigma[0], lr=args.lr[0], **common))
            continue
        lrs = [0.0] if args.mode == "snr" else args.lr
        for delta, lr in product(args.delta, lrs):
            jobs.append(dict(rule="neuron", delta=delta, sigma=0.0, lr=lr, **common))
        for sigma, lr in product(args.sigma, lrs):
            jobs.append(dict(rule="synapse", delta=0.0, sigma=sigma, lr=lr, **common))
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
        knob = f"delta {r['delta']}" if r["rule"] == "neuron" else f"sigma {r['sigma']}"
        print(f"{r['rule']:8s} fan-in {r['fan_in']:4d} {knob:12s} lr {r['lr']:<8g} seed {r['seed']} "
              f"acc {h['acc']:.3f} spikes/ep {h['spikes']:.2f} stuck {h['stuck_off']} corr {h['corr']:+.3f} snr {h['snr']:.2e} "
              f"({r['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
