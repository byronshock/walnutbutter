#!/usr/bin/env python3
"""How much of the class does the output zone carry, read by the critic and by matched filters?

Reads runs/synapse-read-probe/<sweep>/<arm>.npz (docs/synapse-read-probe.py): every epoch's label,
each output's spikes in time bins, and each output's read-synapse escapes. The first half of the
epochs fits each reader, the second half scores it -- the fraction of epochs whose label is the
reader's pick. The network learns on through the probe, slowly at these rates, so the halves are
the same network to within that drift.

The readers:

- critic: the class critic as mnist pays it (§11.9, §11.15) -- the label's class must strictly
  out-count every other on n_k = n_k+ - n_k-, the count read (spikes plus read escapes); a tie loses.
- critic, ties split: the same sums, a tie between m classes worth 1/m, to show what ties cost.
- poisson: the matched filter for independent Poisson outputs -- each output's mean count under each
  label is its template, and class k scores sum_j x_j ln(mu_kj) - mu_kj.
- whitened: the matched filter for Gaussian noise shared across outputs (linear discriminant): the
  templates correlated through the inverse of the pooled covariance, which discounts outputs that
  carry the same noise. Shrunk towards the diagonal by a factor chosen on the fit half.
- whitened, timed: the same, on each output's spikes in each time bin plus its read escapes, so the
  filter can weight when in the epoch a spike falls as well as which output fired it.

Beside them, for each output, how well its own count tells the epochs it should be on (the label
code, 11.10) from those it should be off: d' = (mean on - mean off) / sqrt of the mean of the two
variances; and the temperature at which n_k/T would be the Poisson matched filter's log-likelihood
if every output were alike, T = 1 / ln(on rate / off rate).

Where runs/synapse-read-probe/<SWEEP>-lr0-fresh holds the same arms built fresh from their seeds and never learning
(docs/synapse-read-probe.py --batch, a null sweep), each arm is set against its own seed's untrained read: a random
input-to-output wiring already carries much of the digit, so what an arm learned is its rise above that, not above
chance (seed 1 on the synapse point: 0.52 untrained, September 26, 2026).

Usage: docs/synapse-matched-filter.py [--timed] SWEEP [SWEEP ...]  (the timed reader only with --timed)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = Path(subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=HERE,
                           capture_output=True, text=True, check=True).stdout.strip()).parent
CLASSES, POPULATION = 10, 3
TIMED = "--timed" in sys.argv  # the timed reader: 1,260 features, the slow one, and on the synapse arms it found nothing
sys.argv = [a for a in sys.argv if a != "--timed"]


def code(labels):
    """The label code (11.10), epochs x 60: the label's fire-if-one population and every other fire-if-zero on."""
    ones = (np.repeat(np.arange(CLASSES), POPULATION)[None, :] == labels[:, None])
    return np.concatenate([ones, ~ones], axis=1)


def critic(counts, labels, split_ties=False):
    plus = counts[:, :CLASSES * POPULATION].reshape(len(counts), CLASSES, POPULATION).sum(axis=2)
    minus = counts[:, CLASSES * POPULATION:].reshape(len(counts), CLASSES, POPULATION).sum(axis=2)
    evidence = plus - minus
    mine = evidence[np.arange(len(labels)), labels]
    top = evidence.max(axis=1)
    tied = (evidence == top[:, None]).sum(axis=1)
    if split_ties:
        return float(np.where(mine == top, 1.0 / tied, 0.0).mean())
    return float(((mine == top) & (tied == 1)).mean())


def poisson(fit_x, fit_y, test_x, test_y):
    mu = np.stack([fit_x[fit_y == k].mean(axis=0) for k in range(CLASSES)])  # classes x outputs
    mu = np.maximum(mu, 1e-2)
    scores = test_x @ np.log(mu).T - mu.sum(axis=1)[None, :]
    return float((scores.argmax(axis=1) == test_y).mean())


def _lda(fit_x, fit_y, alpha):
    means = np.stack([fit_x[fit_y == k].mean(axis=0) for k in range(CLASSES)])
    centred = fit_x - means[fit_y]
    cov = centred.T @ centred / (len(fit_x) - CLASSES)
    diag = np.diag(np.diag(cov))
    ridge = 1e-3 * max(float(np.diag(cov).mean()), 1e-9) * np.eye(len(cov))  # an output silent in a bin has no variance
    cov = (1 - alpha) * cov + alpha * diag + ridge
    weights = np.linalg.solve(cov, means.T)  # features x classes
    bias = -0.5 * np.einsum("kf,fk->k", means, weights) + np.log(np.bincount(fit_y, minlength=CLASSES) / len(fit_y))
    return weights, bias


def whitened(fit_x, fit_y, test_x, test_y):
    """The linear discriminant, its shrinkage chosen on the last third of the fit half, then refit on the whole half."""
    cut = 2 * len(fit_x) // 3
    best = max((0.0, 0.01, 0.03, 0.1, 0.3, 1.0), key=lambda a: (
        (lambda w, b: ((fit_x[cut:] @ w + b).argmax(axis=1) == fit_y[cut:]).mean())(*_lda(fit_x[:cut], fit_y[:cut], a))))
    w, b = _lda(fit_x, fit_y, best)
    return float(((test_x @ w + b).argmax(axis=1) == test_y).mean()), best


def per_output(counts, labels):
    on = code(labels)
    rows = []
    for j in range(counts.shape[1]):
        a, b = counts[on[:, j], j], counts[~on[:, j], j]
        spread = np.sqrt((a.var() + b.var()) / 2)
        rows.append((a.mean(), b.mean(), (a.mean() - b.mean()) / spread if spread > 0 else 0.0))
    return np.array(rows)


def arm(path):
    d = np.load(path)
    labels = d["labels"].astype(int)
    spikes, reads = d["spikes"].astype(float), d["reads"].astype(float)
    counts = spikes.sum(axis=2) + reads  # the count read (§5.10, §7.9)
    timed = np.concatenate([spikes.reshape(len(spikes), -1), reads], axis=1)
    half = len(labels) // 2
    fy, ty = labels[:half], labels[half:]
    out = {"arm": path.stem, "epochs": len(labels)}
    out["critic"] = critic(counts[half:], ty)
    out["critic_ties_split"] = critic(counts[half:], ty, split_ties=True)
    out["poisson"] = poisson(counts[:half], fy, counts[half:], ty)
    out["whitened"], out["shrink"] = whitened(counts[:half], fy, counts[half:], ty)
    out["whitened_timed"], out["shrink_timed"] = whitened(timed[:half], fy, timed[half:], ty) if TIMED else (None, None)
    rows = per_output(counts, labels)
    out["dprime"] = rows[:, 2].round(3).tolist()
    out["mean_on"], out["mean_off"] = rows[:, 0].round(3).tolist(), rows[:, 1].round(3).tolist()
    on = code(labels)
    a, b = counts[on].mean(), counts[~on].mean()
    out["zone_on"], out["zone_off"] = float(a), float(b)
    out["temperature_if_alike"] = float(1 / np.log(a / b)) if a > b > 0 else None
    # the signal's time course: in each bin, the zone's mean spikes when an output should be on, less when off
    bins = spikes.shape[2]
    out["signal_by_bin"] = [float(spikes[:, :, t][on].mean() - spikes[:, :, t][~on].mean()) for t in range(bins)]
    out["level_by_bin"] = [float(spikes[:, :, t].mean()) for t in range(bins)]
    return out


def read_folder(folder):
    """Every arm of a probe folder read by every reader, written beside them as matched-filter.json."""
    results = [arm(p) for p in sorted(folder.glob("*.npz"))]
    (folder / "matched-filter.json").write_text(json.dumps(results, indent=1))
    return results


def short(name):
    """An arm's name cut to what tells the arms of one sweep apart: its rate, its hidden count, its seed."""
    knobs = dict(re.findall(r"(lr|hidden_neurons|seed)(-?[0-9.]+)", name))
    return f"lr{knobs.get('lr', '?')}-h{knobs.get('hidden_neurons', '0')}-s{knobs.get('seed', '?')}"


def untrained_key(name):
    """What pairs an arm with its seed's learning-off control: its name without its rate."""
    return re.sub(r"-lr-?[0-9.]+", "", name)


def main():
    """Read each named probe folder, and where runs/synapse-read-probe/<sweep>-lr0-fresh holds the same arms built fresh
    and never learning, set every arm against its own seed's untrained read."""
    for sweep in sys.argv[1:]:
        folder = MAIN / "runs" / "synapse-read-probe" / sweep
        results = read_folder(folder)
        fresh = folder.with_name(f"{sweep}-lr0-fresh")
        controls = {untrained_key(r["arm"]): r for r in read_folder(fresh)} if fresh.is_dir() else {}
        print(f"\n{sweep}: fraction right on the second half of each continuation (fit on the first)"
              + (f", against the same seed built fresh and never learning ({fresh.name})" if controls else ""))
        print(f"{'arm':<18} {'sums':>6} {'poisson':>8} {'whiten':>7} {'timed':>7} | {'untrained: sums':>15} {'whiten':>7} "
              f"{'gain':>7} | d' median, outputs with d' < 0.2")
        groups = {}
        for r in results:
            c = controls.get(untrained_key(r["arm"]))
            dp = np.array(r["dprime"])
            gain = r["whitened"] - c["whitened"] if c else None
            print(f"{short(r['arm']):<18} {r['critic']:6.3f} {r['poisson']:8.3f} {r['whitened']:7.3f} "
                  f"{r['whitened_timed'] if TIMED else float('nan'):7.3f} | {c['critic'] if c else float('nan'):15.3f} "
                  f"{c['whitened'] if c else float('nan'):7.3f} {gain if gain is not None else float('nan'):+7.3f} | "
                  f"{np.median(dp):.2f}, {int((dp < 0.2).sum())}")
            groups.setdefault(short(r["arm"]).rsplit("-s", 1)[0], []).append((r, c))
        print(f"\n{'arm, over seeds':<18} {'n':>2} {'sums':>6} {'whiten':>7} {'untrained':>9} {'gain mean':>9} {'sd':>6}")
        for key, pairs in groups.items():
            gains = [r["whitened"] - c["whitened"] for r, c in pairs if c]
            print(f"{key:<18} {len(pairs):2d} {np.mean([r['critic'] for r, _ in pairs]):6.3f} "
                  f"{np.mean([r['whitened'] for r, _ in pairs]):7.3f} "
                  f"{np.mean([c['whitened'] for _, c in pairs if c]) if gains else float('nan'):9.3f} "
                  f"{np.mean(gains) if gains else float('nan'):+9.3f} {np.std(gains) if gains else float('nan'):6.3f}")


if __name__ == "__main__":
    main()
