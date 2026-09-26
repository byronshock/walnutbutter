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

Usage: docs/synapse-matched-filter.py SWEEP [SWEEP ...]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = Path(subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=HERE,
                           capture_output=True, text=True, check=True).stdout.strip()).parent
CLASSES, POPULATION = 10, 3


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
    out["whitened_timed"], out["shrink_timed"] = whitened(timed[:half], fy, timed[half:], ty)
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


def main():
    for sweep in sys.argv[1:]:
        folder = MAIN / "runs" / "synapse-read-probe" / sweep
        results = [arm(p) for p in sorted(folder.glob("*.npz"))]
        (folder / "matched-filter.json").write_text(json.dumps(results, indent=1))
        print(f"\n{sweep}: fraction right on the second half of each continuation (fit on the first)")
        print(f"{'arm':<14} {'critic':>7} {'ties/m':>7} {'poisson':>8} {'whiten':>7} {'timed':>7}   "
              f"{'on':>5} {'off':>5} {'T alike':>7}  d' median [min, max], outputs with d' < 0.2")
        for r in results:
            short = "lr" + r["arm"].split("-lr")[1].split("-")[0] + "-" + r["arm"].rsplit("-", 1)[1]
            dp = np.array(r["dprime"])
            print(f"{short:<14} {r['critic']:7.3f} {r['critic_ties_split']:7.3f} {r['poisson']:8.3f} "
                  f"{r['whitened']:7.3f} {r['whitened_timed']:7.3f}   {r['zone_on']:5.2f} {r['zone_off']:5.2f} "
                  f"{r['temperature_if_alike'] or float('nan'):7.2f}  {np.median(dp):.2f} [{dp.min():.2f}, {dp.max():.2f}], "
                  f"{int((dp < 0.2).sum())}")


if __name__ == "__main__":
    main()
