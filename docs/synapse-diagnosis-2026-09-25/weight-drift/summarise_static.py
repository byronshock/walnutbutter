"""Tables from drift_static.json: per (rule, drive, lr), seeds pooled or averaged."""
import json, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "drift_static.json"))
groups = collections.OrderedDict()
for r in R:
    groups.setdefault((r["rule"], r["drive"], r["lr"]), []).append(r)
f = lambda xs: f"{np.mean(xs):+.4f}"
print("TABLE A: change on input->output synapses, start (grid_of rebuild) to epoch 25,000; mean over 3 seeds (min..max over seeds)")
print(f"{'rule':8s}{'drive':8s}{'lr':>7s} | {'mean dw':>22s} | {'sd dw':>6s} | {'w0>0 mean dw':>13s} | {'w0<0 mean dw':>13s} | {'frac dw>0':>9s} | flips -to+ / +to- | busy(>0.99)")
for (rule, drive, lr), rs in groups.items():
    m = [r["mean_dw"] for r in rs]
    print(f"{rule:8s}{drive:8s}{lr:7g} | {np.mean(m):+.4f} ({min(m):+.4f}..{max(m):+.4f}) | {np.mean([r['sd_dw'] for r in rs]):6.3f} | "
          f"{f([r['sign_pos']['mean_dw'] for r in rs]):>13s} | {f([r['sign_neg']['mean_dw'] for r in rs]):>13s} | "
          f"{np.mean([r['frac_dw_pos'] for r in rs]):9.3f} | {np.mean([r['flip_neg_to_pos'] for r in rs]):5.1f} / {np.mean([r['flip_pos_to_neg'] for r in rs]):5.1f} | "
          f"{[r['n_busy'] for r in rs]}")
print()
print("TABLE B: mean dw by P(coded input on) over the training split, seeds pooled (edge counts per bin summed over the 3 seeds)")
bins = [(b["lo"], b["hi"]) for b in R[0]["bins"]]
print(f"{'rule':8s}{'drive':8s}{'lr':>7s} | " + " | ".join(f"p[{lo:.2f},{min(hi,1):.2f})" for lo, hi in bins))
for (rule, drive, lr), rs in groups.items():
    cells = []
    for k in range(len(bins)):
        ns = [r["bins"][k]["n"] for r in rs]; ms = [r["bins"][k]["mean_dw"] for r in rs]
        cells.append(f"{np.average(ms, weights=ns):+.4f}")
    print(f"{rule:8s}{drive:8s}{lr:7g} | " + " | ".join(f"{c:>14s}" for c in cells))
print("edges per bin (seed 1):", [b["n"] for b in R[0]["bins"]])
print()
print("TABLE B2: same bins, split by sign of starting weight: mean dw for w0>0 / w0<0, seeds pooled")
for (rule, drive, lr), rs in groups.items():
    cells = []
    for k in range(len(bins)):
        # pooled means need counts by sign; approximate by averaging seed means
        a = [r["bins"][k]["mean_dw_w0pos"] for r in rs if r["bins"][k]["mean_dw_w0pos"] is not None]
        b = [r["bins"][k]["mean_dw_w0neg"] for r in rs if r["bins"][k]["mean_dw_w0neg"] is not None]
        cells.append(f"{np.mean(a):+.3f}/{np.mean(b):+.3f}")
    print(f"{rule:8s}{drive:8s}{lr:7g} | " + " | ".join(f"{c:>14s}" for c in cells))
print()
print("TABLE C: by coded input kind (clock always on / raw pixel bit / its complement), mean dw, seeds averaged")
for (rule, drive, lr), rs in groups.items():
    print(f"{rule:8s}{drive:8s}{lr:7g} | " + " | ".join(f"{k} {np.mean([r['by_kind'][k]['mean_dw'] for r in rs]):+.4f}" for k in ("clock", "raw", "complement")))
print()
print("TABLE D: per output, expected drive a presentation D = sum_i P(input i on) * w_i, in units of the output's threshold;")
print("busy = rate memory > 0.99 at 25k. Mean over outputs, averaged over seeds.")
print(f"{'rule':8s}{'drive':8s}{'lr':>7s} | {'D0/th all':>9s} | {'D0/th busy':>10s} {'quiet':>7s} | {'D25k/th busy':>12s} {'quiet':>7s} | {'dD/th busy':>10s} {'quiet':>7s} | corr(rate,D0) corr(rate,D25k) corr(rate,dD) | fire-if-one/zero busy")
for (rule, drive, lr), rs in groups.items():
    g = lambda k: np.mean([r[k] for r in rs if r[k] is not None])
    print(f"{rule:8s}{drive:8s}{lr:7g} | {g('D0_over_theta_mean'):9.3f} | {g('D0_over_theta_busy'):10.3f} {g('D0_over_theta_quiet'):7.3f} | "
          f"{g('D1_over_theta_busy'):12.3f} {g('D1_over_theta_quiet'):7.3f} | {g('dD_over_theta_busy'):+10.3f} {g('dD_over_theta_quiet'):+7.3f} | "
          f"{g('corr_rate_D0'):13.3f} {g('corr_rate_D1'):15.3f} {g('corr_rate_dD'):13.3f} | {[ (r['fire_if_one_busy'], r['fire_if_zero_busy']) for r in rs]}")
print()
print("TABLE E: per-output mean dw over its fan-in, busy vs quiet outputs, averaged over seeds")
for (rule, drive, lr), rs in groups.items():
    g = lambda k: np.mean([r[k] for r in rs if r[k] is not None])
    print(f"{rule:8s}{drive:8s}{lr:7g} | busy {g('mean_dw_busy'):+.4f}  quiet {g('mean_dw_quiet'):+.4f}")
