"""Re-read the critic investigator's probe records: the per-epoch summed score e over in->out synapses (e_j summed over
the 60 outputs), its mean, the i.i.d. standard error, the lag autocorrelations, and a block standard error."""
import glob, math
import numpy as np
D = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/critic-sign"
tot_s = tot_w = 0.0
for p in sorted(glob.glob(D + "/synapse-*.npz")) + sorted(glob.glob(D + "/neuron-*.npz")):
    z = np.load(p)
    s = z["e_j"].sum(1)
    n = len(s)
    se = s.std(ddof=1) / math.sqrt(n)
    ac = [float(np.corrcoef(s[:-k], s[k:])[0, 1]) for k in (1, 2, 5)]
    b = 50
    blocks = s[: n // b * b].reshape(-1, b).mean(1)
    bse = blocks.std(ddof=1) / math.sqrt(len(blocks))
    from scipy import stats
    sk = stats.skew(s)
    print(f"{p.split('/')[-1]:40s} n {n} mean {s.mean():+.3f} se {se:.3f} z {s.mean()/se:+.2f} | block50 se {bse:.3f} | "
          f"ac1,2,5 {ac[0]:+.3f} {ac[1]:+.3f} {ac[2]:+.3f} | sd {s.std():.2f} skew {sk:+.2f} | first-half {s[:n//2].mean():+.3f} second {s[n//2:].mean():+.3f}")
    if "synapse" in p:
        tot_s += s.sum(); tot_w += n
