"""The critic's four synapse records pooled: the summed score's mean, median, and how much of the mean the largest
epochs carry; a permutation-free check that the positive mean is not one or two epochs."""
import glob, math
import numpy as np
D = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/critic-sign"
S = np.concatenate([np.load(p)["e_j"].sum(1) for p in sorted(glob.glob(D + "/synapse-*.npz"))])
n = len(S)
print(f"pooled n {n} mean {S.mean():+.3f} se {S.std(ddof=1)/math.sqrt(n):.3f} median {np.median(S):+.3f} "
      f"share > 0 {np.mean(S > 0):.3f}")
for q in (0.999, 0.99, 0.95):
    cut = np.quantile(S, q)
    print(f"  without the epochs above the {q} quantile ({cut:.1f}): mean {S[S <= cut].mean():+.3f}")
