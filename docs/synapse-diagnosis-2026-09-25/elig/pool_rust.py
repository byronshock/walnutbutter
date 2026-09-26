"""Pool the Rust repeats of the critic's measurement (fresh seeds 3-10) and set them beside the critic's four records."""
import glob, math
import numpy as np
E = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/elig"
C = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/critic-sign"
def line(name, S):
    se = S.std(ddof=1) / math.sqrt(len(S))
    return f"{name:34s} n {len(S):6d} mean {S.mean():+.3f} se {se:.3f} z {S.mean() / se:+.2f}"
new = []
for s in range(3, 11):
    S = np.load(f"{E}/rust-seed{s}-3000.npz")["S_inout"]
    new.append(S); print(line(f"rust fresh seed {s}", S))
S1 = np.load(f"{E}/rust-seed1-4000.npz")["S_inout"]
Sc1 = np.load(f"{C}/synapse-seed1-fresh-4000.npz")["e_j"].sum(1)
print(line("rust fresh seed 1 (repeat)", S1), "| critic's seed 1 record, epoch by epoch, max |diff|", float(np.abs(S1 - Sc1).max()))
crit = [np.load(p)["e_j"].sum(1) for p in sorted(glob.glob(C + "/synapse-*.npz"))]
for s, st in ((1, 101), (1, 102), (2, 201), (2, 202)):
    S = np.load(f"{E}/rust-seed{s}-stream{st}-4000.npz")["S_inout"]
    new.append(S); print(line(f"rust network {s}, stream {st}", S))
print(line("NEW: the four other-stream runs", np.concatenate(new[8:])))
N = np.concatenate(new); K = np.concatenate(crit)
print(line("NEW: all twelve pooled", N))
print(line("CRITIC: its four pooled", K))
print(line("ALL sixteen pooled", np.concatenate([N, K])))
z = [x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) for x in new + crit]
print("per-record z:", " ".join(f"{v:+.2f}" for v in z), "| sum z^2", round(sum(v * v for v in z), 2), "on", len(z), "df")
print("difference critic - new:", f"{K.mean() - N.mean():+.3f} +- {math.hypot(K.std(ddof=1)/math.sqrt(len(K)), N.std(ddof=1)/math.sqrt(len(N))):.3f}")
