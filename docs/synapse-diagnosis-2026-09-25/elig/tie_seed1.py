"""The object engine's seed-1 run (a Teacher under fast.compare, instrumented) against the critic's Rust record of seed 1,
epoch by epoch: the settled summed score, and the per-wave direct posting summed over the epoch."""
import numpy as np
E = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/elig"
C = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/critic-sign"
z = np.load(f"{E}/mnist-seed1-150.npz")
rows = z["rows"]
crit = np.load(f"{C}/synapse-seed1-fresh-4000.npz")["e_j"].sum(1)[: len(rows)]
print("epochs", len(rows), "| objects settled vs critic's Rust record: max |diff|", float(np.abs(rows[:, 1] - crit).max()),
      "| objects direct per-wave posting vs critic's record: max |diff|", float(np.abs(rows[:, 2] - crit).max()))
print("first five: objects settled", np.round(rows[:5, 1], 6), "critic", np.round(crit[:5], 6))
