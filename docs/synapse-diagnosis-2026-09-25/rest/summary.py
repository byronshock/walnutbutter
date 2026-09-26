"""One line per arm: early (epochs 1-500) against late (4001-5000) activity, rate memories, score, right, selectivity,
and the weights' mean movement; then the eligibility's reach (elig.py's columns) for the same arms."""
import glob
from pathlib import Path

import numpy as np

import importlib.util
spec = importlib.util.spec_from_file_location("an", str(Path(__file__).with_name("analyse.py")))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
D = Path(__file__).parent

print("arm                        | count/out/epoch early->late (spk, read) | outs>0 | rate>.99 at 500 -> 5000 (ones/zeros) | "
      "score early->late | right late | eta2 late (null) | duty 5-95% late | w_mean 0->5000 | |elig|")
for p in sorted(glob.glob(str(D / "*.npz"))):
    meta, res, d = an.arm(p, win=500)
    _, late, _ = an.arm(p, win=1000)
    e, l = res[0], late[-1]
    W = d["weights"]
    so = d["score_out"]
    print(f"{Path(p).stem:27s}| {e['count']:5.2f} -> {l['count']:5.2f} ({l['spk']:4.2f}, {l['read']:4.2f})     | {l['outs_counting']:4.1f}  "
          f"| {e['rate_hi']:2d} -> {res[-1]['rate_hi']:2d} ({res[-1]['rate_hi_ones']:2d}/{res[-1]['rate_hi_zeros']:2d})                   "
          f"| {e['score']:6.2f} -> {l['score']:6.2f} | {l['right']:.3f}      | {l['eta2']:.3f} ({l['eta2_null']:.3f})    "
          f"| {l['duty_mid']:2d}              | {W[0].mean():+.4f} -> {W[-1].mean():+.4f} | {np.abs(so).mean():.2f}")
