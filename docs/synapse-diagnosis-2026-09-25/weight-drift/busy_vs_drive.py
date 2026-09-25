"""Per output at 25k: share busy (rate memory > 0.99) by its expected drive a presentation in threshold units, start and end."""
import json, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
R = json.load(open(HERE / "drift_static.json"))
groups = collections.OrderedDict()
for r in R:
    groups.setdefault((r["rule"], r["drive"], r["lr"]), []).append(r)
edges = [-99, -2, -1, -0.5, 0, 0.5, 1, 2, 99]
lab = ["<-2", "-2..-1", "-1..-.5", "-.5..0", "0..0.5", "0.5..1", "1..2", ">2"]
for key, name in (("D1", "drive at 25k"), ("D0", "drive at start")):
    print(f"share of outputs busy at 25k, by {name} / threshold (n outputs in the cell, 3 seeds pooled)")
    print(f"{'rule':8s}{'drive':8s}{'lr':>7s} | " + " | ".join(f"{l:>11s}" for l in lab))
    for (rule, drive, lr), rs in groups.items():
        D = np.concatenate([np.array(r["outputs"][key]) / np.array(r["outputs"]["theta0"]) for r in rs])
        rate = np.concatenate([r["outputs"]["rate"] for r in rs])
        cells = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (D >= a) & (D < b)
            cells.append(f"{(rate[m] > 0.99).mean():.2f} ({m.sum():3d})" if m.any() else "     -     ")
        print(f"{rule:8s}{drive:8s}{lr:7g} | " + " | ".join(f"{c:>11s}" for c in cells))
    print()
# of outputs busy at 25k, how much of the fan-in's positive and negative weight did they keep
print("sum of positive and of negative weights into an output, p-weighted, start -> 25k, busy vs quiet (mean over outputs, seeds pooled)")
