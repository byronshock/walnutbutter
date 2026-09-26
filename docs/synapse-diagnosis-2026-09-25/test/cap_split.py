"""Cap test: busy (own spike in >99% of epochs) before and after the cap, split by whether the output held a synapse
>= 4 theta/3 at 25k; first 100 and last 500 epochs of the continuation."""
import numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
for rule in ("synapse", "neuron"):
    for s in (1, 2):
        a = np.load(H / f"cap-{rule}-nocap-s{s}-e3000.npz"); b = np.load(H / f"cap-{rule}-cap-s{s}-e3000.npz")
        hold = np.zeros(60, bool); hold[np.unique(a["t"][a["trapped"]])] = True
        def busy(z, lo, hi): return (z["spikes"][lo:hi] > 0).mean(0) > 0.99
        def rate(z, lo, hi): return z["spikes"][lo:hi].mean(0)
        for lo, hi, nm in ((0, 100, "first 100"), (2500, 3000, "last 500")):
            bn, bc = busy(a, lo, hi), busy(b, lo, hi); rn, rc = rate(a, lo, hi), rate(b, lo, hi)
            print(f"{rule} s{s} {nm:9s}: holders {hold.sum():2d}: busy uncapped {int((bn & hold).sum()):2d} capped {int((bc & hold).sum()):2d}, "
                  f"spikes/epoch/output {rn[hold].mean():5.2f} -> {rc[hold].mean():5.2f} | non-holders {(~hold).sum():2d}: busy {int((bn & ~hold).sum()):2d} -> {int((bc & ~hold).sum()):2d}, "
                  f"spikes {rn[~hold].mean():5.2f} -> {rc[~hold].mean():5.2f}")
