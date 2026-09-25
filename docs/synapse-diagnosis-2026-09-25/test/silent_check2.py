"""Per output, in the floor arms at lr 0.01: share of its often-on synapses (0 < w < own line) that did not move in a
250-epoch window, against the output's own spikes and read escapes in that window."""
import re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
exec(open(HERE / "analyse_floor.py").read().split("files = sorted")[0])
for tag in ("floor-1.2-lr0.01-s1-e5000", "floor-1.2-lr0.01-s2-e5000", "floor-0.6-lr0.01-s1-e5000", "floor-0.6-lr0.01-s2-e5000"):
    seed = int(re.search(r"-s(\d)", tag).group(1))
    z = np.load(HERE / f"ft-{tag}.npz"); p, t = meta(seed)
    W, sp, rd, th, fl = z["w"], z["spikes"], z["reads"], z["th"], z["fl"]
    nwin = len(W) - 1
    own = np.array([sp[k*250:(k+1)*250].sum(0) for k in range(nwin)]); rds = np.array([rd[k*250:(k+1)*250].sum(0) for k in range(nwin)])
    X = W[:-1] / th[t]; line = (th - fl)[t]; R = W[:-1] / line
    mv = np.abs(np.diff(W, axis=0)); often = np.broadcast_to(p > 0.5, X.shape)
    mk = often & (X > 0) & (R < 1); un = mk & (mv == 0)
    quiet = own[:, t] == 0  # no own spike in the window (read escapes may still come, at the rest hazard)
    print(f"{tag}: unmoved {un.sum()}/{mk.sum()}; of the unmoved, target had no own spike in the window {(un & quiet).sum()}; "
          f"moved with a spikeless target {((mk & ~un) & quiet).sum()}")
    print(f"   outputs with no own spike in the last window {(own[-1] == 0).sum()}, their read escapes/epoch there {rds[-1][own[-1]==0].sum()/250:.2f}; "
          f"outputs spikeless in >= half the windows {((own == 0).mean(0) >= 0.5).sum()}")
