"""Read cap-*.npz of cap_test.py: over the continuation, the synapses at or above 4 theta/3 (all inputs, and often-on),
how many of the capped ones are back above the line, and busy outputs / spikes in 100-epoch windows."""
import glob, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
for f in sorted(glob.glob(str(HERE / "cap-*.npz"))):
    m = re.match(r".*cap-(\w+)-(cap|nocap)-s(\d)-e(\d+)\.npz", f)
    rule, mode, seed, E = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    z = np.load(f)
    W, sp, rd, trapped, p, t, th, fl = z["w"], z["spikes"], z["reads"], z["trapped"], z["p"], z["t"], z["th"], z["fl"]
    line = (th - fl)[t]; often = p > 0.5
    print(f"-- {rule} rule, {mode}, seed {seed}: {int(trapped.sum())} synapses >= 4th/3 at 25k ({int((trapped & often).sum())} from often-on inputs); "
          f"score trace {np.round(z['trace'], 2).tolist()}")
    print("   epoch | >=4th/3 all | often-on | capped ones back >=4th/3 | capped ones mean w/th | busy (last 100) | spikes/ep | reads/ep")
    for k in range(len(W)):
        e = 100 * k
        if e not in (0, 100, 200, 500, 1000, 1500, 2000, 2500, 3000) or e > E:
            continue
        w = W[k]; above = w >= line
        back = int((above & trapped).sum()); mw = float((w[trapped] / th[t][trapped]).mean()) if trapped.any() else float("nan")
        if e == 0:
            s = sp[:100]; lab = "first 100"
        else:
            s = sp[e - 100:e]; lab = ""
        b = int(((s > 0).mean(0) > 0.99).sum())
        r = rd[max(0, e - 100):max(e, 100)].sum(1).mean()
        print(f"   {e:5d} | {int(above.sum()):11d} | {int((above & often).sum()):8d} | {back:24d} | {mw:21.3f} | {b:6d} {lab:>9s} | {s.sum(1).mean():9.1f} | {r:8.1f}")
    print()
