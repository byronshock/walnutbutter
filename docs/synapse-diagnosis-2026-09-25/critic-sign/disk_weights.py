"""From the 25k sweeps on disk (read only): mean input->output weight and output spikes at the end, every lr, both rules."""
import glob, json, re
from pathlib import Path
R = Path("/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure/runs")
rows = []
for f in sorted(glob.glob(str(R / "synapse-lr-25k/*-network.json")) + glob.glob(str(R / "neuron-lr-25k/*-network.json"))):
    d = json.load(open(f))
    w = d["weights"]
    s = json.load(open(f.replace("-network.json", ".json")))
    lr = re.search(r"lr([0-9.]+)-", f).group(1)
    drive = "charged" if "drivecharged" in f else "rate"
    seed = re.search(r"seed(\d)", f).group(1)
    rows.append((d["exploration"], drive, float(lr), seed, sum(w) / len(w), sum(1 for x in w if x >= 0.999) , sum(1 for x in w if x <= -0.999), sum(s["output_counts_last"]), s["stuck_on"], s["last_tenth"]))
for r in sorted(rows):
    print(f"{r[0]:8s} {r[1]:8s} lr {r[2]:<7g} seed {r[3]}: mean w {r[4]:+.4f}  at +1 {r[5]:4d} at -1 {r[6]:4d}  last-epoch output count {r[7]:4d}  stuck on {r[8]:2d}  last tenth {r[9]:+.3f}")
