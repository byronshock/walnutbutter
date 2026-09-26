import json, glob, re, statistics as st, collections
M = "/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure/runs"
groups = collections.defaultdict(list)
for n in ("neuron-floor-200k", "synapse-floor-200k"):
    for f in sorted(glob.glob(f"{M}/{n}/*-network.json")):
        d = json.load(open(f)); rows = d["progress"]["rows"]; est = d["progress"].get("estimator") or []
        arm = f.split("/")[-1]
        lr = re.search(r"lr([0-9.]+)-", arm).group(1); floor = re.search(r"minimum_potential(-[0-9.]+)", arm).group(1)
        groups[(n.split("-")[0], lr, floor)].append((d["epoch"], [float(r[2]) for r in rows], est[-1]["corr_cum"] if est else float("nan")))
print("accuracy per 25k block (mean over the block's 5k windows, then over seeds); chance 0.100")
for key in sorted(groups):
    arms = groups[key]
    ep = min(a[0] for a in arms)
    blocks = []
    for b in range(0, ep // 25000):
        blocks.append(st.mean(st.mean(a[1][b*5:(b+1)*5]) for a in arms))
    last = st.mean(st.mean(a[1][-3:]) for a in arms)
    print(f"{key[0]:7s} lr {key[1]:6s} floor {key[2]:5s} at epoch {ep:6d}: blocks {' '.join(f'{x:.3f}' for x in blocks)} | last 15k {last:.3f} | corr {st.mean(a[2] for a in arms):.3f} | seeds {len(arms)}")
