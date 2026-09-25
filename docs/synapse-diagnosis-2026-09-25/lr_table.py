import json, glob, math, statistics as st, collections
M = "/home/byron/Documents/code/walnutbutter/.claude/worktrees/synapse-measure/runs"
rows = collections.defaultdict(list)
for n in ("neuron-lr-25k", "synapse-lr-25k"):
    for f in glob.glob(f"{M}/{n}/*.json"):
        if f.endswith("-network.json"): continue
        d = json.load(open(f))
        key = ("neuron" if n.startswith("neuron") else "synapse", d.get("drive", "rate"), d["lr"])
        est = d.get("estimator") or []
        rows[key].append((d["explore_seed"], d["last_tenth"], d["accuracy_last_tenth"], d["stuck_on"], d["stuck_off"],
                          est[-1]["corr_cum"] if est else float("nan"), d["epochs_per_second"], d["mean"]))
print(f"uniform line ln(1/10) = {math.log(0.1):.3f}; accuracy chance 0.100; 25,000 epochs, last tenth = the last 2,500")
print(f"{'rule':8s} {'drive':8s} {'lr':>7s} | {'acc (seeds)':24s} {'mean acc':>8s} | {'last-tenth score':>16s} | {'corr':>6s} | stuck on/off | eps")
for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
    r = sorted(rows[key])
    accs = [x[2] for x in r]
    print(f"{key[0]:8s} {key[1]:8s} {key[2]:7g} | {' '.join(f'{a:.3f}' for a in accs):24s} {st.mean(accs):8.3f} | "
          f"{st.mean(x[1] for x in r):8.3f} ({min(x[1] for x in r):.2f}..{max(x[1] for x in r):.2f}) | {st.mean(x[5] for x in r):6.3f} | "
          f"{st.mean(x[3] for x in r):4.0f}/{st.mean(x[4] for x in r):3.0f} | {st.mean(x[6] for x in r):3.0f}  n={len(r)}")
