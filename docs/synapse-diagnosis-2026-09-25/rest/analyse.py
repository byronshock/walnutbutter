"""Read the rest.py records: per 250-epoch window, output activity, rate memories, score, fraction right, selectivity,
potentials at the read, weights, and the drift the rule applies to each output's fan-in (advantage x eligibility)."""
import glob, json, sys
from pathlib import Path

import numpy as np

D = Path("/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/rest")
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 250
which = sys.argv[2:] or None


def eta2(c, lab):
    """Share of each output's count variance explained by the label (between-class / total), per output; nan if flat."""
    out = np.full(c.shape[1], np.nan)
    tot = c.var(0)
    for j in range(c.shape[1]):
        if tot[j] == 0:
            continue
        means = np.array([c[lab == k, j].mean() if (lab == k).any() else 0.0 for k in range(10)])
        w = np.array([(lab == k).mean() for k in range(10)])
        out[j] = (w * (means - c[:, j].mean()) ** 2).sum() / tot[j]
    return out


def arm(path, win=WIN, rows=True):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    spk, rd = d["spk"].astype(float), d["read"].astype(float)
    c = spk + rd
    V, thr = d["V"], d["thresholds"]
    u = np.clip(V, 0, None) / thr
    lab, rew, right = d["label"], d["reward"], d["right"]
    # the baseline as fast.train keeps it: first reward, then b += 0.05 (r - b), the advantage taken before the move
    b = rew[0]; adv = np.empty_like(rew)
    for e, r in enumerate(rew):
        adv[e] = r - b; b += 0.05 * (r - b)
    so = d["score_out"].astype(float)
    snap_e, rates, W = d["snap_epoch"], d["rates"], d["weights"]
    eo = d["edge_outpos"]
    n = len(rew)
    res = []
    for start in range(0, n, win):
        s = slice(start, min(n, start + win))
        end = s.stop
        k = np.searchsorted(snap_e, end)  # the snapshot at the window's end
        k = min(k, len(snap_e) - 1)
        rt = rates[k]; w = W[k]
        et = eta2(c[s], lab[s])
        rng = np.random.default_rng(0)
        et0 = eta2(c[s], rng.permutation(lab[s]))  # the same with the labels shuffled: the floor eta2 reads at by chance
        duty = (c[s] > 0).mean(0)
        wo = np.array([w[eo == j].sum() for j in range(60)])
        drift = (adv[s, None] * so[s]).mean(0)  # the per-epoch expected change of each output's summed fan-in weight / lr
        res.append({
            "end": end,
            "count": c[s].mean(), "spk": spk[s].mean(), "read": rd[s].mean(),
            "outs_counting": (c[s] > 0).sum(1).mean(),
            "rate_hi": int((rt > 0.99).sum()), "rate_lo": int((rt < 0.01).sum()), "rate_mean": float(rt.mean()),
            "rate_hi_ones": int((rt[:30] > 0.99).sum()), "rate_hi_zeros": int((rt[30:] > 0.99).sum()),
            "score": rew[s].mean(), "right": right[s].mean(),
            "duty_mid": int(((duty > 0.05) & (duty < 0.95)).sum()), "duty_all": int((duty >= 0.95).sum()),
            "eta2": np.nanmean(et), "eta2_null": np.nanmean(et0),
            "u_mean": u[s].mean(), "at_floor": (V[s] <= -0.2222).mean(), "u_hi": (u[s] > 0.9).mean(),
            "w_mean": float(w.mean()), "w_sd": float(w.std()), "w_clamped": float((np.abs(w) >= 0.999999).mean()),
            "fanin_sum": float(wo.mean()),
            "drift": float(drift.mean()), "drift_ones": float(drift[:30].mean()), "drift_zeros": float(drift[30:].mean()),
            "adv_mean": adv[s].mean(), "score_out_mean": so[s].mean(),
        })
    return meta, res, d


def show(path, win=WIN):
    meta, res, d = arm(path, win)
    name = Path(path).stem
    print(f"\n== {name}  ({len(d['reward'])} epochs){'  PARTIAL' if bool(d['partial']) else ''}")
    print("  end  count  spk  read outs>0 r>.99(1/0) r<.01 rmean  score  right dutymid dutyall eta2(null)   u_mean floor  u>.9  "
          "w_mean  w_sd  clamp fanin  drift(1s/0s) x1e3")
    for r in res:
        print(f"{r['end']:5d} {r['count']:5.2f} {r['spk']:5.2f} {r['read']:4.2f} {r['outs_counting']:5.1f}  {r['rate_hi']:2d}({r['rate_hi_ones']:2d}/{r['rate_hi_zeros']:2d})"
              f" {r['rate_lo']:3d}  {r['rate_mean']:.3f} {r['score']:6.2f} {r['right']:.3f}  {r['duty_mid']:3d}   {r['duty_all']:3d}   "
              f"{r['eta2']:.3f}({r['eta2_null']:.3f})  {r['u_mean']:.3f} {r['at_floor']:.3f} {r['u_hi']:.3f} "
              f"{r['w_mean']:+.4f} {r['w_sd']:.3f} {r['w_clamped']:.3f} {r['fanin_sum']:+.3f} "
              f"{1e3*r['drift']:+.2f} ({1e3*r['drift_ones']:+.2f}/{1e3*r['drift_zeros']:+.2f})")


if __name__ == "__main__":
    paths = sorted(glob.glob(str(D / "*.npz")))
    if which:
        paths = [p for p in paths if any(w in Path(p).stem for w in which)]
    for p in paths:
        show(p)
