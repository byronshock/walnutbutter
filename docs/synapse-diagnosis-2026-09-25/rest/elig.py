"""What each output's eligibility (its fan-in's summed score, engine.scores() after the update) moves with, epoch by
epoch: its own spikes, or its read synapse's escapes; how big it is; and the part of the drift the advantage makes of
it. Also: is an epoch with more output activity a worse epoch (the reward against the total count)?"""
import glob, json, sys
from pathlib import Path

import numpy as np

D = Path("/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/rest")
paths = sorted(glob.glob(str(D / "*.npz")))
if sys.argv[1:]:
    paths = [p for p in paths if any(w in Path(p).stem for w in sys.argv[1:])]
print("arm                         |score| sd   corr(score,spk) corr(score,read)  | corr(reward, total count) "
      "| cov(adv,score) per output: all  ones  zeros  (x1e3) | adv x score from spk-part, read-part (x1e3)")
for p in paths:
    d = np.load(p)
    if bool(d["partial"]):
        continue
    spk, rd, so, rew = d["spk"].astype(float), d["read"].astype(float), d["score_out"].astype(float), d["reward"]
    b = rew[0]; adv = np.empty_like(rew)
    for e, r in enumerate(rew):
        adv[e] = r - b; b += 0.05 * (r - b)
    cs, cr = [], []
    ps, pr = [], []
    for j in range(60):
        if so[:, j].std() > 0 and spk[:, j].std() > 0:
            cs.append(np.corrcoef(so[:, j], spk[:, j])[0, 1])
        if so[:, j].std() > 0 and rd[:, j].std() > 0:
            cr.append(np.corrcoef(so[:, j], rd[:, j])[0, 1])
        # split the eligibility into what the output's spike count and read count linearly predict of it
        X = np.column_stack([np.ones(len(rew)), spk[:, j], rd[:, j]])
        coef, *_ = np.linalg.lstsq(X, so[:, j], rcond=None)
        ps.append((adv * (coef[1] * (spk[:, j] - spk[:, j].mean()))).mean())
        pr.append((adv * (coef[2] * (rd[:, j] - rd[:, j].mean()))).mean())
    tot = (spk + rd).sum(1)
    cov = (adv[:, None] * so).mean(0)
    print(f"{Path(p).stem:28s} {np.abs(so).mean():6.3f} {so.std():6.3f}   {np.nanmean(cs):+.3f}          {np.nanmean(cr):+.3f}"
          f"           | {np.corrcoef(rew, tot)[0, 1]:+.3f}                     "
          f"| {1e3 * cov.mean():+7.2f} {1e3 * cov[:30].mean():+7.2f} {1e3 * cov[30:].mean():+7.2f}"
          f"            | {1e3 * np.mean(ps):+7.2f} {1e3 * np.mean(pr):+7.2f}")
