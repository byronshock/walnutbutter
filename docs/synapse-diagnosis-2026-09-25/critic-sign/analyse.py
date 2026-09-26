"""Read the probe_arm.py records and print the numbers: the critic against activity, and the update split by the sign of e."""
import glob, math, sys
from pathlib import Path

import numpy as np

D = Path("/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/critic-sign")
K = np.arange(60)
CLS = (K // 3) % 10
ONE = K < 30


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan"), (float("nan"), float("nan"))
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    half = 1.96 / math.sqrt(n - 3)
    z = math.atanh(r)
    return r, (math.tanh(z - half), math.tanh(z + half))


def mse(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


def fmt(m_se, scale=1.0, p=3):
    m, se = m_se
    return f"{m * scale:+.{p}f} +- {se * scale:.{p}f}"


def one(path, skip=0):
    z = np.load(path)
    lr = float(z["lr"])
    R, b, A, y = z["R"][skip:], z["b"][skip:], z["A"][skip:], z["label"][skip:].astype(int)
    spk, rd = z["spk"][skip:], z["rd"][skip:]
    cnt = spk + rd
    ej, epos, eneg = z["e_j"][skip:], z["epos_j"][skip:], z["eneg_j"][skip:]
    dw, ws = z["dw_j"][skip:], z["wsum_j"][skip:]
    dRp, dRm = z["dRp"][skip:], z["dRm"][skip:]
    n = len(R)
    tot = cnt.sum(1)
    ones_tot = cnt[:, ONE].sum(1)
    zeros_tot = cnt[:, ~ONE].sum(1)
    lab_one = np.array([cnt[t, (CLS == y[t]) & ONE].sum() for t in range(n)])
    lab_zero = np.array([cnt[t, (CLS == y[t]) & ~ONE].sum() for t in range(n)])
    ny = lab_one - lab_zero
    print(f"\n=== {Path(path).stem}  (epochs {int(z['offset']) + skip + 1}-{int(z['offset']) + skip + n}, lr {lr})")
    print(f"R mean {R.mean():.3f} (uniform -2.303); b mean {b.mean():.3f}; A mean {fmt(mse(A))}, sd {A.std():.3f}; "
          f"share A>0 {np.mean(A > 0):.3f}")
    print(f"per epoch: count {tot.mean():.1f} (spikes {spk.sum(1).mean():.1f}, read escapes {rd.sum(1).mean():.1f}); "
          f"fire-if-one {ones_tot.mean():.1f}, fire-if-zero {zeros_tot.mean():.1f}; label's fire-if-one {lab_one.mean():.2f}, "
          f"label's fire-if-zero {lab_zero.mean():.2f}")
    for nm, x in (("total count", tot), ("total spikes", spk.sum(1)), ("total read escapes", rd.sum(1)),
                  ("fire-if-one total", ones_tot), ("fire-if-zero total", zeros_tot), ("label's fire-if-one", lab_one),
                  ("label's fire-if-zero", lab_zero), ("n_y (label evidence)", ny)):
        rA, ciA = corr(A, x)
        rR, ciR = corr(R, x)
        print(f"  corr(A, {nm:22s}) {rA:+.3f} [{ciA[0]:+.3f},{ciA[1]:+.3f}]   corr(R, .) {rR:+.3f}")
    # the update, LR*A*e over input->output synapses, split by the sign of e
    Ep, En = epos.sum(1), eneg.sum(1)
    Up, Un = lr * A * Ep, lr * A * En
    U = Up + Un
    print(f"  score e summed over in->out (no A): e>0 part {fmt(mse(Ep))}, e<0 part {fmt(mse(En))}, net {fmt(mse(Ep + En))}"
          f"  [net/sd of net {np.mean(Ep + En) / np.std(Ep + En):+.4f}]")
    print(f"  update LR*A*e summed, x1e3:  from e>0 {fmt(mse(Up), 1e3, 4)}   from e<0 {fmt(mse(Un), 1e3, 4)}   net {fmt(mse(U), 1e3, 4)}")
    for cond, nm in ((A > 0, "A>0"), (A < 0, "A<0")):
        print(f"    given {nm} ({cond.sum()} epochs): from e>0 {fmt(mse(Up[cond]), 1e3, 4)}   from e<0 {fmt(mse(Un[cond]), 1e3, 4)}"
              f"   net {fmt(mse(U[cond]), 1e3, 4)}   | e net {fmt(mse((Ep + En)[cond]))}")
    covAe = np.mean((A - A.mean()) * ((Ep + En) - (Ep + En).mean()))
    print(f"  E[U] = LR*(E[A]E[e] + Cov(A,e)): LR*E[A]*E[e] = {lr * A.mean() * (Ep + En).mean() * 1e3:+.4f}e-3, "
          f"LR*Cov(A,e) = {lr * covAe * 1e3:+.4f}e-3;  corr(A, net e) {corr(A, Ep + En)[0]:+.3f}")
    print(f"  applied weight change summed over in->out, x1e3: {fmt(mse(dw.sum(1)), 1e3, 4)}; "
          f"sum of in->out weights {ws[0].sum():.3f} -> {ws[-1].sum():.3f} (change {ws[-1].sum() - ws[0].sum():+.3f}); "
          f"at the bounds low/high {int(z['atlow'][skip])}/{int(z['athigh'][skip])} -> {int(z['atlow'][-1])}/{int(z['athigh'][-1])}")
    clip = dw.sum(1) - U  # what the clip to the weight range took off LR*A*e, summed over in->out
    print(f"  applied = unclipped LR*A*e + clip residual; over the run, summed: unclipped {U.sum():+.3f}, clip {clip.sum():+.3f}"
          f" (per epoch x1e3 {fmt(mse(clip), 1e3, 4)}; epochs with clip < 0: {np.mean(clip < -1e-12):.3f}, > 0: {np.mean(clip > 1e-12):.3f})")
    # the critic's marginal: what one more count at output j pays this epoch
    dR = np.where(ONE[None, :], dRp[:, CLS], dRm[:, CLS])
    print(f"  critic's marginal for one more count, R(n+1_j)-R(n): mean over all 60 outputs {fmt(mse(dR.mean(1)), 1, 4)}; "
          f"fire-if-one {dR[:, ONE].mean():+.4f}, fire-if-zero {dR[:, ~ONE].mean():+.4f}")
    w_rd = (dR * rd).sum() / max(rd.sum(), 1)
    w_sp = (dR * spk).sum() / max(spk.sum(), 1)
    print(f"    weighted by where read escapes fell {w_rd:+.4f}; by where spikes fell {w_sp:+.4f}; "
          f"one more count on every output at once: {fmt(mse(dR.sum(1)), 1, 4)} (first order)")
    # per output: the expected update direction of its fan-in, Cov(A, e_j), against its mean marginal
    cov_j = np.array([np.mean((A - A.mean()) * (ej[:, j] - ej[:, j].mean())) for j in range(60)])
    mdR = dR.mean(0)
    act = cnt.mean(0)
    print(f"  per output (60): LR*Cov(A,e_j) x1e3 sum {lr * cov_j.sum() * 1e3:+.4f}, positive for {int((cov_j > 0).sum())}/60; "
          f"corr over outputs with mean marginal {corr(cov_j, mdR)[0]:+.3f}, with mean count {corr(cov_j, act)[0]:+.3f}; "
          f"mean marginal vs mean count {corr(mdR, act)[0]:+.3f}")
    print(f"    applied weight change per output over the run vs its mean count: corr {corr(dw.sum(0), act)[0]:+.3f}; "
          f"outputs whose fan-in sum rose {int((dw.sum(0) > 0).sum())}/60")
    return dict(corrA=corr(A, tot)[0], U=U.mean(), Up=Up.mean(), Un=Un.mean())


if __name__ == "__main__":
    skip = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for p in sorted(glob.glob(str(D / "*.npz"))):
        one(p, skip)
