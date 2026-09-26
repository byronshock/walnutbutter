"""Pool the instrumented object runs by kind (goo 60, mnist's point): waves, posting decisions, the per-wave direct
posting's mean against zero (unconditioned, and conditioned on a posting wave) with its standard error, the conditional
z, the summed eligibility per epoch, the identity's worst case, the draw calibration."""
import glob, math
import numpy as np
E = "/tmp/claude-1000/-home-byron-Documents-code-walnutbutter/5e969395-0125-4ccd-9daf-efdcffa9eda4/scratchpad/diag/elig"
for kind, pat in (("goo 60", "goo60-seed*-5000.npz"), ("mnist", "mnist-seed*-[14]*0.npz")):
    files = sorted(glob.glob(f"{E}/{pat}"))
    Ws, Rs, cal, calu = [], [], np.zeros(2), np.zeros(2)
    for p in files:
        z = np.load(p); Ws.append(z["waves"]); Rs.append(z["rows"]); cal += z["cal"][:2]; calu += z["cal_u"]
    W, R = np.concatenate(Ws), np.concatenate(Rs)
    D, var, ngx = W[:, 1], W[:, 2], W[:, 8]
    sel = ngx > 0
    se = D.std(ddof=1) / math.sqrt(len(D)); sec = D[sel].std(ddof=1) / math.sqrt(sel.sum())
    S = R[:, 1]; seS = S.std(ddof=1) / math.sqrt(len(S))
    Vep = [np.bincount(w[:, 0].astype(int), weights=w[:, 2]) for w in Ws]
    print(f"{kind}: {len(files)} runs {[p.split('/')[-1] for p in files]}")
    print(f"  {len(R)} epochs, {len(W)} waves, {int(sel.sum())} posting waves, {int(ngx.sum())} posting decisions")
    print(f"  per wave unconditioned: mean {D.mean():+.3e} +- {se:.3e} (z {D.mean() / se:+.2f}); conditioned on posting: "
          f"{D[sel].mean():+.3e} +- {sec:.3e} (z {D[sel].mean() / sec:+.2f}); conditional z {D.sum() / math.sqrt(var.sum()):+.2f}")
    print(f"  summed eligibility per epoch: {S.mean():+.4f} +- {seS:.4f} (z {S.mean() / seS:+.2f}); "
          f"identity worst per synapse {R[:, 3].max():.2e}, per epoch sum {np.abs(R[:, 1] - R[:, 2]).max():.2e}")
    print(f"  draw calibration, gated: sum(a - F P) {cal[0]:+.1f} sd {math.sqrt(cal[1]):.1f} z {cal[0] / math.sqrt(cal[1]):+.2f}; "
          f"ungated {calu[0]:+.1f} sd {math.sqrt(calu[1]):.1f} z {calu[0] / math.sqrt(calu[1]):+.2f}")
