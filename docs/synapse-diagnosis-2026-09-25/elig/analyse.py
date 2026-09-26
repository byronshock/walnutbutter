"""Read the instrumented runs: the per-wave direct posting against zero, the settled scores against the direct posting,
the per-epoch summed eligibility, the draw calibration and the state bins. usage: analyse.py FILE.npz..."""
import math, sys
import numpy as np
from instrument import XBINS, MBINS


def mse(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


for p in sys.argv[1:]:
    z = np.load(p)
    W, rows = z["waves"], z["rows"]
    ep, D, var, esc, esce, sil, sile, ng, ngx = W.T
    print(f"\n=== {p.split('/')[-1]}: {len(rows)} epochs, {len(W)} waves, {int((ngx > 0).sum())} posting waves "
          f"(a source gated with X > 0), {int(ngx.sum())} posting decisions")
    # identity
    settled, direct, maxdiff, maxabs = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    print(f"identity: max over epochs of max_k |settled e_k - direct e_k| = {maxdiff.max():.3e} (largest |e_k| {maxabs.max():.3g}); "
          f"max |sum settled - sum direct| per epoch {np.abs(settled - direct).max():.3e}")
    # per-epoch direct from waves equals the direct sum over synapses?
    De = np.bincount(ep.astype(int), weights=D, minlength=int(rows[:, 0].max()) + 1)[rows[:, 0].astype(int)]
    Ve = np.bincount(ep.astype(int), weights=var, minlength=int(rows[:, 0].max()) + 1)[rows[:, 0].astype(int)]
    print(f"  per-epoch sum of per-wave D vs direct sum over synapses: max diff {np.abs(De - direct).max():.3e}")
    m, se = mse(settled)
    print(f"summed eligibility at the read, per epoch: mean {m:+.4f} +- {se:.4f} (iid se, n {len(settled)}), z {m / se:+.2f}; "
          f"conditional z = sum / sqrt(sum Var) = {settled.sum() / math.sqrt(Ve.sum()):+.2f}; "
          f"sd {settled.std():.3f} vs sqrt(mean conditional Var) {math.sqrt(Ve.mean()):.3f}")
    m, se = mse(D)
    print(f"per wave, unconditioned (all {len(D)} waves): mean D {m:+.3e} +- {se:.3e}, z {m / se:+.2f}")
    sel = ngx > 0
    m, se = mse(D[sel])
    print(f"per wave, conditioned on posting ({sel.sum()} waves): mean D {m:+.3e} +- {se:.3e}, z {m / se:+.2f}; "
          f"conditional z {D[sel].sum() / math.sqrt(var[sel].sum()):+.2f}")
    print(f"escape half: sum a c X {esc.sum():.3f} vs its conditional expectation sum F P c X {esce.sum():.3f} (ratio {esc.sum() / esce.sum():.5f}); "
          f"silent half: sum (F-a) m X {sil.sum():.3f} vs sum F e^-m m X {sile.sum():.3f} (ratio {sil.sum() / sile.sum():.5f}); "
          f"expectations equal: {esce.sum():.6f} vs {sile.sum():.6f}")
    cal, cal_u = z["cal"], z["cal_u"]
    print(f"draw calibration: gated decisions sum(a - F P) = {cal[0]:+.2f}, sd {math.sqrt(cal[1]):.2f}, z {cal[0] / math.sqrt(cal[1]):+.2f}; "
          f"ungated {cal_u[0]:+.2f}, sd {math.sqrt(cal_u[1]):.2f}, z {cal_u[0] / math.sqrt(max(cal_u[1], 1e-300)):+.2f}")
    for name, arr, edges in (("X_i", z["xb"], XBINS), ("m_i", z["mb"], MBINS)):
        parts = []
        for b in range(len(edges) - 1):
            s, v, n = arr[b]
            if n:
                parts.append(f"[{edges[b]:g},{edges[b + 1]:g}) n {int(n)} z {s / math.sqrt(v):+.2f}")
        print(f"  by {name}: " + "; ".join(parts))
