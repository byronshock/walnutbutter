"""Weight drift, start (rebuilt by grid_of from the arm's knobs and seed) against the 25k checkpoint, every arm of
runs/synapse-lr-25k and runs/neuron-lr-25k. Read only. Writes per-arm numbers to drift_static.json beside this file."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("rs", "docs/rust-sweep.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from walnutbutter.persistence import wiring_digest
from walnutbutter.mnist import pixel_statistics

HERE = Path(__file__).resolve().parent
SYN_FIX = ("--exploration", "synapse", "--hazard-family", "loglinear", "--synapse-scaling", "count", "--trace-counts", "all")
NEU_FIX = ("--exploration", "neuron", "--drive", "rate")
BINS = [(0.0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 0.98), (0.98, 1.0001)]


def arms():
    for lr in (0.0005, 0.001, 0.002, 0.005, 0.01):
        for drive in ("rate", "charged"):
            for seed in (1, 2, 3):
                yield "synapse", "synapse-lr-25k", {"interval": 100.0, "lr": lr, "threshold": 0.6, "minimum_potential": -0.2,
                       "hidden_neurons": 0.0, "temperature": 2.0, "synapse_hazard": 0.01, "drive": drive, "seed": seed}, SYN_FIX
    for lr in (0.0005, 0.002):
        for seed in (1, 2, 3):
            yield "neuron", "neuron-lr-25k", {"interval": 100.0, "lr": lr, "delta": 0.3125, "threshold": 0.6, "minimum_potential": -0.2,
                   "hidden_neurons": 0.0, "temperature": 2.0, "seed": seed}, NEU_FIX


def analyse(rule, folder, arm, fix):
    g, cli = rs.grid_of("mnist", arm, "hazard", True, None, None, fix)
    name = rs.arm_name(arm)
    d = json.load(open(f"runs/{folder}/{name}-network.json"))
    assert d["wiring_digest"] == wiring_digest(g), name
    assert d["epoch"] == 25000, (name, d["epoch"])
    n_conn = len(g.connections)
    conns = [g.connections[i] for i in range(1, n_conn + 1)]
    w0 = np.array([c.weight for c in conns]); w1 = np.array(d["weights"]); dw = w1 - w0
    on, _ = pixel_statistics(g.clock)
    inrow = {n: i for i, n in enumerate(g.input_row())}
    allidx = {n: i for i, n in enumerate(g.all_neurons())}
    outs = list(dict.fromkeys(g.output_row()))
    oidx = {n: k for k, n in enumerate(outs)}
    p = np.array([on[inrow[c.source]] for c in conns])
    kind = np.array([0 if inrow[c.source] < g.clock else (1 if inrow[c.source] < g.clock + 196 else 2) for c in conns])  # clock, raw, complement
    tgt = np.array([oidx[c.target] for c in conns])
    theta0 = np.array([n.threshold for n in outs])
    theta1 = np.array([d["thresholds"][allidx[n]] for n in outs])
    rate = np.array([d["rates"][allidx[n]] for n in outs])
    lo, hi = g.weight_range
    r = {"rule": rule, "folder": folder, "arm": name, "lr": arm["lr"], "drive": arm.get("drive", "rate"), "seed": arm["seed"],
         "n": int(n_conn), "mean_dw": float(dw.mean()), "sd_dw": float(dw.std()), "median_dw": float(np.median(dw)),
         "mean_abs_dw": float(np.abs(dw).mean()), "frac_dw_pos": float((dw > 0).mean()),
         "mean_w0": float(w0.mean()), "mean_w1": float(w1.mean()),
         "frac_at_hi0": float((w0 >= hi - 1e-9).mean()), "frac_at_hi1": float((w1 >= hi - 1e-9).mean()),
         "frac_at_lo1": float((w1 <= lo + 1e-9).mean()),
         "sign_pos": {"n": int((w0 > 0).sum()), "mean_dw": float(dw[w0 > 0].mean()), "frac_pos": float((dw[w0 > 0] > 0).mean())},
         "sign_neg": {"n": int((w0 < 0).sum()), "mean_dw": float(dw[w0 < 0].mean()), "frac_pos": float((dw[w0 < 0] > 0).mean())},
         "frac_sign_flipped": float((np.sign(w0) != np.sign(w1)).mean()),
         "flip_neg_to_pos": int(((w0 < 0) & (w1 > 0)).sum()), "flip_pos_to_neg": int(((w0 > 0) & (w1 < 0)).sum()),
         "pct": {q: float(np.percentile(dw, q)) for q in (5, 25, 50, 75, 95)},
         "corr_dw_w0": float(np.corrcoef(dw, w0)[0, 1]),
         "corr_dw_p": float(np.corrcoef(dw, p)[0, 1]),
         }
    r["bins"] = []
    for a, b in BINS:
        m = (p >= a) & (p < b)
        if m.sum():
            r["bins"].append({"lo": a, "hi": b, "n": int(m.sum()), "mean_dw": float(dw[m].mean()), "frac_pos": float((dw[m] > 0).mean()),
                              "mean_dw_w0pos": float(dw[m & (w0 > 0)].mean()) if (m & (w0 > 0)).any() else None,
                              "mean_dw_w0neg": float(dw[m & (w0 < 0)].mean()) if (m & (w0 < 0)).any() else None})
    r["by_kind"] = {nm: {"n": int((kind == k).sum()), "mean_dw": float(dw[kind == k].mean())} for k, nm in enumerate(("clock", "raw", "complement")) if (kind == k).any()}
    # per output: expected drive per presentation = sum over its synapses of P(input on) * w, in units of its threshold
    D0 = np.bincount(tgt, weights=p * w0, minlength=len(outs)); D1 = np.bincount(tgt, weights=p * w1, minlength=len(outs))
    fan = np.bincount(tgt, minlength=len(outs))
    mdw = np.bincount(tgt, weights=dw, minlength=len(outs)) / fan
    busy = rate > 0.99
    r["outputs"] = {"rate": rate.tolist(), "theta0": theta0.tolist(), "theta1": theta1.tolist(), "D0": D0.tolist(), "D1": D1.tolist(),
                    "fan": fan.tolist(), "mean_dw": mdw.tolist()}
    r["n_busy"] = int(busy.sum())
    r["theta_changed"] = float(np.abs(theta1 - theta0).max())
    r["D0_over_theta_mean"] = float((D0 / theta0).mean()); r["D1_over_theta_mean"] = float((D1 / theta1).mean())
    r["dD_over_theta_mean"] = float(((D1 - D0) / theta0).mean())
    r["dD_over_theta_busy"] = float(((D1 - D0) / theta0)[busy].mean()) if busy.any() else None
    r["dD_over_theta_quiet"] = float(((D1 - D0) / theta0)[~busy].mean()) if (~busy).any() else None
    r["D0_over_theta_busy"] = float((D0 / theta0)[busy].mean()) if busy.any() else None
    r["D0_over_theta_quiet"] = float((D0 / theta0)[~busy].mean()) if (~busy).any() else None
    r["D1_over_theta_busy"] = float((D1 / theta1)[busy].mean()) if busy.any() else None
    r["D1_over_theta_quiet"] = float((D1 / theta1)[~busy].mean()) if (~busy).any() else None
    r["mean_dw_busy"] = float(mdw[busy].mean()) if busy.any() else None
    r["mean_dw_quiet"] = float(mdw[~busy].mean()) if (~busy).any() else None
    r["corr_rate_D0"] = float(np.corrcoef(rate, D0 / theta0)[0, 1])
    r["corr_rate_D1"] = float(np.corrcoef(rate, D1 / theta1)[0, 1])
    r["corr_rate_dD"] = float(np.corrcoef(rate, (D1 - D0) / theta0)[0, 1])
    # fire-if-one (first half of the zone) against fire-if-zero (second half), complement coding (§5.11)
    half = len(outs) // 2
    r["fire_if_one_busy"] = int(busy[:half].sum()); r["fire_if_zero_busy"] = int(busy[half:].sum())
    return r


if __name__ == "__main__":
    out = []
    for rule, folder, arm, fix in arms():
        r = analyse(rule, folder, arm, fix)
        out.append(r)
        print(f"{rule:7s} {r['drive']:7s} lr {r['lr']:<6g} s{r['seed']}  mean dw {r['mean_dw']:+.4f}  sd {r['sd_dw']:.3f}  "
              f"w0>0 {r['sign_pos']['mean_dw']:+.4f}  w0<0 {r['sign_neg']['mean_dw']:+.4f}  flips -+ {r['flip_neg_to_pos']:3d} +- {r['flip_pos_to_neg']:3d}  "
              f"at+1 {r['frac_at_hi1']:.3f} at-1 {r['frac_at_lo1']:.3f}  busy {r['n_busy']:2d}  dD/th {r['dD_over_theta_mean']:+.3f} "
              f"(busy {r['dD_over_theta_busy'] if r['dD_over_theta_busy'] is None else round(r['dD_over_theta_busy'],3)}, quiet {r['dD_over_theta_quiet'] if r['dD_over_theta_quiet'] is None else round(r['dD_over_theta_quiet'],3)})  "
              f"theta moved {r['theta_changed']:.3g}", flush=True)
    (HERE / "drift_static.json").write_text(json.dumps(out))
