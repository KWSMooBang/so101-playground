"""T010 analysis: pre-check gate, learning curves, eval table, per-object success, episodes-to-reach, tsv, figure.

    python -m so101_playground.t010_analysis --root /outputs/so101-playground/runs/T010 \
        --plot docs/T010_rl_sim_a2.png --tsv /outputs/so101-playground/results.tsv

Reads <root>/queue/done/*.json (one per episode) and <root>/es/<method>.jsonl (one per ES update).
CIs: 95% percentile bootstrap over episodes (2000 resamples); success also gets a Wilson interval.
Eval deltas vs theta0 are paired by (row, repeat): every theta saw the same jitter and episode id.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from so101_playground.t010_es import (AVG_LAST, ETA, GAPS, LAMBDA, METHODS, PAIRS_PER_UPDATE, SIGMA,
                                      theta_oracle)

OBJECTS = ["black pen", "black remote", "blue paintbrush", "red pen", "purple toothbrush", "razor"]
LABEL = {"theta0": "θ0 (k=1)", "oracle": "oracle (k*)", "epred": "① −e_pred", "mixed": "③ −e_pred+λ·succ"}
EVAL_TAGS = {"theta0": "T010-baseline-theta0", "oracle": "T010-oracle", "epred": "T010-epred-eval",
             "mixed": "T010-mixed-eval"}
COLOR = {"epred": "#2a78d6", "mixed": "#eb6834", "oracle": "#1baf7a", "theta0": "#8a8984"}
RNG = np.random.default_rng(0)
B = 2000
TASK = "T010-wam-rl-sim-a2"
METRIC = "Wan2.2_VAE_latent_mse@1(ticks4-16)"
TSV_COLS = ["run_id", "task", "model", "condition", "n_episodes", "success", "e_track_deg", "e_track@h_deg", "metric",
            "e_pred", "e_pred_paired", "e_ideal_paired", "e_exec", "e_pred_minus_none", "e_pred_minus_none_ci95",
            "rho_pred_track", "n_queries", "notes"]


def load(root: Path):
    res = [json.loads(p.read_text()) for p in sorted((root / "queue" / "done").glob("*.json"))]
    es = {}
    for m in METHODS:
        p = root / "es" / f"{m}.jsonl"
        if p.exists() and p.stat().st_size:
            es[m] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return res, es


def boot_ci(x):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan)
    s = x[RNG.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def summarize(eps):
    out = {"n": len(eps)}
    for k in ("success", "e_pred", "e_track_deg", "path_len_deg", "ticks"):
        v = [float(e[k]) for e in eps if e.get(k) is not None]
        out[k] = float(np.mean(v)) if v else math.nan
        out[k + "_ci"] = boot_ci(v)
    out["n_succ"] = int(sum(bool(e["success"]) for e in eps))
    out["success_wilson"] = wilson(out["n_succ"], len(eps))
    out["n_scored"] = int(sum(e.get("n_scored", 0) for e in eps))
    return out


def paired_delta(a, b, key):
    ib = {(e["row"], e.get("rep", 0)): e for e in b}
    d = [float(e[key]) - float(ib[k][key]) for e in a
         if (k := (e["row"], e.get("rep", 0))) in ib and e.get(key) is not None and ib[k].get(key) is not None]
    if not d:
        return math.nan, (math.nan, math.nan), 0
    return float(np.mean(d)), boot_ci(d), len(d)


def gap_of(res):
    gaps = {r.get("gap") for r in res if r.get("gap")}
    return sorted(gaps)[-1] if gaps else "gainlh"


def evals(res, gap):
    out = {}
    for short, tag in EVAL_TAGS.items():
        eps = [r for r in res if r.get("kind") == "eval" and r.get("tag") == tag and r.get("gap", "gainlh") == gap]
        if eps:
            out[short] = sorted(eps, key=lambda e: (e["rep"], e["row"]))
    return out


def reach(es_m, t_star, frac=0.25):
    """First update (and its episode count) where theta_bar is within frac * theta* of theta*."""
    for r in es_m:
        if abs(r["theta_bar"] - t_star) <= frac * t_star:
            return r["update"], r["episodes"]
    return None, None


def training(res, es, t_star):
    out = {}
    for m, recs in es.items():
        eps = sorted([r for r in res if r.get("kind") == "train" and r.get("method") == m],
                     key=lambda e: (e["pair"], e["sign"]))
        d_e = np.array([p["plus"]["e_pred"] - p["minus"]["e_pred"] for r in recs for p in r["pairs"]])
        dth = np.array([p["theta_plus"] - p["theta_minus"] for r in recs for p in r["pairs"]])
        flips = sum(bool(p["plus"]["success"]) != bool(p["minus"]["success"]) for r in recs for p in r["pairs"])
        slope = float(np.sum(d_e * dth) / np.sum(dth ** 2)) if np.sum(dth ** 2) > 0 else math.nan
        bs = []
        for _ in range(B):
            i = RNG.integers(0, len(d_e), len(d_e))
            den = np.sum(dth[i] ** 2)
            bs.append(np.sum(d_e[i] * dth[i]) / den if den > 0 else math.nan)
        u_reach, ep_reach = reach(recs, t_star)
        u_reach_raw = next((r["update"] for r in recs if abs(r["theta"] - t_star) <= 0.25 * t_star), None)
        out[m] = {"episodes": eps, "es": recs, "theta_bar": recs[-1]["theta_bar"], "theta_last": recs[-1]["theta"],
                  "reach_update": u_reach, "reach_episodes": ep_reach, "reach_update_raw": u_reach_raw,
                  "flips": flips, "n_pairs": len(d_e), "median_abs_de": float(np.median(np.abs(d_e))) if len(d_e) else math.nan,
                  "slope_epred_logk": slope, "slope_ci": (float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))),
                  "zero_g": sum(1 for r in recs if r["g"] == 0.0)}
        n8 = 2 * PAIRS_PER_UPDATE
        out[m]["first"] = summarize(eps[:24])
        out[m]["last"] = summarize(eps[-24:])
        out[m]["all"] = summarize(eps)
        out[m]["n8"] = n8
    return out


def fmt(x, ci, f="{:.3f}"):
    return f"{f.format(x)} [{f.format(ci[0])}, {f.format(ci[1])}]"


def report(res, es, gap):
    t_star = theta_oracle(gap)
    ev = evals(res, gap)
    tr = training(res, es, t_star)
    L = [f"# T010 analysis (gap {gap}: stiffness x{GAPS[gap]} on lift+elbow, k* = {math.exp(t_star):.2f}, "
         f"log k* = {t_star:.3f})", ""]
    # pre-check gate
    if "theta0" in ev and "oracle" in ev:
        s0 = sum(e["success"] for e in ev["theta0"])
        so = sum(e["success"] for e in ev["oracle"])
        ok = s0 <= 7 and so >= 10
        L += [f"Pre-check: θ0 {s0}/{len(ev['theta0'])}, oracle {so}/{len(ev['oracle'])} -> gate "
              f"(θ0 ≤ 7/12, oracle ≥ 10/12) {'PASS' if ok else 'FAIL'}", ""]
    for g in sorted({r.get("gap") for r in res if r.get("gap") and r.get("gap") != gap}):
        for short in ("theta0", "oracle"):
            eps = [r for r in res if r.get("tag") == EVAL_TAGS[short] and r.get("gap") == g]
            if eps:
                L.append(f"Pre-check at other gap {g}: {short} {sum(e['success'] for e in eps)}/{len(eps)}")
    summ = {s: summarize(e) for s, e in ev.items()}
    base = ev.get("theta0", [])
    L += ["## Eval (home bin 6 layouts x 2 jittered repeats)", "",
          "| θ | k (lift = elbow) | success (Wilson 95%) | e_track ° | e_pred@1 | Δe_pred vs θ0 | Δsuccess vs θ0 | arm path ° | ticks |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, eps in ev.items():
        x = summ[s]
        k = math.exp(eps[0]["theta1"])
        de = paired_delta(eps, base, "e_pred") if s != "theta0" else None
        ds = paired_delta(eps, base, "success") if s != "theta0" else None
        L.append(f"| {LABEL[s]} | {k:.2f} | {x['n_succ']}/{x['n']}, {x['success']:.2f} "
                 f"[{x['success_wilson'][0]:.2f}, {x['success_wilson'][1]:.2f}] | {fmt(x['e_track_deg'], x['e_track_deg_ci'], '{:.2f}')} "
                 f"| {fmt(x['e_pred'], x['e_pred_ci'])} | {'—' if de is None else fmt(de[0], de[1], '{:+.3f}')} "
                 f"| {'—' if ds is None else fmt(ds[0], ds[1], '{:+.2f}')} | {x['path_len_deg']:.0f} | {x['ticks']:.0f} |")
    L += ["", "Per object (S success, T timeout, F other failure; repeat 0, repeat 1):", "",
          "| layout | " + " | ".join(LABEL[s] for s in ev) + " | training success " + "/".join(tr) + " |",
          "|---|" + "---|" * len(ev) + "---|"]
    for row, name in enumerate(OBJECTS):
        cells = []
        for s, eps in ev.items():
            c = ""
            for e in sorted([e for e in eps if e["row"] == row], key=lambda e: e["rep"]):
                c += "S" if e["success"] else ("T" if e["reason"] == "time_out" else "F")
            cells.append(c)
        trs = []
        for m in tr:
            te = [e for e in tr[m]["episodes"] if e["row"] == row]
            trs.append(f"{np.mean([e['success'] for e in te]):.2f}" if te else "-")
        L.append(f"| {name} | " + " | ".join(cells) + " | " + "/".join(trs) + " |")
    pens = [0, 3]
    L += ["", "Success without/with the pens (rows 0, 3):", ""]
    for s, eps in ev.items():
        np_ = [e for e in eps if e["row"] in pens]
        nn = [e for e in eps if e["row"] not in pens]
        L.append(f"- {LABEL[s]}: pens {sum(e['success'] for e in np_)}/{len(np_)}, "
                 f"other four {sum(e['success'] for e in nn)}/{len(nn)}")
    L += ["", f"## Learning (σ={SIGMA}, η={ETA}, {PAIRS_PER_UPDATE} pairs/update, θ̄ = mean of last {AVG_LAST} iterates)", "",
          "| method | episodes | final θ̄ (k̄) | last iterate θ (k) | |θ̄ − log k*| | first update θ̄ within 25% of log k* (episodes) | raw iterate within 25% | pairs with success flip | median |paired Δe_pred| | d e_pred / d log k (pooled OLS) | train success first 24 → last 24 | train e_track first → last | train e_pred first → last |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, t in tr.items():
        ru = "never" if t["reach_update"] is None else f"update {t['reach_update']} ({t['reach_episodes']} ep)"
        rr = "never" if t["reach_update_raw"] is None else f"update {t['reach_update_raw']}"
        L.append(f"| {LABEL[m]} | {len(t['episodes'])} | {t['theta_bar']:.3f} ({math.exp(t['theta_bar']):.2f}) | "
                 f"{t['theta_last']:.3f} ({math.exp(t['theta_last']):.2f}) | {abs(t['theta_bar'] - t_star):.3f} | {ru} | {rr} | "
                 f"{t['flips']}/{t['n_pairs']} | {t['median_abs_de']:.4f} | {fmt(t['slope_epred_logk'], t['slope_ci'], '{:+.4f}')} | "
                 f"{t['first']['success']:.2f} → {t['last']['success']:.2f} | {t['first']['e_track_deg']:.2f} → {t['last']['e_track_deg']:.2f} | "
                 f"{t['first']['e_pred']:.4f} → {t['last']['e_pred']:.4f} |")
    L += ["", "Per update (k = raw iterate, k̄ = mean of last 5):", ""]
    for m, t in tr.items():
        L.append(f"- {LABEL[m]}: " + ", ".join(f"{r['episodes']}:{r['k']:.2f}/{r['k_bar']:.2f}" for r in t["es"]))
    # per-pair SNR: signal = |slope| * 2 sigma |eps|, noise = residual sd
    for m, t in tr.items():
        d_e = np.array([p["plus"]["e_pred"] - p["minus"]["e_pred"] for r in t["es"] for p in r["pairs"]])
        dth = np.array([p["theta_plus"] - p["theta_minus"] for r in t["es"] for p in r["pairs"]])
        resid = d_e - t["slope_epred_logk"] * dth
        L.append(f"- {LABEL[m]} per-pair SNR: |slope|·2σ = {abs(t['slope_epred_logk']) * 2 * SIGMA:.4f} vs residual sd "
                 f"{np.std(resid):.4f} -> {abs(t['slope_epred_logk']) * 2 * SIGMA / max(np.std(resid), 1e-9):.2f}")
    return "\n".join(L) + "\n", tr, ev, summ


def write_tsv(path: Path, tr, ev, summ, gap):
    t_star = theta_oracle(gap)
    rows = []
    for m, t in tr.items():
        x = t["all"]
        rows.append([f"T010-{m}-train", TASK, "internw0", gap, str(x["n"]), f"{x['success']:.3f}", f"{x['e_track_deg']:.3f}",
                     "", METRIC, f"{x['e_pred']:.4f}", "", "", "", "", "", "", str(x["n_scored"]),
                     f"1-dim ES (shared lift+elbow log k, antithetic, sigma {SIGMA}, eta {ETA}, {PAIRS_PER_UPDATE} pairs/update, "
                     f"RMS-normalised d), reward={m}{f' lambda={LAMBDA}' if m == 'mixed' else ''}; home bin 6 layouts cycled, "
                     f"jitter ±1cm ±5deg; theta_bar(last {AVG_LAST})={t['theta_bar']:.3f} k_bar={math.exp(t['theta_bar']):.2f} "
                     f"last k={math.exp(t['theta_last']):.2f} k*={math.exp(t_star):.2f}; reach 25% at "
                     f"{t['reach_episodes'] if t['reach_episodes'] is not None else 'never'} ep"])
    base = ev.get("theta0", [])
    for s, eps in ev.items():
        x = summ[s]
        de = paired_delta(eps, base, "e_pred") if s != "theta0" else None
        rows.append([EVAL_TAGS[s], TASK, "internw0", gap, str(x["n"]), f"{x['success']:.3f}", f"{x['e_track_deg']:.3f}", "",
                     METRIC, f"{x['e_pred']:.4f}", "", "", "", "" if de is None else f"{de[0]:.4f}",
                     "" if de is None else f"[{de[1][0]:.4f},{de[1][1]:.4f}]", "", str(x["n_scored"]),
                     f"eval home bin 6 layouts x 2 jittered repeats, no exploration; k(lift=elbow)={math.exp(eps[0]['theta1']):.3f}; "
                     f"success Wilson {np.round(x['success_wilson'], 2).tolist()}; e_pred CI {np.round(x['e_pred_ci'], 4).tolist()}; "
                     f"path {x['path_len_deg']:.0f} deg" + ("; e_pred_minus_none = paired delta vs theta0" if de else "")])
    with path.open("a") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


def plot(path: Path, tr, ev, summ, gap):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_star = theta_oracle(gap)
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.2))
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(alpha=0.25, lw=0.6)
    a = ax[0]
    for m, t in tr.items():
        ep = [0] + [r["episodes"] for r in t["es"]]
        a.plot(ep, [1.0] + [r["k"] for r in t["es"]], color=COLOR[m], lw=1, alpha=0.45, ls="--")
        a.plot(ep, [1.0] + [r["k_bar"] for r in t["es"]], color=COLOR[m], lw=2, label=f"{LABEL[m]} k̄ (dashed: raw)")
    a.axhline(math.exp(t_star), color=COLOR["oracle"], lw=1.5, label=f"k* = {math.exp(t_star):.0f}")
    a.axhspan(math.exp(0.75 * t_star), math.exp(1.25 * t_star), color=COLOR["oracle"], alpha=0.08)
    a.set_yscale("log")
    a.set_yticks([0.5, 1, 2, 4, 6])
    a.set_yticklabels(["0.5", "1", "2", "4", "6"])
    a.set_xlabel("training episodes")
    a.set_title("gain k (lift = elbow); band = θ̄ within 25% of log k*")
    a.legend(fontsize=8, frameon=False)
    for i, (key, title) in enumerate((("e_pred", "e_pred@1 (per-update mean)"), ("success", "success (per-update mean)"),
                                      ("e_track_deg", "e_track ° (per-update mean)")), start=1):
        a = ax[i]
        for m, t in tr.items():
            eps = t["episodes"]
            n8 = t["n8"]
            xs = [n8 * (j + 1) for j in range(len(eps) // n8)]
            ys = [np.mean([float(e[key]) for e in eps[j * n8:(j + 1) * n8]]) for j in range(len(eps) // n8)]
            a.plot(xs, ys, color=COLOR[m], lw=2, marker="o", ms=4, label=LABEL[m])
        for s in ("theta0", "oracle"):
            if s in summ:
                a.axhline(summ[s][key], color=COLOR[s], lw=1.2, ls=":", label=f"{LABEL[s]} eval")
        a.set_xlabel("training episodes")
        a.set_title(title)
        if i == 1:
            a.legend(fontsize=8, frameon=False)
    a = ax[4]
    names = list(ev)
    xs = np.arange(len(names))
    for j, s in enumerate(names):
        x = summ[s]
        lo, hi = x["success_wilson"]
        a.bar(j, x["success"], color=COLOR[s], width=0.6)
        a.errorbar(j, x["success"], yerr=[[x["success"] - lo], [hi - x["success"]]], color="#52514e", capsize=3, lw=1)
        a.text(j, 0.03, f"{x['n_succ']}/{x['n']}\ne_tr {x['e_track_deg']:.2f}°\ne_p {x['e_pred']:.3f}", ha="center",
               va="bottom", fontsize=7.5, color="white")
    a.set_xticks(xs)
    a.set_xticklabels([LABEL[s] for s in names], fontsize=8, rotation=12)
    a.set_ylim(0, 1.05)
    a.set_title("eval success (12 ep, Wilson 95%)")
    fig.suptitle(f"T010: 1-dim ES, gap stiffness ×{GAPS[gap]} on lift+elbow, 120 training episodes per method", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--tsv", type=Path, default=None)
    args = ap.parse_args()
    res, es = load(args.root)
    gap = gap_of(res)
    md, tr, ev, summ = report(res, es, gap)
    print(md)
    (args.root / "analysis.md").write_text(md)
    (args.root / "summary.json").write_text(json.dumps(
        {"gap": gap, "eval": {s: {k: v for k, v in x.items()} for s, x in summ.items()},
         "train": {m: {k: v for k, v in t.items() if k not in ("episodes", "es")} for m, t in tr.items()}},
        default=float, indent=1))
    if args.plot:
        plot(args.plot, tr, ev, summ, gap)
    if args.tsv:
        write_tsv(args.tsv, tr, ev, summ, gap)


if __name__ == "__main__":
    main()
