"""T009 analysis: learning curves, eval table, jitter noise check, hacking flags, results.tsv rows, figure.

    python -m so101_playground.t009_analysis --root /outputs/so101-playground/runs/T009 \
        --plot docs/T009_rl_sim.png --tsv /outputs/so101-playground/results.tsv

Reads <root>/queue/done/*.json (one per episode) and <root>/es/<method>.jsonl (one per ES update).
CIs: 95% percentile bootstrap over episodes (2000 resamples). Eval deltas vs theta0 are paired by (row, repeat):
every theta saw the same layout jitter and server noise seed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from so101_playground.t009_es import ARM, LAMBDA, METHODS, THETA_ORACLE

LABEL = {"epred": "① −e_pred", "success": "② success", "mixed": "③ −e_pred+λ·succ"}
EVAL_TAGS = {"theta0": "T009-baseline-theta0", "oracle": "T009-oracle", "epred": "T009-epred-eval",
             "success": "T009-success-eval", "mixed": "T009-mixed-eval"}
RNG = np.random.default_rng(0)
B = 2000


def load(root: Path):
    res = [json.loads(p.read_text()) for p in sorted((root / "queue" / "done").glob("*.json"))]
    es = {}
    for m in METHODS:
        p = root / "es" / f"{m}.jsonl"
        if p.exists():
            es[m] = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return res, es


def boot_ci(x, stat=np.mean):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if len(x) == 0:
        return (math.nan, math.nan)
    s = [stat(x[RNG.integers(0, len(x), len(x))]) for _ in range(B)]
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))


def summarize(eps: list[dict]) -> dict:
    out = {"n": len(eps)}
    for k in ("success", "e_pred", "e_pred_wrist", "e_track_deg", "path_len_deg", "ticks", "sent_minus_cmd_deg"):
        v = [float(e[k]) for e in eps if e.get(k) is not None]
        out[k] = float(np.mean(v)) if v else math.nan
        out[k + "_ci"] = boot_ci(v)
    out["n_scored"] = int(sum(e.get("n_scored", 0) for e in eps))
    return out


def paired_delta(a: list[dict], b: list[dict], key: str):
    """mean(a - b) over episodes matched by (set, row, rep), with a bootstrap CI."""
    ib = {(e["set"], e["row"], e.get("rep", 0)): e for e in b}
    d = [float(e[key]) - float(ib[k][key]) for e in a
         if (k := (e["set"], e["row"], e.get("rep", 0))) in ib and e.get(key) is not None and ib[k].get(key) is not None]
    if not d:
        return math.nan, (math.nan, math.nan), 0
    return float(np.mean(d)), boot_ci(d), len(d)


def evals(res):
    out = {}
    for short, tag in EVAL_TAGS.items():
        eps = [r for r in res if r.get("kind") == "eval" and r.get("tag") == tag]
        if eps:
            out[short] = {s: [e for e in eps if e["set"] == s] for s in ("home", "t004")}
    return out


def jitter_noise(res):
    """|e(rep0) - e(rep1)| at theta0, same layout row and noise seed: jittered vs exact placement."""
    out = {}
    for name, tag in (("jittered", "T009-baseline-theta0"), ("exact", "T009-theta0-nojit")):
        eps = {(e["row"], e["rep"]): e for e in res if e.get("tag") == tag and e["set"] == "home"}
        rows = sorted({r for r, _ in eps})
        for key in ("e_pred", "success", "e_track_deg"):
            d = [abs(float(eps[(r, 0)][key]) - float(eps[(r, 1)][key])) for r in rows
                 if (r, 0) in eps and (r, 1) in eps and eps[(r, 0)].get(key) is not None and eps[(r, 1)].get(key) is not None]
            out.setdefault(name, {})[key] = {"mean": float(np.mean(d)) if d else math.nan, "ci": boot_ci(d),
                                             "n_pairs": len(d), "values": d}
    return out


def training(res, es):
    out = {}
    for m in METHODS:
        eps = sorted([r for r in res if r.get("kind") == "train" and r.get("method") == m],
                     key=lambda r: (r["pair"], r["sign"] != "+"))
        if not eps:
            continue
        ups = es.get(m, [])
        d_abs = [abs(p["plus"]["e_pred"] - p["minus"]["e_pred"]) for u in ups for p in u["pairs"]
                 if p["plus"]["e_pred"] is not None and p["minus"]["e_pred"] is not None]
        succ_flips = sum(1 for u in ups for p in u["pairs"] if bool(p["plus"]["success"]) != bool(p["minus"]["success"]))
        n_pairs = sum(len(u["pairs"]) for u in ups)
        out[m] = {
            "episodes": eps, "updates": ups,
            "theta_final": ups[-1]["theta"] if ups else None,
            "k_final": ups[-1]["k"] if ups else None,
            "dist_final": ups[-1]["dist_oracle"] if ups else None,
            "zero_grad_updates": sum(1 for u in ups if not any(p["d"] != 0 for p in u["pairs"])),
            "pair_abs_d_epred_median": float(np.median(d_abs)) if d_abs else math.nan,
            "success_flip_pairs": succ_flips, "n_pairs": n_pairs,
            "train_success": float(np.mean([e["success"] for e in eps])),
            "train_e_pred": float(np.nanmean([e["e_pred"] if e["e_pred"] is not None else np.nan for e in eps])),
        }
        # learning speed: first episode count at which theta is within half the initial distance of the oracle
        d0 = float(np.linalg.norm(THETA_ORACLE))
        hit = [u["episodes"] for u in ups if u["dist_oracle"] <= 0.5 * d0]
        out[m]["episodes_to_half_dist"] = hit[0] if hit else None
    return out


def hacking(tr, ev):
    """Flags: e_pred better than theta0 while success or motion is lower (eval), and training updates where the
    e_pred reward improved but the episodes moved much less or failed more."""
    flags = []
    base = ev.get("theta0", {}).get("home", [])
    for m in METHODS:
        e = ev.get(m, {}).get("home", [])
        if e and base:
            de, _, _ = paired_delta(e, base, "e_pred")
            ds, _, _ = paired_delta(e, base, "success")
            dp, _, _ = paired_delta(e, base, "path_len_deg")
            bp = np.mean([x["path_len_deg"] for x in base])
            if de < 0 and (ds < 0 or dp < -0.2 * bp):
                flags.append(f"{m} eval: e_pred {de:+.4f} vs theta0 but success {ds:+.2f}, path {dp:+.0f} deg")
        if m in tr:
            eps = tr[m]["episodes"]
            for lo, hi in ((0, 16), (len(eps) - 16, len(eps))):
                pass
            first, last = eps[:16], eps[-16:]
            if len(eps) >= 32:
                fe = np.nanmean([x["e_pred"] for x in first if x["e_pred"] is not None])
                le = np.nanmean([x["e_pred"] for x in last if x["e_pred"] is not None])
                fs, ls = np.mean([x["success"] for x in first]), np.mean([x["success"] for x in last])
                fp, lp = np.mean([x["path_len_deg"] for x in first]), np.mean([x["path_len_deg"] for x in last])
                if le < fe and (ls < fs - 0.15 or lp < 0.8 * fp):
                    flags.append(f"{m} train: e_pred {fe:.4f}->{le:.4f} (first/last 16 ep) while success "
                                 f"{fs:.2f}->{ls:.2f}, path {fp:.0f}->{lp:.0f} deg")
    return flags


# ---------------------------------------------------------------- report
def fmt_ci(x, ci, f="{:.3f}"):
    return f"{f.format(x)} [{f.format(ci[0])}, {f.format(ci[1])}]"


def report(res, es):
    tr, ev, jn = training(res, es), evals(res), jitter_noise(res)
    lines = ["## Training"]
    for m, t in tr.items():
        lines.append(f"- {m}: {len(t['episodes'])} ep, train success {t['train_success']:.2f}, train e_pred "
                     f"{t['train_e_pred']:.4f}; final k = {np.round(t['k_final'], 2).tolist()} "
                     f"(k* = {np.round(np.exp(THETA_ORACLE), 2).tolist()}), |θ−θ*| = {t['dist_final']:.3f} "
                     f"(start {np.linalg.norm(THETA_ORACLE):.3f}); episodes to half distance: {t['episodes_to_half_dist']}; "
                     f"zero-gradient updates {t['zero_grad_updates']}/{len(t['updates'])}; success-flip pairs "
                     f"{t['success_flip_pairs']}/{t['n_pairs']}; median |paired Δe_pred| {t['pair_abs_d_epred_median']:.4f}")
    lines.append("\n## Eval (home bin 6 layouts × 2 jittered repeats)")
    lines.append("| θ | n | success | e_track ° | e_pred@1 | Δe_pred vs θ0 | Δsuccess vs θ0 | path ° | ticks | k |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    base = ev.get("theta0", {}).get("home", [])
    summ = {}
    for short in EVAL_TAGS:
        for s in ("home", "t004"):
            eps = ev.get(short, {}).get(s, [])
            if eps:
                summ[(short, s)] = summarize(eps)
        eps = ev.get(short, {}).get("home", [])
        if not eps:
            continue
        x = summ[(short, "home")]
        de = paired_delta(eps, base, "e_pred") if short != "theta0" else (math.nan, (math.nan, math.nan), 0)
        dsu = paired_delta(eps, base, "success") if short != "theta0" else (math.nan, (math.nan, math.nan), 0)
        k = np.round(eps[0]["k"], 2).tolist()
        lines.append(f"| {short} | {x['n']} | {fmt_ci(x['success'], x['success_ci'], '{:.2f}')} | "
                     f"{fmt_ci(x['e_track_deg'], x['e_track_deg_ci'], '{:.2f}')} | {fmt_ci(x['e_pred'], x['e_pred_ci'], '{:.4f}')} | "
                     f"{'—' if short == 'theta0' else fmt_ci(de[0], de[1], '{:+.4f}')} | "
                     f"{'—' if short == 'theta0' else fmt_ci(dsu[0], dsu[1], '{:+.2f}')} | {x['path_len_deg']:.0f} | "
                     f"{x['ticks']:.0f} | {k} |")
    if any(s == "t004" for _, s in summ):
        lines.append("\n## Eval, T004 rows 0-7 (seen objects, exact layouts, 1 run each)")
        for (short, s), x in summ.items():
            if s == "t004":
                lines.append(f"- {short}: success {x['success']:.2f} ({x['n']} ep), e_track {x['e_track_deg']:.2f}°, "
                             f"e_pred {x['e_pred']:.4f}")
    lines.append("\n## Jitter noise at θ0 (|rep0 − rep1|, same row and noise seed)")
    for name, d in jn.items():
        lines.append(f"- {name}: |Δe_pred| {fmt_ci(d['e_pred']['mean'], d['e_pred']['ci'], '{:.4f}')} "
                     f"(n={d['e_pred']['n_pairs']}), |Δsuccess| {d['success']['mean']:.2f}, |Δe_track| "
                     f"{d['e_track_deg']['mean']:.2f}°; values {np.round(d['e_pred']['values'], 4).tolist()}")
    flags = hacking(tr, ev)
    lines.append("\n## Hacking flags\n" + ("\n".join(f"- {f}" for f in flags) if flags else "- none"))
    return "\n".join(lines), tr, ev, summ, jn, flags


# ---------------------------------------------------------------- tsv
TSV_COLS = ["run_id", "task", "model", "condition", "n_episodes", "success", "e_track_deg", "e_track@h_deg", "metric",
            "e_pred", "e_pred_paired", "e_ideal_paired", "e_exec", "e_pred_minus_none", "e_pred_minus_none_ci95",
            "rho_pred_track", "n_queries", "notes"]
TASK = "T009-wam-rl-sim"
METRIC = "Wan2.2_VAE_latent_mse@1(ticks4-16)"


def write_tsv(path: Path, tr, ev, summ):
    rows = []
    for m, t in tr.items():
        eps = t["episodes"]
        x = summarize(eps)
        rows.append([f"T009-{m}-train", TASK, "internw0", "gainj", str(x["n"]), f"{x['success']:.3f}",
                     f"{x['e_track_deg']:.3f}", "", METRIC, f"{x['e_pred']:.4f}", "", "", "", "", "", "",
                     str(x["n_scored"]),
                     f"ES training (antithetic, sigma 0.2, eta 0.2, 2 pairs/update, RMS-normalised d), reward={m}"
                     f"{f' lambda={LAMBDA}' if m == 'mixed' else ''}; home bin 6 layouts cycled, jitter ±1cm ±5deg; "
                     f"final k={np.round(t['k_final'], 3).tolist()} k*={np.round(np.exp(THETA_ORACLE), 3).tolist()} "
                     f"|theta-theta*|={t['dist_final']:.3f}"])
    base = ev.get("theta0", {}).get("home", [])
    for short, tag in EVAL_TAGS.items():
        if (short, "home") not in summ:
            continue
        x = summ[(short, "home")]
        eps = ev[short]["home"]
        de = paired_delta(eps, base, "e_pred") if short != "theta0" else None
        t4 = summ.get((short, "t004"))
        note = (f"eval home bin 6 layouts x 2 jittered repeats, no exploration; k={np.round(eps[0]['k'], 3).tolist()}; "
                f"success CI {np.round(x['success_ci'], 2).tolist()}; e_pred CI {np.round(x['e_pred_ci'], 4).tolist()}; "
                f"path {x['path_len_deg']:.0f} deg")
        if t4:
            note += f"; T004 rows 0-7: success {t4['success']:.2f}, e_track {t4['e_track_deg']:.2f}, e_pred {t4['e_pred']:.4f}"
        rows.append([tag, TASK, "internw0", "gainj", str(x["n"]), f"{x['success']:.3f}", f"{x['e_track_deg']:.3f}", "",
                     METRIC, f"{x['e_pred']:.4f}", "", "", "", "" if de is None else f"{de[0]:.4f}",
                     "" if de is None else f"[{de[1][0]:.4f},{de[1][1]:.4f}]", "", str(x["n_scored"]),
                     note + ("; e_pred_minus_none column = paired delta vs theta0" if de is not None else "")])
    with path.open("a") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------- figure
def plot(path: Path, tr, ev, summ):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"epred": "#2a6fdb", "success": "#d1495b", "mixed": "#2e9e6a"}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.2))
    ax_ep, ax_su, ax_tr, ax_d = axes[0]
    win = 8
    for m, t in tr.items():
        eps = t["episodes"]
        n = np.arange(1, len(eps) + 1)
        def roll(key):
            v = np.array([np.nan if e.get(key) is None else float(e[key]) for e in eps])
            return np.array([np.nanmean(v[max(0, i - win + 1): i + 1]) for i in range(len(v))])
        c = colors[m]
        ax_ep.plot(n, roll("e_pred"), color=c, label=LABEL[m])
        ax_su.plot(n, roll("success"), color=c, label=LABEL[m])
        ax_tr.plot(n, roll("e_track_deg"), color=c, label=LABEL[m])
        ups = t["updates"]
        ax_d.plot([0] + [u["episodes"] for u in ups], [np.linalg.norm(THETA_ORACLE)] + [u["dist_oracle"] for u in ups],
                  color=c, marker="o", ms=3, label=LABEL[m])
    for short, ls in (("theta0", ":"), ("oracle", "--")):
        if (short, "home") in summ:
            x = summ[(short, "home")]
            for ax, k in ((ax_ep, "e_pred"), (ax_su, "success"), (ax_tr, "e_track_deg")):
                ax.axhline(x[k], color="0.4", ls=ls, lw=1, label=f"eval {short}")
    ax_ep.set_title(f"training e_pred@1 (running mean, {win} ep)")
    ax_su.set_title(f"training success (running mean, {win} ep)")
    ax_tr.set_title(f"training e_track ° (running mean, {win} ep)")
    ax_d.set_title("‖θ − log k*‖")
    for ax in axes[0]:
        ax.set_xlabel("training episodes")
        ax.grid(alpha=0.3)
    ax_ep.legend(fontsize=8)
    # theta trajectories, one panel per learned method, 5 joints each
    jc = plt.cm.viridis(np.linspace(0, 0.9, len(ARM)))
    for ax, m in zip(axes[1][:3], METHODS):
        if m not in tr:
            ax.axis("off")
            continue
        ups = tr[m]["updates"]
        xs = [0] + [u["episodes"] for u in ups]
        for j, name in enumerate(ARM):
            ax.plot(xs, [1.0] + [u["k"][j] for u in ups], color=jc[j], label=name)
            ax.axhline(math.exp(THETA_ORACLE[j]), color=jc[j], ls="--", lw=0.8)
        ax.set_yscale("log", base=2)
        ax.set_ylim(0.45, 4.4)
        ax.set_title(f"{LABEL[m]}: gains k (dashed = k*)")
        ax.set_xlabel("training episodes")
        ax.grid(alpha=0.3)
    axes[1][0].legend(fontsize=7, loc="upper left")
    # eval bars
    ax = axes[1][3]
    names = [s for s in EVAL_TAGS if (s, "home") in summ]
    xpos = np.arange(len(names))
    succ = [summ[(s, "home")]["success"] for s in names]
    sci = np.array([summ[(s, "home")]["success_ci"] for s in names])
    ax.bar(xpos - 0.2, succ, 0.4, color="#888", yerr=[np.array(succ) - sci[:, 0], sci[:, 1] - np.array(succ)],
           capsize=3, label="success")
    ax2 = ax.twinx()
    ep = [summ[(s, "home")]["e_pred"] for s in names]
    eci = np.array([summ[(s, "home")]["e_pred_ci"] for s in names])
    ax2.bar(xpos + 0.2, ep, 0.4, color="#2a6fdb", yerr=[np.array(ep) - eci[:, 0], eci[:, 1] - np.array(ep)],
            capsize=3, label="e_pred@1")
    ax.set_xticks(xpos, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success")
    ax2.set_ylabel("e_pred@1", color="#2a6fdb")
    ax.set_title("eval: home bin 6 × 2 jittered (95% CI)")
    fig.suptitle("T009 stage A: ES gain adapter on frozen InternW0-Δ under an unknown per-joint stiffness gap")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/outputs/so101-playground/runs/T009"))
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--tsv", type=Path, default=None)
    args = ap.parse_args()
    res, es = load(args.root)
    text, tr, ev, summ, jn, flags = report(res, es)
    print(text)
    (args.root / "analysis.md").write_text(text + "\n")
    summary = {"training": {m: {k: v for k, v in t.items() if k not in ("episodes", "updates")} for m, t in tr.items()},
               "eval": {f"{a}/{b}": v for (a, b), v in summ.items()},
               "jitter_noise": jn, "hacking_flags": flags}
    (args.root / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if args.plot:
        plot(args.plot, tr, ev, summ)
    if args.tsv:
        write_tsv(args.tsv, tr, ev, summ)


if __name__ == "__main__":
    main()
