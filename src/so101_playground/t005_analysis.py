#!/usr/bin/env python3
"""T005 analysis: oracle correction strength alpha vs success, e_track and raw e_pred (RESEARCH.md H2).

Inputs: ``<root>/lawam/<tag>/`` (episodes.jsonl, ep*.npz from scripts/wam_gap_eval.py) with tag = ``none`` or
``<gap>-a<alpha>``, and the server log ``<root>/lawam/lam_scores.jsonl`` (rows tagged with the tag).

Per setting (all means query-weighted as in T004 for e_pred; e_track tick-weighted; CIs 95% bootstrap over
episodes, Δ vs none and AUROC CIs episode-paired):
  success, e_track (|cmd - q| arm, deg; cmd = the policy's command before oracle and gap), e_pred (LAM token MSE
  @35), Δ e_pred vs none, oracle exactness |applied - cmd| (deg; 0 = the arm receives exactly what the policy
  commanded), e_pred on the changed / unchanged token regions (mask = top 25% tokens of the observed change
  mse(h_t, h_obs) per query; an approximate arm+object mask).
Per gap: AUROC that e_pred ranks alpha=0 above alpha=1 (episode means; single queries; 4-query windows), with
episode-bootstrap CIs. This is the separability an RL adapter rewarded with -e_pred would see.

  t005_analysis.py --root /outputs/so101-playground/runs/T005 [--tsv results.tsv] [--plot out.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from so101_playground.t004_analysis import episodes_stats

ARM = slice(0, 5)
GAPS = ("offset6", "gain2", "latency6")
ALPHAS = (0.0, 0.5, 1.0, 1.5)
N_BOOT = 2000
WINDOW = 4
TOP_FRAC = 0.25


def tag_of(gap: str, alpha: float) -> str:
    return f"{gap}-a{alpha:g}"


def load_setting(run: Path, scores: list[dict]) -> dict:
    """Per-episode series: e_pred per query (ordered by frame), token-region e_pred, e_track per tick, exactness."""
    eps = {}
    attempts = {}
    for line in (run / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            attempts[r["episode"]] = r.get("attempt")
            eps[r["episode"]] = {"success": bool(r["success"]), "ticks": r["ticks"]}
    rows = [r for r in scores if r.get("condition") == run.name and r["kind"] == "observed"
            and r["episode"] in attempts and r.get("attempt") == attempts[r["episode"]]]
    for e, d in eps.items():
        mine = sorted((r for r in rows if r["episode"] == e), key=lambda r: r["frame"])
        d["e_pred"] = np.array([r["e_pred_lat"] for r in mine])
        d["e_static"] = np.array([r["e_static_lat"] for r in mine])
        chg, unchg = [], []
        for r in mine:
            if "tok_pred" in r:
                tp, ts = np.asarray(r["tok_pred"]), np.asarray(r["tok_static"])
                k = int(round(TOP_FRAC * len(ts)))  # exactly k tokens (64 of 256); ties broken by token index
                m = np.zeros(len(ts), dtype=bool)
                m[np.argsort(-ts, kind="stable")[:k]] = True
                chg.append(tp[m].mean())
                unchg.append(tp[~m].mean())
        d["e_pred_changed"], d["e_pred_unchanged"] = np.array(chg), np.array(unchg)
        z = np.load(run / f"ep{e:03d}.npz", allow_pickle=True)
        h = int(z["hold_steps"])
        cmd, q, app = z["cmd"][h:, ARM], z["q"][h:, ARM], z["applied"][h:, ARM]
        d["e_track"] = np.degrees(np.abs(cmd - q)).mean(1)
        exact = np.degrees(np.abs(app - cmd)).max(1)
        d["applied_err"] = np.degrees(np.abs(app - cmd)).mean(1)
        d["exact_frac"] = float((exact < 0.01).mean())
    return eps


def qmean(eps: dict, key: str, pick) -> float:
    vals = [eps[e][key] for e in pick if len(eps[e][key])]
    return float(np.mean(np.concatenate(vals))) if vals else float("nan")


def boot(fn, ids, rng) -> list[float]:
    vals = [fn(rng.choice(ids, size=len(ids), replace=True)) for _ in range(N_BOOT)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def auroc(pos, neg) -> float:
    """P(pos > neg) + 0.5 P(tie): pos = alpha 0 (gap uncorrected), neg = alpha 1 (corrected)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def windows(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x) // w
    return x[: n * w].reshape(n, w).mean(1) if n else np.zeros(0)


def reward_auroc(a: dict, b: dict, rng, key: str = "e_pred") -> dict:
    """AUROC of e_pred: setting a (uncorrected) vs b (corrected), for three reward granularities."""
    ea, eb = sorted(a), sorted(b)
    common = np.array(sorted(set(ea) & set(eb)))  # same episode index = same layout and seed in both settings
    feats = {
        "episode": lambda d, pick: [d[e][key].mean() for e in pick if len(d[e][key])],
        "query": lambda d, pick: np.concatenate([d[e][key] for e in pick]),
        f"window{WINDOW}": lambda d, pick: np.concatenate([windows(d[e][key], WINDOW) for e in pick]),
    }
    out = {}
    for name, f in feats.items():
        out[f"auroc_{name}"] = auroc(f(a, ea), f(b, eb))
        bs = []
        for _ in range(N_BOOT // 4):  # episode-paired: one resample of episode ids used for both settings
            pick = rng.choice(common, len(common))
            bs.append(auroc(f(a, pick), f(b, pick)))
        out[f"auroc_{name}_ci95"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    return out


def summarize(root: Path) -> dict:
    lam = root / "lawam"
    scores = [json.loads(l) for l in (lam / "lam_scores.jsonl").read_text().splitlines() if l.strip()]
    scores = [r for r in scores if r["episode"] >= 0]
    tags = ["none"] + [tag_of(g, a) for g in GAPS for a in ALPHAS]
    data = {t: load_setting(lam / t, scores) for t in tags if (lam / t / "episodes.jsonl").exists()}
    rng = np.random.default_rng(0)
    summary = {}
    base = data.get("none")
    for t, eps in data.items():
        ids = np.array(sorted(eps))
        s = episodes_stats(lam / t)
        s["n_success"] = int(sum(eps[e]["success"] for e in ids))
        for key in ("e_pred", "e_track", "applied_err", "e_pred_changed", "e_pred_unchanged", "e_static"):
            s[key] = qmean(eps, key, ids)
            s[f"{key}_ci95"] = boot(lambda p: qmean(eps, key, p), ids, rng)
        s["exact_frac"] = float(np.mean([eps[e]["exact_frac"] for e in ids]))
        s["n_scored_queries"] = int(sum(len(eps[e]["e_pred"]) for e in ids))
        if base is not None and t != "none":
            common = np.array(sorted(set(ids) & set(base)))
            for key in ("e_pred", "e_track", "e_pred_changed", "e_pred_unchanged"):
                f = lambda p, k=key: qmean(eps, k, p) - qmean(base, k, p)
                s[f"{key}_minus_none"] = f(common)
                s[f"{key}_minus_none_ci95"] = boot(f, common, rng)
        summary[t] = s
    for g in GAPS:
        a0, a1 = data.get(tag_of(g, 0.0)), data.get(tag_of(g, 1.0))
        if a0 and a1:
            summary[tag_of(g, 1.0)]["reward_vs_a0"] = reward_auroc(a0, a1, rng)
            summary[tag_of(g, 1.0)]["reward_changed_vs_a0"] = reward_auroc(a0, a1, rng, key="e_pred_changed")
        if a0 and base:
            summary[tag_of(g, 0.0)]["reward_vs_none"] = reward_auroc(a0, base, rng)
        if a1 and base:
            summary[tag_of(g, 1.0)]["reward_a1_vs_none"] = reward_auroc(a1, base, rng)
    return summary


# ---------------------------------------------------------------- outputs
TSV_COLS = ["run_id", "task", "model", "condition", "n_episodes", "success", "e_track_deg", "e_track@h_deg", "metric",
            "e_pred", "e_pred_paired", "e_ideal_paired", "e_exec", "e_pred_minus_none", "e_pred_minus_none_ci95",
            "rho_pred_track", "n_queries", "notes"]


def _f(v, fmt="{:.4f}"):
    if v is None:
        return ""
    if isinstance(v, list):
        return "[" + ",".join(fmt.format(x) for x in v) + "]"
    return fmt.format(v)


def write_tsv(path: Path, summary: dict) -> None:
    with path.open("a") as f:
        for t, s in summary.items():
            alpha = re.sub(r".*-a", "", t) if t != "none" else ""
            notes = (f"oracle alpha={alpha or 'n/a'} (T005, no replay); e_track_ci={_f(s['e_track_ci95'], '{:.3f}')}; "
                     f"e_pred_ci={_f(s['e_pred_ci95'])}; |applied-cmd|={s['applied_err']:.3f}deg exact_ticks={s['exact_frac']:.2f}; "
                     f"e_pred changed/unchanged tokens={s['e_pred_changed']:.4f}/{s['e_pred_unchanged']:.4f}")
            if "reward_vs_a0" in s:
                r = s["reward_vs_a0"]
                notes += (f"; AUROC a0>a1 episode={r['auroc_episode']:.2f} query={r['auroc_query']:.2f} "
                          f"win{WINDOW}={r[f'auroc_window{WINDOW}']:.2f}")
            row = [f"T005-{t}", "T005-wam-oracle-correction", "lawam", t.split("-a")[0], str(s["n_episodes"]),
                   f"{s['success']:.3f}", f"{s['e_track']:.3f}", "", "LAM_DINOv3_token_mse@35", f"{s['e_pred']:.4f}",
                   "", "", "", _f(s.get("e_pred_minus_none")), _f(s.get("e_pred_minus_none_ci95")), "",
                   str(s["n_scored_queries"]), notes]
            f.write("\t".join(row) + "\n")


BLUE = "#2a78d6"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"


def plot(summary: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [("e_pred", "raw e_pred\n(LAM token MSE @35)"),
            ("e_pred_changed", "e_pred, changed tokens\n(top 25% by e_static)"),
            ("e_track", "e_track (deg)"), ("success", "success rate")]
    titles = {"offset6": "offset6: +6° Pitch/Elbow", "gain2": "gain2: arm stiffness ×0.25",
              "latency6": "latency6: 200 ms delay"}
    fig, axes = plt.subplots(len(rows), len(GAPS), figsize=(11, 10), sharex=True, sharey="row")
    none = summary.get("none", {})
    for j, g in enumerate(GAPS):
        pts = [(a, summary[tag_of(g, a)]) for a in ALPHAS if tag_of(g, a) in summary]
        xs = [a for a, _ in pts]
        for i, (key, label) in enumerate(rows):
            ax = axes[i, j]
            if key == "success":
                ys = [s["n_success"] / s["n_episodes"] for _, s in pts]
                lo = hi = ys
                nb = none.get("n_success", 0) / max(none.get("n_episodes", 1), 1)
                nlo = nhi = nb
            else:
                ys = [s[key] for _, s in pts]
                lo = [s[f"{key}_ci95"][0] for _, s in pts]
                hi = [s[f"{key}_ci95"][1] for _, s in pts]
                nb = none.get(key)
                nlo, nhi = none.get(f"{key}_ci95", [nb, nb])
            if nb is not None:
                ax.axhspan(nlo, nhi, color=BAND, alpha=0.6, lw=0)
                ax.axhline(nb, color=INK2, lw=1, ls="--")
                if j == 0:
                    ax.text(0.02, nb, " none", color=INK2, fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
            ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.18, lw=0)
            ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.5)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(INK2)
            ax.tick_params(colors=INK2, labelsize=8)
            if j == 0:
                ax.set_ylabel(label, color=INK, fontsize=9)
            if i == 0:
                ax.set_title(titles[g], color=INK, fontsize=10)
            if i == len(rows) - 1:
                ax.set_xlabel("oracle correction strength α (1 = exact inverse)", color=INK2, fontsize=8)
                ax.set_xticks(ALPHAS)
    fig.suptitle("T005: LaWAM raw e_pred under an oracle command correction (8 ep per point; band = none, 95% CI)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--tsv", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()
    summary = summarize(args.root)
    (args.root / "lawam" / "summary.json").write_text(json.dumps(summary, indent=2))
    hdr = f"{'setting':16s} {'succ':>5s} {'e_track':>7s} {'|app-cmd|':>9s} {'exact':>5s} {'e_pred':>7s} {'Δnone':>8s} {'CI':>20s} {'chg Δ':>8s} {'unchg Δ':>8s}"
    print(hdr)
    for t, s in summary.items():
        print(f"{t:16s} {s['n_success']}/{s['n_episodes']:<3d} {s['e_track']:7.3f} {s['applied_err']:9.3f} {s['exact_frac']:5.2f} "
              f"{s['e_pred']:7.4f} {s.get('e_pred_minus_none', 0):+8.4f} {_f(s.get('e_pred_minus_none_ci95')):>20s} "
              f"{s.get('e_pred_changed_minus_none', 0):+8.4f} {s.get('e_pred_unchanged_minus_none', 0):+8.4f}")
    for t, s in summary.items():
        for k in ("reward_vs_a0", "reward_changed_vs_a0", "reward_vs_none", "reward_a1_vs_none"):
            if k in s:
                print(t, k, json.dumps({kk: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v])
                                        for kk, v in s[k].items()}))
    if args.tsv:
        write_tsv(args.tsv, summary)
    if args.plot:
        plot(summary, args.plot)


if __name__ == "__main__":
    main()
