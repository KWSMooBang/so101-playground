#!/usr/bin/env python3
"""T006 figure: InternW0-Δ vs LaWAM on the T004nom and T005 designs.

The two models score in different spaces (InternW0: Wan2.2 VAE latent MSE on ticks 4–16; LaWAM: LAM DINOv3
token MSE at +35), so every value is indexed to the model's own scale: % of its `none` e_pred, or a fraction
of the fault's own effect d(fault obs, nominal obs). LaWAM numbers come from results.tsv (T004nom-lawam-*,
T005-*); InternW0 from runs/T006/{nom,oracle}/internw0/summary.json (t006_analysis.py).

    python -m so101_playground.t006_plot --tsv results.tsv --t006 runs/T006 --out T006_internw0_vs_lawam.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (dataviz reference palette): LaWAM, InternW0
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONDS = ("none", "gain", "gain2", "latency6")
ALPHAS = (0.0, 1.0, 1.5)


def _num(x):
    return float(x) if x not in ("", None) else None


def _ci(text, key):
    m = re.search(key + r"=(-?[\d.]+) ci=\[(-?[\d.]+),(-?[\d.]+)\]", text)
    return [float(m.group(i)) for i in (1, 2, 3)] if m else None


def lawam_from_tsv(path: Path) -> dict:
    rows = {}
    for line in path.read_text().splitlines()[1:]:
        f = line.split("\t")
        rows[f[0]] = f
    nom = {}
    for c in CONDS:
        f = rows.get(f"T004nom-lawam-{c}")
        if f:
            nom[c] = {"diff": _ci(f[17], "paired_diff"), "d_fault": _num(f[10]), "d_obs_nom": _num(f[12])}
    ora = {"none": _num(rows["T005-none"][9])}
    for a in (0, 0.5, 1, 1.5):
        f = rows.get(f"T005-gain2-a{a:g}")
        if f:
            lo, hi = (float(v) for v in f[14].strip("[]").split(","))
            ora[a] = [_num(f[13]), lo, hi]
    return {"nom": nom, "oracle": ora}


def internw0_from_summary(t006: Path) -> dict:
    nom_s = json.loads((t006 / "nom" / "internw0" / "summary.json").read_text())
    ora_s = json.loads((t006 / "oracle" / "internw0" / "summary.json").read_text())
    nom = {c: {"diff": [s["diff@1"], *s["diff@1_ci95"]], "d_fault": s["d_fault@1"], "d_obs_nom": s["d_obs_nom@1"]}
           for c, s in nom_s.items() if "diff@1" in s}
    ora = {"none": ora_s["none"]["e_pred@1"]}
    for a in ALPHAS:
        s = ora_s.get(f"gain2-a{a:g}")
        if s and "e_pred@1_minus_none" in s:
            ora[a] = [s["e_pred@1_minus_none"], *s["e_pred@1_minus_none_ci95"]]
    return {"nom": nom, "oracle": ora}


def _style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.axhline(0, color=INK2, lw=1)


def plot(models: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = {"LaWAM": BLUE, "InternW0-Δ": ORANGE}
    off = {"LaWAM": -0.12, "InternW0-Δ": 0.12}
    for name, m in models.items():
        c = colors[name]
        nom = m["nom"]
        base = nom["none"]["d_fault"]
        xs = [i for i, k in enumerate(CONDS) if k in nom and nom[k]["diff"]]
        # (a) d_fault - d_nominal, % of the model's none-level d_fault
        y = [100 * nom[CONDS[i]]["diff"][0] / base for i in xs]
        lo = [100 * nom[CONDS[i]]["diff"][1] / base for i in xs]
        hi = [100 * nom[CONDS[i]]["diff"][2] / base for i in xs]
        px = [i + off[name] for i in xs]
        axes[0].vlines(px, lo, hi, color=c, lw=2)
        axes[0].plot(px, y, "o", color=c, ms=8, mec="white", mew=1.5, label=name)
        # (b) the same as a fraction of the fault's own effect d(fault obs, nominal obs)
        fx = [i for i in xs if CONDS[i] != "none" and nom[CONDS[i]]["d_obs_nom"]]
        fy = [100 * nom[CONDS[i]]["diff"][0] / nom[CONDS[i]]["d_obs_nom"] for i in fx]
        axes[1].bar([i + off[name] for i in fx], fy, width=0.22, color=c, label=name)
        # (c) oracle: delta e_pred vs none, % of none e_pred
        ora = m["oracle"]
        ax_ = [a for a in sorted(k for k in ora if k != "none")]
        oy = [100 * ora[a][0] / ora["none"] for a in ax_]
        olo = [100 * ora[a][1] / ora["none"] for a in ax_]
        ohi = [100 * ora[a][2] / ora["none"] for a in ax_]
        # points + CI only: InternW0 has no alpha=0.5 run, so no line is drawn between settings
        oxp = [a + off[name] / 3 for a in ax_]
        axes[2].vlines(oxp, olo, ohi, color=c, lw=2)
        axes[2].plot(oxp, oy, "o", color=c, ms=8, mec="white", mew=1.5, label=name)
    for ax in axes:
        _style(ax)
    axes[0].set_xticks(range(len(CONDS)), CONDS)
    axes[0].set_ylabel("d_fault − d_nominal  (% of none d_fault)", color=INK, fontsize=9)
    axes[0].set_title("Nominal replay: > 0 = prediction sits on the fault-free side\n(95% CI, query-paired, episode bootstrap)",
                      color=INK, fontsize=9.5)
    axes[1].set_xticks(range(len(CONDS))[1:], CONDS[1:])
    axes[1].set_ylabel("(d_fault − d_nominal) / d(fault obs, nominal obs)  (%)", color=INK, fontsize=9)
    axes[1].set_title("Share of the fault's own visual effect\nthat the prediction does not follow", color=INK, fontsize=9.5)
    axes[2].set_xticks((0, 0.5, 1, 1.5))
    axes[2].set_xlabel("oracle stiffness correction α (1 = exact inverse)", color=INK2, fontsize=8)
    axes[2].set_ylabel("raw e_pred − none  (% of none e_pred)", color=INK, fontsize=9)
    axes[2].set_title("gain2 oracle (no replay): raw e_pred vs α\n(95% CI, episode-paired; InternW0 ran α ∈ {0, 1, 1.5})", color=INK, fontsize=9.5)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left" if ax is axes[0] else "best")
    fig.suptitle("T006: InternW0-Δ (VAE latent, ticks 4–16) vs LaWAM (LAM tokens, +35), 8 ep per point",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=Path("/outputs/so101-playground/results.tsv"))
    ap.add_argument("--t006", type=Path, default=Path("/outputs/so101-playground/runs/T006"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plot({"LaWAM": lawam_from_tsv(args.tsv), "InternW0-Δ": internw0_from_summary(args.t006)}, args.out)


if __name__ == "__main__":
    main()
