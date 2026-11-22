#!/usr/bin/env python3
"""T006 figure: InternW0-Δ vs LaWAM on the T004nom and T005 designs.

The two models score in different spaces (InternW0: Wan2.2 VAE latent MSE on ticks 4–16; LaWAM: LAM DINOv3
token MSE at +35), so each model gets its own row in its own units, with 95% CIs. Percentages are point labels only
(their CIs, computed inside the bootstrap, are in t006_analysis's summary.json and the doc). LaWAM numbers come from results.tsv (T004nom-lawam-*,
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
    """One row per model, raw units with 95% CIs (each model in its own metric space; no shared axis).
    Percentages are point labels only: (a) d_fault − d_nominal as % of the fault's own effect
    d(fault obs, nominal obs); (b) Δ e_pred as % of the model's none e_pred. Their CIs are in the doc."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"LaWAM": BLUE, "InternW0-Δ": ORANGE}
    units = {"LaWAM": "LAM token MSE @+35", "InternW0-Δ": "VAE latent MSE, ticks 4–16"}
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 3.6 * len(models)), squeeze=False)
    for r, (name, m) in enumerate(models.items()):
        c = colors[name]
        ax = axes[r, 0]
        nom = m["nom"]
        xs = [i for i, k in enumerate(CONDS) if k in nom and nom[k]["diff"]]
        y = [nom[CONDS[i]]["diff"][0] for i in xs]
        ax.vlines(xs, [nom[CONDS[i]]["diff"][1] for i in xs], [nom[CONDS[i]]["diff"][2] for i in xs], color=c, lw=2)
        ax.plot(xs, y, "o", color=c, ms=8, mec="white", mew=1.5)
        for i, v in zip(xs, y):
            k = CONDS[i]
            if k != "none" and nom[k]["d_obs_nom"]:
                ax.annotate(f"{100 * v / nom[k]['d_obs_nom']:.0f}% of fault effect", (i, v), xytext=(9, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK2)
        ax.set_xticks(range(len(CONDS)), CONDS)
        ax.set_xlim(-0.4, len(CONDS) - 0.2)
        ax.set_ylabel(f"{name}\nd_fault − d_nominal\n({units[name]})", color=INK, fontsize=9)
        ora = m["oracle"]
        al = sorted(k for k in ora if k != "none")
        ax2 = axes[r, 1]
        oy = [ora[a][0] for a in al]
        ax2.vlines(al, [ora[a][1] for a in al], [ora[a][2] for a in al], color=c, lw=2)
        ax2.plot(al, oy, "o", color=c, ms=8, mec="white", mew=1.5)
        for a, v in zip(al, oy):
            ax2.annotate(f"{100 * v / ora['none']:+.0f}%", (a, v), xytext=(9, 0), textcoords="offset points",
                         va="center", fontsize=8, color=INK2)
        ax2.set_xticks((0, 0.5, 1, 1.5))
        ax2.set_xlim(-0.15, 1.75)
        ax2.set_ylabel(f"{name}\nraw e_pred − none\n({units[name]})", color=INK, fontsize=9)
        for a_ in (ax, ax2):
            _style(a_)
    axes[0, 0].set_title("Nominal replay: > 0 = prediction on the fault-free side\n(95% CI; label = share of the fault's own effect)",
                         color=INK, fontsize=9.5)
    axes[0, 1].set_title("gain2 oracle, no replay: raw e_pred vs α\n(95% CI, episode-paired; label = % of none e_pred)",
                         color=INK, fontsize=9.5)
    axes[-1, 0].set_xlabel("injected gap", color=INK2, fontsize=8)
    axes[-1, 1].set_xlabel("oracle correction α (1 = exact inverse; InternW0: α ∈ {0, 1, 1.5})",
                           color=INK2, fontsize=8)
    fig.suptitle("T006: InternW0-Δ vs LaWAM, 8 ep per point (each row in its model's own units)", color=INK, fontsize=11)
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
