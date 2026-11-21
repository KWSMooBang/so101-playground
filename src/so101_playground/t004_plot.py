#!/usr/bin/env python3
"""T004 summary figure: prediction error vs tracking error per control-gap condition, one panel per model
(each model has its own distance, so separate y axes = small multiples, not a dual axis).

Per condition: e_pred (prediction vs observed future) and e_ideal (prediction vs perfect-tracking replay),
both on the same replayed queries, plotted against e_track (mean |cmd - q| over arm joints, deg).

  t004_plot.py ROOT OUT.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (dataviz reference palette)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PANELS = {
    "lawam": ("LaWAM: LAM DINOv3 token MSE at +35 ticks", "paired_obs_e_pred_lat", "paired_ideal_e_pred_lat"),
    "flux3": ("FLUX 3 Action: DINOv2 distance, scene view, +16 ticks",
              "paired_e_pred.dino_scene@16", "paired_e_ideal.dino_scene@16"),
}


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    models = [m for m in PANELS if (root / m / "summary.json").exists()]
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6), facecolor=SURFACE, squeeze=False)
    for ax, model in zip(axes[0], models):
        title, kp, ki = PANELS[model]
        summary = json.loads((root / model / "summary.json").read_text())
        rows = [(c, s["e_track_deg"], s.get(kp), s.get(ki)) for c, s in summary.items() if s.get(kp) is not None]
        ax.set_facecolor(SURFACE)
        for c, x, yp, yi in rows:
            ax.plot([x, x], [yi, yp], color=GRID, lw=2, zorder=1)
        ax.scatter([r[1] for r in rows], [r[2] for r in rows], s=64, color=BLUE, marker="o", zorder=3,
                   edgecolors=SURFACE, linewidths=2, label="e_pred: prediction vs observed")
        ax.scatter([r[1] for r in rows], [r[3] for r in rows], s=64, color=ORANGE, marker="s", zorder=3,
                   edgecolors=SURFACE, linewidths=2, label="e_ideal: prediction vs perfect-tracking replay")
        for c, x, yp, _ in rows:
            ax.annotate(c, (x, yp), xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK2)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("e_track: mean |command - joint| (deg)", fontsize=10, color=INK2)
        ax.set_ylabel("distance (lower = better prediction)", fontsize=10, color=INK2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.suptitle("T004: WAM future prediction under injected sim control gaps (same queries for both distances)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
