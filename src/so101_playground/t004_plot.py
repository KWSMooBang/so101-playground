#!/usr/bin/env python3
"""T004 summary figure: prediction error vs tracking error per control-gap condition, one panel per model
(each model has its own distance, so separate y axes = small multiples, not a dual axis).

Per condition: e_pred (prediction vs observed future) and e_ideal (prediction vs perfect-tracking replay),
both on the same replayed queries, plotted against e_track (mean |cmd - q| over arm joints, deg).

  t004_plot.py ROOT_V1 ROOT_V2 OUT.png   (ROOT_V1 = runs/T004, ROOT_V2 = runs/T004_nominal)
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
# (summary root key, model, title, e_pred key, e_ideal key or None). e_ideal is drawn only from runs made with the
# post-review replay (T004_v2); FLUX's replay values are invalid and it shows e_pred over all queries.
PANELS = [
    ("v2", "lawam", "LaWAM: LAM DINOv3 token MSE at +35 ticks (nominal replay)",
     "paired_obs_e_pred_lat", "paired_ideal_e_pred_lat"),
    ("v1", "flux3", "FLUX 3 Action: DINOv2 distance, scene view, +16 ticks (e_pred only)",
     "e_pred.dino_scene@16", None),
]


def main() -> None:
    roots = {"v1": Path(sys.argv[1]), "v2": Path(sys.argv[2])}
    out = Path(sys.argv[3])
    panels = [p for p in PANELS if (roots[p[0]] / p[1] / "summary.json").exists()]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), facecolor=SURFACE, squeeze=False)
    for ax, (rk, model, title, kp, ki) in zip(axes[0], panels):
        summary = json.loads((roots[rk] / model / "summary.json").read_text())
        rows = [(c, s["e_track_deg"], s.get(kp), s.get(ki) if ki else None) for c, s in summary.items()
                if s.get(kp) is not None]
        ax.set_facecolor(SURFACE)
        for c, x, yp, yi in rows:
            if yi is not None:
                ax.plot([x, x], [yi, yp], color=GRID, lw=2, zorder=1)
        ax.scatter([r[1] for r in rows], [r[2] for r in rows], s=64, color=BLUE, marker="o", zorder=3,
                   edgecolors=SURFACE, linewidths=2, label="d_fault: pred vs gap arm")
        if ki:
            ax.scatter([r[1] for r in rows], [r[3] for r in rows], s=64, color=ORANGE, marker="s", zorder=3,
                       edgecolors=SURFACE, linewidths=2, label="d_nominal: pred vs fault-free arm, same cmds")
        # Direct labels; nudge a label below its point when it would collide with one already placed.
        xr = max(r[1] for r in rows) - min(r[1] for r in rows) or 1.0
        yr = max(r[2] for r in rows) - min(r[2] for r in rows) or 1.0
        placed = []
        for c, x, yp, _ in sorted(rows, key=lambda r: r[2], reverse=True):
            clash = any(abs(x - px) < 0.12 * xr and abs(yp - py) < 0.06 * yr for px, py in placed)
            ax.annotate(c, (x, yp), xytext=(6, -13) if clash else (6, 4), textcoords="offset points",
                        fontsize=9, color=INK2)
            placed.append((x, yp))
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
        if ki:
            ax.legend(frameon=False, fontsize=9, loc="center", bbox_to_anchor=(0.35, 0.55), labelcolor=INK)
        else:
            ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    fig.suptitle("T004: WAM future prediction under injected sim control gaps (paired queries)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
