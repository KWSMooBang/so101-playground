#!/usr/bin/env python3
"""T007 figure: (a) Phase A success by family x model; (b) Phase B paired d_fault - d_nominal @1 by subset and
condition, with T006 seen-object Bin as reference; (c) oracle gain2 α0-vs-α1 AUROC by subset, T006 reference.

    python -m so101_playground.t007_plot --out docs/T007_home_objects.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/outputs/so101-playground/runs/T007")
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
# T006 (8 seen-object val_set Bin episodes): paired diff@1 [95% CI] and AUROC α0 vs α1 (query, window4, episode).
T006_DIFF = {"none": (-0.0018, -0.0063, 0.0016), "gain2": (0.0565, 0.0486, 0.0634), "latency6": (0.0815, 0.0701, 0.0945)}
T006_AUROC = {"query": (0.89, 0.83, 0.96), "window4": (0.93, 0.86, 0.99), "episode": (1.00, 1.00, 1.00)}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb8")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color="#ebeae7", lw=0.6)
    ax.set_axisbelow(True)


def panel_a(ax, a):
    fams = ["bin", "move", "next_to", "between", "pooled"]
    w = 0.38
    for i, (m, col, lab) in enumerate((("internw0", BLUE, "InternW0-Δ"), ("lawam", ORANGE, "LaWAM"))):
        if m not in a:
            continue
        x = np.arange(len(fams)) + (i - 0.5) * w
        r = [a[m][f]["rate"] for f in fams]
        ax.bar(x, r, w - 0.04, color=col, label=lab)
        for xi, f in zip(x, fams):
            ax.text(xi, a[m][f]["rate"] + 0.02, f"{a[m][f]['success']}/{a[m][f]['n']}", ha="center", fontsize=7,
                    color=INK2)
    ax.axhline(1.0, color=GRAY, ls="--", lw=1)
    ax.text(1.5, 1.02, "T004–T006 seen-object Bin: 8/8 (both)", ha="center", fontsize=7, color=INK2)
    ax.axhline(0.5, color="#bdbcb8", ls=":", lw=1)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("success (none)", fontsize=8, color=INK2)
    ax.set_title("(a) Phase A: success by family, home-object set, 1 run/layout", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.0))
    style(ax)


def panel_b(ax, b):
    conds = ["none", "gain2", "latency6"]
    subs = [("bin", BLUE, "home bin (all 6)"), ("sim_seen", AQUA, "sim-seen objects (3)"),
            ("real_only", ORANGE, "real-only objects (3)")]
    w = 0.2
    for j, c in enumerate(conds):
        lo, mid = T006_DIFF[c][1], T006_DIFF[c][0]
        x = j - 1.5 * w
        ax.errorbar(x, mid, yerr=[[mid - T006_DIFF[c][1]], [T006_DIFF[c][2] - mid]], fmt="s", color=GRAY, ms=6,
                    capsize=2, lw=1.2, label="T006 seen Bin (8)" if j == 0 else None)
        for i, (s, col, lab) in enumerate(subs):
            d = b["nominal"].get(c, {}).get(s)
            if not d or "diff@1" not in d:
                continue
            v, (l, h) = d["diff@1"], d["diff@1_ci95"]
            ax.errorbar(j + (i - 0.5) * w, v, yerr=[[v - l], [h - v]], fmt="o", color=col, ms=6, capsize=2, lw=1.2,
                        label=lab if j == 0 else None)
    ax.axhline(0, color="#bdbcb8", lw=1)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_ylabel("d_fault − d_nominal @1 (VAE latent MSE)", fontsize=8, color=INK2)
    ax.set_title("(b) Phase B nominal replay: paired gap, 95% CI", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax)


def panel_c(ax, b):
    kinds = ["query", "window4", "episode"]
    subs = [("bin", BLUE, "home bin (all 6)"), ("sim_seen", AQUA, "sim-seen (3)"), ("real_only", ORANGE, "real-only (3)")]
    w = 0.2
    for j, k in enumerate(kinds):
        m, l, h = T006_AUROC[k]
        ax.errorbar(j - 1.5 * w, m, yerr=[[m - l], [h - m]], fmt="s", color=GRAY, ms=6, capsize=2, lw=1.2,
                    label="T006 seen Bin (8)" if j == 0 else None)
        for i, (s, col, lab) in enumerate(subs):
            r = b.get("auroc_a0_vs_a1", {}).get(s, {}).get("e_pred@1")
            if not r:
                continue
            v, (lo, hi) = r[f"auroc_{k}"], r[f"auroc_{k}_ci95"]
            ax.errorbar(j + (i - 0.5) * w, v, yerr=[[v - lo], [hi - v]], fmt="o", color=col, ms=6, capsize=2,
                        lw=1.2, label=lab if j == 0 else None)
    ax.axhline(0.5, color="#bdbcb8", lw=1, ls=":")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("AUROC e_pred@1: gain2 α0 > α1", fontsize=8, color=INK2)
    ax.set_title("(c) Phase B oracle: raw e_pred as reward, 95% CI", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    style(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a = json.loads((ROOT / "summary_A.json").read_text())
    b = json.loads((ROOT / "summary_B.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    panel_a(axs[0], a)
    panel_b(axs[1], b)
    panel_c(axs[2], b)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
