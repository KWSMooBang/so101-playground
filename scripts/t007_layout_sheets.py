#!/usr/bin/env python3
"""T007: real-world layout sheets for tasks/home_objects.jsonl.

One top-view PNG per layout + one PDF with all 30 (and an index HTML), in docs/home_objects/:
  * view from above with the robot at the top of the page (you stand at the front of the table, facing it);
  * object area (bench TABLE_BOUNDS, 39 x 25.5 cm), robot root, shoulder-pan axis (the measuring origin),
    robot keep-out polygon, plastic bin, each object's footprint with name and an arrow along its local +x;
  * 5 cm grid in the measuring frame: fwd = cm from the pan axis away from the robot, side = cm to the robot's
    left (= your right as you face the robot);
  * a table of fwd / side / heading for every item, and the sim overhead render of the start scene when
    present (docs/home_objects/sim_frames/trialNN_overhead.png, written by the Phase A run).

    python scripts/t007_layout_sheets.py     # needs numpy + matplotlib (isaac venv)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "docs/home_objects"
TASKS = HERE / "tasks/home_objects.jsonl"
LAYOUTS = HERE / "tasks/layouts/home_objects_layouts.jsonl"
DIMS = OUT / "asset_dims.json"

ROBOT_XY = np.array([0.05209, 0.18061])
ROBOT_YAW = math.radians(6.195)
PAN_ROOT = np.array([0.0208, -0.0231])
TABLE_BOUNDS = {"x": (-0.14, 0.25), "y": (-0.10, 0.155)}
SO101_BOX = [(0.012, 0.241), (0.0113, 0.1267), (0.0437, 0.1262), (0.0431, 0.0182), (0.0939, 0.0179),
             (0.0939, 0.1262), (0.1256, 0.126), (0.1263, 0.2403)]  # env cfg SO101_BOUNDING_BOX (world)
OVERHEAD_CAM = np.array([0.0, -0.34])
COLORS = {"black pen": "#222222", "red pen": "#d62728", "purple toothbrush": "#9467bd", "razor": "#7f7f7f",
          "black remote": "#444444", "blue paintbrush": "#1f77b4"}
HOME = {"black pen": "black pen", "red pen": "red pen", "purple toothbrush": "toothbrush", "razor": "razor",
        "black remote": "remote", "blue paintbrush": "paintbrush"}


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


PAN_WORLD = ROBOT_XY + rot(ROBOT_YAW) @ PAN_ROOT


def to_sheet(p_world):
    """World (m) -> sheet (cm): x = side (robot's left = page right), y = -fwd (robot at the top, page up)."""
    p = np.atleast_2d(p_world)
    base = (p - ROBOT_XY) @ rot(ROBOT_YAW)  # rows: R^T (p - t)
    side = 100 * (base[:, 0] - PAN_ROOT[0])
    fwd = -100 * (base[:, 1] - PAN_ROOT[1])
    return np.stack([side, -fwd], -1)


def boxes_outline(boxes):
    """Rasterize the union of local boxes (1 mm) and return its boundary as line segments (local m)."""
    b = np.asarray(boxes)
    x0, y0 = b[:, 0].min(), b[:, 1].min()
    nx, ny = int(round((b[:, 2].max() - x0) / 1e-3)) + 2, int(round((b[:, 3].max() - y0) / 1e-3)) + 2
    g = np.zeros((nx + 2, ny + 2), bool)
    for a, c, d, e in b:
        g[int(round((a - x0) / 1e-3)) + 1:int(round((d - x0) / 1e-3)) + 1,
          int(round((c - y0) / 1e-3)) + 1:int(round((e - y0) / 1e-3)) + 1] = True
    segs = []
    for i, j in zip(*np.nonzero(g)):
        X, Y = x0 + (i - 1) * 1e-3, y0 + (j - 1) * 1e-3
        if not g[i - 1, j]:
            segs.append(((X, Y), (X, Y + 1e-3)))
        if not g[i + 1, j]:
            segs.append(((X + 1e-3, Y), (X + 1e-3, Y + 1e-3)))
        if not g[i, j - 1]:
            segs.append(((X, Y), (X + 1e-3, Y)))
        if not g[i, j + 1]:
            segs.append(((X, Y + 1e-3), (X + 1e-3, Y + 1e-3)))
    return np.asarray(segs)


def draw_layout(ax, row, lay):
    # grid
    xs = np.arange(-35, 36, 5)
    ys = np.arange(-65, 11, 5)
    for x in xs:
        ax.axvline(x, color="#dddddd", lw=0.5, zorder=0)
    for y in ys:
        ax.axhline(y, color="#dddddd", lw=0.5, zorder=0)
    ax.axvline(0, color="#aaaaaa", lw=0.8, zorder=0)
    ax.axhline(0, color="#aaaaaa", lw=0.8, zorder=0)
    # object area
    tb = np.array([[TABLE_BOUNDS["x"][0], TABLE_BOUNDS["y"][0]], [TABLE_BOUNDS["x"][1], TABLE_BOUNDS["y"][0]],
                   [TABLE_BOUNDS["x"][1], TABLE_BOUNDS["y"][1]], [TABLE_BOUNDS["x"][0], TABLE_BOUNDS["y"][1]]])
    ax.add_patch(Polygon(to_sheet(tb), closed=True, fill=False, ls="--", ec="#2ca02c", lw=1.0, zorder=1))
    a = to_sheet(tb)
    ax.text(a[:, 0].min() + 0.5, a[:, 1].min() + 0.5, "object area (sim TABLE_BOUNDS)", color="#2ca02c",
            fontsize=6, zorder=1)
    # robot keep-out + root + pan axis
    ax.add_patch(Polygon(to_sheet(np.array(SO101_BOX)), closed=True, fc="#ffe0cc", ec="#ff7f0e", lw=0.8, zorder=1))
    r = to_sheet(ROBOT_XY)[0]
    ax.plot(*r, marker="s", color="#ff7f0e", ms=4, zorder=3)
    ax.annotate("robot root", r, xytext=(r[0] + 2, r[1] + 3), fontsize=6, color="#ff7f0e")
    ax.plot(0, 0, marker="+", color="k", ms=12, mew=1.5, zorder=4)
    ax.annotate("ORIGIN: shoulder-pan axis\n(centre of the base turntable)", (0, 0), xytext=(4, 3), fontsize=6.5)
    # bin
    b = lay["bin"]
    hx, hy = b["footprint_half_extents"]
    yaw = b.get("yaw", b["rpy"][2])
    corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) @ rot(yaw).T + np.array(b["position"][:2])
    ax.add_patch(Polygon(to_sheet(corners), closed=True, fc="#e6f2ff", ec="#1f77b4", lw=1.0, zorder=2))
    c = to_sheet(b["position"][:2])[0]
    ax.text(*c, "plastic bin", ha="center", va="center", fontsize=7, color="#1f77b4", zorder=3)
    # objects
    target = row.get("target", row["objects"][0])
    refs = row.get("referents", [])
    for o in lay["objects"]:
        segs = boxes_outline(o["footprint_boxes"])
        w = segs.reshape(-1, 2) @ rot(o["yaw"]).T + np.array(o["position"][:2])
        s = to_sheet(w).reshape(-1, 2, 2)
        col = COLORS[o["name"]]
        role = "TARGET" if o["name"] == target else ("ref" if o["name"] in refs else "")
        for seg in s:
            ax.plot(seg[:, 0], seg[:, 1], color=col, lw=1.6 if role == "TARGET" else 0.9, zorder=3,
                    solid_capstyle="butt")
        p = np.array(o["position"][:2])
        tip = p + rot(o["yaw"]) @ np.array([0.05, 0.0])
        ps, ts = to_sheet(p)[0], to_sheet(tip)[0]
        ax.annotate("", xy=ts, xytext=ps, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0), zorder=4)
        ax.plot(*ps, marker="o", color=col, ms=3, zorder=4)
        label = HOME[o["name"]] + (f" [{role}]" if role else "")
        ax.annotate(label, ps, xytext=(ps[0] + 1.2, ps[1] - 2.2), fontsize=7, color=col, weight="bold", zorder=5)
    if row["home_family"] == "move":
        d = {"left": (1, 0), "right": (-1, 0), "forward": (0, -1), "backward": (0, 1)}[row["direction"]]
        t = next(o for o in lay["objects"] if o["name"] == target)
        p = np.array(t["position"][:2])
        q = to_sheet(np.stack([p, p + 0.07 * np.array(d)]))
        ax.annotate("", xy=q[1], xytext=q[0], arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2, ls="--"))
    cam = to_sheet(OVERHEAD_CAM)[0]
    ax.text(cam[0], max(cam[1], -51.5), "↓ front of table / overhead camera side", ha="center", fontsize=6.5,
            color="#555555")
    ax.set_xlim(-36, 36)
    ax.set_ylim(-53, 12)
    ax.set_aspect("equal")
    ax.set_xlabel("side [cm]: + = robot's left = YOUR RIGHT (facing the robot)", fontsize=7)
    ax.set_ylabel("fwd [cm] from pan axis (downwards = away from robot)", fontsize=7)
    ax.set_yticks(ys[::2])
    ax.set_yticklabels([f"{-y:g}" for y in ys[::2]], fontsize=6)
    ax.set_xticks(xs[::2])
    ax.tick_params(labelsize=6)


def table_rows(row, lay):
    rb = lay["robot_base_frame"]
    out = [["bin", rb["bin"]["pan_fwd_cm"], rb["bin"]["pan_side_cm"], rb["bin"]["heading_from_fwd_deg"], ""]]
    target = row.get("target", row["objects"][0])
    for o in lay["objects"]:
        r = rb["objects"][o["name"]]
        role = "target" if o["name"] == target else ("ref" if o["name"] in row.get("referents", []) else "")
        out.append([HOME[o["name"]], r["pan_fwd_cm"], r["pan_side_cm"], r["heading_from_fwd_deg"], role])
    return out


def page(row, lay):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    ax = fig.add_axes([0.04, 0.07, 0.52, 0.83])
    draw_layout(ax, row, lay)
    fam = row["home_family"] + (f" ({row['direction']})" if "direction" in row else "")
    fig.suptitle(f"T007 home layout {row['trial_id']:02d}  —  {fam}  —  \"{row['instruction']}\"", fontsize=12,
                 x=0.04, ha="left", y=0.97)
    tx = fig.add_axes([0.60, 0.50, 0.38, 0.40])
    tx.axis("off")
    rows = table_rows(row, lay)
    t = tx.table(cellText=[[r[0], f"{r[1]:.1f}", f"{r[2]:+.1f}", f"{r[3]:+.0f}", r[4]] for r in rows],
                 colLabels=["item", "fwd cm", "side cm", "heading °", "role"], loc="upper left", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.4)
    tx.text(0, -0.02, "Point = asset origin (dot on the drawing; for these objects ≈ the middle of the\n"
                      "footprint). heading = direction of the arrow (asset +x), degrees counter-clockwise\n"
                      "seen from above, from 'fwd' (0 = pointing away from the robot, +90 = pointing to\n"
                      "your left, -90 = to your right). Bin heading is for its long side.",
            fontsize=6.5, va="top", transform=tx.transAxes)
    img = OUT / "sim_frames" / f"trial{row['trial_id']:02d}_overhead.png"
    if img.exists():
        ia = fig.add_axes([0.60, 0.05, 0.38, 0.33])
        ia.imshow(plt.imread(img))
        ia.set_title("sim overhead camera at robot start (match your overhead view)", fontsize=7)
        ia.axis("off")
    return fig


def dims_page(dims):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("T007 bench asset sizes (collision meshes, cm) — compare with your real objects", fontsize=12,
                 x=0.04, ha="left")
    ax = fig.add_axes([0.04, 0.35, 0.92, 0.55])
    ax.axis("off")
    cell = []
    for name, d in dims["objects"].items():
        L, W, H = d["size_cm"]
        cell.append([HOME[name], name, "sim-seen" if name in ("black pen", "black remote", "blue paintbrush")
                     else "real-only", f"{L:.1f}", f"{W:.1f}", f"{H:.1f}"])
    L, W, H = dims["plastic_bin"]["size_cm"]
    cell.append(["plastic bin", "plastic_bin", "—", f"{L:.1f}", f"{W:.1f}", f"{H:.1f}"])
    t = ax.table(cellText=cell, colLabels=["home object", "bench asset", "group", "length (along arrow, +x)",
                                           "width", "height"], loc="upper left", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    fig.text(0.04, 0.30, "Reference frame used on every sheet: origin = shoulder-pan axis of the SO-101 (centre of the "
             "base turntable), sim position\n"
             f"world ({PAN_WORLD[0]:.4f}, {PAN_WORLD[1]:.4f}) m = robot root (0.0521, 0.1806) + Rz(6.195°)·(0.0208, "
             "−0.0231). fwd = away from the robot along the base's\n"
             "facing direction (root −y), side = robot's left (root +x). Table top = z 0. The bin stays at the "
             "bench canonical pose for all 30 layouts.\n"
             "Spatial layouts hold 4 objects (bench convention); bin layouts 1 object.", fontsize=8, va="top")
    return fig


def main():
    rows = [json.loads(l) for l in TASKS.read_text().splitlines()]
    lays = [json.loads(l) for l in LAYOUTS.read_text().splitlines()]
    dims = json.loads(DIMS.read_text())
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "home_objects_layouts.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = dims_page(dims)
        pdf.savefig(fig)
        plt.close(fig)
        for row, lay in zip(rows, lays):
            fig = page(row, lay)
            fig.savefig(OUT / "sheets" / f"layout_{row['trial_id']:02d}.png", dpi=110)
            pdf.savefig(fig)
            plt.close(fig)
    # flat CSV of every pose
    lines = ["trial_id,family,instruction,item,role,pan_fwd_cm,pan_side_cm,heading_from_fwd_deg,"
             "base_x_m,base_y_m,base_yaw_deg,world_x_m,world_y_m,world_yaw_deg"]
    for row, lay in zip(rows, lays):
        rb = lay["robot_base_frame"]
        items = [("plastic bin", rb["bin"], "")] + [
            (o, rb["objects"][o], "target" if o == row.get("target", row["objects"][0])
             else ("ref" if o in row.get("referents", []) else "")) for o in row["objects"]]
        for name, r, role in items:
            lines.append(",".join(str(v) for v in [
                row["trial_id"], row["home_family"], f"\"{row['instruction']}\"", name, role, r["pan_fwd_cm"],
                r["pan_side_cm"], r["heading_from_fwd_deg"], *r["base_xy_m"], r["base_yaw_deg"],
                *r["world_xy_m"], r["world_yaw_deg"]]))
    (OUT / "home_objects_poses.csv").write_text("\n".join(lines) + "\n")
    html = ["<html><head><meta charset='utf-8'><title>T007 home layouts</title></head><body>",
            "<h1>T007 home-object layouts</h1><p>PDF: <a href='home_objects_layouts.pdf'>home_objects_layouts.pdf"
            "</a>; poses: <a href='home_objects_poses.csv'>home_objects_poses.csv</a></p>"]
    for row in rows:
        html.append(f"<h2>{row['trial_id']:02d} {row['home_family']}: {row['instruction']}</h2>"
                    f"<img src='sheets/layout_{row['trial_id']:02d}.png' width='1100'>")
    html.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(html))
    print(f"wrote {pdf_path}, {len(rows)} sheets")


if __name__ == "__main__":
    main()
