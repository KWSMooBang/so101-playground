#!/usr/bin/env python3
"""T007: real-world layout sheets for tasks/home_objects.jsonl.

One top-view PNG per layout + one PDF with all 30 (and an index HTML), in docs/home_objects/:
  * view from above with the robot at the top of the page (you stand at the front of the table, facing it);
  * object area (bench TABLE_BOUNDS, 39 x 25.5 cm), robot root, shoulder-pan axis (the measuring origin),
    robot keep-out polygon, plastic bin, each object's footprint with name and an arrow along its local +x;
  * 5 cm grid in the measuring frame: fwd = cm from the pan axis away from the robot, side = cm to the robot's
    left (= your right as you face the robot);
  * a table of fwd / side / heading for every item, and the sim overhead render of the start scene when
    present (docs/home_objects/sim_frames/trialNN_overhead.jpg, written by the Phase A run).

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


# Overhead camera (bench DomainRandomizationSpec defaults; so101_bench_env_cfg.py camera_overhead, OpenGL
# convention: looks along its -Z, +Y up). Used to draw the sheet geometry onto the sim frame as a check.
CAM_POS = np.array([0.0, -0.34, 0.45722])
CAM_RPY_DEG = (51.361, -0.169, 0.603)
CAM_HFOV_DEG = 74.0
CAM_WH = (640, 480)


def _rpy_matrix(rpy_deg):
    """Extrinsic XYZ (isaacsim euler_angles_to_quat default): R = Rz @ Ry @ Rx."""
    r, p, y = np.radians(rpy_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


CAM_R = _rpy_matrix(CAM_RPY_DEG)


def project(p_world, img_w):
    """World points (N,2 on the table or N,3) -> pixel coords in an image of width img_w (4:3)."""
    p = np.atleast_2d(np.asarray(p_world, float))
    if p.shape[1] == 2:
        p = np.concatenate([p, np.full((len(p), 1), 0.002)], 1)
    c = (p - CAM_POS) @ CAM_R  # rows: R^T (p - c)
    f = (CAM_WH[0] / 2) / math.tan(math.radians(CAM_HFOV_DEG) / 2)
    u = CAM_WH[0] / 2 + f * c[:, 0] / -c[:, 2]
    v = CAM_WH[1] / 2 - f * c[:, 1] / -c[:, 2]
    return np.stack([u, v], -1) * (img_w / CAM_WH[0])


def clock(heading_from_fwd_deg):
    """Arrow direction for a person standing at the front of the table facing the robot.

    heading_from_fwd_deg is counter-clockwise seen from above from 'fwd' (pointing toward that person), so
    +90 points to the person's RIGHT. Returned: degrees clockwise from 12 o'clock (= toward the robot) and the
    clock hour (12 = toward the robot, 3 = your right, 6 = toward you, 9 = your left)."""
    cw = (-(heading_from_fwd_deg + 180.0)) % 360.0
    hour = round(cw / 30.0) % 12 or 12
    return cw, hour


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


def bin_corners(b):
    """Bin footprint rectangle in world xy, footprint_center_offset applied in the bin frame (as the bench
    placement does: layouts._rectangle_local_corners + offset)."""
    hx, hy = b["footprint_half_extents"]
    ox, oy = b.get("footprint_center_offset", [0.0, 0.0])
    yaw = b.get("yaw", b["rpy"][2])
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) + np.array([ox, oy])
    return local @ rot(yaw).T + np.array(b["position"][:2])


def object_outline_world(o):
    segs = boxes_outline(o["footprint_boxes"])
    return (segs.reshape(-1, 2) @ rot(o["yaw"]).T + np.array(o["position"][:2])).reshape(-1, 2, 2)


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
    corners = bin_corners(b)
    ax.add_patch(Polygon(to_sheet(corners), closed=True, fc="#e6f2ff", ec="#1f77b4", lw=1.0, zorder=2))
    c = to_sheet(b["position"][:2])[0]
    ax.text(*c, "plastic bin", ha="center", va="center", fontsize=7, color="#1f77b4", zorder=3)
    # objects
    target = row.get("target", row["objects"][0])
    refs = row.get("referents", [])
    for o in lay["objects"]:
        s = to_sheet(object_outline_world(o).reshape(-1, 2)).reshape(-1, 2, 2)
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
    ax.set_aspect("equal")
    ax.set_xlabel("RIGHT [cm] from the pan axis: + = to YOUR RIGHT, − = to your left (you face the robot)",
                  fontsize=7)
    ax.set_ylabel("TOWARD YOU [cm] from the pan axis (down the page = away from the robot)", fontsize=7)
    ax.text(35.5, 11, "robot side", ha="right", va="top", fontsize=6.5, color="#555555")
    ax.text(-35.5, 0.5, "← your LEFT", fontsize=6.5, color="#555555")
    ax.text(35.5, 0.5, "your RIGHT →", ha="right", fontsize=6.5, color="#555555")
    ax.set_yticks(ys[::2])
    ax.set_yticklabels([f"{-y:g}" for y in ys[::2]], fontsize=6)
    ax.set_xticks(xs[::2])
    ax.tick_params(labelsize=6)
    ax.set_xlim(-36, 36)  # after set_*ticks, which would widen the view
    ax.set_ylim(-53, 12)


def table_rows(row, lay):
    rb = lay["robot_base_frame"]
    items = [("bin", rb["bin"], "")]
    target = row.get("target", row["objects"][0])
    for o in lay["objects"]:
        role = "target" if o["name"] == target else ("ref" if o["name"] in row.get("referents", []) else "")
        items.append((HOME[o["name"]], rb["objects"][o["name"]], role))
    out = []
    for name, r, role in items:
        cw, hour = clock(r["heading_from_fwd_deg"])
        out.append([name, r["pan_fwd_cm"], r["pan_side_cm"], cw, hour, role])
    return out


def draw_sim_overlay(ia, row, lay, img):
    """Sim overhead frame with the sheet's outlines and arrows projected onto it (check + orientation cue)."""
    im = plt.imread(img)
    w = im.shape[1]
    ia.imshow(im)
    for o in lay["objects"]:
        col = COLORS[o["name"]]
        for seg in object_outline_world(o)[::3]:
            q = project(seg, w)
            ia.plot(q[:, 0], q[:, 1], color="#ffe600", lw=0.5, alpha=0.8)
        p = np.array(o["position"][:2])
        q = project(np.stack([p, p + rot(o["yaw"]) @ np.array([0.05, 0.0])]), w)
        ia.annotate("", xy=q[1], xytext=q[0], arrowprops=dict(arrowstyle="-|>", color="#ffe600", lw=1.2))
        ia.text(q[0, 0] + 4, q[0, 1] + 10, HOME[o["name"]], color="#ffe600", fontsize=6, weight="bold")
    bc = project(np.vstack([bin_corners(lay["bin"]), bin_corners(lay["bin"])[:1]]), w)
    ia.plot(bc[:, 0], bc[:, 1], color="#00e5ff", lw=0.8)
    ia.set_xlim(0, w)
    ia.set_ylim(im.shape[0], 0)


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
    t = tx.table(cellText=[[r[0], f"{r[1]:.1f}", f"{r[2]:+.1f}", f"{r[3]:.0f}°", f"{r[4]} o'clock", r[5]]
                           for r in rows],
                 colLabels=["item", "toward\nyou cm", "right cm\n(+R / −L)", "arrow\n° cw", "arrow\nclock", "role"],
                 loc="upper left", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.5)
    for (r_, c_), cell in t.get_celld().items():
        if r_ == 0:
            cell.set_height(cell.get_height() * 1.7)
    tx.text(0, -0.02, "You stand at the front of the table FACING THE ROBOT (see page 1). Distances are from the\n"
                      "shoulder-pan axis to the item's dot. Arrow = the object's +x end as drawn (yellow arrow on\n"
                      "the sim image shows which physical end). Arrow direction as a clock seen from above:\n"
                      "12 = toward the robot, 3 = to your right, 6 = toward you, 9 = to your left; ° = clockwise\n"
                      "from 12. Bin: arrow = its long side (either end).",
            fontsize=6.5, va="top", transform=tx.transAxes)
    img = OUT / "sim_frames" / f"trial{row['trial_id']:02d}_overhead.jpg"
    if img.exists():
        ia = fig.add_axes([0.60, 0.04, 0.38, 0.34])
        draw_sim_overlay(ia, row, lay, img)
        ia.set_title("sim overhead camera at robot start (you stand on this side); yellow = sheet outlines/arrows "
                     "projected", fontsize=6.5)
        ia.axis("off")
    return fig


def guide_page():
    """Page 1: how to read the sheets, from the point of view of a person facing the robot."""
    from matplotlib.patches import Circle, FancyArrowPatch
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("How to read the layout sheets — stand at the FRONT of the table, FACING THE ROBOT", fontsize=13,
                 x=0.04, ha="left", y=0.97)
    ax = fig.add_axes([0.03, 0.05, 0.55, 0.86])
    ax.set_aspect("equal")
    ax.axis("off")
    # table, robot, person (sheet coordinates: x = to your right, y = toward the robot)
    ax.add_patch(Rectangle((-30, -52), 60, 62, fc="#2b2b2b", ec="none", alpha=0.12))
    ax.add_patch(Polygon(to_sheet(np.array(SO101_BOX)), closed=True, fc="#ffe0cc", ec="#ff7f0e", lw=1))
    ax.text(0, 12, "ROBOT (base clamped here)", ha="center", fontsize=10, color="#ff7f0e", weight="bold")
    ax.add_patch(Circle((0, -60), 2.2, fc="#555555"))
    ax.add_patch(Rectangle((-4, -68), 8, 5, fc="#555555"))
    ax.text(0, -72, "YOU (facing the robot)", ha="center", fontsize=10, weight="bold")
    ax.plot(0, 0, marker="+", color="k", ms=18, mew=2)
    ax.text(-1.5, 2.5, "ORIGIN = shoulder-pan axis\n(centre of the base turntable)", fontsize=8.5, ha="right")
    ax.text(7, 7, "robot (arm at rest)", fontsize=7.5, color="#ff7f0e")
    ax.plot([0, 0], [0, -48], color="#888888", ls="--", lw=1)
    ax.text(-1, -49.5, "robot's centre line = the direction the arm reaches\nstraight out when shoulder_pan = 0",
            fontsize=7.5, color="#555555", ha="right", va="top")
    a1 = FancyArrowPatch((0, 0), (0, -22), arrowstyle="-|>", mutation_scale=18, color="#d62728", lw=2.5)
    a2 = FancyArrowPatch((0, 0), (22, 0), arrowstyle="-|>", mutation_scale=18, color="#1f77b4", lw=2.5)
    ax.add_patch(a1)
    ax.add_patch(a2)
    ax.text(-1.2, -19, "TOWARD YOU\n(+ cm)", color="#d62728", fontsize=9, ha="right", weight="bold")
    ax.text(23.5, -1, "RIGHT (+ cm)\nyour right\n= robot's left", color="#1f77b4", fontsize=9, ha="left",
            va="top", weight="bold")
    ax.text(-23, -1.5, "LEFT (− cm)", color="#1f77b4", fontsize=9, ha="center", va="top")
    # worked example: toward you 20, right +10, arrow at 2 o'clock (60° cw)
    ex = np.array([10.0, -20.0])
    ax.plot([0, ex[0]], [ex[1], ex[1]], color="#1f77b4", lw=1, ls=":")
    ax.plot([ex[0], ex[0]], [0, ex[1]], color="#d62728", lw=1, ls=":")
    ang = math.radians(90 - 60)  # 60° clockwise from 12 o'clock
    d = np.array([math.cos(ang), math.sin(ang)])
    body = np.stack([ex - 7 * d, ex + 7 * d])
    ax.plot(body[:, 0], body[:, 1], color="#9467bd", lw=6, solid_capstyle="round", alpha=0.6)
    ax.add_patch(FancyArrowPatch(tuple(ex), tuple(ex + 7 * d), arrowstyle="-|>", mutation_scale=14,
                                 color="#9467bd", lw=1.8))
    ax.plot(*ex, "o", color="#9467bd", ms=6)
    ax.text(ex[0] + 2, ex[1] - 4.5, "example: toward you 20.0, right +10.0,\narrow 60° cw = 2 o'clock",
            fontsize=8.5, color="#9467bd")
    ax.set_xlim(-34, 34)
    ax.set_ylim(-76, 16)
    # clock inset
    cx = fig.add_axes([0.62, 0.50, 0.30, 0.38])
    cx.set_aspect("equal")
    cx.axis("off")
    cx.add_patch(Circle((0, 0), 1, fill=False, lw=1.5))
    for h in range(1, 13):
        t = math.radians(90 - 30 * h)
        cx.text(1.17 * math.cos(t), 1.17 * math.sin(t), str(h), ha="center", va="center", fontsize=11,
                weight="bold" if h % 3 == 0 else "normal")
    cx.text(0, 1.48, "12 = toward the ROBOT", ha="center", fontsize=8.5)
    cx.text(0, -1.5, "6 = toward YOU", ha="center", fontsize=8.5)
    cx.text(1.42, 0, "3 = your\nRIGHT", ha="left", va="center", fontsize=8.5)
    cx.text(-1.42, 0, "9 = your\nLEFT", ha="right", va="center", fontsize=8.5)
    t = math.radians(90 - 60)
    cx.add_patch(FancyArrowPatch((0, 0), (0.85 * math.cos(t), 0.85 * math.sin(t)), arrowstyle="-|>",
                                 mutation_scale=14, color="#9467bd", lw=2))
    cx.set_xlim(-2.1, 2.1)
    cx.set_ylim(-1.8, 1.8)
    cx.set_title("Arrow direction, looking DOWN at the table from where you stand", fontsize=9)
    fig.text(0.60, 0.43,
             "Each layout sheet gives, for every item:\n"
             "  • toward you [cm]: from the pan axis, along the robot's centre line\n"
             "  • right [cm]: across, + to your right, − to your left\n"
             "  • the point is the dot on the drawing (≈ middle of the object)\n"
             "  • arrow: which way the drawn arrow on the object points, as a clock\n"
             "    (and degrees clockwise from 12). The yellow arrow on the sim image\n"
             "    shows which physical end of the object the arrow points to.\n\n"
             "The drawings are top views in this same orientation: robot at the top,\n"
             "you at the bottom, your right on the right of the page — exactly what you\n"
             "see looking down at the table. The sim image is the overhead camera,\n"
             "which sits on your side of the table, so it shows the same left/right.",
             fontsize=8.5, va="top")
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
    fig.text(0.04, 0.30,
             "Reference frame on every sheet (see page 1): origin = shoulder-pan axis of the SO-101 (centre of the base "
             "turntable),\n"
             f"sim world ({PAN_WORLD[0]:.4f}, {PAN_WORLD[1]:.4f}) m = robot root (0.0521, 0.1806) + Rz(6.195°)·(0.0208, "
             "−0.0231).\n"
             "'toward you' = robot root −y (the base's centre line, away from the robot); 'right' = root +x = robot's "
             "left = YOUR right.\n"
             "Table top = z 0. The bin stays at the bench canonical pose in all 30 layouts. Spatial layouts hold 4 "
             "objects (bench convention), bin layouts 1.\n"
             "Object length is along the drawn arrow (+x); for the remote the arrow is across its short side.",
             fontsize=8, va="top")
    return fig


def main():
    rows = [json.loads(l) for l in TASKS.read_text().splitlines()]
    lays = [json.loads(l) for l in LAYOUTS.read_text().splitlines()]
    dims = json.loads(DIMS.read_text())
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "home_objects_layouts.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = guide_page()
        fig.savefig(OUT / "how_to_read.png", dpi=110)
        pdf.savefig(fig)
        plt.close(fig)
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
