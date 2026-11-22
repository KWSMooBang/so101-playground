#!/usr/bin/env python3
"""Align the real overhead camera with the SO-101 Bench sim overhead camera.

Draws where the bench's canonical overhead camera sees the robot base and the object area on the table,
using the bench's own pinhole model (`so101_bench.utils.layout_overlay.OverheadProjector`), over a live
camera feed or a still image. Move the camera until the robot base sits in the circle and the table
markers line up.

    envs/lerobot/.venv/bin/python scripts/overhead_align.py                  # live, /dev/so101_cam_overhead
    envs/lerobot/.venv/bin/python scripts/overhead_align.py --image a.jpg --out a_overlay.jpg

Canonical values (bench `benchmark.py` DomainRandomizationSpec defaults, `assets/so101.py`), world frame
with the table top at z = 0:
  overhead camera at (0.0, -0.34, 0.45722) m, rpy (51.361, -0.169, 0.603) deg, horizontal FOV 74 deg
  robot base at (0.05209, 0.18061) m, yaw 6.195 deg; object area x -0.14..0.25, y -0.10..0.155 m
The live window: Save/Quit buttons (or s / q in the window, or s / q + Enter in the terminal). Save writes a
timestamped snapshot to outputs/camera_check/; overhead_align_latest*.jpg there is rewritten every second.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "third_party/so101_bench/source/so101_bench/so101_bench/utils"
CAMERA_POS = (0.0, -0.34, 0.45722)
CAMERA_RPY = (51.361, -0.169, 0.603)
CAMERA_HFOV = 74.0
BASE_XY, BASE_YAW_DEG = (0.05209, 0.18061), 6.195
OBJECT_AREA = (-0.14, 0.25, -0.10, 0.155)  # x0, x1, y0, y1
FALLBACK_CAMERA = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.4.3:1.0-video-index0"


def _projector():
    spec = importlib.util.spec_from_file_location("_layout_overlay", UTILS / "layout_overlay.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.OverheadProjector(CAMERA_POS, CAMERA_RPY, CAMERA_HFOV)


def draw_targets(image: np.ndarray, projector) -> np.ndarray:
    out = image.copy()
    scale = np.array([out.shape[1] / projector.width, out.shape[0] / projector.height])

    def px(points):
        return np.round(projector.project(np.asarray(points, dtype=float)) * scale).astype(np.int32)

    # Table grid every 5 cm over the object area (thin), its outline (yellow).
    x0, x1, y0, y1 = OBJECT_AREA
    for x in np.arange(-0.15, 0.2501, 0.05):
        cv2.polylines(out, [px([[x, y0, 0], [x, y1, 0]])], False, (90, 90, 90), 1, cv2.LINE_AA)
    for y in np.arange(-0.10, 0.1501, 0.05):
        cv2.polylines(out, [px([[x0, y, 0], [x1, y, 0]])], False, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.polylines(out, [px([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])], True, (0, 220, 255), 2, cv2.LINE_AA)

    # Robot base: center cross, 5 cm radius circle on the table, and its forward direction (red).
    bx, by = BASE_XY
    yaw = math.radians(BASE_YAW_DEG)
    ring = [[bx + 0.05 * math.cos(t), by + 0.05 * math.sin(t), 0] for t in np.linspace(0, 2 * math.pi, 48)]
    cv2.polylines(out, [px(ring)], True, (0, 0, 255), 2, cv2.LINE_AA)
    c = tuple(px([[bx, by, 0]])[0])
    cv2.drawMarker(out, c, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    # The robot faces -y (toward the camera); draw 10 cm along its facing direction.
    fwd = px([[bx, by, 0], [bx + 0.10 * math.sin(yaw), by - 0.10 * math.cos(yaw), 0]])
    cv2.arrowedLine(out, tuple(fwd[0]), tuple(fwd[1]), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(out, "base", (c[0] + 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "object area", tuple(px([[x0, y0, 0]])[0] + [4, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 220, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, help="Still image instead of the live camera.")
    parser.add_argument("--out", type=Path, help="Where to write the overlay for --image.")
    parser.add_argument("--camera", default=None, help="Camera device (default /dev/so101_cam_overhead).")
    args = parser.parse_args()
    projector = _projector()

    if args.image:
        out = draw_targets(cv2.imread(str(args.image)), projector)
        target = args.out or args.image.with_name(args.image.stem + "_overlay.jpg")
        cv2.imwrite(str(target), out)
        print(f"wrote {target}")
        return

    device = args.camera or ("/dev/so101_cam_overhead" if Path("/dev/so101_cam_overhead").exists() else FALLBACK_CAMERA)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    snapshots = ROOT / "outputs" / "camera_check"
    snapshots.mkdir(parents=True, exist_ok=True)
    print(f"{device}: Save/Quit buttons in the window, or s / q in the window or here (then Enter)", flush=True)

    # The env's OpenCV is headless (no imshow), so the window is tkinter.
    import queue
    import threading
    import tkinter
    from PIL import Image, ImageTk

    save_keys, quit_keys = {"s", "S", "ㄴ"}, {"q", "Q", "ㅂ"}  # ㄴ/ㅂ: the same keys with a Korean IME on
    commands: queue.Queue[str] = queue.Queue()

    def read_terminal():
        for line in sys.stdin:
            commands.put(line.strip())

    threading.Thread(target=read_terminal, daemon=True).start()

    root = tkinter.Tk()
    root.title("overhead align")
    label = tkinter.Label(root)
    label.pack()
    buttons = tkinter.Frame(root)
    buttons.pack(fill="x")
    state = {"frame": None, "view": None}

    def save():
        if state["frame"] is None:
            return
        stamp = time.strftime("%H%M%S")
        cv2.imwrite(str(snapshots / f"overhead_align_{stamp}.jpg"), state["frame"])
        cv2.imwrite(str(snapshots / f"overhead_align_{stamp}_overlay.jpg"), state["view"])
        print(f"saved outputs/camera_check/overhead_align_{stamp}.jpg (+ _overlay)", flush=True)

    def on_key(event):
        if event.char in save_keys:
            save()
        elif event.char in quit_keys:
            root.destroy()

    def write_latest():
        # Always-current copy, so the view can be checked without pressing Save.
        if state["frame"] is not None:
            cv2.imwrite(str(snapshots / "overhead_align_latest.jpg"), state["frame"])
            cv2.imwrite(str(snapshots / "overhead_align_latest_overlay.jpg"), state["view"])
        root.after(1000, write_latest)

    def tick():
        while not commands.empty():
            command = commands.get()
            if command in save_keys:
                save()
            elif command in quit_keys:
                root.destroy()
                return
        ok, frame = cap.read()
        if ok:
            state["frame"], state["view"] = frame, draw_targets(frame, projector)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(state["view"], cv2.COLOR_BGR2RGB)))
            label.configure(image=photo)
            label.image = photo
        root.after(15, tick)

    tkinter.Button(buttons, text="Save snapshot (s)", command=save).pack(side="left", expand=True, fill="x")
    tkinter.Button(buttons, text="Quit (q)", command=root.destroy).pack(side="left", expand=True, fill="x")
    root.bind("<Key>", on_key)
    root.after(0, tick)
    root.after(1000, write_latest)
    root.mainloop()
    cap.release()


if __name__ == "__main__":
    main()
