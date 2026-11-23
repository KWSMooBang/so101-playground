#!/usr/bin/env python3
"""T011: first / last overhead (and wrist) frames of every home episode, for the success labels and the object
inventory (contact sheets under <out>/sheets/). Run in envs/internw0 (needs av, PIL)."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

HOME = Path("/outputs/so101-playground/runs/real_robot/home_runs")


def frames_at(path: Path, idx: list[int]) -> dict[int, np.ndarray]:
    n = 0
    with av.open(str(path)) as c:
        n = sum(1 for p in c.demux(video=0) if p.size)
    want = {i if i >= 0 else n + i for i in idx}
    out = {}
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = f.to_ndarray(format="rgb24")
    return {k: out[k if k >= 0 else n + k] for k in idx}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/outputs/so101-playground/runs/T011")
    p.add_argument("--per_sheet", type=int, default=6)
    p.add_argument("--scale", type=float, default=0.6)
    a = p.parse_args()
    out = Path(a.out) / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in HOME.iterdir() if (d / "trajectory.npz").exists())
    W, H = int(640 * a.scale), int(480 * a.scale)
    cols = [("overhead", 0), ("overhead", -60), ("overhead", -1), ("wrist", -1)]
    for s in range(0, len(dirs), a.per_sheet):
        chunk = dirs[s:s + a.per_sheet]
        sheet = Image.new("RGB", (W * len(cols), (H + 18) * len(chunk)), "white")
        dr = ImageDraw.Draw(sheet)
        for r, d in enumerate(chunk):
            fr = {cam: frames_at(d / f"{cam}.mp4", [0, -60, -1]) for cam in ("overhead", "wrist")}
            dr.text((4, r * (H + 18) + 2), f"[{s + r:02d}] {d.name}   cols: first | last-60 | last overhead | last wrist",
                    fill="black")
            for c, (cam, i) in enumerate(cols):
                sheet.paste(Image.fromarray(fr[cam][i]).resize((W, H)), (c * W, r * (H + 18) + 18))
        sheet.save(out / f"sheet_{s // a.per_sheet:02d}.jpg", quality=90)
        print(out / f"sheet_{s // a.per_sheet:02d}.jpg", flush=True)


if __name__ == "__main__":
    main()
