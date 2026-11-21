#!/usr/bin/env python3
"""Side-by-side T004 videos for FLUX 3 Action (flux-action venv: numpy, PIL, av).

For each replayed query s of one episode, 31 frames at 30 fps (j = 1..31, the executed part of the chunk):
  row 1  FLUX predicted canvas, frame j of predicted/ep<e>_f<s>.mp4 (scene | wrist, 256x512)
  row 2  observed canvas after s+j env steps (same resize)
  row 3  ideal-replay canvas after j ticks of perfect tracking from the state at s

  t004_flux_videos.py RUN_DIR OUT.mp4 [--episode 0] [--tag gain]
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:  # no DejaVu on this machine
        return ImageFont.load_default()


def _img(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def _video(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--horizon", type=int, default=31)
    ap.add_argument("--max_queries", type=int, default=12)
    args = ap.parse_args()
    tag = args.tag or args.run.name
    ep = args.episode
    fz = np.load(args.run / f"frames_ep{ep:03d}.npz", allow_pickle=True)
    observed = fz["observed"]
    ideal = {int(q): fz["ideal"][i] for i, q in enumerate(fz["ideal_queries"])}
    font = _font(14)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    W, H, LAB = 512, 256, 18
    with av.open(str(args.out), "w") as c:
        st = c.add_stream("libx264", rate=30)
        st.width, st.height, st.pix_fmt = W, 3 * (H + LAB), "yuv420p"
        st.options = {"crf": "20"}
        for s in sorted(ideal)[: args.max_queries]:
            vp = args.run / "predicted" / f"ep{ep:03d}_f{s:05d}.mp4"
            if not vp.exists():
                continue
            pred = _video(vp)
            for j in range(1, args.horizon + 1):
                if j >= len(pred) or s + j > len(observed) or j > len(ideal[s]):
                    break
                canvas = Image.new("RGB", (W, 3 * (H + LAB)), (20, 20, 20))
                d = ImageDraw.Draw(canvas)
                rows = [(Image.fromarray(pred[j]), f"FLUX predicted  s={s} j={j}"),
                        (_img(observed[s + j - 1]), f"observed (gap: {tag})"),
                        (_img(ideal[s][j - 1]), "ideal replay (perfect tracking)")]
                for r, (im, lab) in enumerate(rows):
                    y = r * (H + LAB)
                    d.text((4, y + 1), lab, font=font, fill="white")
                    canvas.paste(im, (0, y + LAB))
                for pkt in st.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                    c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
