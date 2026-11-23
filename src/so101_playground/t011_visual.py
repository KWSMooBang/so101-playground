#!/usr/bin/env python3
"""T011 scope 6: predicted | observed side by side from `t011_real_epred.py decode` output (<out>/decoded/<dir>.npz).

pred  = VAE decode of [L0, L0 + d1] (the server's prediction for ticks 4-16 after the query),
recon = VAE decode of [L0, L1 observed] (what the VAE can represent of the real clip),
real  = the recorded frames at ticks 0, 4, 8, 12, 16 (mp4, same canvas).
Writes <out>/fig_decoded.png (tick-16 frames for a few queries per episode) and <out>/clips/<dir>.mp4
(per query: query frame, then ticks 4-16; columns real | predicted | VAE recon), using numpy, PIL and ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LABEL = {"20261123_232455": "pen, named (success)", "20261123_232546": "pen, named (failure: toothbrush placed)",
         "20261123_233440": "razor, each (success)", "20261123_234054": "remote, each (failure: left standing)"}


def text(im: Image.Image, xy, s: str) -> None:
    d = ImageDraw.Draw(im)
    x, y = xy
    d.rectangle((x, y, x + 7 * len(s) + 6, y + 14), fill="black")
    d.text((x + 3, y + 1), s, fill="white")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/outputs/so101-playground/runs/T011")
    p.add_argument("--per_ep", type=int, default=2)
    a = p.parse_args()
    out = Path(a.out)
    Q = {}
    for line in (out / "queries.jsonl").open():
        r = json.loads(line)
        Q[(r["dir"], r["q"])] = r
    files = sorted((out / "decoded").glob("*.npz"))
    W, H = 256, 384
    rows = []
    (out / "clips").mkdir(exist_ok=True)
    for f in files:
        name = f.stem
        z = dict(np.load(f))  # load once: npz members decompress on every access
        qs = z["q"].tolist()
        # queries for the figure: the highest-motion query in the first and in the second half of the episode
        half = len(qs) // 2
        pick = []
        for part in (qs[:half], qs[half:]):
            part = [q for q in part if "e_pred@1" in Q[(name, q)]]
            if part:
                pick.append(max(part, key=lambda q: Q[(name, q)]["motion16_deg"]))
        rows.append((name, z, pick[:a.per_ep]))
        # clip
        frames = []
        for i, q in enumerate(qs):
            r = Q[(name, q)]
            for t in range(5):
                im = Image.new("RGB", (3 * W, H + 16), "black")
                for c, key in enumerate(("real", "pred", "recon")):
                    src = z[key][i][t] if not (key != "real" and t == 0) else z["real"][i][0]
                    im.paste(Image.fromarray(src), (c * W, 16))
                text(im, (0, 0), f"q{q} tick +{4 * t}  e_pred@1 {r.get('e_pred@1', float('nan')):.3f}  | real | predicted | VAE recon")
                frames.append(np.asarray(im))
        h, w = frames[0].shape[:2]
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "4",
                                 "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / "clips" / f"{name}.mp4")],
                                stdin=subprocess.PIPE)
        proc.stdin.write(np.stack(frames).tobytes())
        proc.stdin.close()
        proc.wait()
    # figure: rows = episode x picked query; columns = query frame | real +16 | predicted +16 | recon +16
    nrow = sum(len(p) for _, _, p in rows)
    fig = Image.new("RGB", (4 * W + 200, nrow * (H + 4)), "white")
    y = 0
    for name, z, pick in rows:
        qs = z["q"].tolist()
        for q in pick:
            i = qs.index(q)
            r = Q[(name, q)]
            for c, img in enumerate((z["real"][i][0], z["real"][i][4], z["pred"][i][4], z["recon"][i][4])):
                fig.paste(Image.fromarray(img), (200 + c * W, y))
            d = ImageDraw.Draw(fig)
            d.multiline_text((4, y + 8), f"{name[:15]}\n{LABEL.get(name[:15], '')}\nquery {q}\n"
                             f"motion {r['motion16_deg']:.0f} deg\ne_track16 {r['e_track16_deg']:.2f} deg\n"
                             f"e_pred@1 {r['e_pred@1']:.3f}\n (ovh {r['e_pred@1_overhead']:.3f},\n  wrist {r['e_pred@1_wrist']:.3f})\n"
                             f"e_static@1 {r['e_static@1']:.3f}", fill="black", spacing=4)
            y += H + 4
    hdr = Image.new("RGB", (fig.width, 22), "white")
    for c, s in enumerate(("query frame (tick 0)", "real tick +16", "predicted tick +16 (L0+d1)", "VAE recon of real +16")):
        ImageDraw.Draw(hdr).text((200 + c * W + 4, 4), s, fill="black")
    full = Image.new("RGB", (fig.width, fig.height + 22), "white")
    full.paste(hdr, (0, 0))
    full.paste(fig, (0, 22))
    full.save(out / "fig_decoded.png")
    print(out / "fig_decoded.png")


if __name__ == "__main__":
    main()
