#!/usr/bin/env python3
"""Side-by-side T004 videos for LaWAM (run in the flux-action venv: numpy, PIL, av).

Input: a run made with ``wam_gap_eval.py --model lawam --save_frames --replay_episodes 1`` (frames_ep*.npz,
lamfeat_ep*.npz, ep*.npz). For each replayed query s, 35 frames at 30 fps:

  top    observed canvas (overhead | wrist) after s+k env steps  |  ideal-replay canvas after k ticks
  bottom PCA-RGB of LaWAM's predicted LAM features h_t1_pred | observed h at s+35 | ideal-replay h at s+35,
         plus the scores. LaWAM predicts no pixels; its "predicted frame" is h_t1_pred, the 16x16 grid of
         768-d DINOv3 (LAM encoder, layer -2) patch tokens of the overhead view 35 ticks ahead. The three
         maps share one 3-component PCA basis per episode (fit on all its h_t, h_t1_pred, h_obs), so equal
         colours mean equal features.

  t004_videos.py RUN_DIR OUT.mp4 [--episode 0]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _img(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def _pca_maps(feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = sorted(feats)
    x = np.concatenate([feats[k].astype(np.float32) for k in keys])
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    proj = (x - mu) @ vt[:3].T
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    out, i = {}, 0
    for k in keys:
        p = np.clip((proj[i:i + 256] - lo) / (hi - lo + 1e-6), 0, 1)
        i += 256
        grid = (p.reshape(16, 16, 3) * 255).astype(np.uint8)
        out[k] = np.asarray(Image.fromarray(grid).resize((256, 256), Image.NEAREST))
    return out


def exec_summary(root: Path) -> None:
    """Per condition: the control gap itself in LAM space (observed vs ideal-replay frame at s+35), next to
    e_pred (pred vs observed) and e_ideal (pred vs ideal), all on the same queries."""
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = []
        for f in sorted(d.glob("lamfeat_ep*.npz")):
            z = dict(np.load(f))
            for k in z:
                if not k.startswith("h_observed_"):
                    continue
                s = k.removeprefix("h_observed_")
                if f"h_ideal_{s}" not in z or f"h_t1_pred_{s}" not in z:
                    continue
                o, i, p = (z[n].astype(np.float32) for n in (k, f"h_ideal_{s}", f"h_t1_pred_{s}"))
                rows.append((float(((o - i) ** 2).mean()), float(((p - o) ** 2).mean()), float(((p - i) ** 2).mean())))
        if rows:
            a = np.array(rows)
            out[d.name] = {"n_queries": len(rows), "e_exec": float(a[:, 0].mean()), "e_pred": float(a[:, 1].mean()),
                           "e_ideal": float(a[:, 2].mean())}
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, nargs="?")
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--exec_summary", type=Path, default=None,
                    help="Instead: print per-condition e_exec = mse(LAM(observed s+35), LAM(ideal s+35)) and "
                         "e_pred/e_ideal from lamfeat_ep*.npz under DIR/<cond>/.")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=35)
    ap.add_argument("--scores", type=Path, default=None, help="lam_scores.jsonl (default: <run>/../lam_scores.jsonl)")
    ap.add_argument("--tag", default=None, help="condition tag in lam_scores.jsonl (default: run dir name)")
    args = ap.parse_args()
    if args.exec_summary is not None:
        return exec_summary(args.exec_summary)
    ep = args.episode
    fz = np.load(args.run / f"frames_ep{ep:03d}.npz", allow_pickle=True)
    feats = dict(np.load(args.run / f"lamfeat_ep{ep:03d}.npz"))
    maps = _pca_maps(feats)
    observed = fz["observed"]
    ideal = {int(q): fz["ideal"][i] for i, q in enumerate(fz["ideal_queries"])}
    tag = args.tag or args.run.name
    scores = {}
    for line in (args.scores or args.run.parent / "lam_scores.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("condition") == tag and r["episode"] == ep:
            scores[(r["kind"], r["frame"])] = r
    font = ImageFont.truetype(FONT, 15)
    small = ImageFont.truetype(FONT, 13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(args.out), "w") as c:
        st = c.add_stream("libx264", rate=30)
        st.width, st.height, st.pix_fmt = 1024, 548, "yuv420p"
        st.options = {"crf": "20"}
        for s in sorted(ideal):
            keys = (f"h_t1_pred_{s}", f"h_observed_{s}", f"h_ideal_{s}")
            if not all(k in maps for k in keys):
                continue
            so, si = scores.get(("observed", s), {}), scores.get(("ideal", s), {})
            for k in range(1, args.horizon + 1):
                if s + k > len(observed) or k > len(ideal[s]):
                    break
                canvas = Image.new("RGB", (1024, 548), (20, 20, 20))
                canvas.paste(Image.fromarray(_img(observed[s + k - 1])), (0, 18))
                canvas.paste(Image.fromarray(_img(ideal[s][k - 1])), (512, 18))
                for i, key in enumerate(keys):
                    canvas.paste(Image.fromarray(maps[key]), (i * 256, 292))
                d = ImageDraw.Draw(canvas)
                d.text((4, 1), f"observed (gap: {tag})  s={s} +{k}", font=font, fill="white")
                d.text((516, 1), f"ideal replay (perfect tracking)  +{k}", font=font, fill="white")
                for i, lab in enumerate(("LaWAM h_t1_pred (s+35)", "LAM(observed s+35)", "LAM(ideal s+35)")):
                    d.text((i * 256 + 4, 274), lab, font=small, fill="white")
                lines = [f"query s={s}", "",
                         f"e_pred  {so.get('e_pred_lat', float('nan')):.4f}",
                         f"e_ideal {si.get('e_pred_lat', float('nan')):.4f}",
                         f"e_static {so.get('e_static_lat', float('nan')):.4f}", "",
                         "mse vs h_t1_pred,", "LAM DINOv3 tokens", "(LaWAM train loss)"]
                d.multiline_text((776, 300), "\n".join(lines), font=font, fill="white", spacing=4)
                for pkt in st.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                    c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
