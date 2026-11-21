#!/usr/bin/env python3
"""FLUX 3 Action predicted-vs-observed frame distances for T004 (run in the flux-action venv; needs av, lpips,
transformers).

Per query at loop step s, FLUX's decoded prediction ``predicted/ep<e>_f<s>.mp4`` (256x512 scene|wrist canvas)
is compared at horizon j with
  observed  the rollout canvas after s + j + ``--align`` env steps (``frames_ep<e>.npz["observed"][n-1]`` is
            the canvas after n steps; the canvas uses FLUX's own resize: each camera bilinear+antialias to
            256x256, overhead left, wrist right),
  ideal     the kinematic-replay canvas after j + ``--align`` replay ticks from the same state (replay episodes only).
Also ``exec`` = observed vs ideal: the control gap itself in image space (no model involved).

Distances, each on the scene half and the wrist half separately (no other mask):
  l1     mean |a - b| on [0, 1] RGB
  lpips  LPIPS (AlexNet, v0.1) on [-1, 1]
  dino   1 - cosine similarity of DINOv2-small patch tokens (224x224 input, ImageNet norm), averaged over the
         256 patches, i.e. a spatially aligned feature distance.
Stored frames are JPEG q95 (observed/ideal) and H.264 CRF 23 (prediction, bench default); both affect the
floors, not the comparisons across conditions.

``--lag_sweep`` prints e_pred l1 for align in -2..2 on the given runs to pin the frame alignment.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HORIZONS = (1, 4, 8, 16, 24, 31, 40)


def _jpeg_decode(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def _video(path: Path) -> np.ndarray:
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


class Metrics:
    def __init__(self, device: str):
        import lpips
        from transformers import AutoModel

        self.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.dino = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
        """a, b: (N, 256, 512, 3) uint8 -> dict of (N,) arrays per metric x half."""
        ta = torch.from_numpy(a).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        tb = torch.from_numpy(b).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        out = {}
        for half, sl in (("scene", slice(0, 256)), ("wrist", slice(256, 512))):
            x, y = ta[..., sl], tb[..., sl]
            out[f"l1_{half}"] = (x - y).abs().mean(dim=(1, 2, 3)).cpu().numpy()
            out[f"lpips_{half}"] = self.lpips(x * 2 - 1, y * 2 - 1).flatten().cpu().numpy()
            fx = self._dino(x)
            fy = self._dino(y)
            out[f"dino_{half}"] = (1 - F.cosine_similarity(fx, fy, dim=-1).mean(dim=1)).cpu().numpy()
        return out

    def _dino(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        h = self.dino(pixel_values=(x - self.mean) / self.std).last_hidden_state
        return h[:, 1:]


def run_dir(run: Path, metrics: Metrics, align: int, horizons=HORIZONS, max_queries: int | None = None):
    rows = []
    for fe in sorted(run.glob("frames_ep*.npz")):
        ep = int(fe.stem.removeprefix("frames_ep"))
        fz = np.load(fe, allow_pickle=True)
        observed = fz["observed"]
        ideal_q = [int(q) for q in fz["ideal_queries"]]
        ideal = {q: fz["ideal"][i] for i, q in enumerate(ideal_q)}
        d = np.load(run / f"ep{ep:03d}.npz", allow_pickle=True)
        queries = [int(q) for q in d["queries"]][:max_queries]
        for s in queries:
            vp = run / "predicted" / f"ep{ep:03d}_f{s:05d}.mp4"
            if not vp.exists():
                continue
            pred = _video(vp)
            js = [j for j in horizons if 0 <= j < len(pred) and 1 <= s + j + align <= len(observed)]
            if not js:
                continue
            P = np.stack([pred[j] for j in js])
            O = np.stack([_jpeg_decode(observed[s + j + align - 1]) for j in js])
            res = {"pred_obs": metrics(P, O)}
            if s in ideal:
                jj = [k for k, j in enumerate(js) if 1 <= j + align <= len(ideal[s])]
                if jj:
                    I = np.stack([_jpeg_decode(ideal[s][js[k] + align - 1]) for k in jj])
                    res["pred_ideal"] = metrics(P[jj], I)
                    res["obs_ideal"] = metrics(O[jj], I)
                    res["_jj"] = jj
            for k, j in enumerate(js):
                row = {"episode": ep, "frame": s, "j": j}
                for name, m in res.items():
                    if name == "_jj":
                        continue
                    if name != "pred_obs":
                        if k not in res["_jj"]:
                            continue
                        idx = res["_jj"].index(k)
                    else:
                        idx = k
                    for key, v in m.items():
                        row[f"{name}.{key}"] = float(v[idx])
                rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    keys = sorted({k for r in rows for k in r if "." in k})
    for j in sorted({r["j"] for r in rows}):
        rj = [r for r in rows if r["j"] == j]
        for k in keys:
            v = [r[k] for r in rj if k in r]
            if v:
                out[f"{k}@{j}"] = float(np.mean(v))
                out[f"n.{k.split('.')[0]}@{j}"] = len(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--align", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lag_sweep", action="store_true")
    args = ap.parse_args()
    metrics = Metrics(args.device)
    if args.lag_sweep:
        for run in args.runs:
            for a in (-2, -1, 0, 1, 2):
                rows = run_dir(run, metrics, a, horizons=(1, 4, 8, 16, 31), max_queries=6)
                s = summarize(rows)
                print(run.name, "align", a, {k: round(v, 4) for k, v in s.items() if k.startswith(("pred_obs.l1_", "pred_obs.dino_"))})
        return
    for run in args.runs:
        rows = run_dir(run, metrics, args.align)
        with (run / "flux_metrics.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        s = summarize(rows)
        s["align"] = args.align
        (run / "flux_metrics_summary.json").write_text(json.dumps(s, indent=2))
        print(run, len(rows), "rows")


if __name__ == "__main__":
    main()
