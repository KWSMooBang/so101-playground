#!/usr/bin/env python3
"""LaWAM policy server with its latent future prediction exposed (T004, RESEARCH.md H2-P0).

A copy of so101_bench ``scripts/lawam_server.py`` (bench @ 470e09e; the submodule is not edited) with
the same ``POST /act`` contract, plus what the probe needs. Runs in the lerobot env (lerobot.policies.lawam);
uses the standard library HTTP server, so it needs no fastapi.

LaWAM (lerobot ``policies/lawam/vlas/lawam.py``): the LAM vision encoder (DINOv3, layer -2) gives
``h_t`` (256 patch tokens x 768) of the current overhead frame; the VLM predicts a latent action
``pred_action_emb``; the LAM decoder maps (h_t, pred_action_emb) to ``h_t1_pred``, trained with
``mse(h_t1_pred, h_t1_gt)`` where h_t1_gt is the same encoder on the overhead frame
``action_horizon - 1`` = 35 control ticks later (num_video_frames=2 -> indices [0, 35]).

Extra behaviour:
  * Every /act with ``trace`` = {episode, frame} keeps h_t, h_t1_pred, pred_action_emb and the
    preprocessed overhead frame in memory under (episode, frame).
  * ``POST /lam_score`` {"episode", "frame", "kind", "images_png_base64": [overhead, wrist],
    "camera_names"} encodes a later frame (observed rollout frame, or kinematic-replay frame) with the
    same preprocessor and LAM encoder and returns / logs:
        e_pred_lat   mse(h_t1_pred, h_obs)                 (the training loss, fp32)
        cos_pred_lat 1 - mean token cosine(h_t1_pred, h_obs)
        e_static_lat mse(h_t, h_obs)                       (how much the scene moved; "no-motion" baseline)
        e_act_lat    mse(pred_action_emb, LAM teacher latent action of (o_t, o_obs))
        cos_act_lat  1 - cosine(pred_action_emb, teacher latent action)
  * Flow noise is seeded per (episode, frame) so conditions see identical sampling noise for
    identical inputs.
  * Per-request timing and peak GPU memory go to ``<log_dir>/server_requests.jsonl``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lerobot.policies import make_pre_post_processors
from lerobot.policies.lawam.modeling_lawam import LaWAMPolicy

IMAGE_PREFIX = "observation.images."

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
log = logging.getLogger("lawam.probe.server")


def _decode_png(encoded: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


def _seed(episode: Any, frame: Any) -> int:
    return int.from_bytes(hashlib.sha256(f"{episode}:{frame}".encode()).digest()[:4], "little")


class Probe:
    def __init__(self, *, checkpoint: str, device: str, camera_map: dict[str, str], log_dir: Path, keep: int):
        log.info("Loading LaWAM checkpoint %s", checkpoint)
        self.policy = LaWAMPolicy.from_pretrained(checkpoint)
        self.policy.to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )
        self.camera_map = camera_map
        features = set(self.policy.config.image_features)
        targets = {IMAGE_PREFIX + v for v in camera_map.values()}
        if targets != features:
            raise SystemExit(f"--camera_map targets {sorted(targets)} != checkpoint image features {sorted(features)}")
        self.camera_keys = list(camera_map)
        self.checkpoint = checkpoint
        self.device = device
        self.backend = self.policy.model.policy_backend
        self.lam = self.backend.lam
        self.embodiment_id = None
        self.memory: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self.keep = keep
        self.lock = threading.Lock()
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self.req_log = (log_dir / "server_requests.jsonl").open("a")
        self.score_log = (log_dir / "lam_scores.jsonl").open("a")
        # Capture the shared encoding state that predict_action computes and drops.
        self._last_shared = None
        original = self.backend._run_shared_encoding_infer

        def capture(*args, **kwargs):
            out = original(*args, **kwargs)
            self._last_shared = out
            return out

        self.backend._run_shared_encoding_infer = capture
        cfg = self.policy.config
        log.info("Cameras %s, action_horizon %d, n_action_steps %d, lam %s, video frames %s, future_prediction %s",
                 camera_map, cfg.action_horizon, cfg.n_action_steps, cfg.lam_image_feature,
                 cfg.image_observation_delta_indices, cfg.future_prediction)

    def _batch(self, images: dict[str, np.ndarray], instruction: str) -> dict[str, Any]:
        missing = sorted(set(self.camera_keys) - set(images))
        if missing:
            raise ValueError(f"camera mismatch: server expects {self.camera_keys}; missing {missing}")
        batch: dict[str, Any] = {"task": [instruction]}
        for name in self.camera_keys:
            tensor = torch.from_numpy(np.ascontiguousarray(images[name])).permute(2, 0, 1).float().div_(255.0)
            batch[IMAGE_PREFIX + self.camera_map[name]] = tensor.unsqueeze(0)
        return self.preprocessor(batch)

    @torch.inference_mode()
    def predict(self, *, images: dict[str, np.ndarray], instruction: str, trace: dict | None) -> np.ndarray:
        batch = self._batch(images, instruction)
        if trace is not None:
            torch.manual_seed(_seed(trace["episode"], trace["frame"]))
        chunk = self.policy.predict_action_chunk(batch)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        actions = torch.stack([self.postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1)[0]
        if trace is not None and self._last_shared is not None:
            s = self._last_shared
            key = (int(trace["episode"]), int(trace["frame"]))
            self.memory.pop(key, None)  # re-insert at the end: eviction is by insertion order
            self.memory[key] = {
                "h_t": s.h_t.detach().float().clone(),
                "h_t1_pred": s.h_t1_pred.detach().float().clone(),
                "pred_action_emb": s.pred_action_emb.detach().float().clone(),
                "primary_image": batch["primary_image"].detach().clone(),
                "embodiment_id": batch["embodiment_id"].detach().clone(),
            }
            while len(self.memory) > self.keep:  # oldest requests first (runs reuse episode ids)
                del self.memory[next(iter(self.memory))]
        return actions.float().cpu().numpy()

    @torch.inference_mode()
    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = (int(payload["episode"]), int(payload["frame"]))
        if key not in self.memory:
            raise ValueError(f"no stored prediction for episode/frame {key}")
        mem = self.memory[key]
        names = payload.get("camera_names") or self.camera_keys
        images = {n: _decode_png(d) for n, d in zip(names, payload["images_png_base64"])}
        batch = self._batch(images, "score")
        obs_img = batch["primary_image"]
        video = torch.stack([mem["primary_image"], obs_img], dim=1)  # (1, 2, 3, 256, 256), imagenet-normalized
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = self.lam.extract_vision_features(video)
            teacher_in = self.backend._build_lam_teacher_inputs_for_distill(video)
            teacher = self.lam.get_latent_action(videos=teacher_in, states=None, dec_videos=teacher_in,
                                                 predict_future_frame=False,
                                                 embodiment_ids=mem["embodiment_id"])["quantized"]
        h_obs = feats[:, -1].float()
        h_t0 = feats[:, 0].float()
        teacher = teacher.float()
        pred, h_t = mem["h_t1_pred"], mem["h_t"]
        pae = mem["pred_action_emb"]
        out = {
            "episode": key[0], "frame": key[1], "kind": payload.get("kind", "observed"),
            "obs_frame": payload.get("obs_frame"),
            "e_pred_lat": float(F.mse_loss(pred, h_obs)),
            "cos_pred_lat": float(1 - F.cosine_similarity(pred, h_obs, dim=-1).mean()),
            "e_static_lat": float(F.mse_loss(h_t, h_obs)),
            "e_pred_vs_t_lat": float(F.mse_loss(pred, h_t)),
            "h_t_reencode_mse": float(F.mse_loss(h_t0, h_t)),
            "e_act_lat": float(F.mse_loss(pae, teacher)) if pae.shape == teacher.shape else None,
            "cos_act_lat": float(1 - F.cosine_similarity(pae.flatten(1), teacher.flatten(1), dim=-1).mean())
            if pae.shape == teacher.shape else None,
        }
        out.update({k: payload[k] for k in ("condition",) if k in payload})
        self.score_log.write(json.dumps(out) + "\n")
        self.score_log.flush()
        return out


def make_handler(probe: Probe):
    cfg = probe.policy.config

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"status": "ok", "repo_id": probe.checkpoint, "norm_tag": "lawam",
                             "num_images": len(probe.camera_keys), "camera_keys": probe.camera_keys,
                             "history_ticks": 1, "chunk_size": cfg.action_horizon,
                             "n_action_steps": cfg.n_action_steps, "device": probe.device,
                             "lam_horizon": cfg.image_observation_delta_indices[-1]})

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            except Exception as exc:  # noqa: BLE001
                return self._send(400, {"error": f"invalid request: {exc}"})
            start = time.perf_counter()
            try:
                with probe.lock:
                    torch.cuda.reset_peak_memory_stats()
                    if self.path.startswith("/lam_score"):
                        result = probe.score(payload)
                    else:
                        names = payload.get("camera_names") or probe.camera_keys
                        images = {n: _decode_png(d) for n, d in zip(names, payload["images_png_base64"])}
                        trace = payload.get("trace")
                        actions = probe.predict(images=images, instruction=str(payload["instruction"]), trace=trace)
                        result = {"actions": actions.tolist()}
                    dt = (time.perf_counter() - start) * 1000.0
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    probe.req_log.write(json.dumps({"path": self.path, "dt_ms": dt, "peak_alloc_gib": peak,
                                                    "t": time.time()}) + "\n")
                    probe.req_log.flush()
            except ValueError as exc:
                return self._send(400, {"error": f"invalid request: {exc}"})
            except Exception as exc:  # noqa: BLE001
                log.exception("LaWAM request failed")
                return self._send(500, {"error": f"inference failed: {exc}"})
            result["dt_ms"] = dt
            self._send(200, result)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="Checkpoint pretrained_model directory.")
    parser.add_argument("--camera_map", default="overhead=image,wrist=image2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--keep", type=int, default=4, help="Most recent queries kept for scoring.")
    args = parser.parse_args()
    camera_map = dict(pair.split("=", 1) for pair in args.camera_map.split(","))
    probe = Probe(checkpoint=args.repo_id, device=args.device, camera_map=camera_map,
                  log_dir=Path(args.log_dir), keep=args.keep)
    blank = {name: np.zeros((480, 640, 3), dtype=np.uint8) for name in probe.camera_keys}
    for i in range(2):
        start = time.perf_counter()
        out = probe.predict(images=blank, instruction="warmup", trace={"episode": -1, "frame": i})
        log.info("Warmup %d: chunk %s in %.1f ms, peak %.2f GiB", i, out.shape, (time.perf_counter() - start) * 1e3,
                 torch.cuda.max_memory_allocated() / 2**30)
    probe.memory.clear()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(probe))
    log.info("Serving on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
