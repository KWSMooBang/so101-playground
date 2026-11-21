#!/usr/bin/env python3
"""FLUX 3 Action server for T004: so101_bench ``scripts/flux3_server.py`` (bench @ 470e09e, imported, not
edited) with two changes to fit a 24 GB card next to Isaac:

  * the Qwen3-VL text encoder stays on the CPU (flux-action ``set_text_encoder_offload``); it visits the GPU
    once per new instruction (8.9 GB saved);
  * every request's latency and peak allocated GPU memory are appended to ``<log_dir>/server_requests.jsonl``.

Everything else (``--save_predictions`` video capture/decoding, request contract, sampler preset) is the
bench server. Run in the flux-action venv:

  python src/so101_playground/flux3_probe_server.py --package <merged package> --vae <video_vae.safetensors> \
      --text_encoder <text_encoder dir> --save_predictions --log_dir <dir> --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(os.environ.get("SO101_BENCH_ROOT", "")) / "scripts"))
import flux3_server as bench  # noqa: E402
from flux_action.inference.so101 import load_policy  # noqa: E402


class OffloadPolicy(bench.Policy):
    def __init__(self, args: argparse.Namespace):
        self.device = torch.device(args.device)
        source = bench._local_package(Path(args.package), Path(args.vae), Path(args.text_encoder))
        bench.log.info("Loading SO-101 package %s (text encoder offloaded)", args.package)
        self.policy = load_policy(source)
        self.policy.set_text_encoder_offload(True)
        self.checkpoint = args.package
        self.policy.to(self.device).eval()
        # flux-action's offload still moves the encoder to the GPU to encode a new caption, which does not fit
        # next to the DiT + VAE on 24 GB. Encode on the CPU (cached per instruction) and move only the contexts.
        original_contexts = self.policy._contexts

        def cpu_contexts(captions, device, **kwargs):
            out = original_contexts(captions, torch.device("cpu"), **kwargs)
            return [(ctx.to(device), ids.to(device)) for ctx, ids in out]

        self.policy._contexts = cpu_contexts
        cfg = self.policy.config
        self.camera_keys = [key.removeprefix("images.") for key in cfg.camera_keys]
        self.client_to_key = dict(zip(args.client_cameras.split(","), cfg.camera_keys, strict=True))
        self.history_ticks = cfg.n_obs_steps
        self.capture = bench._VideoCapture() if args.save_predictions else None
        self.req_log = (Path(args.log_dir) / "server_requests.jsonl").open("a")
        bench.log.info("chunk %d, executes %d; sampler %s x%d, cfg %.1f; text encoder on %s",
                       cfg.chunk_size, cfg.n_action_steps, cfg.sampler, cfg.num_inference_steps, cfg.guidance_scale,
                       next(self.policy.text_encoder.parameters()).device)

    def predict(self, payload):
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        out = super().predict(payload)
        torch.cuda.synchronize(self.device)
        self.req_log.write(json.dumps({"dt_ms": (time.perf_counter() - start) * 1000.0,
                                       "peak_alloc_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
                                       "reserved_gib": torch.cuda.memory_reserved(self.device) / 2**30,
                                       "trace": payload.get("trace"), "t": time.time()}) + "\n")
        self.req_log.flush()
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--package", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--text_encoder", required=True)
    p.add_argument("--client_cameras", default="overhead,wrist")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--log_dir", required=True)
    args = p.parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    policy = OffloadPolicy(args)
    import base64

    import numpy as np

    cfg = policy.policy.config
    blank = base64.b64encode(bench._png(np.zeros((480, 640, 3), np.uint8))).decode()
    zeros = [[0.0] * cfg.action_dim] * policy.history_ticks
    payload = {"camera_names": list(policy.client_to_key), "images_png_base64": [blank] * 2,
               "snapshot_images_png_base64": [blank] * 2, "instruction": "Place each object in the plastic bin",
               "state_history": zeros, "command_history": zeros}
    for i in range(2):
        start = time.perf_counter()
        policy.predict(payload)
        bench.log.info("Warmup %d: %.1f ms, peak %.2f GiB", i, (time.perf_counter() - start) * 1e3,
                       torch.cuda.max_memory_allocated(policy.device) / 2**30)
    import uvicorn

    uvicorn.run(bench.build_app(policy), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
