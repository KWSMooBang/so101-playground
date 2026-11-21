#!/usr/bin/env python3
"""InternW0-Δ SO-101 policy server with its future prediction exposed (T006, RESEARCH.md H2).

A copy of so101_bench ``scripts/internw0_server.py`` (bench @ 470e09e; the submodule is not edited) with the
same ``POST /act`` contract, plus what the probe needs. Runs in envs/internw0 (InternW0-Delta repo @ 90801ba).

What InternW0-Δ predicts (repo ``src/wam``):
  * Training window: 33 action ticks, video frames every 4 ticks (ticks 0, 4, ..., 32 = 9 frames), encoded by
    the causal Wan2.2 VAE (×16 spatial, ×4 temporal) into 3 latent frames: L0 = tick 0, L1 = ticks 4–16,
    L2 = ticks 20–32. Canvas: overhead above wrist, 384×256 → latent 48×24×16.
  * ``future_delta`` = 96 learned query tokens (one per 2×2 latent patch; num_views 1) appended to the video
    expert. They read only the clean recent + current frame (never the noisy future, never actions), and a
    LayerNorm+Linear head maps them to the latent deltas (L1 − L0, L2 − L1), trained with MSE (weight 0.5).
    The action tokens attend to the delta queries, so the actions are conditioned on this prediction.
  * Online inference does NOT denoise any video: the video expert is a single prefill over (anchor, recent,
    current, delta queries); only the actions are denoised (10 steps). The delta head is never evaluated by
    the repo at inference. This server captures the last video-layer hidden state during that prefill and
    runs ``video_expert.predict_future_delta`` on it, i.e. exactly the tensor the training loss scored.
  → prediction: L̂1 = L0 + d̂1 and L̂2 = L0 + d̂1 + d̂2 in the Wan2.2 VAE latent space.

Extra behaviour:
  * Sessions are keyed by (client run dir, episode) so several Isaac clients can share one server; a session
    restarts when its first frame comes again (a re-run of the same episode).
  * The text encoder (UMT5-XXL, 11 GB) stays on the CPU; each instruction is encoded once and cached.
  * Every /act keeps (L0, d̂1, d̂2, the query canvas) under (dir, episode, frame).
  * ``POST /iw_score`` {"dir", "episode", "frame", "kind": observed|nominal, "offsets": [4, 8, ...],
    "images_png_base64": [[overhead, wrist] per offset]} builds the same canvases, VAE-encodes the clip
    [query canvas, frames at the offsets] (causal VAE: a 5-frame clip gives L0, L1; 9 frames L0, L1, L2) and
    logs, for k in the available latents and region in all / overhead (top 12 latent rows) / wrist:
        e_pred   mse(L̂k, Lk)          (the model's future-delta training loss, in latent space)
        e_static mse(L0, Lk)          (how much the scene moved; the "predict no motion" baseline)
    For kind=nominal it also logs d_obs_nom = mse(Lk_observed, Lk_nominal) when the observed clip of that
    query has been scored (the fault's own effect in latent space).
    It never touches a session: replay frames do not enter the model's memory.
  * Per-request timing and peak GPU memory go to ``<log_dir>/server_requests.jsonl``; scores to
    ``<log_dir>/iw_scores.jsonl``. ``--save_latents_dir`` stores pred/observed latents (fp16) for videos.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("internw0.probe.server")

REGIONS = {"all": slice(None), "overhead": slice(0, 12), "wrist": slice(12, 24)}  # latent rows (24 x 16)


def _decode_png(encoded: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


def _camera_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)


def build_vertical_canvas(overhead: np.ndarray, wrist: np.ndarray, camera_hw: tuple[int, int]) -> torch.Tensor:
    """Training's `concat_multi_camera: vertical`: per-camera resize to camera_hw, stack, scale to [-1, 1]."""
    cams = [
        transforms_F.resize(_camera_tensor(img), size=list(camera_hw),
                            interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
        for img in (overhead, wrist)
    ]
    return torch.cat(cams, dim=-2).mul(2.0).sub(1.0)


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean())


class Policy:
    def __init__(self, args: argparse.Namespace):
        repo = Path(args.repo).resolve()
        sys.path[:0] = [str(repo), str(repo / "src")]
        os.chdir(repo)  # model ids resolve under checkpoints/ (WAM_CHECKPOINT_ROOT) as in training
        from omegaconf import OmegaConf

        import eval.robotwin.policy as rt
        from eval.robotwin.policy import RobotWinPolicySession, build_robotwin_runtime_and_session_config
        from wam.model.modules.codecs import video_latent_codec

        self.codec = video_latent_codec
        self.device = args.device

        # --- text encoder on the CPU. The model's own __init__ moves every submodule to `device`, so build the
        # whole model on the CPU, detach the text encoder, then move the rest to the GPU (the repo's
        # set_runtime_device for VAE stats / VLM, then build_robotwin_shared_runtime's model.to(device)).
        text_holder: dict[str, Any] = {}
        orig_instantiate = rt.instantiate

        def instantiate(cfg_node, *a, **kw):
            if "model_dtype" not in kw:
                return orig_instantiate(cfg_node, *a, **kw)
            target = kw.pop("device")
            model = orig_instantiate(cfg_node, *a, device="cpu", **kw)
            text_holder["enc"] = model.text_encoder
            model.text_encoder = None
            model.set_runtime_device(target)
            return model

        rt.instantiate = instantiate

        # The run's config holds the training pod's absolute paths (VLM under checkpoints/); point them at `repo`.
        cfg = OmegaConf.create(Path(args.train_config).read_text().replace("/root/internw0/InternW0-Delta", str(repo)))
        shape_meta = cfg.data.train.processor.shape_meta
        if [str(m.key) for m in shape_meta.images] != ["overhead", "wrist"]:
            raise SystemExit(f"expected overhead, wrist cameras; got {[m.key for m in shape_meta.images]}")
        if str(cfg.data.train.concat_multi_camera) != "vertical":
            raise SystemExit("expected the vertical two-camera canvas")
        self.camera_hw = tuple(int(x) for x in shape_meta.images[0].shape[-2:])
        self.memory = bool(cfg.model.memory.enabled)
        cfg.EVALUATION = OmegaConf.create({
            "action_horizon": int(cfg.data.train.num_frames) - 1, "replan_steps": args.replan_steps,
            "num_inference_steps": args.num_inference_steps, "action_hz": args.action_hz, "sigma_shift": None,
            "rand_device": "cpu", "tiled": False, "timing_enabled": False, "device": args.device,
            "dataset_stats_path": args.stats,
        })
        cfg.model.mot_compile_mode = "off"
        cfg.model.mot_compile_gradient_checkpointing = False
        cfg.data.train.image_color_jitter.enabled = False
        cfg.data.train.image_color_jitter.apply_to_vlm = False
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        self.runtime, self.session_cfg = build_robotwin_runtime_and_session_config({
            "resolved_config": cfg, "ckpt_setting": args.checkpoint, "dataset_stats_path": args.stats,
            "seed": args.seed, "device": args.device, "mixed_precision": str(cfg.mixed_precision),
        })
        rt.instantiate = orig_instantiate
        model = self.runtime.model
        self.model = model
        text_encoder = text_holder["enc"].eval()
        tokenizer = model.tokenizer
        prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        @torch.inference_mode()
        def encode_prompt_cpu(prompt):
            key = prompt if isinstance(prompt, str) else json.dumps(list(prompt))
            if key not in prompt_cache:
                t = time.time()
                ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
                mask = mask.to(dtype=torch.bool)
                emb = text_encoder(ids, mask).masked_fill(~mask.unsqueeze(-1), 0)
                prompt_cache[key] = (emb.to(model.device), torch.ones_like(mask).to(model.device))
                log.info("encoded prompt on CPU in %.1fs: %r", time.time() - t, key[:80])
            return prompt_cache[key]

        model.encode_prompt = encode_prompt_cpu
        self.video_layout, self.video_view_names = self.session_cfg.video_layout, self.session_cfg.video_view_names
        log.info("model ready in %.0fs; GPU allocated %.2f GiB (peak %.2f)", time.time() - t0,
                 torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_allocated() / 2**30)

        # --- capture the future-delta prediction from the video prefill
        mot = model.mot
        video_expert = model.video_expert
        n_delta = int(model.future_delta_num_tokens)
        self.n_future = int(model.future_delta_num_frames)
        state = {"on": False, "x": None}
        orig_post = mot._apply_post_with_optional_checkpoint
        orig_prefill = mot.prefill_expert_cache

        def post(*a, **kw):
            x = orig_post(*a, **kw)
            if state["on"]:
                state["x"] = x
            return x

        def prefill(expert_name, tokens, *a, **kw):
            state["on"] = expert_name == "video"
            try:
                out = orig_prefill(expert_name, tokens, *a, **kw)
            finally:
                state["on"] = False
            if expert_name == "video":
                x = state["x"]
                meta = {"meta": {"real_video_seq_len": int(x.shape[1]) - n_delta, "num_future_delta_tokens": n_delta}}
                pred = video_expert.predict_future_delta(x, meta)  # [1, F*96, 192]
                self._last_delta = pred.detach().float()
            return out

        mot._apply_post_with_optional_checkpoint = post
        mot.prefill_expert_cache = prefill
        self._last_delta = None

        camera_hw = self.camera_hw

        class SO101Session(RobotWinPolicySession):
            def _build_robotwin_image_tensor(self, observation):
                obs = observation["observation"]
                canvas = build_vertical_canvas(obs["overhead"]["rgb"], obs["wrist"]["rgb"], camera_hw)
                if tuple(canvas.shape[-2:]) != (self.video_height, self.video_width):
                    raise RuntimeError(f"canvas {tuple(canvas.shape)} vs video {(self.video_height, self.video_width)}")
                return canvas.to(device=self.model.device, dtype=self.model.torch_dtype)

            def _build_robotwin_vlm_images(self, observation):
                obs = observation["observation"]
                return torch.stack([torch.from_numpy(np.ascontiguousarray(obs[k]["rgb"])).permute(2, 0, 1)
                                    for k in ("overhead", "wrist")]).to(device=self.model.device)

        self.session_cls = SO101Session
        self.replan_steps = self.session_cfg.replan_steps
        self.action_horizon = self.session_cfg.action_horizon
        self.sessions: OrderedDict[tuple, Any] = OrderedDict()
        self.session_first: dict[tuple, Any] = {}
        self.queries: OrderedDict[tuple, dict] = OrderedDict()  # (dir, episode, frame) -> prediction record
        self.observed: OrderedDict[tuple, torch.Tensor] = OrderedDict()  # observed latents for d_obs_nom
        self.checkpoint = args.checkpoint
        self.lock = threading.Lock()
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.req_log = (log_dir / "server_requests.jsonl").open("a")
        self.score_log = (log_dir / "iw_scores.jsonl").open("a")
        self.save_latents_dir = Path(args.save_latents_dir) if args.save_latents_dir else None
        log.info("InternW0 probe: camera=%s replan=%d denoise=%d future latents=%d delta tokens=%d",
                 camera_hw, self.replan_steps, self.session_cfg.num_inference_steps, self.n_future, n_delta)

    # ------------------------------------------------------------------ latents
    def _latents(self, canvases: list[torch.Tensor]) -> torch.Tensor:
        """[1,3,H,W] canvases in [-1,1] → Wan2.2 VAE latents [C, T, h, w] (fp32), the training encode."""
        video = torch.cat(canvases, dim=0).permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]
        video = video.to(device=self.model.device, dtype=self.model.torch_dtype)
        z = self.codec.encode_video_latents(self.model, video, tiled=False, video_layout=self.video_layout,
                                            video_view_names=self.video_view_names)
        return z[0].float()

    def _unpatch_delta(self, pred: torch.Tensor, latent_hw: tuple[int, int]) -> torch.Tensor:
        """[1, F*96, 192] → [C, F, h, w], inverse of the training F.unfold target (kernel = stride = 2x2)."""
        c = int(self.model.video_expert.in_dim)
        ph, pw = (int(v) for v in self.model.video_expert.patch_size[1:])
        h, w = latent_hw
        x = pred[0].view(self.n_future, h // ph * (w // pw), c * ph * pw).transpose(1, 2)  # [F, C*ph*pw, L]
        x = torch.nn.functional.fold(x, output_size=(h, w), kernel_size=(ph, pw), stride=(ph, pw))  # [F, C, h, w]
        return x.permute(1, 0, 2, 3).contiguous()

    # ------------------------------------------------------------------ /act
    @torch.inference_mode()
    def predict(self, payload: dict[str, Any]) -> np.ndarray:
        names = payload["camera_names"]
        images = dict(zip(names, (_decode_png(x) for x in payload["images_png_base64"]), strict=True))
        state = np.asarray(payload["state"], dtype=np.float32)
        episode = payload.get("episode")
        trace = payload.get("trace") or {}
        frame = trace.get("frame")
        key = (str(trace.get("dir", "")), episode)
        sess = self.sessions.get(key)
        if sess is None or self.session_first.get(key) == frame:
            sess = self.session_cls(self.runtime, self.session_cfg)  # new episode: fresh memory
            self.sessions[key] = sess
            self.session_first[key] = frame
        self.sessions.move_to_end(key)
        while len(self.sessions) > 8:
            old, _ = self.sessions.popitem(last=False)
            self.session_first.pop(old, None)
        while sess.pending_model_actions:  # the client executed the whole previous chunk
            sess.update_obs(None)
        observation = {
            "observation": {"overhead": {"rgb": images["overhead"]}, "wrist": {"rgb": images["wrist"]}},
            "joint_action": {"vector": state},
            "episode": episode,
        }
        self._last_delta = None
        chunk = sess.get_action({"observation": observation, "instruction": str(payload["instruction"])})
        canvas = build_vertical_canvas(images["overhead"], images["wrist"], self.camera_hw)
        l0 = self._latents([canvas])[:, 0]
        delta = self._unpatch_delta(self._last_delta, tuple(l0.shape[-2:]))
        rec = {"canvas": canvas, "l0": l0, "pred": l0[:, None] + torch.cumsum(delta, dim=1)}  # [C, F, h, w]
        self.queries[(key[0], episode, frame)] = rec
        while len(self.queries) > 64:
            self.queries.popitem(last=False)
        if self.save_latents_dir is not None:
            d = self.save_latents_dir / Path(key[0]).name
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"ep{episode:03d}_f{frame:05d}_pred.npz", l0=l0.half().cpu().numpy(),
                                pred=rec["pred"].half().cpu().numpy())
        return np.asarray(chunk, dtype=np.float32)

    # ------------------------------------------------------------------ /iw_score
    @torch.inference_mode()
    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        qkey = (str(payload.get("dir", "")), payload["episode"], payload["frame"])
        rec = self.queries.get(qkey)
        if rec is None:
            raise KeyError(f"no prediction for {qkey}")
        offsets = [int(o) for o in payload["offsets"]]
        if offsets != list(range(4, 4 * len(offsets) + 1, 4)) or len(offsets) % 4:
            raise ValueError(f"offsets must be 4, 8, ... in groups of 4, got {offsets}")
        canvases = [rec["canvas"]] + [build_vertical_canvas(_decode_png(o), _decode_png(w), self.camera_hw)
                                      for o, w in payload["images_png_base64"]]
        z = self._latents(canvases)  # [C, 1 + n/4, h, w]
        kind = str(payload.get("kind", "observed"))
        out: dict[str, Any] = {k: payload.get(k) for k in ("condition", "attempt", "episode", "frame", "replay_mode")}
        out.update(kind=kind, dir=qkey[0], offsets=offsets, l0_check=_mse(z[:, 0], rec["l0"]))
        n = min(int(z.shape[1]) - 1, self.n_future)
        for k in range(1, n + 1):
            pred, obs = rec["pred"][:, k - 1], z[:, k]
            for rname, rs in REGIONS.items():
                sfx = f"@{k}" if rname == "all" else f"@{k}_{rname}"
                out["e_pred" + sfx] = _mse(pred[:, rs], obs[:, rs])
                out["e_static" + sfx] = _mse(rec["l0"][:, rs], obs[:, rs])
            okey = qkey + (k,)
            if kind == "observed":
                self.observed[okey] = obs
                while len(self.observed) > 256:
                    self.observed.popitem(last=False)
            elif okey in self.observed:
                o = self.observed[okey]
                for rname, rs in REGIONS.items():
                    sfx = f"@{k}" if rname == "all" else f"@{k}_{rname}"
                    out["d_obs_nom" + sfx] = _mse(o[:, rs], obs[:, rs])
        if self.save_latents_dir is not None:
            d = self.save_latents_dir / Path(qkey[0]).name
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"ep{qkey[1]:03d}_f{qkey[2]:05d}_{kind}{len(offsets)}.npz",
                                z=z.half().cpu().numpy())
        self.score_log.write(json.dumps(out) + "\n")
        self.score_log.flush()
        return out


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="InternW0-delta SO-101 probe server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "repo_id": policy.checkpoint, "norm_tag": "internw0",
                             "num_images": 2, "camera_keys": ["overhead", "wrist"], "history_ticks": 1,
                             "chunk_size": policy.action_horizon, "n_action_steps": policy.replan_steps,
                             "memory": policy.memory})

    def _run(fn, payload, what):
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with policy.lock:  # one GPU, one request at a time (fastapi runs sync handlers in a thread pool)
            result = fn(payload)
        dt = (time.perf_counter() - start) * 1000.0
        policy.req_log.write(json.dumps({"t": time.time(), "path": what, "dt_ms": dt,
                                         "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
                                         "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}) + "\n")
        policy.req_log.flush()
        return result, dt

    @app.post("/act")
    def act(payload: dict) -> JSONResponse:
        try:
            actions, dt = _run(policy.predict, payload, "act")
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            log.exception("InternW0 inference failed")
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        return JSONResponse({"actions": actions.tolist(), "dt_ms": dt})

    @app.post("/iw_score")
    def iw_score(payload: dict) -> JSONResponse:
        try:
            out, dt = _run(policy.score, payload, "iw_score")
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=400)
        except Exception as exc:  # noqa: BLE001
            log.exception("InternW0 score failed")
            return JSONResponse({"error": f"score failed: {exc}"}, status_code=500)
        out["dt_ms"] = dt
        return JSONResponse(out)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="InternW0-Delta checkout (envs/internw0/InternW0-Delta)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_config", required=True, help="the run's resolved config.yaml")
    parser.add_argument("--stats", required=True, help="the run's dataset_stats.json")
    parser.add_argument("--replan_steps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=10, help="Repo eval default.")
    parser.add_argument("--action_hz", type=float, default=30.0, help="SO-101 Bench control rate.")
    parser.add_argument("--seed", type=int, default=0, help="Repo eval default (configs/sim_robodojo.yaml).")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--save_latents_dir", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    policy = Policy(args)
    import uvicorn

    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
