#!/usr/bin/env python3
"""T011: real-robot e_pred of InternW0-Δ from the home recordings and the T008 server logs (stage B diagnosis).

Run in envs/internw0 (InternW0-Delta venv; only the Wan2.2 VAE is loaded, ~1.4 GB bf16):

    CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $IW/InternW0-Delta/.venv/bin/python -m so101_playground.t011_real_epred encode --out /outputs/so101-playground/runs/T011
    ... t011_real_epred decode --out ... --episodes <dir names>     # predicted | observed clips (scope 6)

`encode` does, for every home episode (recording dir) matched to a server episode (`real_queries.jsonl`):
  * alignment: the q-th `policy_query` step of trajectory.npz is server query q (docs/real_internw0_server.md);
  * l0_check: VAE(query PNG canvas) vs the logged L0 (fp16);
  * codec floor: e_codec = mse(L0_png, L0_mp4), the same frame through the recorder's mp4 (mpeg4, 2.5 Mbit/s);
  * e_pred@k = mse(L̂k, Lk), L̂1 = L0 + d1, L̂2 = L0 + d1 + d2, Lk from the T006 /iw_score clip
    [query PNG canvas, mp4 frames at ticks +4, ..., +16 (+32)] (causal VAE, 5 or 9 frames), and
    e_static@k = mse(L0, Lk), for all / overhead (top 12 latent rows) / wrist;
  * e_pred@1_mp4: the same with the mp4 frame as clip frame 0 and L0 (an all-mp4 control);
  * per query: hold (s, the recorded gap at the query step: the arm waits for the server), joint motion over
    the next 16 ticks, tracking error |cmd(t-1) - q(t)| over the next 16 ticks (arm joints, deg).
Outputs: <out>/queries.jsonl (one line per query), <out>/latents/<dir>.npz (fp16 observed L1/L2 per query).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from PIL import Image

IW = Path("/workspace/so101-playground/.claude/worktrees/wam-internw0/envs/internw0")
VAE_PATH = IW / "weights/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
HOME = Path("/outputs/so101-playground/runs/real_robot/home_runs")
SERVER = Path("/outputs/so101-playground/runs/real_robot/internw0_server")
SERVER_RUNS = ["run_1122_2342", "run_1123_2311"]
CAMERA_HW = (192, 256)  # per camera; canvas 384 x 256 -> latent 48 x 24 x 16
REGIONS = {"all": slice(None), "overhead": slice(0, 12), "wrist": slice(12, 24)}
ARM = slice(0, 5)


# ---------------------------------------------------------------- canvas / VAE (= internw0_probe_server.py)
def _camera_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)


def build_vertical_canvas(overhead: np.ndarray, wrist: np.ndarray) -> torch.Tensor:
    cams = [transforms_F.resize(_camera_tensor(img), size=list(CAMERA_HW),
                                interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True)
            for img in (overhead, wrist)]
    return torch.cat(cams, dim=-2).mul(2.0).sub(1.0)


class VAE:
    """The Wan2.2 VAE alone, loaded the way the InternW0 runtime loads it (bf16, mean/std on the device).
    Encode = the probe server's `_latents`: the server session's video_layout is empty, so the codec takes the
    plain path and encodes the whole 384x256 two-camera canvas in one pass (not per camera slot). Checked
    against the logged L0: plain 9e-5 MSE (fp32), per-slot split 0.010."""

    def __init__(self, device: str):
        repo = IW / "InternW0-Delta"
        sys.path[:0] = [str(repo), str(repo / "src")]
        from wam.model.backbones.wan22.helpers.loader import _load_registered_model
        self.dtype = torch.bfloat16
        self.device = torch.device(device)
        vae = _load_registered_model(str(VAE_PATH), "wan_video_vae", torch_dtype=self.dtype, device="cpu")
        vae.to(device=self.device, dtype=self.dtype)
        vae.mean = vae.mean.to(device=self.device, dtype=self.dtype)
        vae.std = vae.std.to(device=self.device, dtype=self.dtype)
        vae.scale = [vae.mean, 1.0 / vae.std]
        self.vae = vae

    @torch.inference_mode()
    def encode(self, canvases: list[torch.Tensor]) -> torch.Tensor:
        """[1,3,384,256] canvases in [-1,1] -> latents [48, T, 24, 16] (fp32, CPU)."""
        video = torch.cat(canvases, 0).permute(1, 0, 2, 3).unsqueeze(0).to(self.device, self.dtype)
        return self.vae.encode(video, device=self.device)[0].float().cpu()

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> np.ndarray:
        """[48, T, 24, 16] -> uint8 frames [1 + 4(T-1), 384, 256, 3]."""
        v = self.vae.decode(z[None].to(self.device, self.dtype), device=self.device)[0]  # [3, F, H, W]
        return ((v.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean())


# ---------------------------------------------------------------- data
def read_frames(path: Path, keep: set[int]) -> dict[int, np.ndarray]:
    import av
    out = {}
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in keep:
                out[i] = f.to_ndarray(format="rgb24")
    return out


def count_frames(path: Path) -> int:
    import av
    with av.open(str(path)) as c:
        return sum(1 for _ in c.demux(video=0) if _.size)


def server_episodes() -> dict[int, list[dict]]:
    eps: dict[int, list[dict]] = {}
    for run in SERVER_RUNS:
        for line in (SERVER / run / "real" / "real_queries.jsonl").open():
            r = json.loads(line)
            r["run"] = run
            eps.setdefault(int(r["episode"]), []).append(r)
    return {e: sorted(v, key=lambda r: r["q"]) for e, v in eps.items()}


def match(home_dirs: list[Path], server: dict[int, list[dict]]) -> tuple[dict[str, int], int, dict]:
    """Recording dir (home-desktop local time) -> server episode (unix time on the same clock). The home clock's
    UTC offset is the whole-hour offset that best fits; each dir then takes the nearest server episode."""
    naive = {d.name: dt.datetime.strptime(d.name[:15], "%Y%m%d_%H%M%S").replace(tzinfo=dt.timezone.utc).timestamp()
             for d in home_dirs}
    eps = np.array(sorted(server))
    best = None
    for h in range(-14, 15):
        res = [np.min(np.abs(eps - (t - 3600 * h))) for t in naive.values()]
        if best is None or np.median(res) < best[1]:
            best = (h, float(np.median(res)))
    tz = best[0]
    out, resid = {}, {}
    for name, t in naive.items():
        e = int(eps[np.argmin(np.abs(eps - (t - 3600 * tz)))])
        out[name], resid[name] = e, float(e - (t - 3600 * tz))
    return out, tz, resid


# ---------------------------------------------------------------- encode
def encode(args) -> None:
    out = Path(args.out)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    device = args.device
    if device.startswith("cuda"):
        free = torch.cuda.mem_get_info()[0] / 2**30
        print(f"GPU free {free:.1f} GiB", flush=True)
        if free < args.min_free_gib:
            device = "cpu"
            print("-> CPU", flush=True)
    vae = VAE(device)
    server = server_episodes()
    home_dirs = sorted(d for d in HOME.iterdir() if (d / "trajectory.npz").exists())
    if args.episodes:
        home_dirs = [d for d in home_dirs if d.name in args.episodes]
    m, tz, resid = match(sorted(d for d in HOME.iterdir() if (d / "trajectory.npz").exists()), server)
    used = {}
    for name, e in m.items():
        used.setdefault(e, []).append(name)
    align = {"tz_hours": tz, "match": m, "resid_s": resid,
             "dup": {str(e): v for e, v in used.items() if len(v) > 1},
             "unmatched_server": [e for e in server if e not in used]}
    qf = (out / "queries.jsonl").open("a" if args.episodes else "w")
    ep_rows = []
    for d in home_dirs:
        e = m[d.name]
        recs = server[e]
        z = np.load(d / "trajectory.npz")
        meta = json.loads((d / "episode.json").read_text())
        qsteps = np.flatnonzero(z["policy_query"])
        n = len(z["t"])
        n_ov, n_wr = count_frames(d / "overhead.mp4"), count_frames(d / "wrist.mp4")
        row = {"dir": d.name, "episode": e, "tz_resid_s": resid[d.name], "n_steps": n, "n_policy_query": len(qsteps),
               "n_server_q": len(recs), "mp4_frames": [n_ov, n_wr], "instruction": meta["instructions"][0][1],
               "server_instruction": recs[0]["instruction"], "end_reason": meta["end_reason"],
               "restarted_q0": bool(recs[0]["restarted"]), "q_contiguous": [r["q"] for r in recs] == list(range(len(recs)))}
        ep_rows.append(row)
        print(json.dumps(row), flush=True)
        nq = min(len(qsteps), len(recs))
        keep = set()
        for s in qsteps[:nq]:
            keep.update(int(s) + o for o in range(-1, 33))
        frames = {cam: read_frames(d / f"{cam}.mp4", keep) for cam in ("overhead", "wrist")}
        t, st, cmd = z["t"], z["state_sim_deg"], z["command_sim_deg"]
        ep1, l0c = [], []
        lat_L1 = np.full((nq, 48, 24, 16), np.nan, np.float16)
        lat_L2 = np.full((nq, 48, 24, 16), np.nan, np.float16)
        for q in range(nq):
            s = int(qsteps[q])
            r = recs[q]
            qn = np.load(Path(r["npz"]))
            l0 = torch.from_numpy(qn["l0"].astype(np.float32))
            pred = l0[:, None] + torch.cumsum(torch.from_numpy(np.stack([qn["d1"], qn["d2"]], 1).astype(np.float32)), 1)
            qdir = Path(r["npz"]).parent
            png = {cam: np.asarray(Image.open(qdir / f"q{q:04d}_{cam}.png").convert("RGB")) for cam in ("overhead", "wrist")}
            c_png = build_vertical_canvas(png["overhead"], png["wrist"])
            c_mp4 = build_vertical_canvas(frames["overhead"][s], frames["wrist"][s])
            K = 2 if s + 32 < n else (1 if s + 16 < n else 0)
            fut = [build_vertical_canvas(frames["overhead"][s + o], frames["wrist"][s + o]) for o in range(4, 16 * K + 1, 4)]
            zc = vae.encode([c_png] + fut)  # [48, 1+K, 24, 16]
            zm = vae.encode([c_mp4] + fut[:4]) if K else vae.encode([c_mp4])
            pix_png_mp4 = float(np.abs(png["overhead"].astype(np.int16) - frames["overhead"][s]).mean()), \
                float(np.abs(png["wrist"].astype(np.int16) - frames["wrist"][s]).mean())
            nb = {o: float(np.abs(png["overhead"].astype(np.int16) - frames["overhead"][s + o]).mean())
                  for o in (-1, 1) if (s + o) in frames["overhead"]}
            rec = {"pix_png_mp4_prev": nb.get(-1), "pix_png_mp4_next": nb.get(1), "dir": d.name, "episode": e, "q": q, "step": s, "K": K, "run": r["run"],
                   "l0_check": mse(zc[:, 0], l0), "l0_check_max": float((zc[:, 0] - l0).abs().max()),
                   "pix_png_mp4_overhead": pix_png_mp4[0], "pix_png_mp4_wrist": pix_png_mp4[1],
                   "state_match_deg": float(np.abs(np.asarray(r["state"]) - qn["state"]).max()),
                   "memory_frames": r["memory_frames"], "server_latency_s": (r["t_reply"] - r["t_recv"]) if r["t_recv"] else None}
            for rn, rs in REGIONS.items():
                sfx = "" if rn == "all" else f"_{rn}"
                rec["e_codec" + sfx] = mse(zc[:, 0, rs], zm[:, 0, rs])
                for k in range(1, K + 1):
                    rec[f"e_pred@{k}{sfx}"] = mse(pred[:, k - 1, rs], zc[:, k, rs])
                    rec[f"e_static@{k}{sfx}"] = mse(l0[:, rs], zc[:, k, rs])
                if K:
                    rec["e_pred@1_mp4" + sfx] = mse(zm[:, 0, rs] + (pred[:, 0, rs] - l0[:, rs]), zm[:, 1, rs])
            # predicted motion size (|d1|^2) for reference
            rec["pred_motion@1"] = mse(pred[:, 0], l0)
            for k in range(1, K + 1):
                (lat_L1 if k == 1 else lat_L2)[q] = zc[:, k].half().cpu().numpy()
            # hold: the step the runner blocked on the server, and the arm's motion/tracking over the next 16 ticks
            # t is stamped when the step is recorded, after get_action returned: the query wait is t[s] - t[s-1]
            rec["hold_s"] = float(t[s] - t[s - 1]) if s > 0 else None
            rec["dt_median_s"] = float(np.median(np.diff(t)))
            a, b = s + 1, min(s + 17, n)
            rec["motion16_deg"] = float(np.abs(st[min(s + 16, n - 1), ARM] - st[s, ARM]).max())
            rec["path16_deg"] = float(np.abs(np.diff(st[s:b, ARM], axis=0)).sum())
            rec["e_track16_deg"] = float(np.abs(cmd[a - 1:b - 1, ARM] - st[a:b, ARM]).mean()) if b > a else None
            ep1.append(rec.get("e_pred@1", np.nan))
            l0c.append(rec["l0_check"])
            qf.write(json.dumps(rec) + "\n")
        qf.flush()
        np.savez_compressed(out / "latents" / f"{d.name}.npz", L1=lat_L1, L2=lat_L2, qsteps=qsteps[:nq])
        print(f"  {d.name}: {nq} queries, mean e_pred@1 {np.nanmean(ep1):.4f}, max l0_check {max(l0c):.2e}", flush=True)
    if not args.episodes:
        align["episodes"] = ep_rows
        (out / "alignment.json").write_text(json.dumps(align, indent=1))


# ---------------------------------------------------------------- codec floor under motion
def codec(args) -> None:
    """The L0 floor (PNG vs mp4) is measured on a held arm. To see how the recorder's mpeg4 (2.5 Mbit/s) loss grows
    with motion, re-compress each mp4 once more with the same codec and bitrate (second generation, a proxy for
    the first) and take mse(Lk(mp4), Lk(mp4 re-compressed)) for the query clips, frame 0 = the query PNG in both."""
    import subprocess
    import tempfile
    out = Path(args.out)
    vae = VAE(args.device if not args.device.startswith("cuda") or torch.cuda.mem_get_info()[0] / 2**30 >= args.min_free_gib else "cpu")
    server = server_episodes()
    m, _, _ = match(sorted(d for d in HOME.iterdir() if (d / "trajectory.npz").exists()), server)
    f = (out / "codec_motion.jsonl").open("w")
    for name in args.episodes:
        d = HOME / name
        recs = server[m[name]]
        z = np.load(d / "trajectory.npz")
        st, n = z["state_sim_deg"], len(z["t"])
        qsteps = np.flatnonzero(z["policy_query"])
        keep = set()
        for s in qsteps:
            keep.update(int(s) + o for o in range(0, 33, 4))
        fr, fr2 = {}, {}
        with tempfile.TemporaryDirectory(dir=args.tmp) as td:
            for cam in ("overhead", "wrist"):
                br = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=bit_rate",
                                     "-of", "csv=p=0", str(d / f"{cam}.mp4")], capture_output=True, text=True).stdout.strip()
                p2 = Path(td) / f"{cam}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / f"{cam}.mp4"), "-c:v", "mpeg4", "-b:v", br,
                                "-bf", "0", "-pix_fmt", "yuv420p", str(p2)], check=True)
                fr[cam], fr2[cam] = read_frames(d / f"{cam}.mp4", keep), read_frames(p2, keep)
        for q, s in enumerate(qsteps[:len(recs)]):
            s = int(s)
            if s + 16 >= n:
                continue
            K = 2 if s + 32 < n else 1
            png = {cam: np.asarray(Image.open(Path(recs[q]["npz"]).parent / f"q{q:04d}_{cam}.png").convert("RGB")) for cam in ("overhead", "wrist")}
            c0 = build_vertical_canvas(png["overhead"], png["wrist"])
            za = vae.encode([c0] + [build_vertical_canvas(fr["overhead"][s + o], fr["wrist"][s + o]) for o in range(4, 16 * K + 1, 4)])
            zb = vae.encode([c0] + [build_vertical_canvas(fr2["overhead"][s + o], fr2["wrist"][s + o]) for o in range(4, 16 * K + 1, 4)])
            z0 = vae.encode([build_vertical_canvas(fr2["overhead"][s], fr2["wrist"][s])])
            zo = vae.encode([build_vertical_canvas(fr["overhead"][s], fr["wrist"][s])])
            rec = {"dir": name, "q": q, "step": s, "motion16_deg": float(np.abs(st[min(s + 16, n - 1), ARM] - st[s, ARM]).max()),
                   "e_codec2@0": mse(z0[:, 0], zo[:, 0])}
            for rn, rs in REGIONS.items():
                sfx = "" if rn == "all" else f"_{rn}"
                for k in range(1, K + 1):
                    rec[f"e_codec2@{k}{sfx}"] = mse(za[:, k, rs], zb[:, k, rs])
            f.write(json.dumps(rec) + "\n")
        f.flush()
        print(name, flush=True)


# ---------------------------------------------------------------- decode (scope 6)
def decode(args) -> None:
    out = Path(args.out)
    vdir = out / "decoded"
    vdir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device.startswith("cuda") and torch.cuda.mem_get_info()[0] / 2**30 < args.min_free_gib:
        device = "cpu"
    vae = VAE(device)
    server = server_episodes()
    m, _, _ = match(sorted(d for d in HOME.iterdir() if (d / "trajectory.npz").exists()), server)
    for name in args.episodes:
        d = HOME / name
        recs = server[m[name]]
        lat = np.load(out / "latents" / f"{name}.npz")
        qsteps = lat["qsteps"]
        qs = [int(x) for x in args.queries.split(",")] if args.queries else list(range(len(qsteps)))
        qs = [q for q in qs if q < len(qsteps) and not np.isnan(lat["L1"][q].astype(np.float32)).any()]
        keep = set()
        for q in qs:
            keep.update(int(qsteps[q]) + o for o in range(0, 17, 4))
        frames = {cam: read_frames(d / f"{cam}.mp4", keep) for cam in ("overhead", "wrist")}
        pred_v, obs_v, rec_v, real_v = [], [], [], []
        for q in qs:
            qn = np.load(recs[q]["npz"])
            l0 = torch.from_numpy(qn["l0"].astype(np.float32))
            l1h = l0 + torch.from_numpy(qn["d1"].astype(np.float32))
            l1 = torch.from_numpy(lat["L1"][q].astype(np.float32))
            pred_v.append(vae.decode(torch.stack([l0, l1h], 1)))  # [5, 384, 256, 3]: tick 0 (recon), 4..16 predicted
            rec_v.append(vae.decode(torch.stack([l0, l1], 1)))  # VAE reconstruction of the observed clip
            s = int(qsteps[q])
            real_v.append(np.stack([((build_vertical_canvas(frames["overhead"][s + o], frames["wrist"][s + o])[0] + 1)
                                     * 127.5).round().byte().permute(1, 2, 0).numpy() for o in range(0, 17, 4)]))
        np.savez_compressed(vdir / f"{name}.npz", q=np.array(qs), pred=np.stack(pred_v), recon=np.stack(rec_v),
                            real=np.stack(real_v), qsteps=qsteps)
        print(name, len(qs), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["encode", "decode", "codec"])
    p.add_argument("--out", default="/outputs/so101-playground/runs/T011")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min_free_gib", type=float, default=4.0)
    p.add_argument("--episodes", nargs="*", default=[])
    p.add_argument("--queries", default="")
    p.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
    args = p.parse_args()
    torch.set_num_threads(min(16, os.cpu_count() or 4))
    {"encode": encode, "decode": decode, "codec": codec}[args.cmd](args)


if __name__ == "__main__":
    main()
