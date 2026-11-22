#!/usr/bin/env python3
"""Self-test of the real-mode InternW0 server (T008): requests shaped like the bench real runner's.

    python scripts/real_internw0_selftest.py --port 8004 --out <dir>

The runner (so101_real_policy_runner.py → MolmoAct2RemotePolicy._query_server, via scripts/policy_runner.py)
sends {images_png_base64: [overhead, wrist], camera_names, instruction, state (6, LeRobot units), num_steps: 10,
episode: int(time.time()) at the episode start} and no `trace`, once every 16 control ticks. Here the frames are
recorded sim frames (T004 frames_ep000.npz: one JPEG per tick, overhead | wrist side by side, 256×256 each) at
ticks 0, 16, 32, ... and the state is the sim joint trajectory mapped to LeRobot units.

Runs: A (episode e1) 10 queries, B (episode e2, another recording) 10 queries, A again under a new episode id e3
(a reset must reproduce A exactly: the denoiser seed is fixed per call), then A once more with a sim `trace`
(the T006 harness path: per-(dir, episode) session) to compare actions. Prints a JSON summary.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path
from urllib import request

import numpy as np
from PIL import Image

BENCH = Path("/workspace/so101-playground/.claude/worktrees/scaffold-env/third_party/so101_bench")
RUNS = Path("/outputs/so101-playground/runs/T004/lawam_videos")
INSTRUCTION = "Place each object in the plastic bin."


def _calib():
    path = BENCH / "source/so101_bench/so101_bench/utils/lerobot_calibration.py"
    spec = importlib.util.spec_from_file_location("lerobot_calibration", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


def load(cond: str, ticks: list[int]):
    cal = _calib()
    frames = np.load(RUNS / cond / "frames_ep000.npz", allow_pickle=True)["observed"]
    q = np.load(RUNS / cond / "ep000.npz")["q"]  # sim joint radians, bench joint order
    out = []
    for t in ticks:
        img = np.asarray(Image.open(io.BytesIO(frames[t])).convert("RGB"))
        overhead, wrist = img[:, :256], img[:, 256:]
        state = [cal.sim_degrees_to_lerobot_position(j, math.degrees(float(v)))
                 for j, v in zip(cal.LEROBOT_JOINT_ORDER, q[min(t, len(q) - 1)])]
        out.append((overhead, wrist, state))
    return out


def png_b64(a: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode("ascii")


def post(url: str, payload: dict) -> dict:
    req = request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def run(url: str, inputs, episode, trace_dir: str | None = None, ticks=None):
    acts, dts, walls = [], [], []
    for i, (o, w, s) in enumerate(inputs):
        payload = {"images_png_base64": [png_b64(o), png_b64(w)], "camera_names": ["overhead", "wrist"],
                   "instruction": INSTRUCTION, "state": [float(x) for x in s], "num_steps": 10, "episode": episode}
        if trace_dir is not None:
            payload["trace"] = {"dir": trace_dir, "episode": episode, "frame": ticks[i]}
        t = time.perf_counter()
        res = post(url, payload)
        walls.append((time.perf_counter() - t) * 1000)
        if "error" in res:
            raise SystemExit(f"server error: {res['error']}")
        acts.append(np.asarray(res["actions"], dtype=np.float32))
        dts.append(res["dt_ms"])
    return np.stack(acts), dts, walls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8004)
    ap.add_argument("--n", type=int, default=10, help="queries per episode")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/act"
    with request.urlopen(url, timeout=10) as r:
        health = json.loads(r.read())
    print("health", health)
    ticks = [16 * i for i in range(args.n)]
    A, B = load("none", ticks), load("gain", ticks)
    e1 = int(time.time())
    a1, dts_a, walls_a = run(url, A, e1)
    b, dts_b, walls_b = run(url, B, e1 + 1)
    a2, _, _ = run(url, A, e1 + 2)
    a_sim, _, _ = run(url, A, 0, trace_dir="t008_selftest#0", ticks=ticks)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "selftest_actions.npz", a1=a1, b=b, a2=a2, a_sim=a_sim)
    summary = {
        "health": health,
        "queries": {"A": len(a1), "B": len(b), "A_reset": len(a2), "A_sim_trace": len(a_sim)},
        "reset_reproduces_A_max_abs": float(np.abs(a1 - a2).max()),
        "real_vs_sim_trace_max_abs": float(np.abs(a1 - a_sim).max()),
        "A_vs_B_q0_max_abs": float(np.abs(a1[0] - b[0]).max()),
        "server_dt_ms": {"first": dts_a[0], "median": float(np.median(dts_a[1:] + dts_b[1:])),
                         "max": float(max(dts_a[1:] + dts_b[1:]))},
        "client_wall_ms_median": float(np.median(walls_a[1:] + walls_b[1:])),
        "episodes": [e1, e1 + 1, e1 + 2],
    }
    (args.out / "selftest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
