#!/usr/bin/env python3
"""SO-101 Bench sim rollouts of a WAM policy server with an injected control gap (T004, RESEARCH.md H2-P0).

Run from the isaac env (source envs/isaac/env.sh of the checkout that holds the venv). A lean version of
so101_bench ``scripts/molmoact2_eval.py`` (same env construction, reset, initial hold, policy client
``MolmoAct2RemotePolicy`` and success bookkeeping) with three additions; the bench is not edited:

1. Control gap (``so101_playground.control_gap``): ``--condition none|gain|latency|offset|backlash``.
   The policy's command ``cmd`` (sim radians, after the client's 15 deg/tick clamp) goes through the
   gap before ``env.step``; the gain gap edits the actuator cfg before the env is built.
2. Per-tick log per episode (``<out>/ep<e>.npz``): cmd, applied target, measured joint position after
   the step, query ticks, the policy chunk of each query.
3. Prediction targets:
   * ``--model lawam``: 35 ticks after each query (LaWAM's LAM horizon) the observed overhead+wrist frame
     is sent to the server's ``/lam_score`` (kind=observed).
   * ``--model flux3``: the observed scene|wrist canvas (256x512, FLUX's side_by_side resize) of every
     tick is stored as JPEG (q95) in ``<out>/frames_ep<e>.npz`` for offline pixel/feature metrics.
   * ``--model internw0`` (T006): the observed overhead+wrist frames at s+4, s+8, ... are sent to the server's
     ``/iw_score`` at s+16 (offsets 4..16 -> VAE latent 1) and s+32 (offsets 4..32 -> latents 1, 2).
   * ``--replay_episodes N`` / ``--replay_every k``: for the first N episodes, every k-th query snapshots the
     scene; at the end of that chunk the scene is reset to the snapshot, the commands the policy actually
     issued in the chunk (after the 15 deg/tick clamp, before the gap) are executed ideally (see
     ``replay_branch``), the replay frames are kept / scored (kind=ideal), and the end-of-chunk scene is
     restored (and re-rendered) before the rollout continues.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="So101Bench-Bin-v0")
parser.add_argument("--model", choices=["lawam", "flux3", "internw0"], required=True)
parser.add_argument("--condition", default="none")
parser.add_argument("--run_tag", default=None, help="Label for logs and server scores (default: the condition).")
parser.add_argument("--episodes_jsonl", type=Path, required=True)
parser.add_argument("--layouts_jsonl", type=Path, required=True)
parser.add_argument("--num_episodes", type=int, default=None)
parser.add_argument("--episode_offset", type=int, default=0)
parser.add_argument("--out_dir", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1984)
parser.add_argument("--policy_host", default="127.0.0.1")
parser.add_argument("--policy_port", type=int, default=8000)
parser.add_argument("--action_horizon", type=int, default=None)
parser.add_argument("--history_ticks", type=int, default=None)
parser.add_argument("--replay_episodes", type=int, default=0)
parser.add_argument("--save_frames", action="store_true", help="Store canvases (and LaWAM LAM features) for videos.")
parser.add_argument("--replay_every", type=int, default=1, help="Replay every k-th query of a replay episode.")
parser.add_argument("--replay_mode", choices=["teleport", "nominal"], default="teleport",
                    help="teleport: arm placed on the command (perfect tracking); nominal: fault-free PD arm.")
parser.add_argument("--replay_ticks", type=int, default=None, help="Ticks per replay branch (default per model).")
parser.add_argument("--lam_horizon", type=int, default=35)
parser.add_argument("--max_ticks", type=int, default=None, help="Optional cap on control ticks per episode.")
parser.add_argument("--initial_hold_time_s", type=float, default=1.5)
parser.add_argument("--oracle_alpha", type=float, default=0.0,
                    help="T005: oracle correction strength (so101_playground.oracle), between the clamp and the gap.")
parser.add_argument("--token_stats", action="store_true",
                    help="T005: ask /lam_score to log per-token e_pred / e_static (256 values each).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.headless and not getattr(args, "experience", ""):
    args.experience = "isaaclab.python.rendering.kit"  # as molmoact2_eval.py
# replay_ticks: frames needed from the ideal replay (LaWAM scores s+35; FLUX is scored for j <= 31).
MODEL_DEFAULTS = {"lawam": dict(action_horizon=36, history_ticks=1, replay_ticks=35),
                  "flux3": dict(action_horizon=32, history_ticks=8, replay_ticks=31),
                  # InternW0: 16 executed actions per query; its future latents span ticks 4-16 and 20-32.
                  "internw0": dict(action_horizon=16, history_ticks=1, replay_ticks=32)}
IW_OFFSETS = list(range(4, 33, 4))  # video frames of InternW0's 33-tick window after the query frame
for k, v in MODEL_DEFAULTS[args.model].items():
    if getattr(args, k) is None:
        setattr(args, k, v)

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from urllib import request  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import so101_bench.tasks  # noqa: E402,F401
from so101_bench.benchmark import OBJECT_SPLITS, episode_is_executable, load_episode_jsonl  # noqa: E402
from so101_bench.layouts import normalize_layout_object_slots  # noqa: E402
from so101_bench.mdp import mark_benchmark_robot_start  # noqa: E402
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import configure_env_cfg_for_object_pool  # noqa: E402
from so101_bench.utils.lerobot_calibration import LEROBOT_INITIAL_JOINT_POS, lerobot_pose_to_sim_joint_pos  # noqa: E402
from so101_bench.utils.molmoact2 import MolmoAct2RemotePolicy  # noqa: E402

from so101_playground.control_gap import CONDITIONS, JOINT_NAMES, CommandGap, apply_actuator_gap  # noqa: E402
from so101_playground.oracle import OracleCorrection  # noqa: E402

INITIAL = lerobot_pose_to_sim_joint_pos(LEROBOT_INITIAL_JOINT_POS)


# ---------------------------------------------------------------- episode plumbing (as molmoact2_eval.py)
def _object_info(name: str) -> dict:
    for split in OBJECT_SPLITS.values():
        if name in split:
            return split[name]
    return {}


def _reset_params(episode, layout, pool, asset_names):
    idx = {n: i for i, n in enumerate(pool)}
    payload = episode.reset_payload()
    local = [idx[n] for n in episode.objects]
    payload["active_object_ids"] = local
    payload["target_object_id"] = local[episode.target_object_id]
    payload["referent_object_ids"] = [local[i] for i in episode.referent_object_ids]
    remapped = dict(layout)
    objs = []
    for entry in layout.get("objects", []):
        e = dict(entry)
        slot = idx[str(e.get("name") or episode.objects[int(e["slot"])])]
        e["slot"], e["asset_name"] = slot, f"object_{slot + 1}"
        objs.append(e)
    remapped["objects"] = objs
    return {"object_asset_names": asset_names, "object_labels": pool, "task_family": episode.task_family,
            "object_count_range": (len(episode.objects), len(episode.objects)), "active_object_selection": "fixed",
            "fixed_active_object_ids": tuple(payload["active_object_ids"]), "shuffle_object_labels": False,
            "force_bin_all_objects_instruction": False, "episode_spec": payload, "episode_layout": remapped}


def _reset_env(env):
    with torch.inference_mode(False):
        env.reset()
        u = env.unwrapped
        robot = u.scene["robot"]
        ids = [robot.joint_names.index(j) for j in JOINT_NAMES]
        q = _hold(u).unsqueeze(0)
        robot.data.default_joint_pos[:, ids] = q
        robot.data.default_joint_vel[:, ids] = 0.0
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
        robot.set_joint_position_target(q, joint_ids=ids)
        robot.write_data_to_sim()
        u.scene.write_data_to_sim()
        u.sim.forward()
        for _ in range(getattr(u.cfg, "num_rerenders_on_reset", 0) if u.sim.has_rtx_sensors() else 0):
            u.sim.render()
        obs = u.observation_manager.compute(update_history=True)
        u.obs_buf = obs
        return obs


def _hold(u) -> torch.Tensor:
    return torch.tensor([INITIAL[j] for j in JOINT_NAMES], dtype=torch.float32, device=u.device)


# ---------------------------------------------------------------- frames
def _rgb(visual: dict, cam: str) -> torch.Tensor:
    img = visual[f"rgb_{cam}"][0]
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _canvas(visual: dict) -> np.ndarray:
    """FLUX side_by_side canvas: scene(overhead) | wrist, each bilinear+antialias to 256x256 (flux-action packing)."""
    cells = []
    for cam in ("overhead", "wrist"):
        x = _rgb(visual, cam).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        cells.append(x)
    out = torch.cat(cells, dim=-1)[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
    return out.cpu().numpy()


def _jpeg(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="JPEG", quality=95)
    return b.getvalue()


def _png_b64(a: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode("ascii")


def _policy_frame(policy: MolmoAct2RemotePolicy, visual: dict, cam: str) -> np.ndarray:
    return policy._camera_frame(visual, cam)  # same uint8 + resize path the policy request uses


def _post(port: int, path: str, payload: dict) -> dict:
    req = request.Request(f"http://{args.policy_host}:{port}{path}", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except Exception as exc:  # a failed score must not end the rollout
        detail = exc.read().decode(errors="replace") if hasattr(exc, "read") else ""
        print(f"[T004][WARN] {path} failed: {exc} {detail}", flush=True)
        return {}


def _keep_features(feats: dict, kind: str, frame: int, res: dict) -> None:
    """LaWAM features returned by /lam_score (return_features): 256 x 768 fp16 per key."""
    for name in ("h_t", "h_t1_pred", "h_obs"):
        if name in res:
            arr = np.frombuffer(base64.b64decode(res[name]), dtype=np.float16).reshape(256, -1)
            key = f"{name}_{frame}" if name != "h_obs" else f"h_{kind}_{frame}"
            feats[key] = arr


# ---------------------------------------------------------------- ideal (kinematic) replay branch
FLUSH_RENDERS = 3  # re-renders of the restored state so the RTX temporal history holds no replay frames


def replay_branch(env, start_state: dict, targets: list[torch.Tensor], capture: set[int],
                  nominal_stiffness: torch.Tensor, mode: str = "teleport") -> dict[int, dict]:
    """Re-execution of ``targets`` (the commands the policy actually issued, i.e. after the client's
    15 deg/tick clamp and before the gap) from ``start_state`` (the scene at the query tick).

    mode="nominal": the same commands on a fault-free arm. All six joints track them through the nominal PD
    drives (bench stiffness, no delay/offset/backlash, no teleport), exactly as env.step would without a gap.
    This is the arm the WAM's training data was recorded with. mode="teleport" (below): perfect tracking.

    Arm (5 joints): joint positions written at every physics substep (zero velocity), so the arm is exactly
    at the command whatever the actuator does. Jaw: tracked by its PD drive (teleporting it would push
    through a grasped object). Drives use the nominal (no-gap) stiffness during the replay. Objects are
    simulated normally; every tick is rendered as env.step does. Returns visual obs after tick j for j in
    ``capture``. The current scene, drive gains and PD targets are restored afterwards, and the restored state
    is re-rendered FLUSH_RENDERS times so the following real frames carry no replay history."""
    u = env.unwrapped
    scene, robot = u.scene, u.scene["robot"]
    ids = [robot.joint_names.index(j) for j in JOINT_NAMES]
    arm_ids = ids[:5]
    with torch.inference_mode(False):
        now = scene.get_state(is_relative=False)
        saved_target = robot.data.joint_pos_target.clone()
        gap_stiffness = robot.data.joint_stiffness.clone()
        robot.write_joint_stiffness_to_sim(nominal_stiffness)
        scene.reset_to(start_state, is_relative=False)
        out = {}
        zeros = torch.zeros((1, 5), device=u.device)
        for j, tgt in enumerate(targets[: max(capture)], start=1):
            q = tgt.reshape(1, -1).to(u.device)
            robot.set_joint_position_target(q, joint_ids=ids)
            for sub in range(u.cfg.decimation):
                if mode == "teleport":
                    robot.write_joint_state_to_sim(q[:, :5], zeros, joint_ids=arm_ids)
                scene.write_data_to_sim()
                u.sim.step(render=False)
                if sub == u.cfg.decimation - 1:
                    u.sim.render()
                scene.update(dt=u.physics_dt)
            if j in capture:
                vis = u.observation_manager.compute_group("visual")
                out[j] = {k: v.clone() for k, v in vis.items()}
        robot.write_joint_stiffness_to_sim(gap_stiffness)
        scene.reset_to(now, is_relative=False)
        robot.set_joint_position_target(saved_target)
        scene.write_data_to_sim()
        u.sim.forward()
        for _ in range(FLUSH_RENDERS):
            u.sim.render()
    return out


def _clone_obs(obs):
    if isinstance(obs, torch.Tensor):
        return obs.clone()
    if isinstance(obs, dict):
        return {k: _clone_obs(v) for k, v in obs.items()}
    return obs


# ---------------------------------------------------------------- main
def main() -> None:
    gap_cfg = CONDITIONS[args.condition]
    save_frames = args.save_frames or args.model == "flux3"
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    episodes = [e for e in load_episode_jsonl(args.episodes_jsonl) if episode_is_executable(e)]
    episodes = episodes[args.episode_offset:]
    if args.num_episodes is not None:
        episodes = episodes[: args.num_episodes]
    layouts_by_id = {}
    for line in args.layouts_jsonl.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            layouts_by_id[str(row["trial_id"])] = row
    layouts = [normalize_layout_object_slots(layouts_by_id[str((e.metadata or {}).get("trial_id", i))], e.objects, episode_index=i)
               for i, e in enumerate(episodes)]
    pool = []
    for e in episodes:
        pool += [o for o in e.objects if o not in pool]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.scene.robot.init_state.joint_pos = dict(INITIAL)
    nominal_stiffness_cfg = dict(env_cfg.scene.robot.actuators["arm"].stiffness)
    stiffness = apply_actuator_gap(env_cfg, gap_cfg)
    asset_names = configure_env_cfg_for_object_pool(env_cfg, pool)
    env_cfg.events.reset_benchmark_scene.params.update(_reset_params(episodes[0], layouts[0], pool, asset_names))
    t0 = time.time()
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    control_dt = float(u.step_dt)
    hold_steps = max(0, math.ceil(args.initial_hold_time_s / control_dt))
    cameras = {k.replace("camera_", ""): {} for k in u.scene.keys() if k.startswith("camera_")}
    print(f"[T004] env ready in {time.time() - t0:.1f}s; control_dt={control_dt:.4f}; stiffness={stiffness}", flush=True)

    attempt = uuid.uuid4().hex[:12]  # tags this process's rows in append-only logs
    policy = MolmoAct2RemotePolicy(
        device=u.device, cameras=cameras, host=args.policy_host, port=args.policy_port,
        action_horizon=args.action_horizon, camera_names=["overhead", "wrist"], image_size=(640, 480),
        max_joint_step_deg=15.0, joint_frame="lerobot", history_ticks=args.history_ticks,
        # InternW0: the server keys its per-episode session by trace.dir, so it carries this process's attempt
        # id (two clients on the same run dir never share memory); other models need the real dir.
        prediction_trace_dir=f"{out}#{attempt}" if args.model == "internw0" else str(out),
    )
    policy.connect()
    last_chunk = {}
    _orig_query = policy._query_server

    def _query(jp, vis):
        chunk = _orig_query(jp, vis)
        last_chunk["chunk"] = chunk
        return chunk

    policy._query_server = _query
    gap = CommandGap(gap_cfg, u.device)
    oracle = OracleCorrection(gap_cfg, args.oracle_alpha, u.device)
    print(f"[T004] oracle alpha={args.oracle_alpha} kind={oracle.kind}: {oracle.describe()}", flush=True)
    robot = u.scene["robot"]
    ids = [robot.joint_names.index(j) for j in JOINT_NAMES]
    nominal_stiffness = robot.data.joint_stiffness.clone()
    for j, name in enumerate(robot.joint_names):
        if name in nominal_stiffness_cfg:
            nominal_stiffness[:, j] = float(nominal_stiffness_cfg[name])
    tag = args.run_tag or args.condition
    summary_path = out / "episodes.jsonl"
    done_eps = set()
    if summary_path.exists():
        done_eps = {json.loads(l)["episode"] for l in summary_path.read_text().splitlines() if l.strip()}

    for ei, (episode, layout) in enumerate(zip(episodes, layouts)):
        ep_id = ei + args.episode_offset
        if ep_id in done_eps:
            continue
        params = _reset_params(episode, layout, pool, asset_names)
        u.cfg.events.reset_benchmark_scene.params.update(params)
        u.event_manager.get_term_cfg("reset_benchmark_scene").params.update(params)
        obs = _reset_env(env)
        instruction = getattr(u, "so101_bench_instruction", "") or "Place each object in the plastic bin"
        policy.set_language_instruction(instruction)
        policy.reset()
        hold = _hold(u)
        do_replay = ei < args.replay_episodes
        log = {k: [] for k in ("cmd", "sent", "applied", "q", "tick")}
        queries, chunks, pending, frames, ideal_frames, feats = [], [], {}, [], {}, {}
        iw_png: dict[int, list[str]] = {}  # tick -> [overhead, wrist] policy-frame PNGs (InternW0 targets)
        replay_due: dict[int, tuple[int, dict]] = {}  # tick of the chunk's end -> (query tick, scene at query)
        started = False
        step = 0
        success, reason = False, "unknown"
        t_ep = time.time()
        t_inf = 0.0
        while app.is_running():
            if step < hold_steps:
                cmd = hold.clone()
            else:
                if not started:
                    mark_benchmark_robot_start(u, object_asset_names=asset_names, bin_name="plastic_bin",
                                               force_robot_start_time=True)
                    policy.set_episode_initial_observation(obs["visual"])
                    gap.reset(hold)
                    started = True
                policy.trace_episode, policy.trace_frame = ep_id, step
                need_query = not policy.action_queue
                ti = time.time()
                with torch.inference_mode():
                    jp = obs["policy"]["joint_pos_obs"][0].clone()
                    cmd = policy.get_action(jp, obs["visual"]).clone()
                    # Oracle (T005): row index of cmd in the current chunk, and the clamped row k ahead of it.
                    n_rows = min(len(last_chunk["chunk"]), args.action_horizon)
                    row = n_rows - len(policy.action_queue) - 1

                    def lookahead(k, _row=row, _n=n_rows, _jp=jp):
                        r = torch.as_tensor(last_chunk["chunk"][min(_row + k, _n - 1)], dtype=torch.float32,
                                            device=u.device)
                        return policy._limit_joint_step(_jp, policy.mapper.model_positions_to_sim_radians(r))

                    sent = oracle(cmd, jp, lookahead)
                if need_query:
                    t_inf += time.time() - ti
                    chunk = np.asarray(last_chunk["chunk"], dtype=np.float32)
                    queries.append(step)
                    chunks.append(chunk)
                    if args.model == "lawam":
                        pending[step + args.lam_horizon] = step
                    if args.model == "internw0":
                        pending.setdefault(step + 16, []).append(step)  # latent 1 (ticks 4-16, inside the chunk)
                        pending.setdefault(step + 32, []).append(step)  # latents 1, 2 (ticks 20-32: next chunk runs)
                    if do_replay and (len(queries) - 1) % args.replay_every == 0:
                        # Snapshot now; replay at the end of this chunk (InternW0: of the next chunk, so the
                        # replay covers all 32 predicted ticks) with the commands actually issued.
                        due = step + (args.replay_ticks if args.model == "internw0" else args.action_horizon)
                        with torch.inference_mode(False):
                            replay_due[due] = (step, u.scene.get_state(is_relative=False))
            if not started:
                sent = cmd
            applied = gap(sent) if started else sent
            obs, _r, terminated, truncated, info = env.step(applied.unsqueeze(0))
            step += 1
            log["cmd"].append(cmd.cpu().numpy())
            log["sent"].append(sent.cpu().numpy())
            log["applied"].append(applied.cpu().numpy())
            log["q"].append(robot.data.joint_pos[0, ids].cpu().numpy())
            log["tick"].append(step)
            if save_frames:
                frames.append(_jpeg(_canvas(obs["visual"])))
            ended = bool(terminated.any() or truncated.any())
            if args.model == "internw0" and not ended and any(0 < step - q <= 32 and (step - q) % 4 == 0
                                                              for q in queries[-3:]):
                iw_png[step] = [_png_b64(_policy_frame(policy, obs["visual"], c)) for c in ("overhead", "wrist")]
                for k in [k for k in iw_png if k < step - 40]:
                    del iw_png[k]
            # Not on the terminating tick: env.step auto-resets on termination, so that tick's obs is the reset scene
            # (T004 review #5), not the outcome. Such queries (horizon end == last tick) are dropped and counted in
            # the T006 doc; every other query whose horizon ends after the episode is unscorable anyway.
            if args.model == "internw0" and step in pending and not ended:
                for qs in pending.pop(step):
                    offs = IW_OFFSETS[: (step - qs) // 4]
                    _post(args.policy_port, "/iw_score", {"dir": str(out), "attempt": attempt, "episode": ep_id,
                          "frame": qs, "kind": "observed", "condition": tag, "offsets": offs,
                          "images_png_base64": [iw_png[qs + o] for o in offs]})
            if args.model == "lawam" and step in pending and not ended:
                qs = pending.pop(step)
                res = _post(args.policy_port, "/lam_score", {"return_features": save_frames, "dir": str(out),
                    "token_stats": args.token_stats,
                    "attempt": attempt, "episode": ep_id, "frame": qs, "kind": "observed", "condition": tag, "obs_frame": step,
                    "camera_names": ["overhead", "wrist"],
                    "images_png_base64": [_png_b64(_policy_frame(policy, obs["visual"], c)) for c in ("overhead", "wrist")]})
                _keep_features(feats, "observed", qs, res)
            if step in replay_due and not ended:
                qs, start_state = replay_due.pop(step)
                obs = _clone_obs(obs)  # the next query's input, rendered before the replay
                tgts = [torch.as_tensor(c, device=u.device) for c in log["cmd"][qs: qs + args.replay_ticks]]
                capture = ({args.lam_horizon} if args.model == "lawam" and not save_frames
                           else set(IW_OFFSETS) if args.model == "internw0" and not save_frames
                           else set(range(1, len(tgts) + 1)))
                branch = replay_branch(env, start_state, tgts, capture, nominal_stiffness, mode=args.replay_mode)
                if save_frames:
                    ideal_frames[qs] = [_jpeg(_canvas(branch[j])) for j in sorted(branch)]
                if args.model == "lawam" and args.lam_horizon in branch:
                    vis = branch[args.lam_horizon]
                    res = _post(args.policy_port, "/lam_score", {"return_features": save_frames, "dir": str(out),
                        "attempt": attempt, "replay_mode": args.replay_mode, "episode": ep_id, "frame": qs, "kind": "ideal",
                        "condition": tag,
                        "obs_frame": qs + args.lam_horizon, "camera_names": ["overhead", "wrist"],
                        "images_png_base64": [_png_b64(_policy_frame(policy, vis, c)) for c in ("overhead", "wrist")]})
                    _keep_features(feats, "ideal", qs, res)
                if args.model == "internw0" and all(o in branch for o in IW_OFFSETS):
                    _post(args.policy_port, "/iw_score", {"dir": str(out), "attempt": attempt, "episode": ep_id,
                          "frame": qs, "kind": args.replay_mode, "replay_mode": args.replay_mode, "condition": tag,
                          "offsets": IW_OFFSETS, "images_png_base64": [
                              [_png_b64(_policy_frame(policy, branch[o], c)) for c in ("overhead", "wrist")]
                              for o in IW_OFFSETS]})
            if ended:
                term = info.get("log", {})
                success = bool(term.get("Episode_Termination/success", 0.0) > 0.0)
                reason = "success" if success else ("time_out" if bool(truncated.any()) else "failure")
                break
            if args.max_ticks is not None and step >= args.max_ticks:
                reason = "max_ticks"
                break
        np.savez_compressed(
            out / f"ep{ep_id:03d}.npz", hold_steps=hold_steps,
            **{k: np.asarray(v) for k, v in log.items()}, queries=np.asarray(queries),
            chunks=np.asarray(chunks) if chunks else np.zeros((0,)))
        if feats:
            np.savez_compressed(out / f"lamfeat_ep{ep_id:03d}.npz", **feats)
        if save_frames:
            np.savez(out / f"frames_ep{ep_id:03d}.npz", observed=np.array(frames, dtype=object),
                     ideal_queries=np.asarray(sorted(ideal_frames)),
                     ideal=np.array([np.array(ideal_frames[k], dtype=object) for k in sorted(ideal_frames)], dtype=object))
        rec = {"episode": ep_id, "trial_id": (episode.metadata or {}).get("trial_id"), "objects": episode.objects, "condition": tag,
               "attempt": attempt, "replay_mode": args.replay_mode, "oracle_alpha": args.oracle_alpha, "success": success, "reason": reason, "ticks": step, "n_queries": len(queries),
               "wall_s": time.time() - t_ep, "inference_wall_s": t_inf, "replay": do_replay}
        with summary_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[T004] {json.dumps(rec)}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)  # app.close() can hang after a failure inside the loop
    app.close()
