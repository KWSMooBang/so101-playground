#!/usr/bin/env python3
"""T009 Isaac worker: runs single episodes for the ES coordinator (src/so101_playground/t009_es.py).

Run from the isaac env (scripts/run_t009_worker.sh). The env is built once with the injected T009 gap
(``--condition gainj``, per-joint arm stiffness) and an object pool covering both episode sets, then the
worker pulls jobs from a file queue until ``<queue>/STOP`` exists and nothing is pending:

    <queue>/pending/<order>__<job id>.json   written by the coordinator (atomic rename)
    <queue>/running/<job id>.<worker>.json   claimed by this worker (atomic rename)
    <queue>/done/<job id>.json               result (job fields + episode metrics), atomic rename

A job is {"id", "set": "home"|"t004", "row", "theta": [5 floats, log k], "noise_seed", "jitter": [dx m, dy m,
dyaw deg] or null, ...}. The jitter (human re-placement error, drawn by the coordinator) is added to every
object's layout pose before the reset. One episode is
run exactly as scripts/wam_gap_eval.py --model internw0 runs it (same env construction, reset, 1.5 s hold,
bench client with action horizon 16, 15 deg/tick clamp, seed 1984, 30 s limit), with the gain adapter
(so101_playground.gain_adapter) between the clamp and the gap, where T005's oracle acts. The episode id sent
to the policy server is ``noise_seed``: with ``--seed_per_query`` on the server the two members of an
antithetic pair get the same flow noise at the same tick. Server keys (session, predictions, scores) carry
``<attempt>.<job id>``, so no two jobs ever share a key, whichever worker runs them (review 2026-11-23 #1).
The T009 run itself keyed by the process attempt only. The logs were checked clean (docs/T009-rl-sim.md,
review corrections).

Reward signal (real-robot computable): at s+16 of every query the observed overhead+wrist frames at s+4..s+16
go to ``/iw_score`` (latent 1 only); e_pred@1 comes back in the response. The terminating tick is not scored
(env.step has already auto-reset there, T004 review #5).

Per-episode metrics in the result: success, reason, ticks, n_queries, e_pred@1 (mean over scored queries, plus
the wrist / overhead halves and the per-query list), e_track (mean |cmd - q(t+1)| over arm joints and policy
ticks, cmd = the policy's clamped command before adapter and gap, degrees), arm path length (sum over policy
ticks of sum_j |q_j(t+1) - q_j(t)| over the arm joints, degrees), mean |sent - cmd|.

T012 additions (perception gap; all opt-in, T009/T010 behaviour unchanged without them):
  * overhead camera pose delta: job["camera_delta"] (else ``--camera_delta``), e.g. "tilt=4" / "pan=-8"
    (so101_playground.camera_delta: tilt/pan/roll deg, x/y/z cm). It is merged into the reset payload's
    ``episode_spec.domain_randomization`` (``overhead_camera_*`` fields), which the bench applies at every reset;
    the wrist camera is untouched. The policy, the /iw_score frames and the static probe all read the same
    rendered camera, so e_pred is scored on exactly the frames the policy saw. The camera's local pose read back
    from the sensor after the reset is logged (``cam_pos_applied``, ``cam_quat_applied``).
  * ``--score_latents 2``: also scores InternW0's latent 2 (ticks 20-32 after the query) at s+32 (8 offsets);
    e_pred@1 is still scored at s+16 exactly as before.
  * static probe: job["static_queries"] (else ``--static_queries``) N > 0 queries/scores N chunks with the arm
    docked before the rollout, in a separate server session, then resets again (so101_playground.t012_probe).
  * reach error (always logged): XY distance from the gripper TCP to the target object's centre at the first
    commanded jaw close (so101_playground.reach_error), plus per-tick TCP / object xy in the npz.
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
parser.add_argument("--condition", default="gainj")
parser.add_argument("--queue", type=Path, required=True)
parser.add_argument("--out_dir", type=Path, required=True, help="per-episode npz logs")
parser.add_argument("--worker", default="w1")
parser.add_argument("--t004_episodes", type=Path, default=REPO / "tasks/T004_bin1.jsonl")
parser.add_argument("--t004_layouts", type=Path, required=True)
parser.add_argument("--t004_rows", type=int, default=8)
parser.add_argument("--home_episodes", type=Path, default=REPO / "tasks/home_objects.jsonl")
parser.add_argument("--home_layouts", type=Path, default=REPO / "tasks/layouts/home_objects_layouts.jsonl")
parser.add_argument("--home_rows", type=int, default=6, help="the 6 bin rows come first in home_objects.jsonl")
parser.add_argument("--seed", type=int, default=1984)
parser.add_argument("--policy_host", default="127.0.0.1")
parser.add_argument("--policy_port", type=int, default=8000)
parser.add_argument("--action_horizon", type=int, default=16)
parser.add_argument("--history_ticks", type=int, default=1)
parser.add_argument("--initial_hold_time_s", type=float, default=1.5)
parser.add_argument("--max_ticks", type=int, default=None, help="smoke tests only")
parser.add_argument("--poll_s", type=float, default=2.0)
parser.add_argument("--camera_delta", default="", help="T012 default overhead camera delta, e.g. 'tilt=4,pan=-8'")
parser.add_argument("--static_queries", type=int, default=0, help="T012 default static-probe queries per job")
parser.add_argument("--score_latents", type=int, default=1, choices=(1, 2))
parser.add_argument("--save_frames", type=int, default=0,
                    help="smoke: save the policy's overhead/wrist frames of the first N policy ticks (png)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.headless and not getattr(args, "experience", ""):
    args.experience = "isaaclab.python.rendering.kit"  # as molmoact2_eval.py
IW_OFFSETS1 = [4, 8, 12, 16]  # video frames of InternW0's latent 1 (ticks 4-16 after the query)

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from urllib import request  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import so101_bench.tasks  # noqa: E402,F401
from so101_bench.benchmark import episode_is_executable, load_episode_jsonl  # noqa: E402
from so101_bench.layouts import normalize_layout_object_slots  # noqa: E402
from so101_bench.mdp import mark_benchmark_robot_start  # noqa: E402
from so101_bench.tasks.direct.so101_bench.so101_bench_env_cfg import configure_env_cfg_for_object_pool  # noqa: E402
from so101_bench.utils.lerobot_calibration import LEROBOT_INITIAL_JOINT_POS, lerobot_pose_to_sim_joint_pos  # noqa: E402
from so101_bench.utils.molmoact2 import MolmoAct2RemotePolicy  # noqa: E402

from so101_playground.control_gap import ARM_JOINTS, CONDITIONS, JOINT_NAMES, CommandGap, apply_actuator_gap  # noqa: E402
from so101_playground.gain_adapter import GainAdapter  # noqa: E402
from so101_playground import reach_error as reach  # noqa: E402
from so101_playground import t012_probe as probe  # noqa: E402
from so101_playground.camera_delta import CameraDelta, overhead_dr  # noqa: E402

INITIAL = lerobot_pose_to_sim_joint_pos(LEROBOT_INITIAL_JOINT_POS)
N_ARM = len(ARM_JOINTS)


# ---------------------------------------------------------------- episode plumbing (as wam_gap_eval.py)
def _reset_params(episode, layout, pool, asset_names, delta: CameraDelta | None = None):
    idx = {n: i for i, n in enumerate(pool)}
    payload = episode.reset_payload()
    if delta is not None:
        payload["domain_randomization"] = overhead_dr(payload.get("domain_randomization"), delta)
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


def _hold(u) -> torch.Tensor:
    return torch.tensor([INITIAL[j] for j in JOINT_NAMES], dtype=torch.float32, device=u.device)


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


def _png_b64(a: np.ndarray) -> str:
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode("ascii")


def _post(path: str, payload: dict) -> dict:
    req = request.Request(f"http://{args.policy_host}:{args.policy_port}{path}", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except Exception as exc:  # a failed score must not end the rollout
        detail = exc.read().decode(errors="replace") if hasattr(exc, "read") else ""
        print(f"[T009][WARN] {path} failed: {exc} {detail}", flush=True)
        return {}


def _load_set(episodes_jsonl: Path, layouts_jsonl: Path, n: int):
    """First n executable rows with their layouts, matched as wam_gap_eval.py does (offset 0)."""
    episodes = [e for e in load_episode_jsonl(episodes_jsonl) if episode_is_executable(e)][:n]
    by_id = {}
    for line in layouts_jsonl.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            by_id[str(row["trial_id"])] = row
    layouts = [normalize_layout_object_slots(by_id[str((e.metadata or {}).get("trial_id", i))], e.objects, episode_index=i)
               for i, e in enumerate(episodes)]
    return list(zip(episodes, layouts))


def _jitter(layout: dict, jitter) -> dict:
    """Human re-placement error (T009 amendment): every object shifted by (dx, dy) metres in the table frame
    and rotated by dyaw degrees about z. ``jitter`` = [dx, dy, dyaw_deg] drawn by the coordinator; None = exact."""
    if not jitter:
        return layout
    dx, dy, dyaw = float(jitter[0]), float(jitter[1]), math.radians(float(jitter[2]))
    out = dict(layout)
    objs = []
    for entry in layout.get("objects", []):
        e = dict(entry)
        pos = list(e["position"])
        pos[0] += dx
        pos[1] += dy
        e["position"] = pos
        if "yaw" in e:
            e["yaw"] = float(e["yaw"]) + dyaw
        if "rpy" in e:
            rpy = list(e["rpy"])
            rpy[2] = float(rpy[2]) + dyaw
            e["rpy"] = rpy
        objs.append(e)
    out["objects"] = objs
    return out


# ---------------------------------------------------------------- queue
def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _claim(queue: Path, worker: str):
    for p in sorted((queue / "pending").glob("*.json")):
        job_id = p.name.split("__", 1)[1][:-5]
        dst = queue / "running" / f"{job_id}.{worker}.json"
        try:
            os.rename(p, dst)
        except FileNotFoundError:
            continue  # another worker took it
        return json.loads(dst.read_text()), dst
    return None, None


def _requeue_own(queue: Path, worker: str) -> None:
    """Jobs this worker held when it died go back to the front of the queue."""
    for p in (queue / "running").glob(f"*.{worker}.json"):
        job = json.loads(p.read_text())
        os.replace(p, queue / "pending" / f"{job.get('order', '0')}__{job['id']}.json")
        print(f"[T009] requeued {job['id']}", flush=True)


# ---------------------------------------------------------------- main
def main() -> None:
    gap_cfg = CONDITIONS[args.condition]
    for d in ("pending", "running", "done"):
        (args.queue / d).mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _requeue_own(args.queue, args.worker)
    sets = {"t004": _load_set(args.t004_episodes, args.t004_layouts, args.t004_rows),
            "home": _load_set(args.home_episodes, args.home_layouts, args.home_rows)}
    pool = []
    for rows in sets.values():
        for e, _ in rows:
            pool += [o for o in e.objects if o not in pool]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.scene.robot.init_state.joint_pos = dict(INITIAL)
    stiffness = apply_actuator_gap(env_cfg, gap_cfg)
    asset_names = configure_env_cfg_for_object_pool(env_cfg, pool)
    e0, l0 = sets["t004"][0]
    env_cfg.events.reset_benchmark_scene.params.update(_reset_params(e0, l0, pool, asset_names))
    t0 = time.time()
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    control_dt = float(u.step_dt)
    hold_steps = max(0, math.ceil(args.initial_hold_time_s / control_dt))
    cameras = {k.replace("camera_", ""): {} for k in u.scene.keys() if k.startswith("camera_")}
    print(f"[T009] env ready in {time.time() - t0:.1f}s; control_dt={control_dt:.4f}; pool={pool}; "
          f"stiffness={stiffness}", flush=True)

    attempt = uuid.uuid4().hex[:12]
    out = args.out_dir
    policy = MolmoAct2RemotePolicy(
        device=u.device, cameras=cameras, host=args.policy_host, port=args.policy_port,
        action_horizon=args.action_horizon, camera_names=["overhead", "wrist"], image_size=(640, 480),
        max_joint_step_deg=15.0, joint_frame="lerobot", history_ticks=args.history_ticks,
        prediction_trace_dir=f"{out}#{attempt}",
    )
    policy.connect()
    last_chunk: list = [None]
    _query = policy._query_server

    def _recording_query(*a, **kw):  # T012: keep the last raw chunk (probe vs rollout first-chunk check)
        last_chunk[0] = np.asarray(_query(*a, **kw), dtype=np.float32)
        return last_chunk[0]

    policy._query_server = _recording_query
    robot = u.scene["robot"]
    ids = [robot.joint_names.index(j) for j in JOINT_NAMES]
    base_body = robot.body_names.index("base")
    origin = u.scene.env_origins[0].detach().cpu().numpy().astype(float)
    gap = CommandGap(gap_cfg, u.device)
    latents = args.score_latents

    def world_from_base() -> np.ndarray:  # env frame (env origin subtracted), as so101_bench llm_harness
        pos = robot.data.body_pos_w[0, base_body].detach().cpu().numpy().astype(float) - origin
        quat = robot.data.body_quat_w[0, base_body].detach().cpu().numpy().astype(float)
        return reach.world_from_pose(pos, quat_wxyz=quat)

    def object_pose(asset: str):
        a = u.scene[asset]
        return (a.data.root_pos_w[0].detach().cpu().numpy().astype(float) - origin,
                a.data.root_quat_w[0].detach().cpu().numpy().astype(float))

    def frames(o) -> list[str]:
        return [_png_b64(policy._camera_frame(o["visual"], c)) for c in ("overhead", "wrist")]

    def score(key, ep_id, qs, k, png, tag):
        offs = probe.OFFSETS[k]
        return _post("/iw_score", {"dir": str(out), "attempt": key, "episode": ep_id, "frame": qs, "kind": "observed",
                                   "condition": tag, "offsets": offs, "images_png_base64": [png[qs + o] for o in offs]})

    def static_probe(obs, n: int, key: str, ep_id: int, tag: str) -> tuple[dict, np.ndarray | None]:
        """T012: N docked queries in their own server session (t012_probe); the caller resets the env after."""
        sched = probe.probe_schedule(n, hold_steps, latents)
        hold = _hold(u)
        policy.prediction_trace_dir = probe.probe_dir(str(out), key)
        policy.trace_episode = ep_id
        png, recs, chunks, states = {}, {}, [], []
        for step in range(sched["total_ticks"] + 1):
            if step in sched["queries"]:
                policy._clear_episode_state()
                policy.trace_frame = step
                jp = obs["policy"]["joint_pos_obs"][0].clone()
                chunks.append(policy._query_server(jp, obs["visual"]))
                states.append(policy.mapper.sim_radians_to_model_positions(jp).cpu().numpy())
            for qs, k in sched["scores"].get(step, []):
                res = score(probe.probe_attempt(key), ep_id, qs, k, png, tag + ".static")
                if f"e_pred@{k}" in res:
                    recs.setdefault(qs, {})[k] = (res[f"e_pred@{k}"], res[f"e_pred@{k}_overhead"],
                                                  res[f"e_pred@{k}_wrist"], res[f"e_static@{k}"])
            if step == sched["total_ticks"]:
                break
            obs, *_ = env.step(hold.unsqueeze(0))
            if any(0 < step + 1 - q <= 16 * latents and (step + 1 - q) % 4 == 0 for q in sched["queries"]):
                png[step + 1] = frames(obs)
        policy._clear_episode_state()
        policy.prediction_trace_dir = f"{out}#{key}"
        out_d = {"static_n": n, "static_query_ticks": sched["queries"]}
        for k in range(1, latents + 1):
            v = [recs[q][k] for q in sched["queries"] if k in recs.get(q, {})]
            sfx = "" if k == 1 else str(k)
            out_d[f"e_pred_static{sfx}"] = float(np.mean([x[0] for x in v])) if v else None
            out_d[f"e_pred_static{sfx}_overhead"] = float(np.mean([x[1] for x in v])) if v else None
            out_d[f"e_pred_static{sfx}_wrist"] = float(np.mean([x[2] for x in v])) if v else None
            out_d[f"e_pred_static{sfx}_queries"] = [x[0] for x in v]
            out_d[f"e_static_static{sfx}"] = float(np.mean([x[3] for x in v])) if v else None
        # open-loop chunk motion (model units = LeRobot deg / gripper 0-100): max over the chunk of |a - state|
        out_d["static_chunk_motion"] = [float(np.abs(c[:16, :5] - s[:5]).max()) for c, s in zip(chunks, states)]
        return out_d, (np.asarray(chunks) if chunks else None)

    idle_since = time.time()
    while app.is_running():
        job, running_path = _claim(args.queue, args.worker)
        if job is None:
            if (args.queue / "STOP").exists():
                break
            if time.time() - idle_since > 60:
                print("[T009] idle, waiting for jobs", flush=True)
                idle_since = time.time()
            time.sleep(args.poll_s)
            continue
        episode, layout = sets[job["set"]][int(job["row"])]
        layout = _jitter(layout, job.get("jitter"))
        adapter = GainAdapter(job["theta"], u.device)
        delta = CameraDelta.parse(job.get("camera_delta", args.camera_delta))
        n_static = int(job.get("static_queries", args.static_queries))
        max_ticks = job.get("max_ticks", args.max_ticks)
        save_frames = int(job.get("save_frames", args.save_frames))
        ep_id = int(job["noise_seed"])
        key = f"{attempt}.{job['id']}"  # per-job server key (review #1)
        policy.prediction_trace_dir = f"{out}#{key}"
        print(f"[T009] job {job['id']} set={job['set']} row={job['row']} seed={ep_id} {adapter.describe()} "
              f"camera={delta.label()} static={n_static}", flush=True)
        params = _reset_params(episode, layout, pool, asset_names, delta)
        u.cfg.events.reset_benchmark_scene.params.update(params)
        u.event_manager.get_term_cfg("reset_benchmark_scene").params.update(params)
        target = episode.objects[episode.target_object_id]
        target_asset = asset_names[pool.index(target)]
        instruction = None
        static_res, static_chunks = {}, None
        if n_static > 0:
            obs = _reset_env(env)
            instruction = getattr(u, "so101_bench_instruction", "") or "Place each object in the plastic bin"
            policy.set_language_instruction(instruction)
            static_res, static_chunks = static_probe(obs, n_static, key, ep_id, job.get("tag", ""))
        obs = _reset_env(env)  # the rollout always starts from its own fresh reset
        cam = u.scene["camera_overhead"]._view.get_local_poses()
        cam_pos, cam_quat = (np.asarray(c.detach().cpu() if hasattr(c, "detach") else c, dtype=float)[0].tolist()
                             for c in cam)
        instruction = getattr(u, "so101_bench_instruction", "") or "Place each object in the plastic bin"
        policy.set_language_instruction(instruction)
        policy.reset()
        hold = _hold(u)
        log = {k: [] for k in ("cmd", "sent", "q", "tcp", "obj")}
        q_start = None
        queries, scores, scores2, pending = [], [], [], {}
        iw_png: dict[int, list[str]] = {}
        started, step = False, 0
        success, reason = False, "unknown"
        close_det, close = reach.CloseDetector(), {}
        first_chunk = None
        obj_start = None
        t_ep, t_inf = time.time(), 0.0
        while app.is_running():
            if step < hold_steps:
                cmd = hold.clone()
                sent = cmd
            else:
                if not started:
                    mark_benchmark_robot_start(u, object_asset_names=asset_names, bin_name="plastic_bin",
                                               force_robot_start_time=True)
                    policy.set_episode_initial_observation(obs["visual"])
                    gap.reset(hold)
                    q_start = robot.data.joint_pos[0, ids].cpu().numpy()
                    obj_start = object_pose(target_asset)
                    started = True
                policy.trace_episode, policy.trace_frame = ep_id, step
                need_query = not policy.action_queue
                ti = time.time()
                with torch.inference_mode():
                    jp = obs["policy"]["joint_pos_obs"][0].clone()
                    cmd = policy.get_action(jp, obs["visual"]).clone()
                    sent = adapter(cmd, jp)
                if need_query:
                    t_inf += time.time() - ti
                    queries.append(step)
                    pending.setdefault(step + 16, []).append((step, 1))
                    if latents >= 2:
                        pending.setdefault(step + 32, []).append((step, 2))
                    if first_chunk is None:
                        first_chunk = last_chunk[0]
            applied = gap(sent) if started else sent
            obs, _r, terminated, truncated, info = env.step(applied.unsqueeze(0))
            step += 1
            ended = bool(terminated.any() or truncated.any())
            if started:
                log["cmd"].append(cmd.cpu().numpy())
                log["sent"].append(sent.cpu().numpy())
                q_now = robot.data.joint_pos[0, ids].cpu().numpy()
                log["q"].append(q_now)
                wfb = world_from_base()
                tcp_now = reach.tcp(wfb, q_now)
                opos, oquat = object_pose(target_asset)
                log["tcp"].append(tcp_now)
                log["obj"].append(reach.object_centre(target, opos, quat_wxyz=oquat))
                kind = close_det(math.degrees(float(cmd[5])))
                if kind and not ended:  # an auto-reset tick has no valid post-step state
                    centre = log["obj"][-1]
                    lpos, lyaw = reach.layout_target(layout, target)  # layout is already jittered
                    c_layout = reach.object_centre(target, lpos, yaw_rad=lyaw)
                    close = {"close_tick": len(log["cmd"]) - 1, "close_kind": kind,
                             "reach_err_cm": reach.xy_cm(tcp_now, centre),
                             "reach_err_cmd_cm": reach.xy_cm(reach.tcp(wfb, cmd.cpu().numpy()), centre),
                             "reach_err_layout_cm": reach.xy_cm(tcp_now, c_layout),
                             "obj_moved_cm": reach.xy_cm(centre, reach.object_centre(target, obj_start[0],
                                                                                     quat_wxyz=obj_start[1])),
                             "tcp_close": tcp_now.tolist(), "obj_close": centre.tolist(),
                             "tcp_z_close_cm": float(tcp_now[2] * 100.0)}
                if save_frames and len(log["cmd"]) <= save_frames:
                    fd = out / "frames" / job["id"]
                    fd.mkdir(parents=True, exist_ok=True)
                    for c in ("overhead", "wrist"):
                        Image.fromarray(policy._camera_frame(obs["visual"], c)).save(fd / f"{c}_{step:04d}.png")
            if not ended and probe.frames_needed(step, queries, latents):
                iw_png[step] = frames(obs)
                for k in [k for k in iw_png if k < step - 16 * latents - 8]:
                    del iw_png[k]
            if step in pending and not ended:
                for qs, k in pending.pop(step):
                    res = score(key, ep_id, qs, k, iw_png, job.get("tag", ""))
                    if k == 1 and "e_pred@1" in res:
                        scores.append({"frame": qs, "e_pred": res["e_pred@1"], "wrist": res["e_pred@1_wrist"],
                                       "overhead": res["e_pred@1_overhead"], "e_static": res["e_static@1"]})
                    elif k == 2 and "e_pred@2" in res:
                        scores2.append({"frame": qs, "e_pred": res["e_pred@2"], "wrist": res["e_pred@2_wrist"],
                                        "overhead": res["e_pred@2_overhead"], "e_static": res["e_static@2"]})
            if ended:
                term = info.get("log", {})
                success = bool(term.get("Episode_Termination/success", 0.0) > 0.0)
                reason = "success" if success else ("time_out" if bool(truncated.any()) else "failure")
                break
            if max_ticks is not None and step >= int(max_ticks):
                reason = "max_ticks"
                break
        cmd_a = np.asarray(log["cmd"])[:, :N_ARM]
        sent_a = np.asarray(log["sent"])[:, :N_ARM]
        q_a = np.asarray(log["q"])[:, :N_ARM]
        q_prev = np.vstack([q_start[None, :N_ARM], q_a[:-1]])
        extra = {} if static_chunks is None else {"static_chunks": static_chunks}
        if first_chunk is not None:
            extra["first_chunk"] = first_chunk
        np.savez_compressed(out / f"{job['id']}.npz", cmd=np.asarray(log["cmd"]), sent=np.asarray(log["sent"]),
                            q=np.asarray(log["q"]), q_start=q_start, queries=np.asarray(queries),
                            hold_steps=hold_steps, tcp=np.asarray(log["tcp"]), obj=np.asarray(log["obj"]), **extra)
        ep = [s["e_pred"] for s in scores]
        ep2 = [s["e_pred"] for s in scores2]
        res = dict(job)
        res.update({
            "worker": args.worker, "attempt": attempt, "server_key": key, "success": success, "reason": reason, "ticks": step,
            "policy_ticks": len(cmd_a), "n_queries": len(queries), "n_scored": len(scores),
            "e_pred": float(np.mean(ep)) if ep else None,
            "e_pred_wrist": float(np.mean([s["wrist"] for s in scores])) if scores else None,
            "e_pred_overhead": float(np.mean([s["overhead"] for s in scores])) if scores else None,
            "e_static": float(np.mean([s["e_static"] for s in scores])) if scores else None,
            "e_pred_queries": ep,
            "e_track_deg": float(np.degrees(np.abs(cmd_a - q_a)).mean()),
            "path_len_deg": float(np.degrees(np.abs(q_a - q_prev)).sum()),
            "sent_minus_cmd_deg": float(np.degrees(np.abs(sent_a - cmd_a)).mean()),
            "k": adapter.k_arm, "wall_s": time.time() - t_ep, "inference_wall_s": t_inf, "t_end": time.time(),
            "camera_delta": delta.to_dict(), "camera_label": delta.label(),
            "camera_fields": params["episode_spec"]["domain_randomization"],
            "cam_pos_applied": cam_pos, "cam_quat_applied": cam_quat,
            "target": target, "close_tick": None, "close_kind": None, "reach_err_cm": None,
        })
        if latents >= 2:
            res.update({"n_scored2": len(scores2), "e_pred2": float(np.mean(ep2)) if ep2 else None,
                        "e_pred2_wrist": float(np.mean([s["wrist"] for s in scores2])) if scores2 else None,
                        "e_pred2_overhead": float(np.mean([s["overhead"] for s in scores2])) if scores2 else None,
                        "e_static2": float(np.mean([s["e_static"] for s in scores2])) if scores2 else None,
                        "e_pred2_queries": ep2})
        res.update(close)
        res.update(static_res)
        if static_chunks is not None and first_chunk is not None:
            res["probe0_vs_first_chunk_maxdiff"] = float(np.abs(static_chunks[0] - first_chunk).max())
        _atomic_write(args.queue / "done" / f"{job['id']}.json", res)
        running_path.unlink(missing_ok=True)
        print(f"[T009] done {job['id']}: success={success} ticks={step} e_pred={res['e_pred']} "
              f"e_track={res['e_track_deg']:.2f} reach={res.get('reach_err_cm')} "
              f"e_static_probe={res.get('e_pred_static')} wall={res['wall_s']:.0f}s", flush=True)
        idle_since = time.time()
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
