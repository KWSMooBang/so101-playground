#!/usr/bin/env python3
"""Real-arm ablation of the sim-to-real pieces: the joint alignment and the overhead camera pose.

Closed loop on the arm (blocks of the 6 T007 bin trials, one condition per block, blind labels):

    envs/lerobot/.venv/bin/python scripts/ablation.py plan lawam_a --policy lawam \
        --conditions C0 naive off:shoulder_lift=5 rot:4,0 --reps 3
    envs/lerobot/.venv/bin/python scripts/ablation.py run lawam_a      # next unfinished block; repeat
    envs/lerobot/.venv/bin/python scripts/ablation.py run lawam_a -- --policy_port 18005   # extra runner flags
    envs/lerobot/.venv/bin/python scripts/ablation.py status lawam_a
    envs/lerobot/.venv/bin/python scripts/ablation.py analyze lawam_a [--sim sim_results.jsonl]

Offline, no arm (phase 0):

    envs/lerobot/.venv/bin/python scripts/ablation.py jointmap         # naive vs aligned joint angles
    envs/lerobot/.venv/bin/python scripts/ablation.py sensitivity --policy internw0 --episodes 4

Conditions (src/so101_playground/ablation.py): C0, naive, off:<joint>=<deg>[+<joint>=<deg>],
rot:<pitch>,<yaw>[,<roll>].

**plan.** Writes outputs/real_robot/ablation/NAME/schedule.json. Each block runs every trial of --tasks once
under one condition. The block order follows the rows of a cyclic Latin square over the conditions (one row
per rep, condition order shuffled by --seed). Each block gets a random label, the only thing `run`, the
runner window and the results show. Do not open schedule.json before the analysis.

**run.** Starts scripts/policy_runner.py in episode mode for the next block, with:
- the block's condition;
- `--blind`;
- `--quit_when_scored`: the runner docks and quits once all 6 trials are scored.

In the window: stage each trial on the drawn outlines, d, space; y / f scores it (f needs a fail type). Results
go to NAME/<label>/results.jsonl, episodes to NAME/episodes/. The overhead reference for the drift check (3 px,
8 % brightness) is NAME/overhead_reference.jpg, shared by every block.

**analyze.** Per condition: success with a Wilson 95% CI, timeouts, grasp attempts and the worst tracking
error (from each episode's trajectory.npz). Against the baseline: an exact McNemar test on pairs matched by
(trial, rep), and a per-trial table. With --sim (JSONL rows {"trial_id", "success"}, e.g. T007 Phase A), the
baseline is also set against sim per trial.

**jointmap.** How far `naive` moves each joint from the aligned map. Covers our calibration ranges and the
states of logged episodes (outputs/real_robot/runs by default).

**sensitivity.** Open-loop sensitivity. It re-sends logged episodes' query frames and states to the policy
server through the tunnelled port, under each perturbation, and compares the returned action chunks with
an unperturbed pass. A second unperturbed pass is the noise floor (flow sampling). InternW0 keeps memory per
`episode`, so every pass is one fresh episode per logged episode. LaWAM takes no robot state, so state
offsets are skipped for it. The servers log these queries like real ones.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from so101_playground.ablation import (  # noqa: E402
    JOINTS, Condition, apply_joint_condition, mcnemar_exact, overhead_homography, parse_condition, wilson,
    warp_overhead,
)

ABLATION_ROOT = ROOT / "outputs" / "real_robot" / "ablation"
DEFAULT_TASKS = ROOT / "tasks" / "home_objects_bin.jsonl"
DEFAULT_RUNS = ROOT / "outputs" / "real_robot" / "runs"
CONTROL_HZ = 30.0


# ----- plan / status / run -----

def _schedule_path(name: str) -> Path:
    return ABLATION_ROOT / name / "schedule.json"


def _load_schedule(name: str) -> dict:
    path = _schedule_path(name)
    if not path.exists():
        raise SystemExit(f"no schedule {path} (run `plan {name}` first)")
    return json.loads(path.read_text())


def _trial_ids(tasks: Path) -> list[int]:
    return [json.loads(line)["trial_id"] for line in tasks.read_text().splitlines() if line.strip()]


def _block_results(root: Path, label: str) -> dict[int, dict]:
    """Latest result per trial in a block (the runner appends; re-scoring a trial appends again)."""
    path = root / label / "results.jsonl"
    latest = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["trial_id"]] = row
    return latest


def cmd_plan(args) -> None:
    path = _schedule_path(args.name)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists (--force replaces it; scored blocks keep their results by label).")
    conditions = [parse_condition(text) for text in args.conditions]
    names = [c.name for c in conditions]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate conditions in {names}")
    if args.baseline not in names:
        raise SystemExit(f"the baseline {args.baseline!r} is not among the conditions {names}")
    tasks = args.tasks.resolve()
    trials = _trial_ids(tasks)
    rng = random.Random(args.seed)
    order = list(range(len(conditions)))
    rng.shuffle(order)
    blocks = []
    for rep in range(args.reps):
        shift = rep % len(conditions)
        for position in range(len(conditions)):
            condition = conditions[order[(position + shift) % len(conditions)]]
            blocks.append({"block": len(blocks) + 1, "label": f"b{rng.getrandbits(24):06x}", "rep": rep,
                           "condition": condition.name, "runner_args": condition.runner_args()})
    schedule = {
        "name": args.name, "policy": args.policy, "tasks": str(tasks.relative_to(ROOT) if tasks.is_relative_to(ROOT) else tasks),
        "trial_ids": trials, "baseline": args.baseline, "conditions": names, "reps": args.reps, "seed": args.seed,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "blocks": blocks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schedule, indent=2) + "\n")
    minutes = len(blocks) * len(trials) * 1.5
    print(f"wrote {path}: {len(blocks)} blocks x {len(trials)} trials = {len(blocks) * len(trials)} episodes "
          f"(about {minutes / 60:.1f} h at 1.5 min each). Conditions are hidden from `run` and `status`.")


def cmd_status(args) -> None:
    schedule = _load_schedule(args.name)
    root = ABLATION_ROOT / args.name
    n = len(schedule["trial_ids"])
    for block in schedule["blocks"]:
        scored = len(set(_block_results(root, block["label"])) & set(schedule["trial_ids"]))
        reveal = f"  {block['condition']} (rep {block['rep']})" if args.reveal else ""
        print(f"block {block['block']:2d}  {block['label']}  {scored}/{n} scored{reveal}")


def cmd_run(args) -> None:
    schedule = _load_schedule(args.name)
    root = ABLATION_ROOT / args.name
    n = len(schedule["trial_ids"])
    blocks = schedule["blocks"]
    if args.block is not None:
        block = next((b for b in blocks if b["block"] == args.block), None)
        if block is None:
            raise SystemExit(f"no block {args.block} (1..{len(blocks)})")
    else:
        block = next((b for b in blocks if len(set(_block_results(root, b["label"])) & set(schedule["trial_ids"])) < n),
                     None)
        if block is None:
            raise SystemExit("every block is scored: run `analyze`.")
    scored = len(set(_block_results(root, block["label"])) & set(schedule["trial_ids"]))
    command = [
        sys.executable, str(ROOT / "scripts" / "policy_runner.py"),
        "--policy", schedule["policy"], "--episodes_jsonl", str(ROOT / schedule["tasks"]),
        "--results_root", str(root), "--policy_label", block["label"], "--output_root", str(root / "episodes"),
        "--quit_when_scored", "--blind", *block["runner_args"], *args.runner_args,
    ]
    print(f"block {block['block']} of {len(blocks)} (label {block['label']}): {n} trials, {scored} already scored")
    if args.dry_run:
        print("dry run; would start the runner with the block's (hidden) condition")
        return
    returncode = subprocess.call(command, cwd=ROOT)
    scored = len(set(_block_results(root, block["label"])) & set(schedule["trial_ids"]))
    print(f"block {block['block']}: {scored}/{n} scored (runner exit {returncode})")


# ----- analyze -----

def _episode_metrics(episode_dir: str | None) -> dict:
    """Grasp attempts (closing commands), duration and the worst body-joint tracking error of one episode."""
    if not episode_dir or not (Path(episode_dir) / "trajectory.npz").exists():
        return {}
    data = np.load(Path(episode_dir) / "trajectory.npz")
    command, state = data["command_sim_deg"], data["state_sim_deg"]
    if len(command) == 0:
        return {}
    closes, is_open = 0, command[0, 5] > 15.0
    for jaw in command[:, 5]:  # hysteresis: open above 15 deg, closed below 5 deg (jaw limit -12.75)
        if is_open and jaw < 5.0:
            closes, is_open = closes + 1, False
        elif not is_open and jaw > 15.0:
            is_open = True
    settled = slice(int(CONTROL_HZ * 0.5), None)
    error = np.abs(command[settled, :5] - state[settled, :5])
    return {"grasp_attempts": closes, "duration_s": len(command) / CONTROL_HZ,
            "max_tracking_error_deg": float(error.max()) if error.size else None}


def _load_sim(path: Path | None) -> dict[int, bool]:
    if path is None:
        return {}
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        return {int(k): bool(v) for k, v in data.items()}
    return {int(r["trial_id"]): bool(r["success"]) for r in map(json.loads, filter(str.strip, text.splitlines()))}


def cmd_analyze(args) -> None:
    schedule = _load_schedule(args.name)
    root = ABLATION_ROOT / args.name
    baseline = args.baseline or schedule["baseline"]
    rows = []
    for block in schedule["blocks"]:
        for trial, result in _block_results(root, block["label"]).items():
            if trial in schedule["trial_ids"]:
                rows.append({"condition": block["condition"], "rep": block["rep"], "block": block["block"],
                             "trial_id": trial, "success": bool(result["success"]),
                             "timed_out": bool(result.get("timed_out")), "fail_type": result.get("fail_type", []),
                             "episode_dir": result.get("episode_dir"), **_episode_metrics(result.get("episode_dir"))})
    if not rows:
        raise SystemExit("no scored episodes yet")
    by_key = {(r["condition"], r["trial_id"], r["rep"]): r for r in rows}
    summary = {"name": args.name, "policy": schedule["policy"], "baseline": baseline, "conditions": {}}
    lines = [f"# {args.name}: {schedule['policy']}, {len(rows)} scored episodes", "",
             "| condition | success | 95% CI | timeouts | grasp attempts | worst tracking (deg) | vs baseline: b / c, McNemar p |",
             "|---|---|---|---|---|---|---|"]
    for name in schedule["conditions"]:
        mine = [r for r in rows if r["condition"] == name]
        k, n = sum(r["success"] for r in mine), len(mine)
        lo, hi = wilson(k, n)
        attempts = [r["grasp_attempts"] for r in mine if "grasp_attempts" in r]
        tracking = [r["max_tracking_error_deg"] for r in mine if r.get("max_tracking_error_deg") is not None]
        entry = {"success": k, "n": n, "ci95": [lo, hi], "timeouts": sum(r["timed_out"] for r in mine),
                 "mean_grasp_attempts": float(np.mean(attempts)) if attempts else None,
                 "median_worst_tracking_deg": float(np.median(tracking)) if tracking else None}
        paired = ""
        if name != baseline:
            pairs = [(by_key[(baseline, t, rep)]["success"], r["success"]) for r in mine
                     for t, rep in [(r["trial_id"], r["rep"])] if (baseline, t, rep) in by_key]
            b = sum(base and not cond for base, cond in pairs)
            c = sum(cond and not base for base, cond in pairs)
            entry.update(pairs=len(pairs), baseline_only=b, condition_only=c, mcnemar_p=mcnemar_exact(b, c))
            paired = f"{b} / {c} of {len(pairs)}, p = {entry['mcnemar_p']:.3f}"
        summary["conditions"][name] = entry
        fmt = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
        lines.append(f"| {name} | {k}/{n} | {lo:.2f}-{hi:.2f} | {entry['timeouts']} | "
                     f"{fmt(entry['mean_grasp_attempts'], '.1f')} | {fmt(entry['median_worst_tracking_deg'], '.1f')} | {paired} |")
    lines += ["", "b = baseline succeeded and the condition failed; c = the reverse (pairs: same trial and rep).", "",
              "Per trial (successes / episodes):", "",
              "| trial | " + " | ".join(schedule["conditions"]) + (" | sim |" if args.sim else " |"),
              "|---" * (len(schedule["conditions"]) + 1 + bool(args.sim)) + "|"]
    sim = _load_sim(args.sim)
    for trial in schedule["trial_ids"]:
        cells = []
        for name in schedule["conditions"]:
            mine = [r for r in rows if r["condition"] == name and r["trial_id"] == trial]
            cells.append(f"{sum(r['success'] for r in mine)}/{len(mine)}" if mine else "-")
        sim_cell = (" | " + ("ok" if sim[trial] else "fail") if trial in sim else " | -") if args.sim else ""
        lines.append(f"| {trial} | " + " | ".join(cells) + sim_cell + " |")
    fails = {}
    for r in rows:
        for tag in r["fail_type"]:
            fails.setdefault(r["condition"], {}).setdefault(tag, 0)
            fails[r["condition"]][tag] += 1
    summary["fail_types"] = fails
    summary["episodes"] = rows
    lines += ["", "Fail types: " + "; ".join(f"{c}: " + ", ".join(f"{t} {n}" for t, n in sorted(v.items()))
                                             for c, v in fails.items()) if fails else ""]
    text = "\n".join(lines).rstrip() + "\n"
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "summary.md").write_text(text)
    print(text + f"\nwrote {root / 'summary.md'} and summary.json")


# ----- jointmap -----

def cmd_jointmap(args) -> None:
    from so101_playground.bench_units import DEFAULT_ALIGNMENT, lerobot_calibration, real_sim_alignment

    aligned = real_sim_alignment.RealSimAlignment.load(args.alignment or DEFAULT_ALIGNMENT)
    naive = real_sim_alignment.RealSimAlignment.load(args.alignment or DEFAULT_ALIGNMENT)
    apply_joint_condition(naive, parse_condition("naive"), lerobot_calibration.lerobot_position_to_sim_degrees)

    def diff(joint: str, sim_aligned: np.ndarray) -> np.ndarray:
        ticks = [aligned.sim_degrees_to_ticks(joint, float(d), clamp=False) for d in sim_aligned]
        return np.array([naive.ticks_to_sim_degrees(joint, t) for t in ticks]) - sim_aligned

    states = []
    for path in sorted(args.runs.glob("*/trajectory.npz")) + sorted(ABLATION_ROOT.glob("*/episodes/*/trajectory.npz")):
        states.append(np.load(path)["state_sim_deg"])
    states = np.concatenate(states) if states else np.zeros((0, 6))
    print(f"naive - aligned, sim degrees. Logged: {len(states)} states from {args.runs} and the ablation episodes.\n")
    print(f"{'joint':13s} {'over our calibration range':>28s} {'logged: mean':>13s} {'p95 |d|':>8s} {'max |d|':>8s}")
    out = {}
    for j, joint in enumerate(JOINTS):
        cal = aligned.motor_calibration[joint]
        sweep = np.array([aligned.ticks_to_sim_degrees(joint, t) for t in np.linspace(cal.range_min, cal.range_max, 200)])
        whole = diff(joint, sweep)
        logged = diff(joint, states[:, j]) if len(states) else np.zeros(0)
        out[joint] = {"range": [float(whole.min()), float(whole.max())]}
        line = f"{joint:13s} {whole.min():+12.2f} .. {whole.max():+6.2f} deg"
        if len(logged):
            out[joint].update(mean=float(logged.mean()), p95_abs=float(np.percentile(np.abs(logged), 95)),
                              max_abs=float(np.abs(logged).max()))
            line += f" {logged.mean():+13.2f} {np.percentile(np.abs(logged), 95):8.2f} {np.abs(logged).max():8.2f}"
        print(line)
    print("\nRough scale: 1 deg at shoulder_lift moves the gripper tip about 5 mm at a 30 cm reach.")
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n")


# ----- sensitivity -----

def _read_frames(path: Path, keep: set[int]) -> dict[int, np.ndarray]:
    """The RGB frames at the `keep` indices (a whole episode would not fit in memory at 640x480)."""
    import cv2

    capture, frames, index = cv2.VideoCapture(str(path)), {}, 0
    while index <= max(keep, default=-1):
        ok, frame = capture.read()
        if not ok:
            break
        if index in keep:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    return frames


def _sensitivity_conditions(args) -> list[tuple[str, tuple[float, float, float], dict[str, float]]]:
    conditions = [("ref", (0.0, 0.0, 0.0), {})]
    conditions += [(f"ref{k + 2}", (0.0, 0.0, 0.0), {}) for k in range(args.repeats)]
    axes = {"pitch": 0, "yaw": 1, "roll": 2}
    for spec in args.rot:
        axis, _, value = spec.partition(":")
        rotation = [0.0, 0.0, 0.0]
        rotation[axes[axis]] = float(value)
        conditions.append((f"rot {spec}", tuple(rotation), {}))
    for spec in args.state_offset:
        joint, _, value = spec.partition(":")
        if joint not in JOINTS[:5]:
            raise SystemExit(f"state offset on {joint!r}: body joints only")
        conditions.append((f"state {spec}", (0.0, 0.0, 0.0), {joint: float(value)}))
    return conditions


def cmd_sensitivity(args) -> None:
    import types

    import torch

    import policy_runner

    runner = policy_runner._load_runner()
    policy_runner._patch(runner)
    defaults = runner.POLICY_DEFAULTS[args.policy]
    runner_args = runner._parser().parse_args(["--policy", args.policy, "--policy_host", args.host])
    for key, value in defaults.items():
        if hasattr(runner_args, key) and getattr(runner_args, key) is None:
            setattr(runner_args, key, value)
    if args.port:
        runner_args.policy_port = args.port
    client = runner.Runner._make_policy(types.SimpleNamespace(args=runner_args, instruction=""))
    client.connect()

    episodes = [p.parent for p in sorted(args.runs.glob("*/trajectory.npz"))
                if (p.parent / "overhead.mp4").exists() and (p.parent / "wrist.mp4").exists()]
    episodes = [e for e in episodes if np.load(e / "trajectory.npz")["policy_query"].any()][-args.episodes:]
    if not episodes:
        raise SystemExit(f"no logged episodes with policy queries under {args.runs}")
    conditions = _sensitivity_conditions(args)
    if args.policy == "lawam" and any(offsets for _, _, offsets in conditions):
        print("LaWAM takes no robot state: skipping the state offsets.")
        conditions = [c for c in conditions if not c[2]]
    loaded = []
    for episode in episodes:
        data = np.load(episode / "trajectory.npz")
        meta = json.loads((episode / "episode.json").read_text())
        wanted = {int(i) for i in np.flatnonzero(data["policy_query"])}
        overhead, wrist = _read_frames(episode / "overhead.mp4", wanted), _read_frames(episode / "wrist.mp4", wanted)
        queries = sorted(i for i in wanted if i in overhead and i in wrist)
        loaded.append({"dir": episode, "instruction": meta["instructions"][0][1], "state": data["state_sim_deg"],
                       "overhead": overhead, "wrist": wrist, "queries": queries})
    total = sum(len(e["queries"]) for e in loaded) * len(conditions)
    print(f"{len(loaded)} episodes, {total} queries to {args.policy} at {args.host}:{runner_args.policy_port} "
          f"(about {total * 0.6 / 60:.0f} min at 0.6 s each)")

    homographies = {rot: overhead_homography(rot, *runner.FRAME_SIZE) for _, rot, _ in conditions if any(rot)}
    horizon = runner_args.action_horizon
    chunks: dict[str, list[list[np.ndarray]]] = {}
    episode_id = int(time.time() * 1000)
    for name, rotation, offsets in conditions:
        chunks[name] = []
        for episode in loaded:
            episode_id += 1
            client.trace_episode = episode_id  # a fresh server-side episode (InternW0 memory) per pass
            client.set_language_instruction(episode["instruction"])
            per_query = []
            for i in episode["queries"]:
                state = episode["state"][i].copy()
                for joint, value in offsets.items():
                    state[JOINTS.index(joint)] += value
                overhead = episode["overhead"][i]
                if any(rotation):
                    overhead = warp_overhead(overhead, homographies[rotation])
                visual = {"rgb_overhead": overhead, "rgb_wrist": episode["wrist"][i]}
                with contextlib.redirect_stdout(io.StringIO()):
                    chunk = client._query_server(torch.tensor(np.radians(state), dtype=torch.float32), visual)
                sim = client.mapper.model_positions_to_sim_radians(torch.as_tensor(chunk[:horizon], dtype=torch.float32))
                per_query.append(np.degrees(sim.cpu().numpy().astype(np.float64)))
            chunks[name].append(per_query)
        print(f"  {name}: done", flush=True)

    def deviation(name: str) -> tuple[float, float, float]:
        body, worst, jaw = [], [], []
        for ref_episode, episode in zip(chunks["ref"], chunks[name]):
            for ref_chunk, chunk in zip(ref_episode, episode):
                rows = min(len(ref_chunk), len(chunk))
                d = np.abs(chunk[:rows] - ref_chunk[:rows]).mean(axis=0)
                body.append(d[:5].mean())
                worst.append(d[:5].max())
                jaw.append(d[5])
        return float(np.mean(body)), float(np.mean(worst)), float(np.mean(jaw))

    floor = [deviation(name) for name, _, _ in conditions if name.startswith("ref") and name != "ref"]
    floor_body = float(np.mean([f[0] for f in floor])) if floor else float("nan")
    table = {}
    print(f"\nMean |chunk - ref| over the executed {horizon} actions, sim degrees "
          f"(noise floor = repeated unperturbed passes):\n")
    print(f"{'condition':22s} {'body mean':>10s} {'worst joint':>12s} {'gripper':>8s} {'x floor':>8s}")
    for name, _, _ in conditions:
        if name == "ref":
            continue
        body, worst, jaw = deviation(name)
        table[name] = {"body_mean_deg": body, "worst_joint_deg": worst, "gripper_deg": jaw,
                       "x_noise_floor": body / floor_body if floor_body > 0 else None}
        ratio = f"{body / floor_body:8.1f}" if floor_body > 0 else "       -"
        print(f"{name:22s} {body:10.2f} {worst:12.2f} {jaw:8.2f} {ratio}")
    out = args.out or ABLATION_ROOT / f"sensitivity_{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": args.policy, "episodes": [str(e["dir"]) for e in loaded],
                               "action_horizon": horizon, "table": table}, indent=2) + "\n")
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="write a blinded, counterbalanced block schedule")
    p.add_argument("name")
    p.add_argument("--policy", required=True, choices=("internw0", "lawam"))
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--baseline", default="C0")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("status", help="scored trials per block")
    p.add_argument("name")
    p.add_argument("--reveal", action="store_true", help="show each block's condition (unblinds)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("run", help="run the next unfinished block on the arm")
    p.add_argument("name")
    p.add_argument("--block", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.set_defaults(func=cmd_run)  # extra policy_runner.py flags go after a lone `--`

    p = sub.add_parser("analyze", help="success, CIs, paired tests, per-trial table")
    p.add_argument("name")
    p.add_argument("--baseline", default=None)
    p.add_argument("--sim", type=Path, default=None, help="JSONL {trial_id, success} (or JSON {id: bool}) from sim")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("jointmap", help="naive vs aligned joint angles (offline)")
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--alignment", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_jointmap)

    p = sub.add_parser("sensitivity", help="open-loop action change under perturbations (needs the server)")
    p.add_argument("--policy", required=True, choices=("internw0", "lawam"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--episodes", type=int, default=4, help="the most recent N logged episodes")
    p.add_argument("--repeats", type=int, default=1, help="extra unperturbed passes for the noise floor")
    p.add_argument("--rot", nargs="*", default=["pitch:2", "pitch:4", "pitch:8", "yaw:2", "yaw:4", "yaw:8"])
    p.add_argument("--state_offset", nargs="*", default=["shoulder_lift:2", "shoulder_lift:5", "shoulder_lift:10"])
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_sensitivity)

    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    args.runner_args = argv[split + 1:]
    args.func(args)


if __name__ == "__main__":
    main()
