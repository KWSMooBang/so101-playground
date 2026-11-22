#!/usr/bin/env python3
"""T007: task + layout set from the objects the human has at home (sim2real pairs).

30 episodes, canonical appearance only (no domain randomization):
  * bin, 1 object: one per object (6)
  * move: 8 of the 6 objects x 4 directions
  * next_to: 8 ordered (target, referent) pairs
  * between: 8 (target, referent, referent) triples
Spatial scenes hold 4 objects, as every spatial episode of the bench val set does; the other objects are
drawn from the same 6 home objects, so the real scene needs nothing the human does not have.

Every target is the target of 4 spatial episodes in total (3 families + one extra); the 6 extra targets are
spread one per object over the families, half sim-seen and half real-only per family. Move directions are
balanced (2 each). Instructions come from the bench template (``task_generation.instruction_for_row``),
so a pen keeps its colour only when both pens are in the scene.

Layouts come from the bench's footprint-aware sampler (``layouts.generate_episode_layout``) with the bin at
its canonical pose 0 for all 30 episodes (spatial families always use pose 0 in the bench; fixing it for bin
too means the human never moves the bin). Move layouts additionally need a clear 7 cm corridor in the move
direction inside TABLE_BOUNDS (the bench success needs 2 in = 5.1 cm of progress); a layout failing that is
resampled. The bench validators (``episode_spec_from_json``, ``row_with_canonical_instruction_metadata``,
``validate_move_episode_footprints``) run on every row.

Each layout row also carries the robot-base-frame poses (``robot_base_frame``): the bench robot root
(ROBOT_BASE_TRANSLATION, ROBOT_BASE_YAW_DEG) and, for the human, coordinates from the shoulder-pan axis.

    source <scaffold-env>/envs/isaac/env.sh   # numpy + SO101_BENCH_ROOT; no Isaac app is started
    python scripts/t007_make_home_objects.py            # writes tasks/home_objects.jsonl + layouts
    python scripts/t007_make_home_objects.py --check    # re-validate the saved files
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np

BENCH = Path(os.environ["SO101_BENCH_ROOT"])
sys.path.insert(0, str(BENCH / "scripts"))
from task_generation import env_cfg_constants, instruction_for_row, load_modules  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
TASKS = HERE / "tasks/home_objects.jsonl"
LAYOUTS = HERE / "tasks/layouts/home_objects_layouts.jsonl"
SEED = 20261122
GENERATED_AT = "2026-11-22T00:00:00+00:00"

SIM_SEEN = ("black pen", "black remote", "blue paintbrush")
REAL_ONLY = ("red pen", "purple toothbrush", "razor")
OBJECTS = SIM_SEEN + REAL_ONLY
HOME_NAME = {"black pen": "black pen", "red pen": "red pen", "purple toothbrush": "toothbrush",
             "razor": "razor", "black remote": "remote", "blue paintbrush": "paintbrush"}
DIRECTIONS = ("left", "right", "forward", "backward")
# Robot-centric directions as scored by the bench (mdp/terminations.py:_direction_vectors), world frame.
DIRECTION_XY = {"left": (1.0, 0.0), "right": (-1.0, 0.0), "forward": (0.0, -1.0), "backward": (0.0, 1.0)}
MOVE_CORRIDOR_M = 0.07
SPATIAL = ("move", "next_to", "between")
N_SPATIAL = 8

# Bench robot root (so101_bench_env_cfg.py ROBOT_BASE_TRANSLATION / ROBOT_BASE_YAW_DEG) and the shoulder-pan
# joint origin in the root frame (SO-ARM101-USD.usd, joint "Rotation"; scripts/t007_asset_dims.py).
ROBOT_BASE_XY = (0.05209, 0.18061)
ROBOT_BASE_YAW_DEG = 6.195
PAN_AXIS_ROOT_XY = (0.0208, -0.0231)
TABLE_BOUNDS = {"x": (-0.14, 0.25), "y": (-0.10, 0.155)}


def group(name: str) -> str:
    return "sim_seen" if name in SIM_SEEN else "real_only"


# ------------------------------------------------------------------ frames
def world_to_base(xy, yaw=None):
    c, s = math.cos(math.radians(ROBOT_BASE_YAW_DEG)), math.sin(math.radians(ROBOT_BASE_YAW_DEG))
    dx, dy = xy[0] - ROBOT_BASE_XY[0], xy[1] - ROBOT_BASE_XY[1]
    p = (c * dx + s * dy, -s * dx + c * dy)
    return p, (None if yaw is None else yaw - math.radians(ROBOT_BASE_YAW_DEG))


def base_to_pan(p_base):
    """Human coordinates (cm) from the shoulder-pan axis: fwd = away from the robot (root -y),
    side = robot's left (root +x; your right when you face the robot from the front of the table)."""
    return (100 * -(p_base[1] - PAN_AXIS_ROOT_XY[1]), 100 * (p_base[0] - PAN_AXIS_ROOT_XY[0]))


def pose_record(xy, yaw):
    p, yb = world_to_base(xy, yaw)
    fwd, side = base_to_pan(p)
    # Heading of the asset's local +x axis on the table, degrees CCW (seen from above) from the robot's
    # forward direction (root -y).
    heading = (math.degrees(yb) - (-90.0) + 180.0) % 360.0 - 180.0
    return {"world_xy_m": [round(xy[0], 5), round(xy[1], 5)], "world_yaw_deg": round(math.degrees(yaw), 2),
            "base_xy_m": [round(p[0], 5), round(p[1], 5)], "base_yaw_deg": round(math.degrees(yb), 2),
            "pan_fwd_cm": round(fwd, 1), "pan_side_cm": round(side, 1), "heading_from_fwd_deg": round(heading, 1)}


# ------------------------------------------------------------------ plan
def make_rows(benchmark, attempt: int):
    rng = random.Random(f"{SEED}:plan:{attempt}")
    colors = set(benchmark.COLORS)
    # One extra spatial target per object, each family gets one sim-seen and one real-only extra.
    seen, real = list(SIM_SEEN), list(REAL_ONLY)
    rng.shuffle(seen)
    rng.shuffle(real)
    extras = {fam: [seen[i], real[i]] for i, fam in enumerate(SPATIAL)}
    in_scene = Counter()
    rows = []
    for name in OBJECTS:
        rows.append(dict(family="bin", objects=[name], n_objects=1, target=name))
    used_pairs, used_triples = set(), set()
    dir_order = list(DIRECTIONS) * 2
    rng.shuffle(dir_order)
    for fam in SPATIAL:
        targets = list(OBJECTS) + extras[fam]
        rng.shuffle(targets)
        as_ref = Counter()
        for k, target in enumerate(targets):
            others = sorted((o for o in OBJECTS if o != target), key=lambda o: (as_ref[o], in_scene[o], rng.random()))
            row = dict(family=fam, target=target)
            if fam == "move":
                row["direction"] = dir_order[k]
                refs = []
            elif fam == "next_to":
                refs = next([o] for o in others if (target, o) not in used_pairs)
                used_pairs.add((target, refs[0]))
            else:
                refs = next(list(p) for p in ((a, b) for i, a in enumerate(others) for b in others[i + 1:])
                            if (target, frozenset(p)) not in used_triples)
                used_triples.add((target, frozenset(refs)))
            as_ref.update(refs)
            rest = [o for o in sorted(OBJECTS, key=lambda o: (in_scene[o], rng.random()))
                    if o != target and o not in refs][: 3 - len(refs)]
            objs = [target, *refs, *rest]
            rng.shuffle(objs)
            row.update(objects=objs, n_objects=4)
            if refs:
                row["referents"] = refs
            in_scene.update(objs)
            rows.append(row)
    out = []
    for i, r in enumerate(rows):
        fam = r.pop("family")
        rec = dict(ood_key=benchmark_split(benchmark, r["target"]), appearance_profile="canonical", repeat_index=0,
                   objects=r["objects"], n_objects=r["n_objects"])
        if fam != "bin":
            rec["target"] = r["target"]
        for key in ("referents", "direction"):
            if key in r:
                rec[key] = r[key]
        rec["instruction"] = instruction_for_row(rec, fam, colors)
        rec["home_family"] = fam
        rec["home_target"] = r["target"]
        rec["home_target_group"] = group(r["target"])
        rec["trial_id"] = i
        out.append(rec)
    return out


def benchmark_split(benchmark, name):
    return next(s for s, objs in benchmark.OBJECT_SPLITS.items() if name in objs)


def validate(rows, layout_rows, benchmark):
    specs = []
    fam_count, dirs = Counter(), Counter()
    targets = {f: Counter() for f in SPATIAL}
    for i, row in enumerate(rows):
        assert row["trial_id"] == i
        assert row["appearance_profile"] == "canonical" and "domain_randomization" not in row
        assert set(row["objects"]) <= set(OBJECTS) and len(set(row["objects"])) == len(row["objects"])
        spec = benchmark.episode_spec_from_json(row)
        specs.append(spec)
        fam = row["home_family"]
        assert spec.task_family == fam, (i, spec.task_family, fam)
        normalized = benchmark.row_with_canonical_instruction_metadata(row)
        for field in ("target", "referents", "direction"):
            assert normalized.get(field) == row.get(field), (i, field, normalized.get(field), row.get(field))
        fam_count[fam] += 1
        if fam == "move":
            dirs[row["direction"]] += 1
        if fam in SPATIAL:
            targets[fam][row["target"]] += 1
    assert fam_count == Counter(bin=6, move=8, next_to=8, between=8), fam_count
    assert set(dirs.values()) == {2}, dirs
    assert sorted(r["objects"][0] for r in rows if r["home_family"] == "bin") == sorted(OBJECTS)
    total = Counter()
    for fam in SPATIAL:
        assert set(targets[fam]) == set(OBJECTS), (fam, targets[fam])
        assert sum(targets[fam][o] for o in SIM_SEEN) == 4, (fam, targets[fam])
        total.update(targets[fam])
    assert set(total.values()) == {4}, total
    nt = [(r["target"], r["referents"][0]) for r in rows if r["home_family"] == "next_to"]
    assert len(nt) == len(set(nt))
    bt = [(r["target"], frozenset(r["referents"])) for r in rows if r["home_family"] == "between"]
    assert len(bt) == len(set(bt))
    benchmark.validate_move_episode_footprints(specs)
    if layout_rows is not None:
        assert len(layout_rows) == len(rows)
        for i, (row, lay) in enumerate(zip(rows, layout_rows)):
            assert lay["trial_id"] == lay["episode_index"] == i
            assert lay["instruction"] == row["instruction"] and lay["task_family"] == row["home_family"]
            assert [o["name"] for o in lay["objects"]] == row["objects"]
            assert lay["bin"]["pose_index"] == 0
            check_layout_geometry(lay, row)
    return dict(episodes=len(rows), families=dict(fam_count), move_directions=dict(dirs),
                spatial_targets={f: dict(c) for f, c in targets.items()},
                objects_in_scenes=dict(Counter(o for r in rows for o in r["objects"])))


# ------------------------------------------------------------------ geometry checks
def footprint_points(boxes, xy, yaw, step=0.004):
    """Sample points over the union of local footprint boxes, placed in the world."""
    pts = []
    for x0, y0, x1, y1 in boxes:
        xs = np.arange(x0, x1 + 1e-9, step) if x1 - x0 > step else np.array([(x0 + x1) / 2])
        ys = np.arange(y0, y1 + 1e-9, step) if y1 - y0 > step else np.array([(y0 + y1) / 2])
        g = np.stack(np.meshgrid(np.append(xs, x1), np.append(ys, y1)), -1).reshape(-1, 2)
        pts.append(g)
    p = np.concatenate(pts)
    c, s = math.cos(yaw), math.sin(yaw)
    return p @ np.array([[c, s], [-s, c]]) + np.asarray(xy)


def inside_boxes(points, boxes, xy, yaw, margin=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    local = (points - np.asarray(xy)) @ np.array([[c, -s], [s, c]])
    hit = np.zeros(len(points), bool)
    for x0, y0, x1, y1 in boxes:
        hit |= ((local[:, 0] >= x0 - margin) & (local[:, 0] <= x1 + margin)
                & (local[:, 1] >= y0 - margin) & (local[:, 1] <= y1 + margin))
    return hit


def bin_yaw(lay):
    b = lay["bin"]
    return b["yaw"] if "yaw" in b else b["rpy"][2]


def bin_boxes(lay):
    hx, hy = lay["bin"]["footprint_half_extents"]
    ox, oy = lay["bin"].get("footprint_center_offset", [0.0, 0.0])
    return [[ox - hx, oy - hy, ox + hx, oy + hy]]


def check_layout_geometry(lay, row):
    """Move corridor: the target swept MOVE_CORRIDOR_M in the move direction stays on the table and clear of
    the other objects and the bin."""
    if row["home_family"] != "move":
        return
    objs = {o["name"]: o for o in lay["objects"]}
    t = objs[row["target"]]
    dx, dy = DIRECTION_XY[row["direction"]]
    base = footprint_points(t["footprint_boxes"], t["position"][:2], t["yaw"])
    for frac in np.linspace(0.1, 1.0, 10):
        p = base + frac * MOVE_CORRIDOR_M * np.array([dx, dy])
        assert (p[:, 0] > TABLE_BOUNDS["x"][0]).all() and (p[:, 0] < TABLE_BOUNDS["x"][1]).all() and \
            (p[:, 1] > TABLE_BOUNDS["y"][0]).all() and (p[:, 1] < TABLE_BOUNDS["y"][1]).all(), \
            f"trial {lay['trial_id']}: move corridor leaves the table"
        for name, o in objs.items():
            if name != row["target"]:
                assert not inside_boxes(p, o["footprint_boxes"], o["position"][:2], o["yaw"], 0.005).any(), \
                    f"trial {lay['trial_id']}: move corridor hits {name}"
        assert not inside_boxes(p, bin_boxes(lay), lay["bin"]["position"][:2], bin_yaw(lay), 0.005).any(), \
            f"trial {lay['trial_id']}: move corridor hits the bin"


def add_base_frame(lay):
    lay["robot_base_frame"] = {
        "note": "world -> bench robot root: p_base = Rz(-yaw_robot)(p - t_robot); pan_* are cm from the "
                "shoulder-pan axis, fwd = away from the robot (root -y), side = robot's left (root +x); "
                "heading = asset local +x, deg CCW from fwd (top view)",
        "robot_root_world_xy_m": list(ROBOT_BASE_XY), "robot_root_world_yaw_deg": ROBOT_BASE_YAW_DEG,
        "pan_axis_root_xy_m": list(PAN_AXIS_ROOT_XY),
        "bin": pose_record(lay["bin"]["position"][:2], bin_yaw(lay)),
        "objects": {o["name"]: pose_record(o["position"][:2], o["yaw"]) for o in lay["objects"]},
    }
    return lay


def make_layouts(rows, benchmark, layouts):
    k = env_cfg_constants()
    pose0 = (k["BIN_RANDOM_POSES"][0],)
    region0 = [k["VALID_OBJECT_SPAWN_REGIONS"][0]]
    history, out = [], []
    for i, row in enumerate(rows):
        spec = benchmark.episode_spec_from_json(row)
        for trial in range(200):
            lay = layouts.generate_episode_layout(
                spec, episode_index=i, rng=random.Random(f"{SEED}:layout:{i}:{trial}"),
                bin_random_poses=pose0, valid_spawn_regions=region0, table_object_z=k["TABLE_OBJECT_Z"],
                seed=SEED, generated_at=GENERATED_AT, robot_bounding_box=k["SO101_BOUNDING_BOX"],
                placement_history=list(history))
            lay["bin"]["pose_index"] = 0
            try:
                check_layout_geometry(lay, row)
            except AssertionError as exc:
                print(f"[layout {i} try {trial}] rejected: {exc}")
                continue
            lay["placement"]["home_resample_trials"] = trial
            history.extend(tuple(o["position"][:2]) for o in lay["objects"])
            out.append(add_base_frame(lay))
            break
        else:
            raise RuntimeError(f"no valid layout for row {i}")
    return out


def encode(rows):
    return "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cwd = os.getcwd()
    os.chdir(BENCH)  # load_modules / footprint paths are bench-relative
    benchmark, layouts = load_modules()
    os.chdir(cwd)
    if args.check:
        rows = [json.loads(l) for l in TASKS.read_text().splitlines()]
        lays = [json.loads(l) for l in LAYOUTS.read_text().splitlines()]
    else:
        if TASKS.exists() or LAYOUTS.exists():
            raise FileExistsError("home_objects files exist; use --check (delete them to regenerate)")
        for attempt in range(200):
            try:
                rows = make_rows(benchmark, attempt)
                validate(rows, None, benchmark)
                break
            except (AssertionError, StopIteration) as exc:
                print(f"[plan attempt {attempt}] rejected: {exc!r}")
        else:
            raise RuntimeError("no valid plan")
        lays = make_layouts(rows, benchmark, layouts)
        TASKS.parent.mkdir(parents=True, exist_ok=True)
        LAYOUTS.parent.mkdir(parents=True, exist_ok=True)
        TASKS.write_text(encode(rows))
        LAYOUTS.write_text(encode(lays))
    print(json.dumps(validate(rows, lays, benchmark), indent=1))


if __name__ == "__main__":
    main()
