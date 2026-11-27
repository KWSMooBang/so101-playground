"""T012: reach error at the first commanded jaw close (task-relevant perception error).

Event: the first policy tick at which the commanded jaw (the policy's clamped command, sim joint ``Jaw``, degrees)
goes below CLOSE_BELOW_DEG (-8) after having been above OPEN_ABOVE_DEG (0) earlier in the episode. InternW0
opens the jaw only 4-14 deg for thin objects, so a plain sign change would fire on small corrections; -8 deg is
past the point where the pads meet on a thin object (pad gap 4.6 mm, ``so101_kinematics._JAW_GAP_TABLE``).
That rule never fires on the thick black remote (4.5 cm): on all 40 T010 remote successes InternW0 opens to ~38 deg
and closes only to ~7 deg (it commands the object's width, not a squeeze). A second, relative rule covers that:
after an opening of at least REL_MIN_OPEN_DEG (20), the first tick at least REL_DROP_DEG (20) below the running
maximum. The event is the earlier of the two; ``kind`` says which fired ("abs" = the T012 brief's rule, "rel").
Thin objects never open 20 deg, so on them only the brief's rule can fire. Episodes with no close (e.g. the black
pen dithering at 1-10 deg until the timeout) have no reach error and are counted separately.

Reach error: XY distance (cm) between the gripper TCP (``so101_kinematics.tool_pose(...).tcp``, 8 mm off the fixed
finger, 12 mm above its tip) and the target object's centre, at that tick. The TCP comes from the measured joint
angles q(t+1) after the tick (where the gripper actually is when it closes); the TCP of the commanded pose is
logged too. The object centre is the centre of its collision-mesh box (docs/home_objects/asset_dims.json) placed
at the object's root pose; online it is the live root pose at the close tick (the gripper may already have pushed
the object), offline (recorded T009/T010 trajectories, which kept no object poses) the layout pose + jitter.

No Isaac import: ``so101_kinematics`` is loaded by path (as the bench intends for geometry tools).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

OPEN_ABOVE_DEG = 0.0
CLOSE_BELOW_DEG = -8.0
REL_MIN_OPEN_DEG = 20.0
REL_DROP_DEG = 20.0
REPO = Path(__file__).resolve().parents[2]
ASSET_DIMS = REPO / "docs/home_objects/asset_dims.json"
_DEFAULT_BENCH = Path("/workspace/so101-playground/.claude/worktrees/scaffold-env/third_party/so101_bench")
# Bench robot root pose (so101_bench_env_cfg ROBOT_BASE_TRANSLATION / ROBOT_BASE_YAW_DEG). The kinematics' "base"
# body frame is the articulation root (the Rotation joint's localPos0 equals its position in the root frame,
# asset_dims.json robot_joints_root_frame). Used offline only; the worker reads the live body pose.
ROBOT_BASE_TRANSLATION = (0.05209, 0.18061, -0.03102)
ROBOT_BASE_YAW_DEG = 6.195

_KIN = None
_DIMS = None


def kinematics():
    """so101_bench.utils.so101_kinematics, loaded by file path (no Isaac Lab import)."""
    global _KIN
    if _KIN is None:
        root = Path(os.environ.get("SO101_BENCH_ROOT", _DEFAULT_BENCH))
        path = root / "source/so101_bench/so101_bench/utils/so101_kinematics.py"
        spec = importlib.util.spec_from_file_location("_t012_so101_kinematics", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclasses look the module up while the file executes
        spec.loader.exec_module(mod)
        _KIN = mod
    return _KIN


def first_close_tick(jaw_cmd_deg, open_above: float = OPEN_ABOVE_DEG, close_below: float = CLOSE_BELOW_DEG):
    """The brief's rule alone: index of the first sample < close_below after a sample > open_above, or None."""
    armed = False
    for t, j in enumerate(np.asarray(jaw_cmd_deg, dtype=float)):
        if j > open_above:
            armed = True
        elif armed and j < close_below:
            return t
    return None


class CloseDetector:
    """Online detector, one call per policy tick with the commanded jaw (deg); returns the event kind once."""

    def __init__(self):
        self.armed, self.fired, self.max_open, self.t = False, None, -math.inf, 0

    def __call__(self, jaw_cmd_deg: float) -> str | None:
        hit = None
        j = float(jaw_cmd_deg)
        if self.fired is None:
            if j > OPEN_ABOVE_DEG:
                self.armed = True
            elif self.armed and j < CLOSE_BELOW_DEG:
                hit = "abs"
            if hit is None and self.max_open >= REL_MIN_OPEN_DEG and j <= self.max_open - REL_DROP_DEG:
                hit = "rel"
            self.max_open = max(self.max_open, j)
            self.fired = hit
        self.t += 1
        return hit


def detect_close(jaw_cmd_deg):
    """(tick, kind) of the first close event ("abs" | "rel"), or (None, None)."""
    det = CloseDetector()
    for t, j in enumerate(np.asarray(jaw_cmd_deg, dtype=float)):
        kind = det(j)
        if kind:
            return t, kind
    return None, None


def world_from_pose(pos, quat_wxyz=None, yaw_deg: float | None = None) -> np.ndarray:
    m = np.eye(4)
    if quat_wxyz is not None:
        m[:3, :3] = kinematics()._quat_wxyz_to_matrix(quat_wxyz)
    elif yaw_deg is not None:
        m[:3, :3] = kinematics()._rot_z(math.radians(yaw_deg))
    m[:3, 3] = pos
    return m


def default_world_from_base() -> np.ndarray:
    return world_from_pose(ROBOT_BASE_TRANSLATION, yaw_deg=ROBOT_BASE_YAW_DEG)


def tcp(world_from_base: np.ndarray, arm_q_rad) -> np.ndarray:
    return np.asarray(kinematics().tool_pose(world_from_base, np.asarray(arm_q_rad, dtype=float)[:5]).tcp)


def asset_dims() -> dict:
    global _DIMS
    if _DIMS is None:
        _DIMS = json.loads(ASSET_DIMS.read_text())["objects"]
    return _DIMS


def centre_offset_root(name: str) -> np.ndarray:
    """Collision-box centre in the object's root frame (m, xyz); zeros for an unknown asset."""
    d = asset_dims().get(name)
    if d is None:
        return np.zeros(3)
    return (np.asarray(d["min_m"]) + np.asarray(d["max_m"])) / 2.0


def object_centre(name: str, pos, quat_wxyz=None, yaw_rad: float | None = None) -> np.ndarray:
    if quat_wxyz is not None:
        r = kinematics()._quat_wxyz_to_matrix(quat_wxyz)
    else:
        r = kinematics()._rot_z(float(yaw_rad or 0.0))
    return np.asarray(pos, dtype=float) + r @ centre_offset_root(name)


def xy_cm(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]) * 100.0)


def layout_target(layout: dict, target_name: str, jitter=None):
    """(position, yaw rad) of the named object in a layout row after the coordinator's jitter (as t009_worker)."""
    for e in layout["objects"]:
        if e.get("name") == target_name:
            pos = list(map(float, e["position"]))
            yaw = float(e.get("yaw", (e.get("rpy") or [0, 0, 0])[2]))
            if jitter:
                pos[0] += float(jitter[0])
                pos[1] += float(jitter[1])
                yaw += math.radians(float(jitter[2]))
            return pos, yaw
    raise KeyError(target_name)


def reach_error_offline(npz, layout: dict, target_name: str, jitter=None, world_from_base=None) -> dict:
    """Reach error of a recorded T009/T010/T012 episode (``episodes/<id>.npz``: cmd, q per policy tick, rad).

    The object is assumed still at its (jittered) layout pose at the close tick; T012 episodes also record the
    live pose, so the two can be compared on them."""
    d = np.load(npz) if not isinstance(npz, dict) else npz
    jaw = np.degrees(np.asarray(d["cmd"])[:, 5])
    t, kind = detect_close(jaw)
    out = {"close_tick": t, "close_kind": kind, "reach_err_cm": None, "reach_err_cmd_cm": None}
    if t is None:
        return out
    wfb = default_world_from_base() if world_from_base is None else world_from_base
    pos, yaw = layout_target(layout, target_name, jitter)
    c = object_centre(target_name, pos, yaw_rad=yaw)
    out["reach_err_cm"] = xy_cm(tcp(wfb, np.asarray(d["q"])[t]), c)
    out["reach_err_cmd_cm"] = xy_cm(tcp(wfb, np.asarray(d["cmd"])[t]), c)
    out["object_centre"] = c.tolist()
    return out
