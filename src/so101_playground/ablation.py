"""Inference-time perturbations for the real-arm ablation (scripts/ablation.py, scripts/policy_runner.py).

A condition string names one perturbation of what the policy sees or how its joint values reach the motors:

    C0                                   none (aligned joints, camera as mounted)
    naive                                no sim alignment: our LeRobot `.pos` read as the bench's `.pos`
    off:shoulder_lift=5[+elbow_flex=-3]  the alignment's zero shifted by these sim degrees (body joints only)
    rot:<pitch>,<yaw>[,<roll>]           the policy's overhead image re-rendered as if the camera were
                                         rotated by these degrees (positive pitch tilts it down, positive
                                         yaw turns it right, positive roll rolls it clockwise seen from
                                         behind)

Joint conditions replace `RealSimAlignment.ticks_to_sim_degrees` / `sim_degrees_to_ticks` on the alignment
instance the runner uses, so the arm's state reading, its goals, the dock pose and the gripper floor all go
through the perturbed map, consistently in both directions.

- `naive` is the LeRobot-only pipeline. Ticks become our `.pos` through our calibration ranges, and the
  bench reads that `.pos` as its own (`lerobot_position_to_sim_degrees`). Both steps are linear, so the map
  is a line per joint. It is fitted on two interior points, which keeps it clear of the bench's clamps,
  and extended linearly. Goals clamp to the calibration range as in the aligned map.
- `off` makes the policy's angle q land at physical q + delta, and reads physical p as p - delta. It is an
  alignment measured delta degrees wrong. The gripper is excluded (its floor protects the printed jaws),
  and offsets are capped at 10 deg.

`rot` is a homography H = K R^T K^-1 with the bench's overhead intrinsics (74 deg horizontal FOV). That is
exact for a camera rotating about its optical centre, but it cannot model a translated camera (parallax).
The uncovered border is black, an artifact a real rotated camera would not show. Only the frames sent to
the policy are warped: the window, the drift check and the recorded videos stay raw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BODY_JOINTS = JOINTS[:5]
MAX_OFFSET_DEG = 10.0
OVERHEAD_HFOV_DEG = 74.0  # the bench's canonical overhead camera (scripts/overhead_align.py)


@dataclass
class Condition:
    name: str
    joint_map: str = "aligned"  # aligned | naive
    offsets_deg: dict[str, float] = field(default_factory=dict)
    rotate_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll

    def runner_args(self) -> list[str]:
        """The scripts/policy_runner.py flags that apply this condition."""
        args = []
        if self.joint_map != "aligned":
            args.append(f"--joint_map={self.joint_map}")
        if self.offsets_deg:
            args.append("--joint_offset_deg=" + ",".join(f"{j}={v:g}" for j, v in self.offsets_deg.items()))
        if any(self.rotate_deg):
            args.append("--overhead_rotate_deg=" + ",".join(f"{v:g}" for v in self.rotate_deg))
        return args


def parse_offsets(text: str, sep: str = ",") -> dict[str, float]:
    offsets = {}
    for item in filter(None, text.split(sep)):
        joint, _, value = item.partition("=")
        joint = joint.strip()
        if joint not in BODY_JOINTS:
            raise ValueError(f"joint offset on {joint!r}: only {', '.join(BODY_JOINTS)} (never the gripper)")
        offsets[joint] = float(value)
        if abs(offsets[joint]) > MAX_OFFSET_DEG:
            raise ValueError(f"{joint} offset {offsets[joint]} deg exceeds the {MAX_OFFSET_DEG:g} deg cap")
    return offsets


def parse_rotation(text: str) -> tuple[float, float, float]:
    values = [float(v) for v in text.split(",")]
    if not 1 <= len(values) <= 3:
        raise ValueError(f"rotation {text!r}: expected pitch[,yaw[,roll]]")
    return tuple(values + [0.0] * (3 - len(values)))


def parse_condition(text: str) -> Condition:
    text = text.strip()
    if text == "C0":
        return Condition(text)
    if text == "naive":
        return Condition(text, joint_map="naive")
    kind, _, spec = text.partition(":")
    if kind == "off":
        return Condition(text, offsets_deg=parse_offsets(spec, sep="+"))
    if kind == "rot":
        return Condition(text, rotate_deg=parse_rotation(spec))
    raise ValueError(f"unknown condition {text!r} (C0, naive, off:<joint>=<deg>[+...], rot:<pitch>,<yaw>[,<roll>])")


# ----- joint maps -----

def naive_lines(motor_calibration: dict, pos_to_sim: Callable[[str, float], float]) -> dict[str, tuple[float, float]]:
    """Per joint (a, b) with sim_deg = a * ticks + b for the LeRobot-only pipeline (see module doc)."""
    lines = {}
    for joint in JOINTS:
        lo, hi = motor_calibration[joint].range_min, motor_calibration[joint].range_max
        if joint == "gripper":
            positions = (20.0, 80.0)
            ticks = [lo + p / 100.0 * (hi - lo) for p in positions]
        else:
            positions = (-40.0, 40.0)
            ticks = [lo + (p + 100.0) / 200.0 * (hi - lo) for p in positions]
        sims = [pos_to_sim(joint, p) for p in positions]
        a = (sims[1] - sims[0]) / (ticks[1] - ticks[0])
        lines[joint] = (a, sims[0] - a * ticks[0])
    return lines


def apply_joint_condition(alignment, condition: Condition, pos_to_sim: Callable[[str, float], float]) -> None:
    """Replace the alignment instance's tick <-> sim maps by the condition's (no-op for aligned, no offsets).

    Call it on a freshly loaded alignment, before anything edits its motor_calibration: the naive lines
    use the LeRobot calibration's ranges, which the runner later narrows for the gripper floor.
    """
    if condition.joint_map == "aligned" and not condition.offsets_deg:
        return
    to_sim, to_ticks = alignment.ticks_to_sim_degrees, alignment.sim_degrees_to_ticks
    if condition.joint_map == "naive":
        lines = naive_lines(alignment.motor_calibration, pos_to_sim)

        def to_sim(joint: str, ticks: float) -> float:
            a, b = lines[joint]
            return a * ticks + b

        def to_ticks(joint: str, sim_degrees: float, *, clamp: bool = True) -> float:
            a, b = lines[joint]
            ticks = (sim_degrees - b) / a
            if clamp:
                calibration = alignment.motor_calibration[joint]
                ticks = min(calibration.range_max, max(calibration.range_min, ticks))
            return ticks
    elif condition.joint_map != "aligned":
        raise ValueError(f"unknown joint map {condition.joint_map!r}")
    offsets = {joint: condition.offsets_deg.get(joint, 0.0) for joint in JOINTS}
    alignment.ticks_to_sim_degrees = lambda joint, ticks: to_sim(joint, ticks) - offsets[joint]
    alignment.sim_degrees_to_ticks = (
        lambda joint, sim_degrees, *, clamp=True: to_ticks(joint, sim_degrees + offsets[joint], clamp=clamp)
    )


# ----- overhead rotation -----

def overhead_homography(rotate_deg: tuple[float, float, float], width: int = 640, height: int = 480,
                        hfov_deg: float = OVERHEAD_HFOV_DEG) -> np.ndarray:
    """Pixel map from the mounted camera's image to the image of the camera rotated by (pitch, yaw, roll)."""
    pitch, yaw, roll = (math.radians(v) for v in rotate_deg)
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    k = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    # Columns are the rotated camera's axes in the mounted camera's frame (OpenCV: x right, y down, z forward).
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), math.sin(pitch)], [0, -math.sin(pitch), math.cos(pitch)]])
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    r = ry @ rx @ rz
    return k @ r.T @ np.linalg.inv(k)


def warp_overhead(image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    return cv2.warpPerspective(image, homography, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


# ----- statistics -----

def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant pairs (b: baseline only, c: condition only)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)
