"""Convert joint values between our follower's LeRobot `.pos` and the SO-101 Bench's `.pos`.

LeRobot's `.pos` (use_degrees=false) normalizes each motor's ticks by that arm's own calibration ranges, so
one physical pose reads differently on two arms. The bench defines its dataset and policy `.pos` as a fixed
bijection of sim joint angles (`so101_bench.utils.lerobot_calibration`, from the arm that recorded its
data). Our follower's sim alignment gives the tick at which each joint matches the sim zero pose, so:

    ours .pos --our calibration ranges--> ticks --alignment zero_tick--> sim deg --bench bijection--> bench .pos

and back. Both directions stay in float (LeRobot truncates goal ticks to int; here nothing is rounded).
Bench `.pos` covers the sim range of the bench arm's calibration (its +-100 ends), within the USD joint
limits; values past that saturate. Our arm's ranges land within 1-5 deg of the bench arm's in sim degrees;
our folded shoulder_lift (about -107 deg) is past the bench's -103.2 and saturates at -100.

Our calibration comes from LeRobot's calibration file, which is what normalized the recorded values, and
must match the snapshot stored in the alignment (else the alignment is stale). A dataset recorded before
a recalibration cannot be converted with today's files.

The bench modules are loaded by path: `so101_bench` is not installed in the lerobot env, and both modules
are kept free of torch/Isaac imports for this use.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH_UTILS = ROOT / "third_party" / "so101_bench" / "source" / "so101_bench" / "so101_bench" / "utils"
DEFAULT_ALIGNMENT = ROOT / "configs" / "real_robot" / "so101_follower_arm_sim_alignment.json"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _load_bench(name: str) -> ModuleType:
    module_name = f"_so101_bench_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, BENCH_UTILS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # before exec: dataclasses look their module up
    spec.loader.exec_module(module)
    return module


lerobot_calibration = _load_bench("lerobot_calibration")
real_sim_alignment = _load_bench("real_sim_alignment")
assert tuple(lerobot_calibration.LEROBOT_JOINT_ORDER) == JOINTS


def lerobot_calibration_path(robot_id: str) -> Path:
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    return HF_LEROBOT_CALIBRATION / "robots" / "so_follower" / f"{robot_id}.json"


class BenchUnits:
    """Our follower's `.pos` <-> bench `.pos`, for arrays whose last axis is the 6 joints in `JOINTS` order."""

    def __init__(self, alignment_path: Path = DEFAULT_ALIGNMENT, calibration_path: Path | None = None):
        self.alignment_path = Path(alignment_path)
        self.alignment = real_sim_alignment.RealSimAlignment.load(self.alignment_path)
        self.calibration_path = Path(calibration_path or lerobot_calibration_path(self.alignment.robot_id))
        self.calibration = json.loads(self.calibration_path.read_text())
        mismatches = []
        for joint in JOINTS:
            snapshot, current = self.alignment.motor_calibration[joint], self.calibration[joint]
            for key in ("homing_offset", "range_min", "range_max"):
                if getattr(snapshot, key) != current[key]:
                    mismatches.append(f"{joint}.{key}: calibration {current[key]}, alignment {getattr(snapshot, key)}")
            if current["drive_mode"]:
                mismatches.append(f"{joint}: drive_mode {current['drive_mode']} (not handled)")
        if mismatches:
            raise ValueError(
                f"{self.calibration_path} does not match the calibration {self.alignment_path} was measured with:\n  "
                + "\n  ".join(mismatches)
            )

    # Our LeRobot normalization (motors_bus._normalize/_unnormalize with RANGE_M100_100 / RANGE_0_100), in float.
    def _pos_to_ticks(self, joint: str, pos: float) -> float:
        lo, hi = self.calibration[joint]["range_min"], self.calibration[joint]["range_max"]
        if joint == "gripper":
            return min(100.0, max(0.0, pos)) / 100.0 * (hi - lo) + lo
        return (min(100.0, max(-100.0, pos)) + 100.0) / 200.0 * (hi - lo) + lo

    def _ticks_to_pos(self, joint: str, ticks: float) -> float:
        lo, hi = self.calibration[joint]["range_min"], self.calibration[joint]["range_max"]
        norm = (min(hi, max(lo, ticks)) - lo) / (hi - lo)
        return norm * 100.0 if joint == "gripper" else norm * 200.0 - 100.0

    def _map(self, values: np.ndarray, per_joint) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != len(JOINTS):
            raise ValueError(f"last axis must be the {len(JOINTS)} joints, got shape {values.shape}")
        out = np.empty_like(values)
        for j, joint in enumerate(JOINTS):
            column = values[..., j]
            out[..., j] = np.fromiter((per_joint(joint, float(v)) for v in column.ravel()), np.float64,
                                      column.size).reshape(column.shape)
        return out

    def to_sim_degrees(self, pos: np.ndarray) -> np.ndarray:
        """Our `.pos` -> sim joint degrees, not clamped to the USD limits."""
        return self._map(pos, lambda joint, v: self.alignment.ticks_to_sim_degrees(joint, self._pos_to_ticks(joint, v)))

    def to_bench(self, pos: np.ndarray) -> np.ndarray:
        """Our `.pos` -> bench `.pos`."""
        sim = self.to_sim_degrees(pos)
        return self._map(sim, lerobot_calibration.sim_degrees_to_lerobot_position)

    def from_bench(self, pos: np.ndarray) -> np.ndarray:
        """Bench `.pos` -> our `.pos` (e.g. to run a bench-unit policy through LeRobot on our follower)."""
        sim = self._map(pos, lerobot_calibration.lerobot_position_to_sim_degrees)
        return self._map(sim, lambda joint, v: self._ticks_to_pos(joint, self.alignment.sim_degrees_to_ticks(joint, v)))

    @staticmethod
    def usd_limits_deg() -> np.ndarray:
        """(6, 2) sim joint limits the bench mapping clamps to."""
        return np.array([lerobot_calibration.USD_SIM_JOINT_LIMITS_DEG[joint] for joint in JOINTS])

    def describe(self) -> dict:
        return {
            "alignment": str(self.alignment_path.relative_to(ROOT) if self.alignment_path.is_relative_to(ROOT)
                             else self.alignment_path),
            "alignment_created_at": self.alignment.created_at,
            "robot_id": self.alignment.robot_id,
            "calibration": {joint: self.calibration[joint] for joint in JOINTS},
        }
