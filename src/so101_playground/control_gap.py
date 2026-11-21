"""Control-gap injection for SO-101 Bench sim rollouts (T004, RESEARCH.md H2-P0).

A gap sits between the joint target the policy commands (``cmd``, sim radians, after the bench
client's 15 deg/tick clamp) and what the simulated actuators receive. Nothing in the bench is
edited: command-side gaps transform the action passed to ``env.step``; the gain gap edits the
robot's actuator cfg before the env is built.

Conditions (joint order: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw, i.e. LeRobot
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper):

  none      nominal bench actuators, applied = cmd.
  gain      PD stiffness x0.5 on the five arm joints (Jaw kept, so grip force is unchanged);
            damping unchanged. applied = cmd. Steady-state droop under gravity doubles and the arm lags
            more on fast motions.
  latency   applied(t) = cmd(t - 3): every command reaches the actuators 3 control ticks (100 ms at
            30 Hz) late. The first 3 ticks of control repeat the hold pose.
  offset    applied = cmd + 3 deg on Pitch (shoulder_lift) and Elbow (elbow_flex), constant. The joint
            encoders report the true angle, so the policy sees the arm settle 3 deg away from where it
            commanded (a command-side calibration error).
  backlash  per-joint dead band of +-1.5 deg on the five arm joints: the applied target only moves
            once the command leaves a 3 deg window centred on it, then trails it by 1.5 deg.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

import torch

JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
ARM_JOINTS = JOINT_NAMES[:5]


@dataclass(frozen=True)
class GapConfig:
    name: str
    stiffness_scale: float = 1.0
    delay_ticks: int = 0
    offset_deg: tuple[float, ...] = (0.0,) * 6
    backlash_deg: float = 0.0
    description: str = ""
    extra: dict = field(default_factory=dict)


CONDITIONS: dict[str, GapConfig] = {
    "none": GapConfig("none", description="nominal"),
    "gain": GapConfig("gain", stiffness_scale=0.5, description="arm PD stiffness x0.5 (Jaw kept, damping kept)"),
    "latency": GapConfig("latency", delay_ticks=3, description="commands delayed 3 ticks (100 ms)"),
    "offset": GapConfig(
        "offset", offset_deg=(0.0, 3.0, 3.0, 0.0, 0.0, 0.0), description="+3 deg on Pitch and Elbow targets"
    ),
    "backlash": GapConfig("backlash", backlash_deg=1.5, description="+-1.5 deg dead band on arm targets"),
}


def apply_actuator_gap(env_cfg, cfg: GapConfig) -> dict:
    """Scale the robot's arm stiffness in an (unbuilt) env cfg. Returns the stiffness actually used."""
    actuator = env_cfg.scene.robot.actuators["arm"]
    stiffness = dict(actuator.stiffness)
    if cfg.stiffness_scale != 1.0:
        for joint in ARM_JOINTS:
            stiffness[joint] = float(stiffness[joint]) * cfg.stiffness_scale
        actuator.stiffness = stiffness
    return stiffness


class CommandGap:
    """Stateful command -> applied-target transform for one env (tensors of shape (6,), sim radians)."""

    def __init__(self, cfg: GapConfig, device: str | torch.device):
        self.cfg = cfg
        self.device = device
        self.offset = torch.tensor([math.radians(v) for v in cfg.offset_deg], dtype=torch.float32, device=device)
        self.band = math.radians(cfg.backlash_deg)
        self.arm_mask = torch.tensor([n in ARM_JOINTS for n in JOINT_NAMES], device=device)
        self._queue: deque[torch.Tensor] = deque()
        self._backlash_state: torch.Tensor | None = None

    def reset(self, hold: torch.Tensor) -> None:
        """Start of robot control: the delay line and dead band start at the hold pose."""
        hold = hold.detach().clone().float()
        self._queue = deque([hold.clone() for _ in range(self.cfg.delay_ticks)])
        self._backlash_state = hold.clone()

    def __call__(self, cmd: torch.Tensor) -> torch.Tensor:
        x = cmd.detach().clone().float()
        if self.cfg.delay_ticks > 0:
            self._queue.append(x)
            x = self._queue.popleft()
        if self.band > 0.0:
            state = self._backlash_state if self._backlash_state is not None else x.clone()
            upper, lower = state + self.band, state - self.band
            moved = torch.where(x > upper, x - self.band, torch.where(x < lower, x + self.band, state))
            state = torch.where(self.arm_mask, moved, x)
            self._backlash_state = state
            x = state.clone()
        return x + self.offset
