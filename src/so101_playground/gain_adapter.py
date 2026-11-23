"""Per-joint gain adapter (T009, RESEARCH.md H2-P1 stage A).

The adapter acts exactly where the T005 oracle acts: on the command stream, after the bench client's
15 deg/tick clamp and before the gap.

    policy chunk -> client clamp -> cmd -> [adapter(theta)] -> sent -> [gap] -> applied -> env.step

    sent_j = q_j + k_j * (cmd_j - q_j)   for the five arm joints (ARM_JOINTS order), Jaw unchanged,
    k = clip(exp(theta), K_MIN, K_MAX),  theta = log k in R^5.

q is the measured joint position the client clamped against at this tick (the policy's own observation).
theta = 0 is the identity. For a stiffness-only gap with per-joint scale s, k* = 1/s is the T005 oracle
form (gain2 oracle at alpha = 1, per joint).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from so101_playground.control_gap import ARM_JOINTS, JOINT_NAMES

K_MIN, K_MAX = 0.5, 4.0
THETA_MIN, THETA_MAX = math.log(K_MIN), math.log(K_MAX)


def clip_theta(theta: Sequence[float]) -> list[float]:
    return [min(max(float(t), THETA_MIN), THETA_MAX) for t in theta]


def gains(theta: Sequence[float]) -> list[float]:
    return [math.exp(t) for t in clip_theta(theta)]


class GainAdapter:
    def __init__(self, theta: Sequence[float], device: str | torch.device):
        if len(theta) != len(ARM_JOINTS):
            raise ValueError(f"theta must have {len(ARM_JOINTS)} entries, got {len(theta)}")
        self.k_arm = gains(theta)
        k = self.k_arm + [1.0] * (len(JOINT_NAMES) - len(ARM_JOINTS))
        self.k = torch.tensor(k, dtype=torch.float32, device=device)

    def describe(self) -> str:
        return "sent = q + k*(cmd - q), k = " + ", ".join(f"{n} {v:.3f}" for n, v in zip(ARM_JOINTS, self.k_arm))

    def __call__(self, cmd: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        x = cmd.detach().clone().float()
        q = q.detach().float()
        return q + self.k * (x - q)
