"""Oracle corrections for injected control gaps (T005, RESEARCH.md H2).

An oracle correction stands in for the residual adapter that P1 would train with RL. It knows the injected
gap and corrects for it with strength ``alpha``. It acts exactly where an adapter would: on the command
stream, **after** the bench client's 15 deg/tick clamp and **before** the gap (``control_gap.CommandGap``).
It may only change the command sent; the sim's actuator parameters are never touched.

    policy chunk -> client clamp -> cmd -> [oracle(alpha)] -> sent -> [gap] -> applied -> env.step

Formulas (joint order as ``control_gap.JOINT_NAMES``; all in sim radians):

  offset / offset6   sent = cmd - alpha * offset. The gap adds +offset, so alpha=1 is exact.
  gain / gain2       stiffness-only gap (k -> s*k, damping kept). The PD torque k*(target - q) - d*qd is
                     reproduced by amplifying the tracking error on the five arm joints:
                         sent = q + (1 + alpha * (1/s - 1)) * (cmd - q)
                     with q = the latest measured joint position (the policy's own observation at this tick).
                     Approximate: q moves during the tick's physics substeps while the amplified target is
                     held, so the effective gain is not exactly restored. Jaw unchanged (its stiffness is kept).
  latency / latency6 delay of D ticks. The policy issues a chunk, so the adapter can send later rows early:
                     at chunk index i it sends row min(i + k, H - 1) of the current chunk with
                     k = round(alpha * D), passed through the same 15 deg/tick clamp as the client applies to
                     row i (relative to the current measured q). After the D-tick delay the arm receives row i
                     at index i, except: (a) the first k rows of every chunk are never applied (they would have
                     to be sent before the chunk exists), the previous chunk's last row is held instead, and
                     (b) the clamp is evaluated D ticks early, against an older q. Both are measured
                     (``|applied - cmd|`` per tick) and reported.

alpha=0 is the identity (sent = cmd) for every gap.
"""

from __future__ import annotations

from typing import Callable

import torch

from so101_playground.control_gap import ARM_JOINTS, JOINT_NAMES, GapConfig


def oracle_kind(cfg: GapConfig) -> str:
    """Which oracle formula fits a gap condition."""
    kinds = []
    if any(v != 0.0 for v in cfg.offset_deg):
        kinds.append("offset")
    if cfg.stiffness_scale != 1.0:
        kinds.append("gain")
    if cfg.delay_ticks > 0:
        kinds.append("latency")
    if cfg.backlash_deg > 0.0:
        kinds.append("backlash")
    if len(kinds) > 1 or "backlash" in kinds:
        raise ValueError(f"no oracle for condition {cfg.name} ({kinds})")
    return kinds[0] if kinds else "none"


class OracleCorrection:
    """Stateless per tick: ``sent = oracle(cmd, q, lookahead)``.

    cmd        (6,) the client's command for this tick (after the clamp).
    q          (6,) measured joint position the client clamped against (sim radians).
    lookahead  callable k -> (6,) clamped command of the row k ahead in the current chunk (held at the chunk's
               last row); only used by the latency oracle.
    """

    def __init__(self, cfg: GapConfig, alpha: float, device: str | torch.device):
        self.cfg = cfg
        self.alpha = float(alpha)
        self.kind = oracle_kind(cfg) if self.alpha != 0.0 else "identity"
        self.arm_mask = torch.tensor([n in ARM_JOINTS for n in JOINT_NAMES], device=device)
        self.offset = torch.deg2rad(torch.tensor(cfg.offset_deg, dtype=torch.float32, device=device))
        self.gain_factor = 1.0 + self.alpha * (1.0 / cfg.stiffness_scale - 1.0)
        self.lead = int(round(self.alpha * cfg.delay_ticks))

    def describe(self) -> str:
        if self.kind == "offset":
            return f"sent = cmd - {self.alpha} * offset"
        if self.kind == "gain":
            return f"sent = q + {self.gain_factor:.3f} * (cmd - q) on arm joints"
        if self.kind == "latency":
            return f"sent = chunk row i+{self.lead} (clamped), held at the chunk's last row"
        return "identity"

    def __call__(self, cmd: torch.Tensor, q: torch.Tensor, lookahead: Callable[[int], torch.Tensor]) -> torch.Tensor:
        x = cmd.detach().clone().float()
        if self.kind == "offset":
            return x - self.alpha * self.offset
        if self.kind == "gain":
            q = q.detach().float()
            amp = q + self.gain_factor * (x - q)
            return torch.where(self.arm_mask, amp, x)
        if self.kind == "latency":
            return lookahead(self.lead).detach().clone().float() if self.lead > 0 else x
        return x
