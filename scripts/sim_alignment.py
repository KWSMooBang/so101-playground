#!/usr/bin/env python3
"""Run the bench's sim alignment tool on the home-desktop follower, under LeRobot 0.7.0.

    envs/lerobot/.venv/bin/python scripts/sim_alignment.py measure    # guided capture (torque off), writes the JSON
    envs/lerobot/.venv/bin/python scripts/sim_alignment.py live       # stream sim angles under the saved alignment
    envs/lerobot/.venv/bin/python scripts/sim_alignment.py verify     # TORQUE ON: visits sim poses

Wraps third_party/so101_bench/scripts/so101_sim_alignment.py (see its docstring for the procedure) with two
changes, without editing the submodule:

- `ReadOnlyArm.close()` calls `bus.port_handler.closePort()`, which LeRobot 0.7.0's buses no longer have
  (they use rustypot). It raised AttributeError in `measure`'s `finally`, before the result was saved, so a
  full measurement was lost. Here it closes the port with `bus.disconnect(disable_torque=False)`: same effect,
  the torque is left as it was.
- Our defaults: the follower's by-id port, robot id `so101_follower_arm` (its LeRobot calibration id) and the
  output `configs/real_robot/so101_follower_arm_sim_alignment.json` (tracked; the bench default is inside
  the submodule). Any of --port/--robot_id/--output given on the command line wins.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = ROOT / "third_party" / "so101_bench" / "scripts" / "so101_sim_alignment.py"
DEFAULTS = {
    "--port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CBC453103-if00",
    "--robot_id": "so101_follower_arm",
    "--output": str(ROOT / "configs" / "real_robot" / "so101_follower_arm_sim_alignment.json"),
}


def _load_bench_script():
    spec = importlib.util.spec_from_file_location("so101_sim_alignment", BENCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # not __main__, so its main() does not run yet
    return module


def _close(self) -> None:
    self.bus.disconnect(disable_torque=False)


def main() -> None:
    if Path("/dev/so101_follower").exists():
        DEFAULTS["--port"] = "/dev/so101_follower"
    given = {arg.split("=", 1)[0] for arg in sys.argv[1:]}
    for flag, value in DEFAULTS.items():
        if flag not in given:
            sys.argv += [flag, value]
    bench = _load_bench_script()
    bench.ReadOnlyArm.close = _close
    bench.main()


if __name__ == "__main__":
    main()
