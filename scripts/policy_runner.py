#!/usr/bin/env python3
"""Run the bench's real-robot policy runner on the home-desktop follower, under LeRobot 0.7.0.

    envs/lerobot/.venv/bin/python scripts/policy_runner.py --no_policy             # cameras, overlay, docking
    envs/lerobot/.venv/bin/python scripts/policy_runner.py --policy internw0 \
        --instruction "Place the pen in the plastic bin."                          # server on localhost:8004

Wraps third_party/so101_bench/scripts/so101_real_policy_runner.py (its docstring lists the keys; q docks,
turns the torque off and quits) without editing the submodule:

- `RealArm` closes the port with `bus.port_handler.closePort()`, which LeRobot 0.7.0's buses no longer
  have. The buses get a `port_handler` whose `closePort()` is `bus.disconnect(disable_torque=False)`: the
  same effect, and the runner still decides about the torque itself.
- The runner lowers the gripper's EEPROM `Min_Position_Limit` to sim's jaw limit at every start. On our arm
  that limit (tick 2061) sits above the LeRobot calibration's range_min (2012), and a changed EEPROM value
  makes the next teleop/record ask whether to keep the calibration file (typing `c` there recalibrates and
  invalidates the sim alignment). Here the limit is applied only as the runner's software clamp, which
  every goal it sends already passes through; the EEPROM keeps the calibration file's value.
- No local policy servers. The runner starts servers on its own GPU when none answers; ours run on the GPU
  server behind an SSH tunnel, so a missing server is an error naming the port to forward. The runner's
  `stop_strays` (pgrep + kill of other policies' servers) is skipped.
- The overhead camera is a UVC camera here, not DroidCam: it opens as MJPG like the wrist camera (two
  640x480 YUYV streams do not fit on the shared hub). Both keep OpenCV's default 4 capture buffers: the
  runner's 1 halves our cameras' rate to about 16 fps when both stream (30 fps alone). Its reader thread
  keeps only the newest frame either way.
- Our defaults: follower port, robot id `so101_follower_arm`, our alignment, both cameras' udev names, and
  outputs under outputs/real_robot/ (the bench defaults are inside the submodule). Any of them given on
  the command line wins.

- Two more policies, the ones our GPU-server experiments run (T004-T007): `internw0` (InternW0-Δ) and
  `lawam` (LaWAM), both on the bench's HTTP `/act` protocol through the runner's MolmoAct2RemotePolicy
  client, with the client settings of scripts/wam_gap_eval.py (exp/wam-* branches): LeRobot joint frame,
  overhead + wrist only (no init frames), 10 denoising steps, 16 (internw0) / 36 (lawam) actions per
  query. `internw0` is the default policy. InternW0 keeps per-episode memory. The server must reset it
  only when the request's `episode` changes, which the runner bumps at every episode start. The bench's
  `internw0_server.py` does that, and so does our `internw0_probe_server.py --real`
  (exp/real-internw0-server). The plain probe server keys sessions by `trace.frame`, which this client does
  not send, so it would restart the memory on every query.
  Keys 1-6 / p switch among internw0, lawam, groot, g05, pi05, molmoact2 (the LLM agents are left out).

Ablation flags (src/so101_playground/ablation.py has the details; scripts/ablation.py sets them per block):
  --joint_map naive                    LeRobot `.pos` read as the bench's, no sim alignment
  --joint_offset_deg J=DEG[,J=DEG]     alignment zero shifted (body joints, |DEG| <= 10)
  --overhead_rotate_deg P[,Y[,R]]      the policy's overhead image as if the camera were rotated
  --quit_when_scored                   episode mode: dock and quit once every trial has a result
  --blind                              print nothing that reveals the condition

Policy ports (--policy_port overrides): internw0 8004, lawam 8005, groot 5555 (ZMQ), g05 8001, pi05 8002,
molmoact2 8003 (HTTP `/act`), all on --policy_host 127.0.0.1.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

from so101_playground.ablation import Condition, apply_joint_condition, overhead_homography, parse_offsets, \
    parse_rotation, warp_overhead

ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPTS = ROOT / "third_party" / "so101_bench" / "scripts"
OUTPUTS = ROOT / "outputs" / "real_robot"


def _device(name: str, fallback: str) -> str:
    return name if Path(name).exists() else fallback


HTTP_CLIENT = {"use_overhead_init": False, "use_wrist_init": False, "joint_frame": "lerobot", "num_steps": 10}
OUR_POLICIES = {
    "internw0": {"policy_port": 8004, "action_horizon": 16, **HTTP_CLIENT},
    "lawam": {"policy_port": 8005, "action_horizon": 36, **HTTP_CLIENT},
}
POLICY_ORDER = ("internw0", "lawam", "groot", "g05", "pi05", "molmoact2")

DEFAULTS = {
    "--policy": "internw0",
    "--follower_port": _device("/dev/so101_follower", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CBC453103-if00"),
    "--robot_id": "so101_follower_arm",
    "--alignment": str(ROOT / "configs" / "real_robot" / "so101_follower_arm_sim_alignment.json"),
    "--wrist_camera": _device("/dev/so101_cam_wrist", "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.4.2:1.0-video-index0"),
    "--overhead_camera": _device("/dev/so101_cam_overhead", "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.4.3:1.0-video-index0"),
    "--reference_dir": str(OUTPUTS / "sim_camera_reference"),
    "--output_root": str(OUTPUTS / "runs"),
    "--results_root": str(OUTPUTS / "mmrv"),
}


def _load_runner():
    sys.path.insert(0, str(BENCH_SCRIPTS))  # the runner imports so101_real_recording from its own directory
    spec = importlib.util.spec_from_file_location("so101_real_policy_runner", BENCH_SCRIPTS / "so101_real_policy_runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # not __main__, so its main() does not run yet
    return module


class _PortHandler:
    def __init__(self, bus):
        self.bus = bus

    def closePort(self) -> None:  # noqa: N802 - the scservo_sdk name the runner calls
        self.bus.disconnect(disable_torque=False)


def _patch(runner, condition: Condition | None = None, *, quit_when_scored: bool = False, blind: bool = False) -> None:
    import cv2
    from lerobot.motors.feetech import FeetechMotorsBus

    FeetechMotorsBus.port_handler = property(_PortHandler)

    def set_gripper_min_tick(self, tick: int) -> None:
        eeprom = self.bus.read("Min_Position_Limit", "gripper", normalize=False)
        deg = self.alignment.ticks_to_sim_degrees("gripper", tick)
        if not blind:  # the floor's tick depends on the joint map
            print(f"[INFO]: Gripper goals clamped at tick {tick} (sim {deg:.2f} deg) by the runner; "
                  f"EEPROM Min_Position_Limit left at {eeprom}.", flush=True)
        snapshot = self.alignment.motor_calibration["gripper"]
        self.alignment.motor_calibration["gripper"] = replace(snapshot, range_min=tick)

    runner.RealArm.set_gripper_min_tick = set_gripper_min_tick

    open_capture = runner.LatestFrameCamera._open

    def open_uvc(self):
        self.fourcc = self.fourcc or "MJPG"
        capture = open_capture(self)
        if capture is not None:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
        return capture

    runner.LatestFrameCamera._open = open_uvc

    def ensure(self, policy: str, port: int, progress=print) -> None:
        if not runner.server_is_up(policy, self.host, port):
            protocol = "ZMQ" if policy == "groot" else "HTTP /act"
            raise RuntimeError(
                f"no {policy} server ({protocol}) answering on {self.host}:{port}. Start it on the GPU server and "
                f"forward the port: ssh -N -L {port}:<server address, e.g. its container's IP>:{port} <gpu-server>"
            )
        progress(f"{policy} server answering on {self.host}:{port}.")

    runner.PolicyServerManager.ensure = ensure
    runner.PolicyServerManager.stop_strays = lambda self, keep, progress=print: None

    runner.POLICY_DEFAULTS.update(OUR_POLICIES)  # before the parser reads its --policy choices
    runner.POLICY_ORDER = POLICY_ORDER

    if condition is not None:
        _patch_condition(runner, condition)
    if quit_when_scored:
        save_result = runner.Runner.save_result

        def save_then_quit(self, *args, **kwargs) -> None:
            save_result(self, *args, **kwargs)
            done = self._results()
            if self.episodes and all(row["trial_id"] in done for row in self.episodes):
                print("[INFO]: Every trial has a result: docking, then quitting.", flush=True)
                self.quit_after_dock = True
                self.start_dock()
                self.message = "Every trial has a result: docking, then quitting."

        runner.Runner.save_result = save_then_quit


def _patch_condition(runner, condition: Condition) -> None:
    """Apply an ablation condition (src/so101_playground/ablation.py) to the runner's alignment and policy."""
    load = runner.alignment_lib.RealSimAlignment.load

    def load_perturbed(path):
        alignment = load(path)
        apply_joint_condition(alignment, condition, runner.calibration_lib.lerobot_position_to_sim_degrees)
        return alignment

    runner.alignment_lib.RealSimAlignment.load = staticmethod(load_perturbed)

    if any(condition.rotate_deg):
        homography = overhead_homography(condition.rotate_deg, *runner.FRAME_SIZE)
        make_policy = runner.Runner._make_policy

        def make_rotated(self):
            policy = make_policy(self)
            get_action = policy.get_action

            def rotated(joint_positions, visual_obs):
                visual_obs = {**visual_obs, "rgb_overhead": warp_overhead(visual_obs["rgb_overhead"], homography)}
                return get_action(joint_positions, visual_obs)

            policy.get_action = rotated
            return policy

        runner.Runner._make_policy = make_rotated


def _ablation_args() -> tuple[Condition, argparse.Namespace]:
    """Take our ablation flags out of sys.argv (the bench parser rejects unknown flags)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--joint_map", choices=("aligned", "naive"), default="aligned")
    parser.add_argument("--joint_offset_deg", default="")
    parser.add_argument("--overhead_rotate_deg", default="")
    parser.add_argument("--quit_when_scored", action="store_true")
    parser.add_argument("--blind", action="store_true")
    ours, rest = parser.parse_known_args(sys.argv[1:])
    sys.argv = sys.argv[:1] + rest
    condition = Condition(
        "cli",
        joint_map=ours.joint_map,
        offsets_deg=parse_offsets(ours.joint_offset_deg),
        rotate_deg=parse_rotation(ours.overhead_rotate_deg) if ours.overhead_rotate_deg else (0.0, 0.0, 0.0),
    )
    return condition, ours


def main() -> None:
    condition, ours = _ablation_args()
    given = {arg.split("=", 1)[0] for arg in sys.argv[1:]}
    for flag, value in DEFAULTS.items():
        if flag not in given:
            sys.argv += [flag, value]
    runner = _load_runner()
    _patch(runner, condition, quit_when_scored=ours.quit_when_scored, blind=ours.blind)
    if not ours.blind and condition.runner_args():
        print(f"[ABLATION]: {' '.join(condition.runner_args())}", flush=True)
    runner.main()


if __name__ == "__main__":
    main()
