# so101-playground

A playground for SO-101 benchmarks, simulators and frameworks, and for operating and evaluating a
real SO-101 (overhead + wrist cameras) remotely. See `RESEARCH.md` for goals and decisions.

## Layout

```
third_party/so101_bench   submodule, github.com/5hadytru/so101_bench @ 470e09e (main, 2026-11-04)
third_party/lerobot       submodule, github.com/huggingface/lerobot  @ v0.7.0
src/so101_playground/     our package (bench_units.py: our joint units <-> the bench's)
scripts/                  our scripts (isaac_smoke.py, teleop.sh, record.sh, convert_to_bench_units.py,
                          sim_alignment.py, overhead_align.py)
configs/real_robot/       our follower's sim alignment (bench joint frame)
envs/isaac/               Isaac env: setup_isaac.sh, env.sh, Vulkan ICD manifest
envs/lerobot/             LeRobot env: setup_lerobot.sh, env.sh, udev/ (stable /dev/so101_* names)
RESEARCH.md EXPERIMENTS.md
```

External code lives in `third_party/` as pinned submodules and is never edited in place; if upstream
needs a change, point the submodule at a fork. The venvs, the Isaac Lab checkout
(`envs/isaac/IsaacLab`), assets and outputs are gitignored.

## Setup

Two environments, because Isaac Sim 5.1 needs Python 3.11 and LeRobot needs Python >= 3.12.
Requirements: `uv`, `git`, `cmake`, an RTX GPU, glibc >= 2.35.

```bash
git clone --recurse-submodules https://github.com/KWSMooBang/so101-playground
cd so101-playground            # existing clone: git submodule update --init

envs/lerobot/setup_lerobot.sh  # Python 3.12 -> envs/lerobot/.venv
envs/isaac/setup_isaac.sh      # Python 3.11 -> envs/isaac/.venv + envs/isaac/IsaacLab (v2.3.2)
```

The venvs hold absolute paths, so build them in the checkout you will use (a second build is fast
once uv's cache is warm).

### lerobot env

`setup_lerobot.sh` installs `third_party/lerobot` editable with the extras
`feetech,remote,core_scripts,evaluation,training,smolvla`, then the bench plugin
(`third_party/so101_bench/integrations/lerobot`) and this package.

```bash
source envs/lerobot/env.sh     # activates the venv; exports SO101_BENCH_ROOT, ISAACLAB_SH, SO101_BENCH_ISAAC_VENV
python -c "import lerobot, torch; print(lerobot.__version__, torch.cuda.is_available())"
```

### Home desktop devices

The home desktop (RTX 3060 12GB, Ubuntu 24.04) runs the arms and cameras from the lerobot env; it
needs no isaac env. `envs/lerobot/udev/99-so101.rules` gives the devices stable names:

| Link | Device | Matched by |
|---|---|---|
| `/dev/so101_leader` | leader arm | USB serial `5CBC452337` |
| `/dev/so101_follower` | follower arm | USB serial `5CBC453103` |
| `/dev/so101_cam_wrist` | wrist camera | hub port `.4.2` |
| `/dev/so101_cam_overhead` | overhead camera | hub port `.4.3` |

```bash
envs/lerobot/udev/install_udev.sh   # copies the rules to /etc/udev/rules.d (sudo), then: ls -l /dev/so101_*
```

- Both cameras are the same module (1e45:0209) with the same USB serial, so `/dev/v4l/by-id` shows
  only one of them. The rules use the physical port of the 7-port VIA hub instead: keep each camera on
  its marked hub port (swapping the cables swaps the names; the hub itself can go in any PC port).
  Both run 640x480 MJPG at about 28 fps together; dim light lowers the rate (auto exposure).
- On first plug-in the hub's downstream chip once failed to enumerate (`can't set config #1, error -71`
  in `journalctl -k`) and every port behind it was invisible. Re-plugging the hub fixed it.

`scripts/teleop.sh` runs leader -> follower teleoperation with both cameras (`use_degrees=false`, the
bench datasets' units). The rerun viewer (rerun 0.33.1) freezes within about a minute and blocks the
teleop loop (and the exit, before the follower's torque is released); it does the same with no robot
attached. Use `--display_data=false` for anything longer than a quick look.

### Follower sim alignment

The bench maps policy joint values to motor ticks through a per-arm alignment (`zero_tick` per joint at
the sim zero pose). Ours is `configs/real_robot/so101_follower_arm_sim_alignment.json`, measured
2026-11-22 with a phone level. `scripts/sim_alignment.py` wraps the bench's `so101_sim_alignment.py`
with our port/id/output defaults and a fix for LeRobot 0.7.0 (the bench closes the port with
`bus.port_handler`, which 0.7.0 no longer has; the crash came before the save). The bench's
`so101_real_policy_runner.py` and `so101_gripper_test.py` have the same `port_handler` call.

```bash
envs/lerobot/.venv/bin/python scripts/sim_alignment.py measure                 # torque off, guided
envs/lerobot/.venv/bin/python scripts/sim_alignment.py measure --only gripper  # re-measure one joint
envs/lerobot/.venv/bin/python scripts/sim_alignment.py live                    # read-only sim angles
```

The alignment stores the motors' homing offsets; **do not re-run LeRobot calibration on the follower**,
or the bench runner refuses to start and the alignment has to be measured again.

### Overhead camera alignment

The overhead camera should see the scene as the bench's canonical overhead camera does: about 52 cm in
front of the robot base, 46 cm above the table, tilted 51 deg from vertical, 74 deg horizontal FOV.
`scripts/overhead_align.py` draws the robot base target and the object area (5 cm grid) from the bench's
own camera model over the live feed (a tkinter window; Save/Quit buttons). It also rewrites
`outputs/camera_check/overhead_align_latest*.jpg` every second.

```bash
envs/lerobot/.venv/bin/python scripts/overhead_align.py
```

Aligned 2026-11-22: the robot base and two marks at the object area's near corners (28 cm in front of the
base axis, ±19.5 cm) land within about 15 px of the targets. The camera's "86 deg" spec fits a diagonal
FOV (about 73 deg horizontal), so no crop is needed. Keep the camera fixed afterwards: the bench runner
rejects more than 3 px drift against a reference frame.

### Recording

`scripts/record.sh NAME "TASK"` records teleoperated episodes with `lerobot-record` into
`outputs/datasets/NAME` (local only, no Hub upload; an existing NAME gets the new episodes appended). The
format matches the bench real dataset (`5hadytru/so101_bench_real`): LeRobot v3.0, 30 fps, AV1 480x640
`observation.images.{wrist,overhead}`, 6-D `.pos` state and action. Only `robot_type` differs
(`so_follower` from LeRobot 0.7.0, `so101_follower` in the bench's). Device paths and arm arguments are
shared with `teleop.sh` in `scripts/so101_devices.sh`.

```bash
scripts/record.sh pilot "Place the pen in the plastic bin" --dataset.num_episodes=3
```

It waits for Enter, then records right away. Keys work in any window: Right arrow ends the episode (or the
reset) early, Left arrow discards and re-records it, Esc stops. Episodes run up to 60 s, resets up to 30 s.
lerobot parses CLI values as YAML (`a: b` becomes a dict), so the script passes the task quoted.

Pilot 2026-11-22: 3 episodes of 20-24 s, 29.9 Hz with no tick over budget, streaming AV1 encoding, 33 MB.
The `.pos` values are in our follower's calibration, not the bench arm's (`REAL_SO101_CALIBRATION`).
Through our sim alignment they differ by about 1-3 units per joint (wrist_roll scale about 5%), so convert
them before mixing with bench data (next section).

### Bench joint units

The bench's `.pos` is a fixed bijection of sim joint angles (`so101_bench.utils.lerobot_calibration`);
LeRobot's `.pos` depends on each arm's calibration. `src/so101_playground/bench_units.py` maps between
them through our sim alignment: our `.pos` -> ticks (our LeRobot calibration file, checked against the
alignment's snapshot) -> sim degrees -> bench `.pos`, and back (`BenchUnits.to_bench` / `from_bench`).

```bash
envs/lerobot/.venv/bin/python scripts/convert_to_bench_units.py outputs/datasets/pilot   # -> pilot_bench
```

The script copies the dataset, converts `observation.state` and `action` (the leader's `.pos`, which the
follower turns into goal ticks with its own calibration), recomputes their stats, sets `robot_type` to
`so101_follower` and writes `meta/bench_units.json`. Videos are copied unchanged.

Bench `.pos` cannot go past the bench arm's calibration range. In the pilot, 31% of frames saturate on
shoulder_lift and 38% on gripper. All of them are at mechanical ends: the folded rest pose has our lift at
-107 deg, past the bench's -103.2 back stop, and the closed gripper is at -14 deg, past the jaw limit of
-12.75. The bench data reads about the same there: the median episode minimum is -99.4 for lift and 1.1
for gripper.
As a cross-check, the two arms' calibration range ends land within about 2 deg of each other in sim
degrees, except shoulder_lift's back stop (4.7 deg) and the closed gripper. Those ranges came from each
arm's own LeRobot calibration, independent of the phone-level alignment.

### isaac env

`setup_isaac.sh` installs `isaacsim[all,extscache]==5.1.0` (pip, pypi.nvidia.com; brings torch 2.7.0
and numpy 1.26.0), Isaac Lab v2.3.2 (`isaaclab` extension 0.54.2, the version the bench tests; no RL
frameworks) and `third_party/so101_bench/source/so101_bench`. Installing Isaac Sim accepts the NVIDIA
Omniverse EULA (`OMNI_KIT_ACCEPT_EULA=YES`).

The Isaac Lab extensions are installed with `uv pip install -e` under `envs/isaac/constraints.txt`
and `build-constraints.txt`, not with `isaaclab.sh --install`: that script fails to build
`flatdict` with current setuptools (and still exits 0, leaving `isaaclab` uninstalled) and upgrades
numpy to 2.x. `isaaclab.sh -p` still works for running scripts.

```bash
source envs/isaac/env.sh
python third_party/so101_bench/scripts/download_assets.py      # ~2 GB of USD assets, once
CUDA_VISIBLE_DEVICES=0 python scripts/isaac_smoke.py --headless                     # empty stage
CUDA_VISIBLE_DEVICES=0 python scripts/isaac_smoke.py --headless --enable_cameras \
    --task So101Bench-Bin-v0 --steps 20                                                 # bench env, random actions
```

**Vulkan in our container.** The container mounts the NVIDIA driver libraries but not the Vulkan ICD
manifest, so by default Vulkan only sees mesa and Isaac Sim finds no RTX GPU. `envs/isaac/vulkan.sh`
(sourced by both `env.sh` files) sets `VK_DRIVER_FILES` to `envs/isaac/nvidia_icd.json`, which points
the loader at the driver's `libEGL_nvidia.so.0`. It does nothing when the system already has an
NVIDIA manifest. Check with `vulkaninfo --summary` after sourcing `env.sh`.

The same file sets `__EGL_VENDOR_LIBRARY_FILENAMES` to `envs/isaac/nvidia_egl_vendor.json` for EGL.

**GPU selection.** Use `CUDA_VISIBLE_DEVICES=N` (N = GPU index from `nvidia-smi`; the commands above use 0). `--device cuda:N` with N > 0 hangs
("GPUs other than cuda:0 are not currently supported") and allocates memory on every GPU.

**System library needed: `libglu1-mesa`.** Isaac Sim's MDL SDK (`libneuray.so`) links `libGLU.so.1`,
which our container lacks. Without it the headless empty-stage test passes, but any env with cameras
fails: MDL does not load, the RTX shaders fail to compile and the process aborts with
`CUDA error: an illegal memory access` in `omni.physx.tensors` during env creation. Needs
`apt-get install libglu1-mesa` (as of 2026-11-21 not installed; bench env not yet stepped).

## Roles

- **LeRobot**: real-robot control, teleoperation, data collection, small policies (ACT, SmolVLA).
- **SO-101 Bench**: sim evaluation and sim2real. Its evaluator and real-robot runners talk to a policy
  server over HTTP (`POST /act`: camera PNGs, instruction, 6-D joint state -> chunk of joint targets).
- Our policies are evaluated in sim through the bench's LeRobot plugin:
  `lerobot-eval --env.type=so101_bench --policy.path=... --eval.n_episodes=1` (run from the lerobot
  env; the plugin starts the simulator from the isaac env as a subprocess).

## Remote control architecture

The arm (USB serial), the cameras and the robot client run on the home desktop. Policy inference runs
on the GPU server. Only the policy port crosses the network, through an SSH tunnel opened from the
home desktop.

```
home desktop                                         GPU server
────────────────────────────                         ───────────────────────────
SO-101 follower (USB serial)                         policy server
overhead + wrist cameras                               bench:   scripts/*_server.py   HTTP POST /act  :8000
robot client                                           lerobot: lerobot-policy-server Zenoh tcp       :7447
  bench real runner  ──┐
  lerobot-rollout    ──┴── ssh -L 8000:localhost:8000 -L 7447:localhost:7447 user@gpu-server ──▶
```

- Bench path: start a `scripts/*_server.py` on the GPU server (`--port 8000`); the home-side runner
  uses `--policy_port 8000` against the tunnelled localhost port.
- LeRobot path: `lerobot-policy-server` on the GPU server; on the home desktop
  `lerobot-rollout --inference.type=remote --inference.endpoint=tcp/127.0.0.1:7447`. In LeRobot 0.7.0
  asynchronous/remote inference uses Zenoh (the `remote` extra), not the gRPC server of older releases.
- Both sides need the same lerobot version; the home desktop needs only the `feetech,remote,core_scripts`
  extras and no GPU.
