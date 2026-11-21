# so101-playground

A playground for SO-101 benchmarks, simulators and frameworks, and for operating and evaluating a
real SO-101 (overhead + wrist cameras) remotely. See `RESEARCH.md` for goals and decisions.

## Layout

```
third_party/so101_bench   submodule, github.com/5hadytru/so101_bench @ 470e09e (main, 2026-11-04)
third_party/lerobot       submodule, github.com/huggingface/lerobot  @ v0.7.0
src/so101_playground/     our package
scripts/                  our scripts (isaac_smoke.py)
envs/isaac/               Isaac env: setup_isaac.sh, env.sh, Vulkan ICD manifest
envs/lerobot/             LeRobot env: setup_lerobot.sh, env.sh
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

### WAM control-gap probe (T004)

Extra envs: `envs/lawam/setup_lawam.sh` (LaWAM, its own lerobot venv) and `envs/flux3/setup_flux3.sh`
(flux-action venv + FLUX 3 weights, gated). The isaac-side launcher `scripts/run_t004_eval.sh` sources
`$ISAAC_ENV/envs/isaac/env.sh`. `ISAAC_ENV` defaults to the checkout that built the isaac venv on our GPU
server (`/workspace/so101-playground/.claude/worktrees/scaffold-env`); set `ISAAC_ENV=<your checkout>`
anywhere else. Method, commands and results: `docs/T004-wam-control-gap.md`.

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
