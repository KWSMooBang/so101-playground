"""Headless smoke test for the Isaac environment.

Without --task: start Isaac Sim, create an empty stage, step physics a few times.
With --task: create one SO-101 Bench env and step it with random actions (no policy).

    source envs/isaac/env.sh
    CUDA_VISIBLE_DEVICES=<gpu> python scripts/isaac_smoke.py --headless
    CUDA_VISIBLE_DEVICES=<gpu> python scripts/isaac_smoke.py --headless --enable_cameras \
        --task So101Bench-Bin-v0 --steps 100 --save_frames outputs/smoke
"""

import argparse
import time

t0 = time.time()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default=None, help="SO-101 Bench task id; omit for the empty-stage test.")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--action", choices=["random", "zero"], default="random")
parser.add_argument("--save_frames", type=str, default=None, help="Directory for one PNG per camera (last step).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app
t_app = time.time()
print(f"[smoke] app started in {t_app - t0:.1f}s", flush=True)

import torch


class GpuMemPoller:
    """Peak memory used on the visible GPU (whole device: containers often hide per-process usage)."""

    def __init__(self, period: float = 0.5):
        import threading

        self.baseline = self.read()
        self.peak = self.baseline
        self._stop = threading.Event()
        threading.Thread(target=self._run, args=(period,), daemon=True).start()

    @staticmethod
    def read() -> int:
        free, total = torch.cuda.mem_get_info(0)
        return (total - free) // 2**20

    def _run(self, period: float) -> None:
        while not self._stop.wait(period):
            self.peak = max(self.peak, self.read())

    def report(self) -> str:
        self._stop.set()
        self.peak = max(self.peak, self.read())
        return (
            f"device peak {self.peak} MiB (before env creation {self.baseline} MiB, "
            f"torch peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB)"
        )


def save_frames(env, out_dir: str) -> None:
    from pathlib import Path

    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, sensor in env.unwrapped.scene.sensors.items():
        output = getattr(getattr(sensor, "data", None), "output", None)
        if output is None or "rgb" not in output.keys():
            continue
        rgb = output["rgb"][0, ..., :3].cpu().numpy()
        path = out / f"{args.task}_{name}.png"
        Image.fromarray(rgb).save(path)
        print(f"[smoke] saved {path} {rgb.shape} mean={rgb.mean():.1f} std={rgb.std():.1f}", flush=True)


def empty_stage() -> None:
    import isaaclab.sim as sim_utils

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim.reset()
    for _ in range(args.steps):
        sim.step()
    print(f"[smoke] empty stage: {args.steps} steps OK on {sim.device}", flush=True)


def bench_env() -> None:
    import gymnasium as gym

    import so101_bench.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    mem = GpuMemPoller()
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=cfg)
    print(f"[smoke] observation space: {env.observation_space}", flush=True)
    print(f"[smoke] action space: {env.action_space}", flush=True)
    obs, _ = env.reset()
    t_reset = time.time()
    print(f"[smoke] env created + reset in {t_reset - t_app:.1f}s", flush=True)
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            if args.action == "random":
                actions = 2 * torch.rand_like(actions) - 1
            obs, rew, terminated, truncated, _ = env.step(actions)
    dt = time.time() - t_reset
    print(f"[smoke] {args.task}: {args.steps} {args.action} steps OK in {dt:.1f}s ({args.steps / dt:.1f} steps/s)", flush=True)

    def shapes(x, prefix=""):
        if isinstance(x, dict):
            for k, v in x.items():
                shapes(v, f"{prefix}{k}.")
        elif hasattr(x, "shape"):
            print(f"[smoke]   obs {prefix[:-1]}: {tuple(x.shape)} {x.dtype}", flush=True)

    shapes(obs)
    if args.save_frames:
        save_frames(env, args.save_frames)
    print(f"[smoke] gpu mem: {mem.report()}", flush=True)
    env.close()


if __name__ == "__main__":
    import os
    import threading

    code = 1
    try:
        bench_env() if args.task else empty_stage()
        print(f"[smoke] PASS total {time.time() - t0:.1f}s", flush=True)
        code = 0
    finally:
        # Kit sometimes hangs on shutdown in headless containers; do not hold the GPU if it does.
        threading.Timer(30.0, lambda: os._exit(code)).start()
        simulation_app.close()
        os._exit(code)
