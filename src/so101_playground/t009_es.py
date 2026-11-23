"""T009 coordinator: episodic antithetic ES over the 5-dim gain adapter, and eval job submission.

No GPU, no Isaac: it writes jobs to the file queue served by scripts/t009_worker.py and reads their results.
Run with any Python 3 that has numpy.

    python -m so101_playground.t009_es train --method epred   --root R    # reward 1: -e_pred only
    python -m so101_playground.t009_es train --method success --root R    # reward 2: success only
    python -m so101_playground.t009_es train --method mixed   --root R    # reward 3: -e_pred + LAMBDA*success
    python -m so101_playground.t009_es eval  --tag theta0 --theta 0,0,0,0,0 --root R
    python -m so101_playground.t009_es eval  --tag oracle --theta oracle --root R
    python -m so101_playground.t009_es eval  --tag theta0-nojit --theta 0,0,0,0,0 --no_jitter --root R

Update rule (identical for all three rewards; only R differs):
  theta_0 = 0 (k = 1). Pair p uses home bin layout row p % 6, episode id ("noise seed") 1000 + p shared by both
  members (InternW0-Δ's flow noise is in fact fixed for every call, see internw0_probe_server.py
  --seed_per_query, so the policy noise is common to all episodes anyway; what differs within a pair is theta,
  the layout jitter and physics/render nondeterminism), and its own independent layout jitter per
  member (dx, dy ~ U(-1, 1) cm, dyaw ~ U(-5, 5) deg). eps_p ~ N(0, I_5), delta_p = SIGMA * eps_p.
  Members: theta + delta_p and theta - delta_p (each clipped to [log 0.5, log 4] when run).
  d_p = R(theta + delta_p) - R(theta - delta_p) (ties -> 0; a missing reward -> 0).
  Every 2 pairs:  scale = RMS of all d so far in this run (incl. this batch); s_p = clip(d_p / scale, -3, 3)
  (scale = 0 -> s = 0, zero gradient);  g = mean_p(s_p * eps_p);  theta <- clip(theta + ETA * g).
  32 pairs = 64 episodes = 16 updates. eps_p, the jitter draws and the noise seeds depend only on p, so the three
  methods see the same perturbation and placement sequence (common random numbers); they diverge only through
  theta.

Eval (no exploration noise): theta fixed, home bin rows 0-5 x 2 jittered repeats (12 episodes); the jitter of
(row, repeat) and the noise seed 5000 + row are the same for every theta, so evals are paired across thetas, and
the two repeats of a row form a jittered theta-pair at equal noise seed (used for the jitter noise check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GAP_SCALE = (1.0, 0.35, 0.35, 0.6, 1.0)
THETA_ORACLE = [math.log(1.0 / s) for s in GAP_SCALE]
THETA_MIN, THETA_MAX = math.log(0.5), math.log(4.0)

SIGMA = 0.2
ETA = 0.2
N_PAIRS = 32
PAIRS_PER_UPDATE = 2
SHAPE_CLIP = 3.0
LAMBDA = 0.05  # reward 3: a success flip = 2 x the typical |paired e_pred diff| (0.026, T006 per-episode data)
HOME_ROWS = 6
JITTER_XY_M = 0.01
JITTER_YAW_DEG = 5.0
SEED = 9009
METHODS = ("epred", "success", "mixed")


def clip(theta):
    return [min(max(float(t), THETA_MIN), THETA_MAX) for t in theta]


def jitter(rng: np.random.Generator) -> list[float]:
    return [float(rng.uniform(-JITTER_XY_M, JITTER_XY_M)), float(rng.uniform(-JITTER_XY_M, JITTER_XY_M)),
            float(rng.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG))]


def pair_draws(p: int):
    """eps (5,), jitter of the + and - member. Depends on p only (common across methods)."""
    rng = np.random.default_rng([SEED, p])
    eps = rng.standard_normal(len(ARM))
    return eps, jitter(rng), jitter(rng)


def eval_jitter(row: int, rep: int) -> list[float]:
    return jitter(np.random.default_rng([SEED, 7777, row, rep]))


def reward(method: str, res: dict) -> float | None:
    e = res.get("e_pred")
    succ = 1.0 if res.get("success") else 0.0
    if method == "success":
        return succ
    if e is None:
        return None
    if method == "epred":
        return -float(e)
    if method == "mixed":
        return -float(e) + LAMBDA * succ
    raise ValueError(method)


# ---------------------------------------------------------------- queue
class Queue:
    def __init__(self, root: Path):
        self.q = root / "queue"
        for d in ("pending", "running", "done"):
            (self.q / d).mkdir(parents=True, exist_ok=True)

    def known(self, job_id: str) -> bool:
        return ((self.q / "done" / f"{job_id}.json").exists()
                or any((self.q / "pending").glob(f"*__{job_id}.json"))
                or any((self.q / "running").glob(f"{job_id}.*.json")))

    def submit(self, job: dict) -> None:
        if self.known(job["id"]):
            return
        job = dict(job, submitted=time.time())
        name = f"{job['order']}__{job['id']}.json"
        tmp = self.q / "pending" / f".{name}.tmp"
        tmp.write_text(json.dumps(job))
        os.replace(tmp, self.q / "pending" / name)

    def result(self, job_id: str) -> dict | None:
        p = self.q / "done" / f"{job_id}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def wait(self, ids: list[str], poll: float = 5.0) -> list[dict]:
        while True:
            res = [self.result(i) for i in ids]
            if all(r is not None for r in res):
                return res
            time.sleep(poll)


# ---------------------------------------------------------------- train
def train(args) -> None:
    method = args.method
    root = args.root
    queue = Queue(root)
    log_path = root / "es" / f"{method}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")  # rebuilt from the queue results on every (re)start: the run is deterministic in them
    theta = [0.0] * len(ARM)
    ds: list[float] = []
    n_pairs = args.pairs
    for u0 in range(0, n_pairs, PAIRS_PER_UPDATE):
        batch = list(range(u0, min(u0 + PAIRS_PER_UPDATE, n_pairs)))
        ids, draws = [], {}
        for p in batch:
            eps, jit_plus, jit_minus = pair_draws(p)
            draws[p] = eps
            for sign, jit in (("+", jit_plus), ("-", jit_minus)):
                th = clip([t + (1 if sign == "+" else -1) * SIGMA * e for t, e in zip(theta, eps)])
                job_id = f"{method}-p{p:02d}{'P' if sign == '+' else 'M'}"
                queue.submit({"id": job_id, "order": f"1_{p:02d}_{method}", "kind": "train", "method": method,
                              "tag": f"T009-{method}-train", "pair": p, "sign": sign, "set": "home",
                              "row": p % HOME_ROWS, "theta": th, "theta_center": list(theta),
                              "noise_seed": 1000 + p, "jitter": jit})
                ids.append(job_id)
        res = {r["id"]: r for r in queue.wait(ids)}
        rec_pairs = []
        for p in batch:
            rp, rm = res[f"{method}-p{p:02d}P"], res[f"{method}-p{p:02d}M"]
            Rp, Rm = reward(method, rp), reward(method, rm)
            d = 0.0 if (Rp is None or Rm is None) else Rp - Rm
            ds.append(d)
            rec_pairs.append({"pair": p, "row": p % HOME_ROWS, "R_plus": Rp, "R_minus": Rm, "d": d,
                              "eps": draws[p].tolist(),
                              "plus": {k: rp.get(k) for k in ("success", "e_pred", "e_track_deg", "path_len_deg", "ticks")},
                              "minus": {k: rm.get(k) for k in ("success", "e_pred", "e_track_deg", "path_len_deg", "ticks")}})
        scale = float(np.sqrt(np.mean(np.square(ds)))) if ds else 0.0
        g = np.zeros(len(ARM))
        for rp in rec_pairs:
            s = 0.0 if scale == 0.0 else float(np.clip(rp["d"] / scale, -SHAPE_CLIP, SHAPE_CLIP))
            rp["s"] = s
            g += s * np.asarray(rp["eps"])
        g /= len(rec_pairs)
        theta_before = list(theta)
        theta = clip([t + ETA * gi for t, gi in zip(theta, g)])
        rec = {"update": u0 // PAIRS_PER_UPDATE, "episodes": 2 * (u0 + len(batch)), "theta_before": theta_before,
               "theta": theta, "k": [math.exp(t) for t in theta], "g": g.tolist(), "scale": scale,
               "dist_oracle": float(np.linalg.norm(np.asarray(theta) - np.asarray(THETA_ORACLE))),
               "pairs": rec_pairs, "t": time.time()}
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{method}] update {rec['update']:2d} ep {rec['episodes']:3d} k={np.round(rec['k'], 2).tolist()} "
              f"|theta-theta*|={rec['dist_oracle']:.3f} d={[round(r['d'], 4) for r in rec_pairs]}", flush=True)
    (root / "es" / f"{method}.final.json").write_text(json.dumps({"method": method, "theta": theta,
                                                                  "k": [math.exp(t) for t in theta]}))
    if not args.no_eval:
        submit_eval(queue, f"T009-{method}-eval", theta, jittered=True, reps=args.eval_reps, t004=args.t004)


# ---------------------------------------------------------------- eval
def submit_eval(queue: Queue, tag: str, theta, jittered: bool, reps: int, t004: bool, order: str = "2",
                rows=None) -> list[str]:
    ids = []
    th = clip(theta)
    # theta hash in the id (review #3): a re-run with a different theta never reuses an earlier eval result.
    # The T009 run used ids without it; its eval thetas were checked equal to the final thetas (docs, review).
    h = hashlib.sha1(json.dumps([round(t, 9) for t in th]).encode()).hexdigest()[:8]
    for rep in range(reps):
        for row in (range(HOME_ROWS) if rows is None else rows):
            job_id = f"eval-{tag}-{h}-home{row}-r{rep}"
            queue.submit({"id": job_id, "order": f"{order}_{tag}_{rep}{row}", "kind": "eval", "tag": tag,
                          "set": "home", "row": row, "rep": rep, "theta": clip(theta), "noise_seed": 5000 + row,
                          "jitter": eval_jitter(row, rep) if jittered else None})
            ids.append(job_id)
    if t004:
        for row in range(8):
            job_id = f"eval-{tag}-{h}-t004{row}"
            queue.submit({"id": job_id, "order": f"3_{tag}_{row}", "kind": "eval", "tag": tag, "set": "t004",
                          "row": row, "rep": 0, "theta": clip(theta), "noise_seed": 6000 + row, "jitter": None})
            ids.append(job_id)
    return ids


def evaluate(args) -> None:
    theta = THETA_ORACLE if args.theta == "oracle" else [float(x) for x in args.theta.split(",")]
    ids = submit_eval(Queue(args.root), args.tag, theta, jittered=not args.no_jitter, reps=args.eval_reps,
                      t004=args.t004, order=args.order,
                      rows=None if args.rows is None else [int(r) for r in args.rows.split(",")])
    print(f"submitted {len(ids)} jobs for {args.tag} theta={np.round(clip(theta), 3).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--method", choices=METHODS, required=True)
    t.add_argument("--pairs", type=int, default=N_PAIRS)
    t.add_argument("--no_eval", action="store_true")
    t.add_argument("--eval_reps", type=int, default=2)
    t.add_argument("--t004", action="store_true", help="also evaluate T004 rows 0-7 once (budget permitting)")
    e = sub.add_parser("eval")
    e.add_argument("--tag", required=True)
    e.add_argument("--theta", required=True, help="'oracle' or 5 comma-separated log-gains")
    e.add_argument("--no_jitter", action="store_true")
    e.add_argument("--eval_reps", type=int, default=2)
    e.add_argument("--t004", action="store_true")
    e.add_argument("--order", default="2")
    e.add_argument("--rows", default=None, help="comma-separated home rows (default all 6)")
    for p in (t, e):
        p.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
