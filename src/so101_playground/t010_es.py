"""T010 coordinator (H2-P1 stage A2): 1-dim antithetic ES over a shared lift+elbow gain, and eval/pre-check jobs.

No GPU, no Isaac: it writes jobs to the file queue served by scripts/t009_worker.py (started with
``--condition gainlh``) and reads their results. Run with any Python 3 that has numpy.

    python -m so101_playground.t010_es eval  --tag T010-baseline-theta0 --theta 0 --root R    # pre-check
    python -m so101_playground.t010_es eval  --tag T010-oracle --theta oracle --root R         # pre-check
    python -m so101_playground.t010_es train --method epred --root R    # reward 1: -e_pred only
    python -m so101_playground.t010_es train --method mixed --root R    # reward 3: -e_pred + LAMBDA*success

Gap (fixed, unknown to the learner): arm PD stiffness x GAP_S on shoulder_lift and elbow_flex, others nominal
(control_gap ``gainlh``, s = 0.25; ``gainlh20``, s = 0.2, if the pre-check gate fails). Oracle k* = 1/s.

Adapter: one scalar theta = log k shared by shoulder_lift and elbow_flex (sent = q + k (cmd - q) on those two
joints only; pan, wrist_flex, wrist_roll at k = 1). Sent to the worker as the 5-vector (0, t, t, 0, 0).
theta_0 = 0, clipped to k in [0.5, 6].

ES (identical for both rewards; only R and the eps stream differ):
  pair p: home bin row p % 6, episode id 1000 + p, independent jitter per member (dx, dy ~ U(-1, 1) cm,
  dyaw ~ U(-5, 5) deg). Layout order, episode ids and jitter draws depend only on p (same for both methods).
  eps_p ~ N(0, 1) from a per-method stream (independent across methods, unlike T009's shared eps).
  Members theta + SIGMA eps_p and theta - SIGMA eps_p (clipped). d_p = R(+) - R(-).
  Every PAIRS_PER_UPDATE = 4 pairs: scale = RMS of all d so far in this run (incl. this batch),
  s_p = clip(d_p / scale, -3, 3), g = mean_p(s_p eps_p), theta <- clip(theta + ETA g).
  Step size: with a consistent full-SNR gradient d_p = c eps_p, s_p = eps_p / rms(eps) and
  E[g] = E[eps^2] / sqrt(E[eps^2]) = 1, so ETA = 0.25 moves theta by ~0.25 per update (pure noise: d independent
  of eps gives E[g] = 0 and sd(g) ~ 1/sqrt(4) = 0.5, i.e. a +-0.125 random walk per update).
  Reported adapter: theta_bar = mean of the last 5 post-update iterates (the raw iterate is logged too).
  60 pairs = 120 episodes = 15 updates per method.

Eval (no exploration noise): theta fixed, home bin rows 0-5 x 2 jittered repeats (12 episodes); the jitter of
(row, repeat) and the episode id 5000 + row are the same for every theta, so evals are paired across thetas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from so101_playground.t009_es import Queue, jitter

GAPS = {"gainlh": 0.25, "gainlh20": 0.2}
K_MIN, K_MAX = 0.5, 6.0
THETA_MIN, THETA_MAX = math.log(K_MIN), math.log(K_MAX)

SIGMA = 0.3
ETA = 0.25
N_PAIRS = 60
PAIRS_PER_UPDATE = 4
SHAPE_CLIP = 3.0
AVG_LAST = 5
LAMBDA = 0.025  # reward 3: a success flip = 2 x T009's median |paired e_pred diff| (0.0125)
HOME_ROWS = 6
SEED = 10010
METHODS = ("epred", "mixed")
METHOD_STREAM = {"epred": 1, "mixed": 3}


def theta_oracle(gap: str) -> float:
    return math.log(1.0 / GAPS[gap])


def clip(t: float) -> float:
    return min(max(float(t), THETA_MIN), THETA_MAX)


def vec(t: float) -> list[float]:
    """1-dim theta -> the worker's 5-dim log-gain vector (pan, lift, elbow, wrist_flex, wrist_roll)."""
    t = clip(t)
    return [0.0, t, t, 0.0, 0.0]


def pair_eps(method: str, p: int) -> float:
    return float(np.random.default_rng([SEED, METHOD_STREAM[method], p]).standard_normal())


def pair_jitter(p: int):
    rng = np.random.default_rng([SEED, 0, p])
    return jitter(rng), jitter(rng)


def eval_jitter(row: int, rep: int) -> list[float]:
    return jitter(np.random.default_rng([SEED, 7777, row, rep]))


def reward(method: str, res: dict) -> float | None:
    e = res.get("e_pred")
    if e is None:
        return None
    succ = 1.0 if res.get("success") else 0.0
    if method == "epred":
        return -float(e)
    if method == "mixed":
        return -float(e) + LAMBDA * succ
    raise ValueError(method)


def theta_bar(iterates: list[float]) -> float:
    return float(np.mean(iterates[-AVG_LAST:]))


# ---------------------------------------------------------------- train
def train(args) -> None:
    method, root = args.method, args.root
    queue = Queue(root)
    t_star = theta_oracle(args.gap)
    log_path = root / "es" / f"{method}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")  # rebuilt from the queue results on every (re)start: deterministic in them
    theta, iterates, ds = 0.0, [], []
    for u0 in range(0, args.pairs, PAIRS_PER_UPDATE):
        batch = list(range(u0, min(u0 + PAIRS_PER_UPDATE, args.pairs)))
        ids = []
        for p in batch:
            eps = pair_eps(method, p)
            for sign, jit in zip("+-", pair_jitter(p)):
                th = clip(theta + (1 if sign == "+" else -1) * SIGMA * eps)
                job_id = f"t010-{method}-p{p:02d}{'P' if sign == '+' else 'M'}"
                queue.submit({"id": job_id, "order": f"1_{p:02d}_{method}", "kind": "train", "method": method,
                              "tag": f"T010-{method}-train", "pair": p, "sign": sign, "set": "home",
                              "row": p % HOME_ROWS, "theta1": th, "theta": vec(th), "theta_center": theta,
                              "eps": eps, "noise_seed": 1000 + p, "jitter": jit, "gap": args.gap})
                ids.append(job_id)
        res = {r["id"]: r for r in queue.wait(ids)}
        rec_pairs = []
        for p in batch:
            rp, rm = res[f"t010-{method}-p{p:02d}P"], res[f"t010-{method}-p{p:02d}M"]
            Rp, Rm = reward(method, rp), reward(method, rm)
            d = 0.0 if (Rp is None or Rm is None) else Rp - Rm
            ds.append(d)
            rec_pairs.append({"pair": p, "row": p % HOME_ROWS, "R_plus": Rp, "R_minus": Rm, "d": d,
                              "eps": pair_eps(method, p), "theta_plus": rp["theta1"], "theta_minus": rm["theta1"],
                              **{f"{side}": {k: r.get(k) for k in ("success", "e_pred", "e_track_deg",
                                                                   "path_len_deg", "ticks", "reason")}
                                 for side, r in (("plus", rp), ("minus", rm))}})
        scale = float(np.sqrt(np.mean(np.square(ds))))
        g = 0.0
        for rp in rec_pairs:
            s = 0.0 if scale == 0.0 else float(np.clip(rp["d"] / scale, -SHAPE_CLIP, SHAPE_CLIP))
            rp["s"] = s
            g += s * rp["eps"]
        g /= len(rec_pairs)
        theta_before = theta
        theta = clip(theta + ETA * g)
        iterates.append(theta)
        tb = theta_bar(iterates)
        rec = {"update": u0 // PAIRS_PER_UPDATE, "episodes": 2 * (u0 + len(batch)), "theta_before": theta_before,
               "theta": theta, "k": math.exp(theta), "theta_bar": tb, "k_bar": math.exp(tb), "g": g,
               "scale": scale, "theta_star": t_star, "pairs": rec_pairs, "t": time.time()}
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{method}] update {rec['update']:2d} ep {rec['episodes']:3d} k={rec['k']:.2f} k_bar={rec['k_bar']:.2f} "
              f"(k*={math.exp(t_star):.2f}) g={g:+.2f} d={[round(r['d'], 4) for r in rec_pairs]}", flush=True)
    tb = theta_bar(iterates)
    (root / "es" / f"{method}.final.json").write_text(json.dumps(
        {"method": method, "theta_bar": tb, "k_bar": math.exp(tb), "theta_last": theta, "k_last": math.exp(theta),
         "iterates": iterates}))
    if not args.no_eval:
        submit_eval(queue, f"T010-{method}-eval", tb, gap=args.gap)


# ---------------------------------------------------------------- eval
def submit_eval(queue: Queue, tag: str, t: float, gap: str, order: str = "2", reps: int = 2) -> list[str]:
    th = clip(t)
    h = hashlib.sha1(json.dumps(round(th, 9)).encode()).hexdigest()[:8]
    ids = []
    for rep in range(reps):
        for row in range(HOME_ROWS):
            job_id = f"t010-eval-{tag}-{h}-home{row}-r{rep}"
            queue.submit({"id": job_id, "order": f"{order}_{tag}_{rep}{row}", "kind": "eval", "tag": tag,
                          "set": "home", "row": row, "rep": rep, "theta1": th, "theta": vec(th),
                          "noise_seed": 5000 + row, "jitter": eval_jitter(row, rep), "gap": gap})
            ids.append(job_id)
    return ids


def evaluate(args) -> None:
    t = theta_oracle(args.gap) if args.theta == "oracle" else float(args.theta)
    ids = submit_eval(Queue(args.root), args.tag, t, gap=args.gap, order=args.order)
    print(f"submitted {len(ids)} jobs for {args.tag} theta={clip(t):.4f} k={math.exp(clip(t)):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--method", choices=METHODS, required=True)
    t.add_argument("--pairs", type=int, default=N_PAIRS)
    t.add_argument("--no_eval", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--tag", required=True)
    e.add_argument("--theta", required=True, help="'oracle' or log k (shared lift+elbow)")
    e.add_argument("--order", default="0")
    for p in (t, e):
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--gap", choices=tuple(GAPS), default="gainlh")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
