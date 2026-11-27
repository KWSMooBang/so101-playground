"""T012 coordinator (H2 perception gap, stage A): overhead-camera pose sweep, no learning.

No GPU, no Isaac: writes jobs to the file queue served by scripts/t009_worker.py (started with
``--condition none --score_latents 2``) and returns. Run with any Python 3 that has numpy.

    python -m so101_playground.t012_coord sweep --root R                      # 9 conditions x 12 episodes
    python -m so101_playground.t012_coord sweep --root R --conds tilt+12      # the Delta* follow-up, if needed
    python -m so101_playground.t012_coord smoke --root R                      # 1 short episode per axis

Conditions (so101_playground.camera_delta): d0, tilt+-4, tilt+-8, pan+-4, pan+-8 (deg).
Episodes: home bin rows 0-5 x 2 jittered repeats. The jitter of (row, rep) and the episode id 5000 + row are
T010's eval draws (t010_es.eval_jitter), the same for every condition, so every condition is paired with d0.
Every job runs ``STATIC_QUERIES`` = 3 static-probe queries first. No adapter (theta = 0), no control gap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from so101_playground.camera_delta import CameraDelta
from so101_playground.t009_es import Queue
from so101_playground.t010_es import eval_jitter

HOME_ROWS = 6
REPS = 2
STATIC_QUERIES = 3
CONDS = {
    "d0": "",
    "tilt+4": "tilt=4", "tilt-4": "tilt=-4", "tilt+8": "tilt=8", "tilt-8": "tilt=-8",
    "pan+4": "pan=4", "pan-4": "pan=-4", "pan+8": "pan=8", "pan-8": "pan=-8",
    "tilt+5": "tilt=5", "tilt-5": "tilt=-5", "pan+5": "pan=5", "pan-5": "pan=-5",  # smoke contact sheet
    # follow-ups (gate: only if nothing up to 8 deg breaks success)
    "tilt+12": "tilt=12", "tilt-12": "tilt=-12", "pan+12": "pan=12", "pan-12": "pan=-12",
}
STAGE_A = ["d0", "tilt+4", "tilt-4", "tilt+8", "tilt-8", "pan+4", "pan-4", "pan+8", "pan-8"]


def job(cond: str, row: int, rep: int, prefix: str = "t012", order: str = "1", **extra) -> dict:
    spec = CONDS[cond]
    CameraDelta.parse(spec)  # validate
    return {**{"id": f"{prefix}-{cond}-home{row}-r{rep}", "order": f"{order}_{rep}{row}_{cond}", "kind": "stageA",
            "tag": f"T012A-{cond}", "cond": cond, "camera_delta": spec, "static_queries": STATIC_QUERIES,
            "set": "home", "row": row, "rep": rep, "theta1": 0.0, "theta": [0.0] * 5,
            "noise_seed": 5000 + row, "jitter": eval_jitter(row, rep), "gap": "none"}, **extra}


def sweep(args) -> None:
    q = Queue(args.root)
    n = 0
    # interleave conditions within each (rep, row) so a partial run is still balanced across conditions
    for rep in range(REPS):
        for row in range(HOME_ROWS):
            for cond in args.conds:
                q.submit(job(cond, row, rep))
                n += 1
    print(f"submitted {n} jobs: {args.conds} x {HOME_ROWS} rows x {REPS} reps")


def smoke(args) -> None:
    """Contact sheet (first policy frame at d0, tilt +-5, pan +-5; 1 policy tick, no probe) and one short
    end-to-end episode per axis (d0, tilt+4, pan+4; razor row, <= 300 ticks, with probe, @2 scoring, reach error)."""
    q = Queue(args.root)
    for cond in ["d0", "tilt+5", "tilt-5", "pan+5", "pan-5"]:
        q.submit(job(cond, row=0, rep=0, prefix="t012sheet", order="0", kind="sheet", static_queries=0,
                     max_ticks=50, save_frames=1))
    for cond in ["d0", "tilt+4", "pan+4"]:
        q.submit(job(cond, row=5, rep=0, prefix="t012smoke", order="0", kind="smoke", max_ticks=300))
    print("submitted 5 contact-sheet jobs and 3 smoke episodes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--conds", nargs="+", default=STAGE_A, choices=tuple(CONDS))
    m = sub.add_parser("smoke")
    for p in (s, m):
        p.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    sweep(args) if args.cmd == "sweep" else smoke(args)


if __name__ == "__main__":
    main()
