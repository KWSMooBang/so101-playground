"""T012 static probe and scoring schedule (pure Python; used by scripts/t009_worker.py and tests/test_t012.py).

Static probe (``--static_queries N`` or job["static_queries"]): before the real rollout of a job the worker
  1. resets the env exactly as for the rollout (same layout, jitter and camera pose) and holds the docked pose for
     the usual ``hold_steps`` (1.5 s), so the probe sees the same settled scene as the rollout's first query;
  2. sends N /act queries at ticks hold_steps + 16 i (i < N) with the arm kept docked (the chunks are never
     executed) and scores each against the docked frames that follow, @1 at +16 and @2 at +32;
  3. resets the env again (the rollout starts from a fresh reset, so its timeout clock and physics are those of
     a job without a probe) and clears the client's action queue.
Isolation: the probe's queries carry trace dir ``<run dir>#<key>.static`` and its scores ``attempt=<key>.static``,
the rollout's ``<run dir>#<key>`` / ``<key>``. The server keys sessions by (trace dir, episode) and predictions by
(trace dir, episode, frame), so the probe has its own policy session and never enters the rollout's memory.
The rollout's first query is at the same tick on the same scene as probe query 0 with a fresh session, so with
InternW0's deterministic flow noise their chunks must match: the worker logs the max difference as a check.

e_pred_static(Delta) = mean probe e_pred@1 (and @2): the error of the model's predicted future against a scene in
which nothing moves, at that camera pose. It is the visual floor in the sense of the brief: it moves with the view
only, never with what the arm does.
"""

from __future__ import annotations

QUERY_EVERY = 16  # = action horizon; InternW0 latent k covers ticks 16(k-1)+4 .. 16k after the query
OFFSETS = {1: [4, 8, 12, 16], 2: [4, 8, 12, 16, 20, 24, 28, 32]}


def probe_dir(run_dir: str, key: str) -> str:
    return f"{run_dir}#{probe_attempt(key)}"


def probe_attempt(key: str) -> str:
    return f"{key}.static"


def rollout_dir(run_dir: str, key: str) -> str:
    return f"{run_dir}#{key}"


def probe_schedule(n: int, hold_steps: int, latents: int = 2) -> dict:
    """Query ticks, score ticks {tick: [(query tick, latent)]} and the total ticks the probe needs."""
    queries = [hold_steps + QUERY_EVERY * i for i in range(n)]
    scores: dict[int, list[tuple[int, int]]] = {}
    for q in queries:
        for k in range(1, latents + 1):
            scores.setdefault(q + QUERY_EVERY * k, []).append((q, k))
    total = (max(scores) if scores else hold_steps)
    return {"queries": queries, "scores": scores, "total_ticks": total}


def frames_needed(step: int, queries: list[int], latents: int) -> bool:
    """True if the frame after ``step`` ticks is one some recent query will be scored on."""
    return any(0 < step - q <= QUERY_EVERY * latents and (step - q) % 4 == 0 for q in queries[-(latents + 1):])


def score_due(step: int, queries: list[int], latents: int) -> list[tuple[int, int]]:
    """(query tick, latent k) pairs whose score window ends at ``step``."""
    return [(q, k) for q in queries[-(latents + 1):] for k in range(1, latents + 1) if step - q == QUERY_EVERY * k]
