#!/usr/bin/env python3
"""T004 analysis: per-condition success, e_track, e_pred, e_ideal (RESEARCH.md H2-P0).

Inputs per run dir ``<root>/<model>/<condition>/``: ``episodes.jsonl`` and ``ep*.npz`` from
scripts/wam_gap_eval.py, plus ``<root>/<model>/lam_scores.jsonl`` (LaWAM; lawam_probe_server.py's log, rows
tagged with the condition) or ``flux_metrics.jsonl`` (FLUX; flux_pixel_metrics.py).

Definitions
  e_track       mean |cmd(t) - q(t+1)| over the five arm joints and all policy-controlled ticks, deg.
                cmd = the policy's target before the gap; q = measured joint position after the step.
  e_track@h     per query, mean over arm joints of |cmd(s+h-1) - q(s+h)| (the command issued just before
                the predicted frame vs the arm at that frame), averaged over queries.
  e_pred        LaWAM: mse(h_t1_pred, LAM(observed overhead frame s+35)) (its training loss). FLUX: see
                flux_pixel_metrics.py.
  e_ideal       the same distance against the kinematic-replay frame (perfect tracking of the same chunk from
                the same state).
  e_static      mse(h_t, LAM(observed frame s+35)): the "nothing moves" baseline, for scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM = slice(0, 5)


def _spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return None
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def track_stats(run: Path, horizon: int) -> tuple[dict, dict]:
    """Returns (summary, per-query e_track@horizon keyed by (episode, query_frame))."""
    per_tick, per_query = [], {}
    for f in sorted(run.glob("ep[0-9]*.npz")):
        d = np.load(f, allow_pickle=True)
        ep = int(f.stem[2:])
        cmd, q, tick = d["cmd"], d["q"], d["tick"]  # row i: cmd issued at tick i, q after the step
        hold = int(d["hold_steps"])
        err = np.degrees(np.abs(cmd[:, ARM] - q[:, ARM])).mean(axis=1)
        per_tick.append(err[hold:])
        for s in d["queries"]:
            s = int(s)
            # row index of the command issued at loop step s is s (tick counts from 1 after the step)
            i = s + horizon - 1
            if i < len(err):
                per_query[(ep, s)] = (float(err[i]), float(err[s:i + 1].mean()))
    allerr = np.concatenate(per_tick) if per_tick else np.zeros(0)
    vals = list(per_query.values())
    return ({"e_track_deg": float(allerr.mean()) if allerr.size else None,
             "e_track_at_h_deg": float(np.mean([v[0] for v in vals])) if vals else None,
             "e_track_chunk_deg": float(np.mean([v[1] for v in vals])) if vals else None},
            per_query)


def episodes_stats(run: Path) -> dict:
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    return {"n_episodes": len(rows), "success": float(np.mean([r["success"] for r in rows])) if rows else None,
            "n_success": int(sum(r["success"] for r in rows)),
            "mean_ticks": float(np.mean([r["ticks"] for r in rows])) if rows else None,
            "wall_s": float(sum(r["wall_s"] for r in rows)), "inference_wall_s": float(sum(r["inference_wall_s"] for r in rows)),
            "n_queries": int(sum(r["n_queries"] for r in rows))}


def lawam_stats(run: Path, per_query_track: dict) -> dict:
    p = run.parent / "lam_scores.jsonl"  # one server log for all conditions, rows tagged by condition
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    rows = [r for r in rows if r.get("condition") == run.name and r["episode"] >= 0]
    obs = {(r["episode"], r["frame"]): r for r in rows if r["kind"] == "observed"}
    ide = {(r["episode"], r["frame"]): r for r in rows if r["kind"] == "ideal"}
    out = {"n_obs_scores": len(obs), "n_ideal_scores": len(ide)}
    for name, d in (("obs", obs), ("ideal", ide)):
        for k in ("e_pred_lat", "cos_pred_lat", "e_static_lat", "e_act_lat", "cos_act_lat"):
            v = [r[k] for r in d.values() if r.get(k) is not None]
            out[f"{name}_{k}"] = float(np.mean(v)) if v else None
    paired = [k for k in obs if k in ide]
    out["n_paired"] = len(paired)
    if paired:
        out["paired_obs_e_pred_lat"] = float(np.mean([obs[k]["e_pred_lat"] for k in paired]))
        out["paired_ideal_e_pred_lat"] = float(np.mean([ide[k]["e_pred_lat"] for k in paired]))
    keys = [k for k in obs if k in per_query_track]
    out["spearman_e_pred_vs_e_track_at_h"] = _spearman([obs[k]["e_pred_lat"] for k in keys],
                                                       [per_query_track[k][0] for k in keys])
    out["spearman_e_pred_vs_e_track_chunk"] = _spearman([obs[k]["e_pred_lat"] for k in keys],
                                                        [per_query_track[k][1] for k in keys])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="/outputs/so101-playground/runs/T004")
    ap.add_argument("--model", required=True)
    ap.add_argument("--horizon", type=int, default=35)
    ap.add_argument("--conditions", default="none,gain,latency,offset,backlash")
    ap.add_argument("--tsv", type=Path, default=None, help="Append one row per condition (results.tsv).")
    args = ap.parse_args()
    summary = {}
    for cond in args.conditions.split(","):
        run = args.root / args.model / cond
        if not (run / "episodes.jsonl").exists():
            continue
        s = episodes_stats(run)
        tr, pq = track_stats(run, args.horizon)
        s.update(tr)
        if args.model == "lawam":
            s.update(lawam_stats(run, pq))
        else:
            fm = run / "flux_metrics_summary.json"
            if fm.exists():
                s.update(json.loads(fm.read_text()))
        summary[cond] = s
    out = args.root / args.model / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.tsv and args.model == "lawam":
        cols = ["run_id", "task", "model", "condition", "n_episodes", "success", "e_pred@35", "e_pred@35_paired", "e_ideal@35_paired",
                "e_static@35", "e_track_deg", "e_track@35_deg", "e_act_lat", "rho_pred_track", "n_queries", "notes"]
        new = not args.tsv.exists() or not args.tsv.read_text().strip()
        with args.tsv.open("a") as f:
            if new:
                f.write("\t".join(cols) + "\n")
            for cond, s in summary.items():
                def g(k, fmt="{:.4f}"):
                    v = s.get(k)
                    return "" if v is None else fmt.format(v)
                row = [f"T004-{cond}", "T004-wam-pred-gap-probe", "lawam", cond, str(s["n_episodes"]), g("success", "{:.3f}"),
                       g("obs_e_pred_lat"), g("paired_obs_e_pred_lat"), g("paired_ideal_e_pred_lat"), g("obs_e_static_lat"), g("e_track_deg", "{:.3f}"),
                       g("e_track_at_h_deg", "{:.3f}"), g("obs_e_act_lat"), g("spearman_e_pred_vs_e_track_chunk", "{:.3f}"),
                       str(s["n_queries"]),
                       "LaWAM full FT u45k; Bin 1-obj val_set; e_* = LAM DINOv3 token MSE at +35 ticks; "
                       "paired = replayed queries (every 2nd), same query set for e_pred and e_ideal"]
                f.write("\t".join(row) + "\n")


if __name__ == "__main__":
    main()
