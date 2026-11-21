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
import re
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


def _boot_ci(values: dict, n: int = 2000, seed: int = 0) -> list[float] | None:
    """95% CI of the mean over queries, resampling whole episodes (keys are (episode, frame))."""
    eps = sorted({k[0] for k in values})
    if len(eps) < 2:
        return None
    by_ep = {e: [v for (ee, _), v in values.items() if ee == e] for e in eps}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        pick = rng.choice(eps, size=len(eps), replace=True)
        means.append(np.mean(np.concatenate([by_ep[e] for e in pick])))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def delta_vs_none(summary: dict, n: int = 2000, seed: int = 0) -> None:
    """(condition - none) for every per-query series stored under "_q:<name>" ({episode: [values]}).
    Point estimate = difference of query-weighted means (the same weighting as every mean in the tables);
    95% CI = episode-paired bootstrap: resample episode ids (same layouts in both conditions) and recompute
    both query-weighted means over the resampled episodes."""
    if "none" not in summary:
        return
    rng = np.random.default_rng(seed)
    for key, base in summary["none"].items():
        if not key.startswith("_q:"):
            continue
        name = key[3:]
        for cond, s in summary.items():
            cur = s.get(key, {})
            eps = sorted(set(base) & set(cur))
            if cond == "none" or len(eps) < 2:
                continue
            def qmean(series, pick):
                return float(np.mean(np.concatenate([series[e] for e in pick])))
            s[f"{name}_minus_none"] = qmean(cur, eps) - qmean(base, eps)
            boots = []
            for _ in range(n):
                pick = rng.choice(eps, size=len(eps), replace=True)
                boots.append(qmean(cur, pick) - qmean(base, pick))
            s[f"{name}_minus_none_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def _per_query(values: dict) -> dict:
    """{(episode, frame): v} -> {episode: [v, ...]} for delta_vs_none."""
    out: dict = {}
    for (e, _), v in values.items():
        out.setdefault(str(e), []).append(float(v))
    return out


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
    # Only rows of the attempt that completed each episode (episodes.jsonl "attempt"); a crashed or re-run attempt
    # leaves rows with another attempt id in the append-only log. Legacy rows without ids: the last row per key wins.
    attempts = {}
    for line in (run / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            if e.get("attempt"):
                attempts[e["episode"]] = e["attempt"]
    rows = [r for r in rows if r["episode"] not in attempts or r.get("attempt") == attempts[r["episode"]]]
    obs = {(r["episode"], r["frame"]): r for r in rows if r["kind"] == "observed"}
    ide = {(r["episode"], r["frame"]): r for r in rows if r["kind"] == "ideal"}
    out = {"n_obs_scores": len(obs), "n_ideal_scores": len(ide)}
    for name, d in (("obs", obs), ("ideal", ide)):
        for k in ("e_pred_lat", "cos_pred_lat", "e_static_lat", "e_act_lat", "cos_act_lat", "e_exec_lat"):
            v = [r[k] for r in d.values() if r.get(k) is not None]
            out[f"{name}_{k}"] = float(np.mean(v)) if v else None
    paired = [k for k in obs if k in ide]
    out["n_paired"] = len(paired)
    if paired:
        out["paired_obs_e_pred_lat"] = float(np.mean([obs[k]["e_pred_lat"] for k in paired]))
        out["paired_ideal_e_pred_lat"] = float(np.mean([ide[k]["e_pred_lat"] for k in paired]))
        diff = {k: obs[k]["e_pred_lat"] - ide[k]["e_pred_lat"] for k in paired}
        out["paired_diff"] = float(np.mean(list(diff.values())))
        out["paired_diff_ci95"] = _boot_ci(diff)
        out["paired_obs_gt_ideal_frac"] = float(np.mean([v > 0 for v in diff.values()]))
        out["_q:paired_diff"] = _per_query(diff)
        exe = {k: r["e_exec_lat"] for k, r in ide.items() if r.get("e_exec_lat") is not None}
        if exe:
            out["e_exec_lat_ci95"] = _boot_ci(exe)
    ratio = {k: r["e_pred_lat"] / r["e_static_lat"] for k, r in obs.items() if r["e_static_lat"] > 0}
    out["obs_e_pred_over_static_median"] = float(np.median(list(ratio.values()))) if ratio else None
    out["obs_e_pred_ci95"] = _boot_ci({k: r["e_pred_lat"] for k, r in obs.items()})
    out["_q:e_pred"] = _per_query({k: r["e_pred_lat"] for k, r in obs.items()})
    keys = [k for k in obs if k in per_query_track]
    out["spearman_e_pred_vs_e_track_at_h"] = _spearman([obs[k]["e_pred_lat"] for k in keys],
                                                       [per_query_track[k][0] for k in keys])
    out["spearman_e_pred_vs_e_track_chunk"] = _spearman([obs[k]["e_pred_lat"] for k in keys],
                                                        [per_query_track[k][1] for k in keys])
    return out


FLUX_METRICS = ("l1_scene", "l1_wrist", "lpips_scene", "lpips_wrist", "dino_scene", "dino_wrist")
FLUX_HORIZONS = (8, 16, 31)


def flux_stats(run: Path) -> dict:
    p = run / "flux_metrics.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    out = {"n_metric_rows": len(rows)}
    # per-query chunk-mean e_track for the rank correlation
    _, pq = track_stats(run, 16)
    for j in FLUX_HORIZONS:
        rj = [r for r in rows if r["j"] == j]
        for m in FLUX_METRICS:
            name = f"{m}@{j}"
            obs = {(r["episode"], r["frame"]): r[f"pred_obs.{m}"] for r in rj if f"pred_obs.{m}" in r}
            if not obs:
                continue
            out[f"e_pred.{name}"] = float(np.mean(list(obs.values())))
            out[f"e_pred.{name}_ci95"] = _boot_ci(obs)
            out[f"_q:e_pred.{name}"] = _per_query(obs)
            ide = {(r["episode"], r["frame"]): r[f"pred_ideal.{m}"] for r in rj if f"pred_ideal.{m}" in r}
            exe = {(r["episode"], r["frame"]): r[f"obs_ideal.{m}"] for r in rj if f"obs_ideal.{m}" in r}
            paired = [k for k in ide if k in obs]
            if paired:
                diff = {k: obs[k] - ide[k] for k in paired}
                out[f"paired_e_pred.{name}"] = float(np.mean([obs[k] for k in paired]))
                out[f"paired_e_ideal.{name}"] = float(np.mean([ide[k] for k in paired]))
                out[f"paired_diff.{name}"] = float(np.mean(list(diff.values())))
                out[f"paired_diff.{name}_ci95"] = _boot_ci(diff)
                out[f"e_exec.{name}"] = float(np.mean(list(exe.values())))
            keys = [k for k in obs if k in pq]
            out[f"rho_track.{name}"] = _spearman([obs[k] for k in keys], [pq[k][1] for k in keys])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="/outputs/so101-playground/runs/T004")
    ap.add_argument("--model", required=True)
    ap.add_argument("--horizon", type=int, default=35)
    ap.add_argument("--conditions", default="none,gain,latency,offset,backlash,gain2,latency6,offset6")
    ap.add_argument("--tsv", type=Path, default=None, help="Append one row per condition (results.tsv).")
    ap.add_argument("--run_prefix", default="T004", help="run_id prefix: <prefix>-<model>-<condition>.")
    ap.add_argument("--ideal_invalid", action="store_true",
                    help="Runs made with the pre-review replay (unclamped chunk targets, gap actuators, no RTX flush): "
                         "write e_ideal / obs-ideal / e_exec as 'invalid'.")
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
            s.update(flux_stats(run))
        summary[cond] = s
    delta_vs_none(summary)
    for s in summary.values():
        for k in [k for k in s if k.startswith("_q:")]:
            s.pop(k)
    out = args.root / args.model / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.tsv:
        write_tsv(args.tsv, args.model, summary, prefix=args.run_prefix, ideal_invalid=args.ideal_invalid)


TSV_COLS = ["run_id", "task", "model", "condition", "n_episodes", "success", "e_track_deg", "e_track@h_deg", "metric",
            "e_pred", "e_pred_paired", "e_ideal_paired", "e_exec", "e_pred_minus_none", "e_pred_minus_none_ci95",
            "rho_pred_track", "n_queries", "notes"]
FLUX_PRIMARY = "dino_scene@16"  # fixed before the sweep; others go to notes


def write_tsv(path: Path, model: str, summary: dict, prefix: str = "T004", ideal_invalid: bool = False) -> None:
    new = not path.exists() or not path.read_text().strip()
    with path.open("a") as f:
        if new:
            f.write("\t".join(TSV_COLS) + "\n")
        for cond, s in summary.items():
            def g(k, fmt="{:.4f}"):
                v = s.get(k)
                if v is None:
                    return ""
                if isinstance(v, list):
                    return "[" + ",".join(fmt.format(x) for x in v) + "]"
                return fmt.format(v)
            if model == "lawam":
                metric = "LAM_DINOv3_token_mse@35"
                vals = [g("obs_e_pred_lat"), g("paired_obs_e_pred_lat"), g("paired_ideal_e_pred_lat"), "",
                        g("e_pred_minus_none"), g("e_pred_minus_none_ci95"), g("spearman_e_pred_vs_e_track_chunk", "{:.3f}")]
                notes = (f"LaWAM full FT u45k; e_static={g('obs_e_static_lat')}; e_act_lat={g('obs_e_act_lat')}; "
                         f"paired_diff={g('paired_diff')} ci={g('paired_diff_ci95')}; ideal replay on every 2nd query")
            else:
                m = FLUX_PRIMARY
                metric = f"FLUX_{m}"
                vals = [g(f"e_pred.{m}"), g(f"paired_e_pred.{m}"), g(f"paired_e_ideal.{m}"), g(f"e_exec.{m}"),
                        g(f"e_pred.{m}_minus_none"), g(f"e_pred.{m}_minus_none_ci95"), g(f"rho_track.{m}", "{:.3f}")]
                extra = []
                for k in ("lpips_scene@16", "lpips_wrist@16", "l1_scene@16", "dino_wrist@16", "dino_scene@8", "dino_scene@31"):
                    extra.append(f"{k}: pred={g('e_pred.' + k)} ideal={g('paired_e_ideal.' + k)} exec={g('e_exec.' + k)}")
                notes = "FLUX3 bench LoRA raw u3000; " + "; ".join(extra)
            if ideal_invalid:
                notes = notes.split("; paired_diff=")[0] + ("; " + notes.split("; ")[-1] if "; paired_diff=" in notes else "")
                notes = re.sub(r" ideal=\S* exec=[^;]*", "", notes)
                vals[2] = "invalid"  # e_ideal_paired
                if model != "lawam":
                    vals[3] = "invalid"  # e_exec
                notes = ("e_ideal/e_exec INVALID: pre-review replay (unclamped targets, gap actuators, RTX history; "
                         "review 2026-11-21 #2-4). " + notes)
            else:
                notes = "replay v2 (issued cmds, nominal gains, arm teleported per substep, RTX flush). " + notes
            row = [f"{prefix}-{model}-{cond}", "T004-wam-pred-gap-probe", model, cond, str(s["n_episodes"]),
                   g("success", "{:.3f}"), g("e_track_deg", "{:.3f}"), g("e_track_at_h_deg", "{:.3f}"), metric, *vals,
                   str(s["n_queries"]), notes]
            f.write("\t".join(row) + "\n")

if __name__ == "__main__":
    main()
