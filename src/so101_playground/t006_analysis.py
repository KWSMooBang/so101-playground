#!/usr/bin/env python3
"""T006 analysis: InternW0-Δ future-delta probe (RESEARCH.md H2), T004nom and T005 designs.

Inputs: ``<root>/internw0/<tag>/{episodes.jsonl, ep*.npz}`` from scripts/wam_gap_eval.py --model internw0 and
the server's ``iw_scores.jsonl`` (internw0_probe_server.py; rows carry the client's run dir and attempt id; only
rows of the attempt that completed each episode are used).

Metric (Wan2.2 VAE latent space, 48 x 24 x 16 per latent frame; see internw0_probe_server.py):
  e_pred@k   mse(L0 + cumsum(d̂)[k], Lk_obs). k=1 (primary): video ticks 4–16, inside the executed 16-action
             chunk, scored at s+16. k=2: ticks 20–32 (the next chunk's actions run), scored at s+32.
  d_nominal  the same against the nominal-replay clip (fault-free PD arm, same commands, same start scene).
  d_obs_nom  mse(Lk_obs, Lk_nominal): the fault's own effect in latent space.
  e_static   mse(L0, Lk_obs): the "no motion" baseline.
Regions: all / overhead (top 12 latent rows) / wrist (bottom 12).

Means are query-weighted; CIs are 95% bootstrap over episodes; Δ vs none is episode-paired (same layouts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from so101_playground.t004_analysis import episodes_stats
from so101_playground.t005_analysis import auroc, boot, qmean, reward_auroc, windows  # noqa: F401

ARM = slice(0, 5)
NOM_CONDS = ("none", "gain", "gain2", "latency6")
ORACLE = ("none", "gain2-a0", "gain2-a1", "gain2-a1.5")
KS = (1, 2)
REGIONS = ("", "_overhead", "_wrist")


def load_scores(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def load_setting(run: Path, scores: list[dict]) -> dict:
    """Per-episode arrays (ordered by query frame) of every metric, plus e_track per tick."""
    eps, attempts = {}, {}
    for line in (run / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            attempts[r["episode"]] = r.get("attempt")
            eps[r["episode"]] = {"success": bool(r["success"]), "ticks": r["ticks"]}
    rows = [r for r in scores if r.get("dir") == str(run) and r["episode"] in attempts
            and r.get("attempt") == attempts[r["episode"]]]
    for e, d in eps.items():
        mine = [r for r in rows if r["episode"] == e]
        obs1 = {r["frame"]: r for r in mine if r["kind"] == "observed" and len(r["offsets"]) == 4}
        obs2 = {r["frame"]: r for r in mine if r["kind"] == "observed" and len(r["offsets"]) == 8}
        nom = {r["frame"]: r for r in mine if r["kind"] == "nominal"}
        d["l0_check"] = max([r["l0_check"] for r in mine], default=0.0)
        for k, obs in ((1, obs1), (2, obs2)):
            fr = sorted(obs)
            for reg in REGIONS:
                d[f"e_pred@{k}{reg}"] = np.array([obs[f][f"e_pred@{k}{reg}"] for f in fr])
                d[f"e_static@{k}{reg}"] = np.array([obs[f][f"e_static@{k}{reg}"] for f in fr])
            # paired queries: observed (k-horizon row) and nominal both scored
            pf = sorted(f for f in nom if f in obs2 and (k == 2 or f in obs1))
            for reg in REGIONS:
                src = obs1 if k == 1 else obs2
                d[f"d_fault@{k}{reg}"] = np.array([src[f][f"e_pred@{k}{reg}"] for f in pf])
                d[f"d_nominal@{k}{reg}"] = np.array([nom[f][f"e_pred@{k}{reg}"] for f in pf])
                d[f"diff@{k}{reg}"] = d[f"d_fault@{k}{reg}"] - d[f"d_nominal@{k}{reg}"]
                d[f"d_obs_nom@{k}{reg}"] = np.array([nom[f].get(f"d_obs_nom@{k}{reg}", np.nan) for f in pf])
        z = np.load(run / f"ep{e:03d}.npz", allow_pickle=True)
        h = int(z["hold_steps"])
        cmd, q, app = z["cmd"][h:, ARM], z["q"][h:, ARM], z["applied"][h:, ARM]
        d["e_track"] = np.degrees(np.abs(cmd - q)).mean(1)
        d["applied_err"] = np.degrees(np.abs(app - cmd)).mean(1)
    return eps


SERIES = ([f"{m}@{k}{reg}" for m in ("e_pred", "e_static", "d_fault", "d_nominal", "diff", "d_obs_nom")
           for k in KS for reg in REGIONS] + ["e_track", "applied_err"])


def summarize(root: Path, tags, scores_path: Path) -> dict:
    scores = load_scores(scores_path)
    base_dir = root / "internw0"
    data = {t: load_setting(base_dir / t, scores) for t in tags if (base_dir / t / "episodes.jsonl").exists()}
    rng = np.random.default_rng(0)
    summary = {}
    base = data.get("none")
    for t, eps in data.items():
        ids = np.array(sorted(eps))
        s = episodes_stats(base_dir / t)
        s["l0_check_max"] = float(max(eps[e]["l0_check"] for e in ids))
        for key in SERIES:
            if not any(len(eps[e][key]) for e in ids):
                continue
            s[key] = qmean(eps, key, ids)
            s[f"{key}_ci95"] = boot(lambda p, k=key: qmean(eps, k, p), ids, rng)
            s[f"n:{key}"] = int(sum(len(eps[e][key]) for e in ids))
            if base is not None and t != "none" and any(len(base[e][key]) for e in base):
                common = np.array(sorted(set(ids) & set(base)))
                f = lambda p, k=key: qmean(eps, k, p) - qmean(base, k, p)
                s[f"{key}_minus_none"] = f(common)
                s[f"{key}_minus_none_ci95"] = boot(f, common, rng)
        summary[t] = s
    a0, a1 = data.get("gain2-a0"), data.get("gain2-a1")
    for k in KS:
        for reg in REGIONS:
            key = f"e_pred@{k}{reg}"
            if a0 and a1:
                summary["gain2-a1"].setdefault("reward_vs_a0", {})[key] = reward_auroc(a0, a1, rng, key=key)
            if a1 and base:
                summary["gain2-a1"].setdefault("reward_a1_vs_none", {})[key] = reward_auroc(a1, base, rng, key=key)
            if a0 and base:
                summary["gain2-a0"].setdefault("reward_vs_none", {})[key] = reward_auroc(a0, base, rng, key=key)
    return summary


# ---------------------------------------------------------------- outputs
TSV_COLS = ["run_id", "task", "model", "condition", "n_episodes", "success", "e_track_deg", "e_track@h_deg", "metric",
            "e_pred", "e_pred_paired", "e_ideal_paired", "e_exec", "e_pred_minus_none", "e_pred_minus_none_ci95",
            "rho_pred_track", "n_queries", "notes"]


def _f(v, fmt="{:.4f}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, list):
        return "[" + ",".join(fmt.format(x) for x in v) + "]"
    return fmt.format(v)


def write_tsv(path: Path, summary: dict, design: str) -> None:
    with path.open("a") as f:
        for t, s in summary.items():
            if design == "nom":
                run_id, task = f"T006nom-internw0-{t}", "T006-wam-internw0-probe"
                notes = (f"InternW0-Δ step18200 future_delta, nominal replay every 2nd query; "
                         f"d_fault-d_nominal@1={_f(s.get('diff@1'))} ci={_f(s.get('diff@1_ci95'))}; "
                         f"Δ(diff)@1 vs none={_f(s.get('diff@1_minus_none'))} ci={_f(s.get('diff@1_minus_none_ci95'))}; "
                         f"d_obs_nom@1={_f(s.get('d_obs_nom@1'))}; e_static@1={_f(s.get('e_static@1'))}; "
                         f"@2: e_pred={_f(s.get('e_pred@2'))} diff={_f(s.get('diff@2'))} ci={_f(s.get('diff@2_ci95'))}; "
                         f"wrist@1 diff={_f(s.get('diff@1_wrist'))} overhead@1 diff={_f(s.get('diff@1_overhead'))}")
            else:
                run_id, task = f"T006-internw0-{t}", "T006-wam-internw0-probe"
                r = (s.get("reward_vs_a0") or {}).get("e_pred@1", {})
                notes = (f"InternW0-Δ oracle (T005 design, no replay); "
                         f"AUROC a0 vs a1 @1 query={_f(r.get('auroc_query'), '{:.2f}')} "
                         f"window4={_f(r.get('auroc_window4'), '{:.2f}')} episode={_f(r.get('auroc_episode'), '{:.2f}')}; "
                         f"@2: e_pred={_f(s.get('e_pred@2'))} Δ={_f(s.get('e_pred@2_minus_none'))}")
            row = [run_id, task, "internw0", t.split("-a")[0], str(s["n_episodes"]), _f(s["success"], "{:.3f}"),
                   _f(s.get("e_track"), "{:.3f}"), "", "Wan2.2_VAE_latent_mse@1(ticks4-16)", _f(s.get("e_pred@1")),
                   _f(s.get("d_fault@1")), _f(s.get("d_nominal@1")), _f(s.get("d_obs_nom@1")),
                   _f(s.get("e_pred@1_minus_none")), _f(s.get("e_pred@1_minus_none_ci95")), "",
                   str(s.get("n:e_pred@1", 0)), notes]
            f.write("\t".join(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="/outputs/so101-playground/runs/T006/<design dir>")
    ap.add_argument("--design", choices=["nom", "oracle"], required=True)
    ap.add_argument("--scores", type=Path, default=Path("/outputs/so101-playground/runs/T006/server/iw_scores.jsonl"))
    ap.add_argument("--tsv", type=Path, default=None)
    args = ap.parse_args()
    tags = NOM_CONDS if args.design == "nom" else ORACLE
    summary = summarize(args.root, tags, args.scores)
    out = args.root / "internw0" / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1)[:20000])
    if args.tsv:
        write_tsv(args.tsv, summary, args.design)


if __name__ == "__main__":
    main()
