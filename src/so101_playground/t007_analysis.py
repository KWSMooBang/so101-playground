#!/usr/bin/env python3
"""T007 analysis: home-object task set (tasks/home_objects.jsonl).

Phase A  success of InternW0-Δ and LaWAM, `none`, all 30 layouts x 1 run: by family, by target object, by
         target group (sim-seen = black pen / black remote / blue paintbrush; real-only = red pen / purple
         toothbrush / razor).
Phase B  InternW0-Δ only, T006 metrics (e_pred@1 in Wan2.2 VAE latent space; nominal replay d_fault, d_nominal,
         paired d_fault - d_nominal, d(fault obs, nominal obs); oracle gain2 α0 vs α1 Δ and AUROC with an
         episode-paired bootstrap), for every subset: each family, pooled, sim-seen, real-only.
         The oracle baseline (`none`, no replay) is the Phase A InternW0 run: the same setting T006 used.

Episode id == trial_id (full task file, --episode_offset ranges).

    python -m so101_playground.t007_analysis --phase A --tsv /outputs/so101-playground/results.tsv
    python -m so101_playground.t007_analysis --phase B --tsv /outputs/so101-playground/results.tsv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from so101_playground.t005_analysis import boot, qmean, reward_auroc
from so101_playground.t006_analysis import _f, both, load_scores, load_setting, paired

ROOT = Path("/outputs/so101-playground/runs/T007")
TASKS = Path(__file__).resolve().parents[2] / "tasks/home_objects.jsonl"
FAMILIES = ("bin", "move", "next_to", "between")
GROUPS = ("sim_seen", "real_only")
MODELS = ("internw0", "lawam")
NOM = ("none", "gain2", "latency6")
# T004-T006 seen-object Bin references (8 val_set single-object Bin rows), none condition.
REF_BIN = {"internw0": "8/8 (T006 harness check and oracle none)", "lawam": "8/8 (T004nom / T005 none)"}


def tasks():
    return {r["trial_id"]: r for r in map(json.loads, TASKS.read_text().splitlines())}


def subsets(task, ids):
    """name -> episode ids: each family, pooled, sim-seen / real-only target."""
    out = {f: [e for e in ids if task[e]["home_family"] == f] for f in FAMILIES}
    out["pooled"] = list(ids)
    for g in GROUPS:
        out[g] = [e for e in ids if task[e]["home_target_group"] == g]
    return {k: np.array(sorted(v)) for k, v in out.items() if len(v)}


def episodes(run: Path):
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    return {r["episode"]: r for r in rows}  # the last row of an episode wins (resume writes no duplicates)


# ---------------------------------------------------------------- Phase A
def phase_a():
    task = tasks()
    out = {}
    for m in MODELS:
        run = ROOT / "A" / m / "none"
        if not (run / "episodes.jsonl").exists():
            continue
        eps = episodes(run)
        s = {"n": len(eps)}
        for name, ids in subsets(task, sorted(eps)).items():
            k = int(sum(eps[e]["success"] for e in ids))
            s[name] = {"n": len(ids), "success": k, "rate": k / len(ids),
                       "reasons": dict(sorted(_count(eps[e]["reason"] for e in ids).items()))}
        by_obj = defaultdict(list)
        for e, r in eps.items():
            by_obj[(task[e]["home_family"], task[e]["home_target"])].append(bool(r["success"]))
        s["by_target"] = {f"{f}|{o}": [int(sum(v)), len(v)] for (f, o), v in sorted(by_obj.items())}
        tgt = defaultdict(list)
        for e, r in eps.items():
            tgt[task[e]["home_target"]].append(bool(r["success"]))
        s["by_object"] = {o: [int(sum(v)), len(v)] for o, v in sorted(tgt.items())}
        s["per_episode"] = {int(e): bool(eps[e]["success"]) for e in sorted(eps)}
        out[m] = s
    return out


def _count(it):
    c = defaultdict(int)
    for x in it:
        c[x] += 1
    return c


# ---------------------------------------------------------------- Phase B
SERIES = ("e_pred@1", "e_pred@1_wrist", "e_pred@1_overhead", "d_fault@1", "d_nominal@1", "diff@1", "diff@1_wrist",
          "diff@1_overhead", "d_obs_nom@1", "e_static@1", "diff@2", "e_pred@2", "e_track")


def restrict(eps, ids):
    ids = set(int(i) for i in ids)
    return {e: v for e, v in eps.items() if e in ids}


def stats_subset(eps, base, ids, rng, with_ratio):
    s = {"n": int(len(ids)), "success": int(sum(eps[e]["success"] for e in ids))}
    if len(ids) == 0:
        return s
    for key in SERIES:
        if not any(len(eps[e][key]) for e in ids):
            continue
        s[key] = qmean(eps, key, ids)
        s[key + "_ci95"] = boot(lambda p, k=key: qmean(eps, k, p), ids, rng)
        s["n:" + key] = int(sum(len(eps[e][key]) for e in ids))
        if base is not None and any(len(base.get(e, {}).get(key, [])) for e in ids):
            common = both(restrict(eps, ids), restrict(base, ids), key)
            if len(common):
                f = lambda p, k=key: qmean(eps, k, p) - qmean(base, k, p)
                s[key + "_minus_none"] = f(common)
                s[key + "_minus_none_ci95"] = boot(f, common, rng)
    if with_ratio:
        pid = np.array([e for e in ids if len(eps[e]["diff@1"]) and len(eps[e]["d_obs_nom@1"])])
        if len(pid) >= 2:
            f = lambda p: 100 * qmean(eps, "diff@1", p) / qmean(eps, "d_obs_nom@1", p)
            s["pct_diff@1_of_d_obs_nom"] = f(pid)
            s["pct_diff@1_of_d_obs_nom_ci95"] = boot(f, pid, rng)
    return s


def phase_b():
    task = tasks()
    scores = load_scores(ROOT / "server/internw0/iw_scores.jsonl")
    rng = np.random.default_rng(0)
    nom = {t: load_setting(ROOT / "B/internw0" / t, scores) for t in NOM if (ROOT / "B/internw0" / t / "episodes.jsonl").exists()}
    orc = {t: load_setting(ROOT / "B/internw0" / t, scores) for t in ("gain2-a0", "gain2-a1")
           if (ROOT / "B/internw0" / t / "episodes.jsonl").exists()}
    a_none = load_setting(ROOT / "A/internw0/none", scores)
    out = {"nominal": {}, "oracle": {}}
    ids_all = sorted(set().union(*[set(v) for v in list(nom.values()) + list(orc.values())])) if nom or orc else []
    subs = subsets(task, ids_all)
    for t, eps in nom.items():
        base = nom.get("none") if t != "none" else None
        out["nominal"][t] = {name: stats_subset(eps, base, np.array([e for e in ids if e in eps]), rng, True)
                             for name, ids in subs.items()}
    for t, eps in orc.items():
        out["oracle"][t] = {name: stats_subset(eps, a_none, np.array([e for e in ids if e in eps]), rng, False)
                            for name, ids in subs.items()}
    out["oracle"]["none(A)"] = {name: stats_subset(a_none, None, np.array([e for e in ids if e in a_none]), rng, False)
                                for name, ids in subs.items()}
    if len(orc) == 2:
        a0, a1 = orc["gain2-a0"], orc["gain2-a1"]
        out["auroc_a0_vs_a1"] = {}
        for name, ids in subs.items():
            r = {}
            for key in ("e_pred@1", "e_pred@1_wrist"):
                pa, pb = paired(restrict(a0, ids), restrict(a1, ids), key)
                if len(pa) >= 2:
                    r[key] = reward_auroc(pa, pb, rng, key=key)
            out["auroc_a0_vs_a1"][name] = r
    return out


# ---------------------------------------------------------------- results.tsv
TSV_COLS = ["run_id", "task", "model", "condition", "n_episodes", "success", "e_track_deg", "e_track@h_deg", "metric",
            "e_pred", "e_pred_paired", "e_ideal_paired", "e_exec", "e_pred_minus_none", "e_pred_minus_none_ci95",
            "rho_pred_track", "n_queries", "notes"]
TASK = "T007-home-object-set"


def tsv_a(path, a):
    with path.open("a") as f:
        for m, s in a.items():
            for name in (*FAMILIES, "pooled", *GROUPS):
                if name not in s:
                    continue
                x = s[name]
                row = [f"T007A-{m}-none", TASK, m, "none", str(x["n"]), f"{x['rate']:.3f}", "", "", "", "", "", "",
                       "", "", "", "", "", f"subset={name}; home_objects canonical, 1 run/layout; "
                       f"{x['success']}/{x['n']}; reasons={x['reasons']}; ref seen-object Bin {REF_BIN[m]}"]
                f.write("\t".join(row) + "\n")


def tsv_b(path, b):
    with path.open("a") as f:
        for design in ("nominal", "oracle"):
            for t, subs in b[design].items():
                if t == "none(A)":
                    continue
                for name, s in subs.items():
                    if s["n"] == 0:
                        continue
                    if design == "nominal":
                        run_id = f"T007B-internw0-{t}"
                        notes = (f"subset={name}; nominal replay every 2nd query; d_fault-d_nominal@1={_f(s.get('diff@1'))} "
                                 f"ci={_f(s.get('diff@1_ci95'))}; d_obs_nom@1={_f(s.get('d_obs_nom@1'))}; share of fault "
                                 f"effect={_f(s.get('pct_diff@1_of_d_obs_nom'), '{:.1f}')}% "
                                 f"ci={_f(s.get('pct_diff@1_of_d_obs_nom_ci95'), '{:.1f}')}")
                    else:
                        run_id = f"T007B-internw0-{t}"
                        r = b.get("auroc_a0_vs_a1", {}).get(name, {}).get("e_pred@1", {})
                        notes = (f"subset={name}; oracle gain2, no replay, Δ vs Phase A none; AUROC a0 vs a1 @1 "
                                 f"query={_f(r.get('auroc_query'), '{:.2f}')} {_f(r.get('auroc_query_ci95'), '{:.2f}')} "
                                 f"window4={_f(r.get('auroc_window4'), '{:.2f}')} episode={_f(r.get('auroc_episode'), '{:.2f}')}")
                    row = [run_id, TASK, "internw0", t.split("-a")[0], str(s["n"]), f"{s['success'] / s['n']:.3f}",
                           _f(s.get("e_track"), "{:.3f}"), "", "Wan2.2_VAE_latent_mse@1(ticks4-16)", _f(s.get("e_pred@1")),
                           _f(s.get("d_fault@1")), _f(s.get("d_nominal@1")), _f(s.get("d_obs_nom@1")),
                           _f(s.get("e_pred@1_minus_none")), _f(s.get("e_pred@1_minus_none_ci95")), "",
                           str(s.get("n:e_pred@1", 0)), notes]
                    f.write("\t".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--tsv", type=Path, default=None)
    args = ap.parse_args()
    res = phase_a() if args.phase == "A" else phase_b()
    out = ROOT / f"summary_{args.phase}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:6000])
    if args.tsv:
        (tsv_a if args.phase == "A" else tsv_b)(args.tsv, res)


if __name__ == "__main__":
    main()
