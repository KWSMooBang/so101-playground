#!/usr/bin/env python3
"""T011 analysis: real e_pred (from t011_real_epred.py encode) vs sim, success labels, noise floor, figures.

    python -m so101_playground.t011_analysis --out /outputs/so101-playground/runs/T011 \
        [--tsv /outputs/so101-playground/results.tsv]

Inputs: <out>/queries.jsonl, <out>/alignment.json; sim per-query scores from T007 (home bin, A + B none) and
T009 (θ0 eval) iw_scores.jsonl and their episode npz (motion / tracking at each query).
Outputs: <out>/episodes.csv, <out>/summary.json, <out>/fig_*.png, results.tsv rows T011-real-<group>.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HOME = Path("/outputs/so101-playground/runs/real_robot/home_runs")
RUNS = Path("/outputs/so101-playground/runs")
ARM = slice(0, 5)
EXPECTED_PAIRED = 0.0125  # T009: median |paired Δe_pred| of the mixed run

# ---------------------------------------------------------------- success labels (scope 3)
# Labelled by eye from the first / last-60 / last overhead frames (runs/T011/sheets/). No human labels were sent.
# index = position in the sorted home_runs listing. (target object, label 1/0, confidence, note)
LABELS = {
    0: ("pen", 1, "high", "black pen in bin; table also had red pen, remote, razor"),
    1: ("remote", 0, "high", "wrong object: razor placed in bin, remote untouched"),
    2: ("brush", 0, "high", "bin empty; brush = black makeup brush"),
    3: ("brush", 0, "high", "bin empty; human hand rearranges objects mid-episode"),
    4: ("brush", 0, "high", "bin empty"),
    5: ("razor", 1, "high", "razor in bin"),
    6: ("pen", 1, "high", "pen in bin; table: pen, toothbrush, remote, razor"),
    7: ("pen", 0, "high", "wrong object: toothbrush placed in bin"),
    8: ("pen", 0, "high", "bin empty"),
    9: ("pen", 1, "high", "pen in bin"),
    10: ("pen", 0, "high", "bin empty"),
    11: ("pen", 1, "high", "pen in bin"),
    12: ("pen", 1, "high", "pen in bin"),
    13: ("razor", 1, "high", "razor in bin"),
    14: ("razor", 0, "high", "wrong object: toothbrush placed in bin"),
    15: ("razor", 0, "high", "wrong object: toothbrush placed in bin"),
    16: ("razor", 0, "high", "wrong object: toothbrush placed in bin"),
    17: ("razor", 1, "high", ""), 18: ("razor", 1, "high", ""), 19: ("razor", 1, "high", ""),
    20: ("razor", 1, "high", ""), 21: ("razor", 1, "high", ""),
    22: ("remote", 1, "high", "in bin, diagonal"), 23: ("remote", 1, "high", "in bin, leaning on wall"),
    24: ("remote", 1, "high", "flat in bin"),
    25: ("remote", 0, "high", "remote left standing on its end next to the bin; 1600 steps"),
    26: ("remote", 1, "high", "in bin, leaning on wall"),
    27: ("toothbrush", 0, "high", "toothbrush still on table"),
    28: ("toothbrush", 1, "high", ""), 29: ("toothbrush", 1, "high", ""), 30: ("toothbrush", 1, "high", ""),
    31: ("toothbrush", 1, "high", "standing in bin, head up"),
    32: ("pen", 1, "high", ""), 33: ("pen", 1, "high", "multicolour pen"), 34: ("pen", 1, "high", ""),
    35: ("pen", 1, "high", ""), 36: ("pen", 1, "high", ""),
    37: ("glue stick", 0, "high", "glue stick untouched; arm ends extended (not in dock pose)"),
    38: ("glue stick", 1, "high", ""), 39: ("glue stick", 1, "high", ""),
    40: ("glue stick", 0, "high", "glue stick untouched; arm ends extended"),
    41: ("glue stick", 1, "high", "in bin (far corner)"),
    42: ("pencil case", 1, "medium", "standing in bin, top above the rim"),
    43: ("pencil case", 0, "low", "resting across the bin rim, about half inside"),
    44: ("pencil case", 1, "medium", "inside, but the bin is pushed and tilted"),
}
# real object -> sim home bin layout (T007 trial id; T009 home<i>); makeup brush ~ paintbrush (closest asset)
SIM_HOME = {"pen": [0, 3], "remote": [1], "brush": [2], "toothbrush": [4], "razor": [5]}
SIM_NAMES = ["black pen", "black remote", "blue paintbrush", "red pen", "purple toothbrush", "razor"]


def group_of(name: str, label: tuple) -> str:
    obj = label[0].replace(" ", "")
    if "each_object" in name:
        return f"each-{obj}"
    return f"named-{obj}-{name[:8]}"


# ---------------------------------------------------------------- helpers
def boot_ci(f, n: int, rng, B: int = 4000):
    vals = [f(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


# ---------------------------------------------------------------- chunk profile (one definition, real and sim)
def chunk_profile(cmd16: np.ndarray, q17: np.ndarray) -> dict:
    """cmd16[k-1] = command issued at chunk tick k (k = 1..16); q17[0] = arm state at the query (before tick 1),
    q17[k] = state after command k has been applied for one control tick. Degrees, arm joints only.
    real: cmd[s:s+16], state[s:s+17] (state is read at the start of a step, before that step's command);
    sim:  cmd[i:i+16], [q[i-1]] + q[i:i+16] (q is logged after env.step), i = log index of the query step."""
    c, q = cmd16[:, ARM], q17[:, ARM]
    trk = np.abs(c - q[1:]).mean(1)  # per tick
    spd = np.abs(np.diff(q, axis=0)).max(1)  # max joint speed, deg / tick
    return {"motion16_deg": float(np.abs(q[16] - q[0]).max()), "e_track16_deg": float(trk.mean()),
            "e_track_t1_8": float(trk[:8].mean()), "e_track_t9_16": float(trk[8:].mean()), "spd_t1": float(spd[0]),
            "trk_k": trk.tolist(), "spd_k": spd.tolist(), "cmd_spd_k": np.abs(np.diff(np.vstack([q[:1], c]), axis=0)).max(1).tolist()}


# ---------------------------------------------------------------- real
def load_real(out: Path) -> tuple[list[dict], dict]:
    Q = [json.loads(l) for l in (out / "queries.jsonl").open()]
    by = defaultdict(list)
    for r in Q:
        by[r["dir"]].append(r)
    dirs = sorted(d.name for d in HOME.iterdir() if (d / "trajectory.npz").exists())
    eps = []
    for i, name in enumerate(dirs):
        qs = sorted(by[name], key=lambda r: r["q"])
        z = np.load(HOME / name / "trajectory.npz")
        st, cmd = z["state_sim_deg"], z["command_sim_deg"]
        qsteps = np.flatnonzero(z["policy_query"])
        for r in qs:
            sq = int(qsteps[r["q"]])
            assert sq == r["step"]
            if sq + 16 < len(st):
                r.update(chunk_profile(cmd[sq:sq + 16], st[sq:sq + 17]))
            else:
                r["e_track16_deg"] = None
        lab = LABELS[i]
        e = {"idx": i, "dir": name, "date": name[:8], "instruction": json.loads((HOME / name / "episode.json").read_text())["instructions"][0][1],
             "group": group_of(name, lab), "target": lab[0], "success": lab[1], "confidence": lab[2], "note": lab[3],
             "n_steps": len(st), "n_queries": len(qs), "server_episode": qs[0]["episode"] if qs else None}
        e["e_track_deg"] = float(np.abs(cmd[:-1, ARM] - st[1:, ARM]).mean())
        moving = np.abs(np.diff(st[:, ARM], axis=0)).max(1) > 0.3
        e["e_track_moving_deg"] = float(np.abs(cmd[:-1, ARM] - st[1:, ARM])[moving].mean()) if moving.any() else np.nan
        for key in ["e_pred@1", "e_pred@2", "e_static@1", "e_static@2", "e_codec", "e_pred@1_mp4", "e_pred@1_overhead",
                    "e_pred@1_wrist", "e_static@1_overhead", "e_static@1_wrist", "e_codec_overhead", "e_codec_wrist",
                    "pred_motion@1", "hold_s", "motion16_deg", "e_track16_deg", "l0_check", "server_latency_s"]:
            v = [r[key] for r in qs if r.get(key) is not None]
            e[key] = float(np.mean(v)) if v else np.nan
        e["n_q@1"] = sum("e_pred@1" in r for r in qs)
        e["l0_check_max"] = max((r["l0_check"] for r in qs), default=np.nan)
        eps.append(e)
    return eps, by


# ---------------------------------------------------------------- sim
def load_sim() -> list[dict]:
    """Per-query sim rows (observed, @1 from 4-offset clips, @2 from 8-offset clips) with the object and the
    arm's motion / tracking at the query, from T007 (A + B none, bin layouts 0-5) and T009 θ0 eval."""
    rows = []
    srcs = [(RUNS / "T007/server/internw0/iw_scores.jsonl", lambda r: r["condition"] == "none" and r["episode"] < 6
             and ("/A/" in r["dir"] or "/B/" in r["dir"])),
            (RUNS / "T009/server/iw_scores.jsonl", lambda r: r["condition"] in ("T009-baseline-theta0", "T009-theta0-nojit"))]
    per = defaultdict(dict)
    for f, keep in srcs:
        for line in f.open():
            r = json.loads(line)
            if r["kind"] != "observed" or not keep(r):
                continue
            key = (r["dir"], r["condition"], r["episode"], r["attempt"])
            d = per[key].setdefault(r["frame"], {})
            if len(r["offsets"]) == 4:
                d.update({k: v for k, v in r.items() if k.endswith("@1") or "@1_" in k})
            else:
                d.update({k: v for k, v in r.items() if "@2" in k})
    t009_npz = defaultdict(list)
    for p in (RUNS / "T009/episodes").glob("eval-T009-*theta0*-home*.npz"):
        t009_npz[(("nojit" in p.name), int(p.name.split("home")[1][0]))].append(p)
    for (dr, cond, ep, att), frames in per.items():
        src = "T009" if "T009" in cond else ("T007B" if "/B/" in dr else "T007A")
        home = ep - 5000 if src == "T009" else ep
        npz = None
        fr = sorted(frames)
        if src == "T009":
            for p in t009_npz[("nojit" in cond, home)]:
                z = np.load(p, allow_pickle=True)
                if set(fr) <= set(z["queries"].tolist()):
                    npz = z
                    break
        else:
            p = Path(dr) / f"ep{ep:03d}.npz"
            if p.exists():
                npz = np.load(p, allow_pickle=True)
        for qi, f in enumerate(fr):
            d = dict(frames[f], src=src, home=home, obj=SIM_NAMES[home], ep=f"{src}:{cond}:{ep}:{att}", frame=f, qi=qi)
            # T007 (wam_gap_eval) logs every step from 0; T009 logs only after the hold, so index = step - hold
            off = int(npz["hold_steps"]) if (npz is not None and src == "T009") else 0
            i = f - off
            if npz is not None and i + 16 <= len(npz["q"]) and (i >= 1 or "q_start" in npz):
                q, cmd = np.degrees(npz["q"]), np.degrees(npz["cmd"])
                prev = q[i - 1] if i >= 1 else np.degrees(npz["q_start"])
                d.update(chunk_profile(cmd[i:i + 16], np.vstack([prev[None], q[i:i + 16]])))
                d["first_query"] = qi == 0
            rows.append(d)
    return rows


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/outputs/so101-playground/runs/T011")
    p.add_argument("--tsv", default="")
    a = p.parse_args()
    out = Path(a.out)
    rng = np.random.default_rng(0)
    eps, byq = load_real(out)
    sim = load_sim()
    align = json.loads((out / "alignment.json").read_text())
    Q = [r for d in byq.values() for r in d]
    S: dict = {}

    # --- 1. alignment / sanity
    S["alignment"] = {
        "home_episodes": len(eps), "server_episodes": len(align["match"]), "tz_hours": align["tz_hours"],
        "max_abs_resid_s": max(abs(v) for v in align["resid_s"].values()),
        "duplicates": align["dup"], "unmatched_server": align["unmatched_server"],
        "n_queries": len(Q), "queries_eq_policy_query": all(x["n_policy_query"] == x["n_server_q"] for x in align["episodes"]),
        "mp4_frames_eq_steps": all(x["mp4_frames"] == [x["n_steps"]] * 2 for x in align["episodes"]),
        "instruction_match": all(x["instruction"] == x["server_instruction"] for x in align["episodes"]),
        "l0_check_max": max(r["l0_check"] for r in Q), "l0_check_absmax": max(r["l0_check_max"] for r in Q),
        "state_match_max_deg": max(r["state_match_deg"] for r in Q),
        "pix_png_mp4_overhead": float(np.mean([r["pix_png_mp4_overhead"] for r in Q])),
        "pix_png_mp4_wrist": float(np.mean([r["pix_png_mp4_wrist"] for r in Q])),
        "pix_png_mp4_prev": float(np.nanmean([r["pix_png_mp4_prev"] if r["pix_png_mp4_prev"] is not None else np.nan for r in Q])),
        "pix_png_mp4_next": float(np.nanmean([r["pix_png_mp4_next"] if r["pix_png_mp4_next"] is not None else np.nan for r in Q])),
        "frac_aligned_best": float(np.mean([r["pix_png_mp4_overhead"] <= min(x for x in (r["pix_png_mp4_prev"], r["pix_png_mp4_next"]) if x is not None)
                                            for r in Q if r["pix_png_mp4_prev"] is not None and r["pix_png_mp4_next"] is not None])),
    }
    ec = np.array([r["e_codec"] for r in Q])
    S["codec"] = {"e_codec_mean": float(ec.mean()), "e_codec_median": float(np.median(ec)), "e_codec_p90": float(np.percentile(ec, 90)),
                  "overhead": float(np.mean([r["e_codec_overhead"] for r in Q])), "wrist": float(np.mean([r["e_codec_wrist"] for r in Q])),
                  "e_pred@1_png_vs_mp4_frame0": float(np.mean([r["e_pred@1"] - r["e_pred@1_mp4"] for r in Q if "e_pred@1" in r]))}
    # --- 2. real e_pred
    q1 = [r for r in Q if "e_pred@1" in r]
    q2 = [r for r in Q if "e_pred@2" in r]
    hold = np.array([r["hold_s"] for r in Q if r["hold_s"] is not None])
    lat = np.array([r["server_latency_s"] for r in Q if r.get("server_latency_s")])
    S["real"] = {"n_q@1": len(q1), "n_q@2": len(q2),
                 "e_pred@1_query_mean": float(np.mean([r["e_pred@1"] for r in q1])),
                 "e_pred@2_query_mean": float(np.mean([r["e_pred@2"] for r in q2])),
                 "e_pred@1_ep_mean": float(np.mean([e["e_pred@1"] for e in eps])),
                 "e_static@1_query_mean": float(np.mean([r["e_static@1"] for r in q1])),
                 "e_pred@1_overhead": float(np.mean([r["e_pred@1_overhead"] for r in q1])),
                 "e_pred@1_wrist": float(np.mean([r["e_pred@1_wrist"] for r in q1])),
                 "hold_s_median": float(np.median(hold)), "hold_s_p10": float(np.percentile(hold, 10)),
                 "hold_s_p90": float(np.percentile(hold, 90)), "hold_s_max": float(hold.max()),
                 "hold_ticks_median": float(np.median(hold) * 30), "dt_median_s": float(np.median([r["dt_median_s"] for r in Q])),
                 "server_latency_median_s": float(np.median(lat)), "first_query_latency_median_s": float(np.median([r["server_latency_s"] for r in Q if r["q"] == 0 and r.get("server_latency_s")]))}
    # by group / object
    by_group = defaultdict(list)
    for e in eps:
        by_group[e["group"]].append(e)
        by_group[f"obj-{e['target'].replace(' ', '')}"].append(e)
        by_group["named" if "each" not in e["group"] else "each"].append(e)
        by_group["all"].append(e)
    S["groups"] = {}
    for g, L in by_group.items():
        S["groups"][g] = {"n": len(L), "success": float(np.mean([e["success"] for e in L])),
                          "e_pred@1": float(np.mean([e["e_pred@1"] for e in L])),
                          "e_pred@1_sd": float(np.std([e["e_pred@1"] for e in L], ddof=1)) if len(L) > 1 else np.nan,
                          "e_pred@2": float(np.nanmean([e["e_pred@2"] for e in L])),
                          "e_static@1": float(np.mean([e["e_static@1"] for e in L])),
                          "e_track_deg": float(np.mean([e["e_track_deg"] for e in L])),
                          "n_queries": int(sum(e["n_q@1"] for e in L))}
    # --- 4. sim comparison
    sim1 = [r for r in sim if "e_pred@1" in r]

    def sim_stats(rows):
        if not rows:
            return {}
        epm = defaultdict(list)
        for r in rows:
            epm[r["ep"]].append(r["e_pred@1"])
        return {"n_ep": len(epm), "n_q": len(rows), "e_pred@1_query": float(np.mean([r["e_pred@1"] for r in rows])),
                "e_pred@1_ep": float(np.mean([np.mean(v) for v in epm.values()])),
                "e_static@1": float(np.mean([r["e_static@1"] for r in rows])),
                "e_pred@1_overhead": float(np.mean([r["e_pred@1_overhead"] for r in rows])),
                "e_pred@1_wrist": float(np.mean([r["e_pred@1_wrist"] for r in rows])),
                "e_pred@2": float(np.mean([r["e_pred@2"] for r in rows if "e_pred@2" in r])) if any("e_pred@2" in r for r in rows) else None}
    S["sim"] = {"all": sim_stats(sim1)}
    for src in ("T007A", "T007B", "T009"):
        S["sim"][src] = sim_stats([r for r in sim1 if r["src"] == src])
    S["real_vs_sim_by_object"] = {}
    for obj, homes in SIM_HOME.items():
        real_e = [e for e in eps if e["target"] == obj]
        rq = [r for e in real_e for r in byq[e["dir"]] if "e_pred@1" in r]
        ss = sim_stats([r for r in sim1 if r["home"] in homes])
        S["real_vs_sim_by_object"][obj] = {"real_n_ep": len(real_e), "real_e_pred@1_ep": float(np.mean([e["e_pred@1"] for e in real_e])),
                                           "real_e_pred@1_query": float(np.mean([r["e_pred@1"] for r in rq])),
                                           "real_e_static@1": float(np.mean([r["e_static@1"] for r in rq])),
                                           "real_success": float(np.mean([e["success"] for e in real_e])), "sim": ss,
                                           "excess": float(np.mean([r["e_pred@1"] for r in rq])) - ss["e_pred@1_query"]}
    # visual-gap offset vs control gap: e_pred@1 against the arm's motion over the next 16 ticks
    bins = [0, 2, 5, 10, 20, 40, 1e9]
    def binned(rows, key="e_pred@1"):
        res = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            v = [r[key] for r in rows if key in r and r.get("motion16_deg") is not None and lo <= r["motion16_deg"] < hi]
            res.append({"lo": lo, "hi": hi, "n": len(v), "mean": float(np.mean(v)) if v else None,
                        "ci": boot_ci(lambda i, v=np.array(v): v[i].mean(), len(v), rng, 1000) if len(v) > 2 else None})
        return res
    real_m = [r for r in q1]
    sim_m = [r for r in sim1 if r.get("motion16_deg") is not None]
    S["motion_bins"] = {"real": binned(real_m), "sim": binned(sim_m), "real_static": binned(real_m, "e_static@1"),
                        "sim_static": binned(sim_m, "e_static@1"),
                        "real_overhead": binned(real_m, "e_pred@1_overhead"), "sim_overhead": binned(sim_m, "e_pred@1_overhead"),
                        "real_wrist": binned(real_m, "e_pred@1_wrist"), "sim_wrist": binned(sim_m, "e_pred@1_wrist")}

    def ols(rows):
        x = np.array([r["motion16_deg"] for r in rows]); y = np.array([r["e_pred@1"] for r in rows])
        A = np.c_[np.ones_like(x), x]
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        def f(i):
            return np.linalg.lstsq(A[i], y[i], rcond=None)[0]
        bs = np.array([f(rng.integers(0, len(x), len(x))) for _ in range(1000)])
        return {"intercept": float(coef[0]), "slope_per_deg": float(coef[1]),
                "intercept_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(), "slope_ci": np.percentile(bs[:, 1], [2.5, 97.5]).tolist()}
    S["motion_ols"] = {"real": ols([r for r in real_m if r["motion16_deg"] < 60]), "sim": ols([r for r in sim_m if r["motion16_deg"] < 60])}
    first_real = [r["e_pred@1"] for r in q1 if r["q"] == 0]
    first_sim = [r["e_pred@1"] for r in sim1 if r.get("qi") == 0]
    static_real = [r["e_pred@1"] for r in q1 if r["motion16_deg"] < 2]
    static_sim = [r["e_pred@1"] for r in sim_m if r["motion16_deg"] < 2]
    S["offset"] = {"first_query_real": float(np.mean(first_real)), "first_query_sim": float(np.mean(first_sim)),
                   "static_real": float(np.mean(static_real)), "static_sim": float(np.mean(static_sim)),
                   "n_static_real": len(static_real), "n_static_sim": len(static_sim),
                   "static_real_motion_med": float(np.median([r["motion16_deg"] for r in q1 if r["motion16_deg"] < 2]))}
    # --- 5. RL usability
    # noise floor: groups of repeated same-object / same-instruction episodes
    rep = {g: L for g, L in by_group.items() if not g.startswith(("obj-", "all", "named", "each")) or g.startswith(("each-", "named-"))}
    rep = {g: L for g, L in rep.items() if len(L) >= 3}
    # motion-adjusted episode value: per-query e_pred@1 minus the real motion slope (a covariate that a gain change
    # does not target directly), averaged per episode
    slope = S["motion_ols"]["real"]["slope_per_deg"]
    for e in eps:
        rs = [r for r in byq[e["dir"]] if "e_pred@1" in r]
        e["e_pred@1_motion_adj"] = float(np.mean([r["e_pred@1"] - slope * r["motion16_deg"] for r in rs]))

    def noise(key: str, skip: tuple = ()) -> tuple[dict, dict]:
        diffs_all, diffs_succ, resid_all, per = [], [], [], {}
        gs = [g for g in rep if g not in skip]
        for g in gs:
            L = rep[g]
            v = np.array([e[key] for e in L]); s = np.array([e["success"] for e in L])
            pair = [abs(v[i] - v[j]) for i in range(len(v)) for j in range(i + 1, len(v))]
            vs = v[s == 1]
            pair_s = [abs(vs[i] - vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
            diffs_all += pair; diffs_succ += pair_s; resid_all += list(v - v.mean())
            per[g] = {"n": len(L), "n_success": int(s.sum()), "sd": float(v.std(ddof=1)), "mean": float(v.mean()),
                      "median_abs_pair_diff": float(np.median(pair)),
                      "median_abs_pair_diff_success": float(np.median(pair_s)) if pair_s else None}
        pooled = {"n_groups": len(gs), "n_pairs": len(diffs_all), "median_abs_pair_diff": float(np.median(diffs_all)),
                  "median_abs_pair_diff_success_only": float(np.median(diffs_succ)),
                  "pooled_within_sd": float(np.sqrt(np.sum(np.square(resid_all)) / (len(resid_all) - len(gs)))),
                  "expected_paired_effect_T009": EXPECTED_PAIRED, "sim_jitter_abs_diff_T009": 0.0076}
        return per, pooled
    S["noise"], S["noise_pooled"] = noise("e_pred@1")
    # the 11-22 brush trio: all failures, a human rearranges objects inside episode 3
    _, S["noise_pooled_excl_brush"] = noise("e_pred@1", ("named-brush-20261122",))
    _, S["noise_pooled_motion_adj_excl_brush"] = noise("e_pred@1_motion_adj", ("named-brush-20261122",))
    # success vs failure, within groups (group-centred)
    sv, fv, sv_raw, fv_raw = [], [], [], []
    within = {}
    for g, L in rep.items():
        s = np.array([e["success"] for e in L]); v = np.array([e["e_pred@1"] for e in L])
        if 0 < s.sum() < len(s):
            within[g] = {"succ": float(v[s == 1].mean()), "fail": float(v[s == 0].mean()), "diff_fail_minus_succ": float(v[s == 0].mean() - v[s == 1].mean()),
                         "n_s": int(s.sum()), "n_f": int((s == 0).sum())}
            c = v.mean()
            sv += list(v[s == 1] - c); fv += list(v[s == 0] - c)
    allv = np.array([e["e_pred@1"] for e in eps]); alls = np.array([e["success"] for e in eps])
    S["success_vs_fail"] = {"within_groups": within, "auroc_fail_higher_group_centred": auroc(fv, sv),
                            "pooled_fail_minus_succ": float(allv[alls == 0].mean() - allv[alls == 1].mean()),
                            "auroc_fail_higher_pooled": auroc(allv[alls == 0], allv[alls == 1]),
                            "n_success": int(alls.sum()), "n_fail": int((alls == 0).sum())}
    # tracking correlation
    def demean(rows, key):
        m = defaultdict(list)
        for r in rows:
            m[r["dir"] if "dir" in r else r["ep"]].append(r[key])
        mm = {k: np.mean(v) for k, v in m.items()}
        return [r[key] - mm[r["dir"] if "dir" in r else r["ep"]] for r in rows]
    qt = [r for r in q1 if r.get("e_track16_deg") is not None]
    st_ = [r for r in sim1 if r.get("e_track16_deg") is not None]
    S["tracking"] = {"real_query_spearman": spearman([r["e_pred@1"] for r in qt], [r["e_track16_deg"] for r in qt]),
                     "real_query_spearman_within_ep": spearman(demean(qt, "e_pred@1"), demean(qt, "e_track16_deg")),
                     "real_query_spearman_static_vs_track_within_ep": spearman(demean(qt, "e_static@1"), demean(qt, "e_track16_deg")),
                     "real_query_spearman_motion": spearman([r["e_pred@1"] for r in qt], [r["motion16_deg"] for r in qt]),
                     "real_ep_spearman": spearman([e["e_pred@1"] for e in eps], [e["e_track_deg"] for e in eps]),
                     "real_e_track_deg_mean": float(np.mean([e["e_track_deg"] for e in eps])),
                     "real_e_track_moving_deg_mean": float(np.nanmean([e["e_track_moving_deg"] for e in eps])),
                     "real_e_track16_query_mean": float(np.mean([r["e_track16_deg"] for r in qt])),
                     "sim_query_spearman": spearman([r["e_pred@1"] for r in st_], [r["e_track16_deg"] for r in st_]),
                     "sim_query_spearman_within_ep": spearman(demean(st_, "e_pred@1"), demean(st_, "e_track16_deg")),
                     "sim_e_track16_query_mean": float(np.mean([r["e_track16_deg"] for r in st_]))}
    # partial: e_pred vs tracking controlling for motion (residualise both on motion bins)
    def resid_on(rows, y, x="motion16_deg"):
        xs = np.array([r[x] for r in rows]); ys = np.array([r[y] for r in rows])
        A = np.c_[np.ones_like(xs), xs, xs ** 2]
        return ys - A @ np.linalg.lstsq(A, ys, rcond=None)[0]
    S["tracking"]["real_partial_spearman_given_motion"] = spearman(resid_on(qt, "e_pred@1"), resid_on(qt, "e_track16_deg"))
    S["tracking"]["sim_partial_spearman_given_motion"] = spearman(resid_on(st_, "e_pred@1"), resid_on(st_, "e_track16_deg")) if all(r.get("motion16_deg") is not None for r in st_) else None

    def fit(rows, keys):
        X = np.column_stack([np.ones(len(rows))] + [[r[k] for r in rows] for k in keys])
        y = np.array([r["e_pred@1"] for r in rows])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        bs = []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
        return {"keys": ["1"] + keys, "coef": c.tolist(), "ci": np.percentile(bs, [2.5, 97.5], axis=0).T.tolist(),
                "r2": float(1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    rq = [r for r in qt if r["motion16_deg"] < 60]
    sq = [r for r in st_ if r.get("motion16_deg") is not None and r["motion16_deg"] < 60]
    reg = {}
    for nm, rows in (("real", rq), ("sim", sq)):
        for keys in (["motion16_deg"], ["e_track16_deg"], ["motion16_deg", "e_track16_deg"]):
            reg[f"{nm}:{'+'.join(keys)}"] = fit(rows, keys)
        reg[f"{nm}:mean_e_track16"] = float(np.mean([r["e_track16_deg"] for r in rows]))
        reg[f"{nm}:mean_e_pred@1"] = float(np.mean([r["e_pred@1"] for r in rows]))
        reg[f"{nm}:e_track16_by_motion"] = {f"{lo}-{hi}": float(np.mean([r["e_track16_deg"] for r in rows if lo <= r["motion16_deg"] < hi]))
                                             for lo, hi in ((0, 2), (2, 10), (10, 20), (20, 40))}
    (a0, a1), (b0, b1) = reg["real:e_track16_deg"]["coef"], reg["sim:e_track16_deg"]["coef"]
    tr, ts = reg["real:mean_e_track16"], reg["sim:mean_e_track16"]
    reg["decomposition"] = {"total_excess": reg["real:mean_e_pred@1"] - reg["sim:mean_e_pred@1"],
                            "intercept_offset": a0 - b0, "more_tracking_error": a1 * (tr - ts), "steeper_slope": (a1 - b1) * ts}
    S["track_regression"] = reg
    # --- stop-and-restart (L1 question): real chunks start from rest after the 0.8 s query hold; sim pauses the
    # physics, so the arm resumes with its velocity. Per-tick profiles at matched chunk motion, and how much of the
    # tracking-linked excess the early (ticks 1-8) part carries.
    rs_ = {}
    mbins = ((0, 2), (2, 10), (10, 20), (20, 40))
    for nm, rows in (("real", rq), ("sim", sq)):
        prof = {}
        for lo, hi in mbins + ((10, 40),):
            R_ = [r for r in rows if lo <= r["motion16_deg"] < hi and "trk_k" in r]
            prof[f"{lo}-{hi}"] = {"n": len(R_), "trk_k": np.mean([r["trk_k"] for r in R_], 0).round(3).tolist(),
                                  "spd_k": np.mean([r["spd_k"] for r in R_], 0).round(3).tolist(),
                                  "cmd_spd_k": np.mean([r["cmd_spd_k"] for r in R_], 0).round(3).tolist(),
                                  "e_track_t1_8": float(np.mean([r["e_track_t1_8"] for r in R_])),
                                  "e_track_t9_16": float(np.mean([r["e_track_t9_16"] for r in R_])),
                                  "e_track_t4_8": float(np.mean([np.mean(r["trk_k"][3:8]) for r in R_])),
                                  "spd_t1": float(np.mean([r["spd_t1"] for r in R_])),
                                  "spd_t4_8": float(np.mean([np.mean(r["spd_k"][3:8]) for r in R_])),
                                  "e_pred@1": float(np.mean([r["e_pred@1"] for r in R_]))}
        rs_[nm] = {"profile": prof, "fit_early_late": fit(rows, ["e_track_t1_8", "e_track_t9_16"])}
    # counterfactual: give each real chunk sim's early/late tracking ratio at its motion bin (late part kept)
    cf = []
    ce = rs_["real"]["fit_early_late"]["coef"][1]
    for r in rq:
        for lo, hi in mbins:
            if lo <= r["motion16_deg"] < hi:
                ps = rs_["sim"]["profile"][f"{lo}-{hi}"]
                ratio = ps["e_track_t1_8"] / ps["e_track_t9_16"]
                cf.append(ce * max(0.0, r["e_track_t1_8"] - ratio * r["e_track_t9_16"]))
    dec = reg["decomposition"]
    rs_["restart_share"] = {"coef_early_real": ce, "e_pred_reduction_if_sim_early_late_ratio": float(np.mean(cf)),
                            "of_total_excess": float(np.mean(cf) / dec["total_excess"]),
                            "of_tracking_linked_excess": float(np.mean(cf) / (dec["more_tracking_error"] + dec["steeper_slope"]))}
    # sim-only check: at matched motion, does starting a chunk slowly (low tick-1 speed) raise e_pred in sim?
    sm = [r for r in sq if r["motion16_deg"] >= 10]
    rs_["sim_entry_speed_check"] = {"n": len(sm), "fit": fit(sm, ["motion16_deg", "spd_t1"]),
                                    "spearman_e_pred_vs_spd_t1": spearman([r["e_pred@1"] for r in sm], [r["spd_t1"] for r in sm])}
    S["restart"] = rs_
    cm = out / "codec_motion.jsonl"
    if cm.exists():
        R = [json.loads(l) for l in cm.open()]
        Qd = {(r["dir"], r["q"]): r for r in Q}
        S["codec_motion"] = {"n": len(R), "episodes": sorted({r["dir"] for r in R}),
                             "L0_second_gen": float(np.mean([r["e_codec2@0"] for r in R])),
                             "L0_first_gen_same_queries": float(np.mean([Qd[(r["dir"], r["q"])]["e_codec"] for r in R])),
                             "L1_second_gen_by_motion": {f"{lo}-{hi}": float(np.mean([r["e_codec2@1"] for r in R if lo <= r["motion16_deg"] < hi]))
                                                         for lo, hi in ((0, 2), (2, 10), (10, 20), (20, 40))},
                             "L1_second_gen": float(np.mean([r["e_codec2@1"] for r in R])),
                             "L2_second_gen": float(np.mean([r["e_codec2@2"] for r in R if "e_codec2@2" in r]))}
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    cols = ["idx", "dir", "date", "instruction", "group", "target", "success", "confidence", "note", "n_steps", "n_queries",
            "n_q@1", "server_episode", "e_pred@1", "e_pred@2", "e_static@1", "e_static@2", "e_pred@1_overhead", "e_pred@1_wrist",
            "e_codec", "e_pred@1_mp4", "pred_motion@1", "hold_s", "server_latency_s", "motion16_deg", "e_track16_deg", "e_track_deg",
            "e_track_moving_deg", "e_pred@1_motion_adj", "l0_check_max"]
    with (out / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in eps:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in e.items()})
    print(json.dumps({k: S.get(k) for k in ("alignment", "codec", "codec_motion", "real", "offset", "motion_ols", "noise_pooled",
                                             "noise_pooled_excl_brush", "noise_pooled_motion_adj_excl_brush", "success_vs_fail",
                                             "tracking", "track_regression")}, indent=1, default=float))
    if a.tsv:
        write_tsv(Path(a.tsv), S, eps)
    plots(out, eps, byq, sim, S)


def write_tsv(path: Path, S: dict, eps: list[dict]) -> None:
    header = path.read_text().splitlines()[0].split("\t")
    keep = [l for l in path.read_text().splitlines() if not l.startswith("T011-real-")]
    order = ["all", "named", "each"] + sorted(g for g in S["groups"] if g.startswith("obj-"))
    rows = []
    for g in order:
        s = S["groups"][g]
        notes = (f"real SO-101 home runs 2026-11-22/23, InternW0-Δ server T008 (--real), no RL; success = T011 visual label "
                 f"(no human labels); e_static@1={s['e_static@1']:.4f}; e_pred@2={s['e_pred@2']:.4f}; ep-mean e_pred sd={s['e_pred@1_sd']:.4f}; "
                 f"mp4 codec floor e_codec={S['codec']['e_codec_mean']:.4f}; e_track=|cmd(t-1)-q(t)| arm")
        r = {"run_id": f"T011-real-{g.replace('obj-', '')}", "task": "T011-real-epred-analysis", "model": "internw0", "condition": "real",
             "n_episodes": s["n"], "success": f"{s['success']:.3f}", "e_track_deg": f"{s['e_track_deg']:.3f}",
             "metric": "Wan2.2_VAE_latent_mse@1(ticks4-16)", "e_pred": f"{s['e_pred@1']:.4f}", "n_queries": s["n_queries"], "notes": notes}
        rows.append("\t".join(str(r.get(h, "")) for h in header))
    path.write_text("\n".join(keep + rows) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------- figures
def plots(out: Path, eps, byq, sim, S) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_REAL, C_SIM, C_FAIL = "#2a6fdb", "#9a9a9a", "#d1495b"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: real vs sim e_pred@1 by object (episode means) + codec floor
    objs = ["pen", "razor", "remote", "toothbrush", "brush", "glue stick", "pencil case"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    sim_ep = defaultdict(list)
    for r in sim:
        if "e_pred@1" in r:
            sim_ep[(r["home"], r["ep"])].append(r["e_pred@1"])
    for i, o in enumerate(objs):
        L = [e for e in eps if e["target"] == o]
        x = i + 0.12 + np.linspace(-0.06, 0.06, len(L))
        ax[0].scatter(x, [e["e_pred@1"] for e in L], s=22, facecolors=[C_REAL if e["success"] else "white" for e in L],
                      edgecolors=[C_REAL if e["success"] else C_FAIL for e in L], linewidths=1.2, zorder=3)
        if o in SIM_HOME:
            v = [np.mean(vv) for (h, _), vv in sim_ep.items() if h in SIM_HOME[o]]
            ax[0].scatter(i - 0.15 + np.linspace(-0.05, 0.05, len(v)), v, s=18, color=C_SIM, zorder=3)
    ax[0].axhline(S["codec"]["e_codec_mean"], color="k", lw=0.8, ls=":")
    ax[0].text(len(objs) - 0.5, S["codec"]["e_codec_mean"] + 0.004, "mp4 codec floor (L0)", ha="right", fontsize=8)
    ax[0].set_xticks(range(len(objs)), objs)
    ax[0].set_ylabel("episode-mean e_pred@1 (VAE latent MSE)")
    ax[0].set_ylim(0, None)
    ax[0].set_title("Real (blue; open red = failure) vs sim θ0 home bin (grey)", fontsize=9, loc="left")
    # right: motion-binned
    mb = S["motion_bins"]
    for key, c, lab in (("real", C_REAL, "real e_pred@1"), ("sim", C_SIM, "sim e_pred@1"),
                        ("real_static", C_REAL, "real e_static@1"), ("sim_static", C_SIM, "sim e_static@1")):
        b = [x for x in mb[key] if x["n"] >= 5]
        xs = [np.mean([x["lo"], min(x["hi"], 60)]) for x in b]
        ax[1].plot(xs, [x["mean"] for x in b], "o-" if "static" not in key else "s--", color=c, ms=3.5, lw=1.2,
                   alpha=1 if "static" not in key else 0.6, label=lab)
    ax[1].set_xlabel("arm motion over next 16 ticks (max |Δq|, deg)")
    ax[1].set_ylabel("per-query mean")
    ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("Excess grows with motion?", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_real_vs_sim.png", dpi=150)
    plt.close(fig)

    # Fig 2: within-group spread (noise floor) and success vs failure
    groups = [g for g in S["noise"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, g in enumerate(groups):
        L = [e for e in eps if e["group"] == g]
        v = np.array([e["e_pred@1"] for e in L])
        ax.scatter(np.full(len(L), i) + np.linspace(-0.12, 0.12, len(L)), v - v.mean(), s=22,
                   facecolors=[C_REAL if e["success"] else "white" for e in L],
                   edgecolors=[C_REAL if e["success"] else C_FAIL for e in L], linewidths=1.2)
    for y in (EXPECTED_PAIRED / 2, -EXPECTED_PAIRED / 2):
        ax.axhline(y, color="k", lw=0.7, ls=":")
    ax.text(len(groups) - 0.6, EXPECTED_PAIRED / 2 + 0.002, "±½ × sim paired effect (0.0125)", fontsize=7, ha="right")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(range(len(groups)), [g.replace("named-", "named\n").replace("each-", "each\n") for g in groups], fontsize=7)
    ax.set_ylabel("episode e_pred@1 − group mean")
    nb = S["noise_pooled_excl_brush"]
    ax.set_title(f"Repeated real episodes (filled = success). Excl. 11-22 brush: within-group SD {nb['pooled_within_sd']:.4f}, "
                 f"median |pair diff| {nb['median_abs_pair_diff']:.4f}; all: {S['noise_pooled']['pooled_within_sd']:.4f} / "
                 f"{S['noise_pooled']['median_abs_pair_diff']:.4f}", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_noise_floor.png", dpi=150)
    plt.close(fig)

    # Fig 3: per-query e_pred@1 along each episode (query index), real vs sim, + e_track16 correlation
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for e in eps:
        rs = [r for r in byq[e["dir"]] if "e_pred@1" in r]
        ax[0].plot([r["q"] for r in rs], [r["e_pred@1"] for r in rs], color=C_REAL if e["success"] else C_FAIL, lw=0.6, alpha=0.35)
    sq = defaultdict(list)
    for r in sim:
        if "e_pred@1" in r:
            sq[r["ep"]].append((r["qi"], r["e_pred@1"]))
    for v in sq.values():
        v.sort()
        ax[0].plot([a for a, _ in v], [b for _, b in v], color=C_SIM, lw=0.6, alpha=0.35)
    ax[0].set_xlim(0, 60)
    ax[0].set_xlabel("query index in episode")
    ax[0].set_ylabel("e_pred@1")
    ax[0].set_title("Per query: real (blue success, red failure) vs sim (grey)", fontsize=9, loc="left")
    qt = [r for d in byq.values() for r in d if "e_pred@1" in r and r.get("e_track16_deg") is not None]
    ax[1].scatter([r["e_track16_deg"] for r in qt], [r["e_pred@1"] for r in qt], s=4, color=C_REAL, alpha=0.4, label="real")
    st_ = [r for r in sim if "e_pred@1" in r and r.get("e_track16_deg") is not None]
    ax[1].scatter([r["e_track16_deg"] for r in st_], [r["e_pred@1"] for r in st_], s=4, color=C_SIM, alpha=0.4, label="sim")
    ax[1].set_xlabel("tracking error over next 16 ticks (deg)")
    ax[1].set_ylabel("e_pred@1")
    ax[1].legend(frameon=False, fontsize=7)
    t = S["tracking"]
    ax[1].set_title(f"ρ real {t['real_query_spearman']:.2f} (within-ep {t['real_query_spearman_within_ep']:.2f}); "
                    f"sim {t['sim_query_spearman']:.2f}", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_per_query.png", dpi=150)
    plt.close(fig)

    # Fig 4: chunk tick profiles (moving chunks, 10-40 deg over the chunk): real starts from rest after the hold
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    k = np.arange(1, 17)
    for nm, c in (("real", C_REAL), ("sim", C_SIM)):
        pr = S["restart"][nm]["profile"]["10-40"]
        ax[0].plot(k, pr["spd_k"], "o-", color=c, ms=3, label=f"{nm} arm (n={pr['n']})")
        ax[0].plot(k, pr["cmd_spd_k"], "--", color=c, lw=1, label=f"{nm} command")
        ax[1].plot(k, pr["trk_k"], "o-", color=c, ms=3, label=nm)
    ax[0].set_xlabel("tick in chunk (query = 0)"); ax[0].set_ylabel("max joint speed (deg / tick)")
    ax[0].set_title("Real: tick 1 includes the 0.8 s hold (arm catches up), then stalls and re-accelerates", fontsize=8, loc="left")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_xlabel("tick in chunk"); ax[1].set_ylabel("|cmd(t) − q after t| (deg, arm mean)")
    ax[1].set_title("Tracking error per tick (latent L1 = ticks 4–16)", fontsize=8, loc="left")
    ax[1].axvspan(4, 16, color="k", alpha=0.04)
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_restart.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
