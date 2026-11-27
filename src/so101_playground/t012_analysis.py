"""T012 stage A analysis: overhead-camera pose sweep (perception gap), no learning.

    python -m so101_playground.t012_analysis --root R [--plot docs/T012_camera_gap.png] [--tsv results.tsv]
    python -m so101_playground.t012_analysis --root R --sheet docs/T012_contact_sheet.png   # smoke contact sheet

Reads <root>/queue/done/*.json (kind stageA). Every condition ran the same (row, rep) episodes as d0 (same jitter,
same episode id), so deltas vs d0 are paired by (row, rep). CIs: 95% percentile bootstrap over the paired
episodes (2000 resamples, seed 0); success also gets a Wilson interval.

Quantities per episode:
  e_pred, e_pred2          rollout mean e_pred@1 / @2 (InternW0 future-delta error, VAE latent MSE)
  e_pred_static(2)         mean over the 3 static-probe queries (arm docked, nothing moves): the visual floor
  motion part              e_pred - e_pred_static: what the rollout adds on top of the floor at that view
  reach_err_cm             TCP-to-object-centre XY at the first commanded jaw close (None if it never closed)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PENS = {"black pen", "red pen"}
ROWS = ["black pen", "black remote", "blue paintbrush", "red pen", "purple toothbrush", "razor"]
TASK = "T012-camera-gap"
METRIC = "Wan2.2_VAE_latent_mse@1(ticks4-16)"
B = 2000


def load(root: Path, kind: str = "stageA") -> dict[str, list[dict]]:
    by = {}
    for f in sorted((root / "queue" / "done").glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("kind") != kind:
            continue
        by.setdefault(r["cond"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: (r["rep"], r["row"]))
    return by


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_ci(x, fn=np.mean, seed: int = 0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return float(fn(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(base: list[dict], cond: list[dict], key) -> list[float]:
    """cond - base per (row, rep), both values present."""
    b = {(r["row"], r["rep"]): r for r in base}
    out = []
    for r in cond:
        o = b.get((r["row"], r["rep"]))
        if o is None:
            continue
        x, y = key(r), key(o)
        if x is not None and y is not None:
            out.append(x - y)
    return out


def g(k):
    return lambda r: r.get(k)


def motion(r):
    return None if r.get("e_pred") is None or r.get("e_pred_static") is None else r["e_pred"] - r["e_pred_static"]


def motion2(r):
    return None if r.get("e_pred2") is None or r.get("e_pred_static2") is None else r["e_pred2"] - r["e_pred_static2"]


def axis_of(cond: str):
    if cond == "d0":
        return None, 0.0
    ax = "tilt" if cond.startswith("tilt") else "pan"
    return ax, float(cond[len(ax):])


def rank(x):
    x = np.asarray(x, dtype=float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    # average ties
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def boot_spearman(a, b, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a))
        bs.append(spearman(a[i], b[i]))
    bs = [v for v in bs if np.isfinite(v)]
    if not bs:
        return spearman(a, b), float("nan"), float("nan")
    return spearman(a, b), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def auroc(score, label):
    """P(score of a positive > score of a negative), ties 1/2. label 1 = failure."""
    s, y = np.asarray(score, float), np.asarray(label, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def boot_auroc(score, label, seed=0):
    s, y = np.asarray(score, float), np.asarray(label, int)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(s), len(s))
        v = auroc(s[i], y[i])
        if np.isfinite(v):
            bs.append(v)
    if not bs:
        return auroc(s, y), float("nan"), float("nan")
    return auroc(s, y), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def residualize(y, groups):
    """y minus its group mean for each grouping in turn (additive fixed effects, 10 alternating passes)."""
    y = np.asarray(y, float).copy()
    for _ in range(10):
        for gr in groups:
            gr = np.asarray(gr)
            for v in np.unique(gr):
                m = gr == v
                y[m] -= y[m].mean()
    return y


def summarize(by: dict[str, list[dict]]) -> dict:
    base = by.get("d0", [])
    out = {}
    for cond, rs in by.items():
        n = len(rs)
        k = sum(bool(r["success"]) for r in rs)
        s = {"n": n, "success": k, "wilson": wilson(k, n),
             "timeouts": sum(r["reason"] == "time_out" for r in rs),
             "pens": (sum(bool(r["success"]) for r in rs if r["target"] in PENS), sum(r["target"] in PENS for r in rs)),
             "others": (sum(bool(r["success"]) for r in rs if r["target"] not in PENS),
                        sum(r["target"] not in PENS for r in rs)),
             "per_object": {ROWS[row]: "".join("S" if r["success"] else ("T" if r["reason"] == "time_out" else "F")
                                               for r in sorted(rs, key=lambda r: r["rep"]) if r["row"] == row)
                            for row in range(6)},
             "no_close": sum(r.get("reach_err_cm") is None for r in rs)}
        for name, key in [("e_pred", g("e_pred")), ("e_pred2", g("e_pred2")), ("e_pred_static", g("e_pred_static")),
                          ("e_pred_static2", g("e_pred_static2")), ("motion", motion), ("motion2", motion2),
                          ("e_pred_overhead", g("e_pred_overhead")), ("e_pred_wrist", g("e_pred_wrist")),
                          ("e_pred_static_overhead", g("e_pred_static_overhead")),
                          ("e_track_deg", g("e_track_deg")), ("path_len_deg", g("path_len_deg")),
                          ("reach_err_cm", g("reach_err_cm")), ("ticks", g("ticks"))]:
            s[name] = boot_ci([key(r) for r in rs])
            if cond != "d0" and base:
                s["d_" + name] = boot_ci(paired(base, rs, key))
        s["reach_median"] = float(np.median([r["reach_err_cm"] for r in rs if r.get("reach_err_cm") is not None])) \
            if any(r.get("reach_err_cm") is not None for r in rs) else float("nan")
        if cond != "d0" and base:
            sb = {(r["row"], r["rep"]): r["success"] for r in base}
            d = [int(r["success"]) - int(sb[(r["row"], r["rep"])]) for r in rs if (r["row"], r["rep"]) in sb]
            s["d_success"] = boot_ci(d)
        chk = [r.get("probe0_vs_first_chunk_maxdiff") for r in rs if r.get("probe0_vs_first_chunk_maxdiff") is not None]
        s["probe_check_max"] = max(chk) if chk else None
        s["static_chunk_motion"] = float(np.mean([np.mean(r["static_chunk_motion"]) for r in rs
                                                  if r.get("static_chunk_motion")])) if rs else float("nan")
        out[cond] = s
    return out


def correlations(by: dict[str, list[dict]]) -> dict:
    rs = [r for v in by.values() for r in v]
    res = {}
    have = [r for r in rs if r.get("reach_err_cm") is not None and r.get("e_pred") is not None]
    e = np.array([r["e_pred"] for r in have])
    m = np.array([motion(r) if motion(r) is not None else np.nan for r in have])
    st = np.array([r.get("e_pred_static") if r.get("e_pred_static") is not None else np.nan for r in have])
    re_ = np.array([r["reach_err_cm"] for r in have])
    cond = np.array([r["cond"] for r in have])
    row = np.array([r["row"] for r in have])
    res["n"] = len(have)
    res["pooled_e_pred"] = boot_spearman(e, re_)
    # within condition and layout: remove condition and row means from both (the object/view confounds)
    res["within_e_pred"] = boot_spearman(residualize(e, [cond, row]), residualize(re_, [cond, row]))
    ok = np.isfinite(m)
    res["within_motion"] = boot_spearman(residualize(m[ok], [cond[ok], row[ok]]), residualize(re_[ok], [cond[ok], row[ok]]))
    res["within_static"] = boot_spearman(residualize(st[ok], [cond[ok], row[ok]]), residualize(re_[ok], [cond[ok], row[ok]]))
    # across conditions: paired deltas vs d0 (same row, rep): does a view that raises e_pred also raise reach error?
    base = by.get("d0", [])
    de, dr, dm, ds = [], [], [], []
    bmap = {(r["row"], r["rep"]): r for r in base}
    for c, v in by.items():
        if c == "d0":
            continue
        for r in v:
            o = bmap.get((r["row"], r["rep"]))
            if o is None or None in (r.get("reach_err_cm"), o.get("reach_err_cm"), r.get("e_pred"), o.get("e_pred")):
                continue
            de.append(r["e_pred"] - o["e_pred"])
            dr.append(r["reach_err_cm"] - o["reach_err_cm"])
            dm.append((motion(r) or 0) - (motion(o) or 0))
            ds.append((r.get("e_pred_static") or 0) - (o.get("e_pred_static") or 0))
    res["paired_delta_e_pred"] = boot_spearman(de, dr) if len(de) > 3 else None
    res["paired_delta_motion"] = boot_spearman(dm, dr) if len(dm) > 3 else None
    res["paired_delta_static"] = boot_spearman(ds, dr) if len(ds) > 3 else None
    res["n_paired"] = len(de)
    # condition means
    cm = [(np.mean([r["e_pred"] for r in v if r.get("e_pred") is not None]),
           np.mean([r["reach_err_cm"] for r in v if r.get("reach_err_cm") is not None]),
           np.mean([r["e_pred_static"] for r in v if r.get("e_pred_static") is not None]))
          for v in by.values()]
    res["cond_means_e_pred_vs_reach"] = spearman([a for a, _, _ in cm], [b for _, b, _ in cm])
    res["cond_means_static_vs_reach"] = spearman([c for _, _, c in cm], [b for _, b, _ in cm])
    # failure AUROC (pooled over all conditions): e_pred, static floor, motion part, reach error
    all_ = [r for r in rs if r.get("e_pred") is not None]
    y = [0 if r["success"] else 1 for r in all_]
    res["auroc"] = {
        "e_pred": boot_auroc([r["e_pred"] for r in all_], y),
        "e_pred2": boot_auroc([r.get("e_pred2") if r.get("e_pred2") is not None else np.nan for r in all_], y)
        if all(r.get("e_pred2") is not None for r in all_) else None,
        "e_pred_static": boot_auroc([r.get("e_pred_static") or 0 for r in all_], y),
        "motion": boot_auroc([motion(r) or 0 for r in all_], y),
        "reach_err (no close = inf)": boot_auroc([r["reach_err_cm"] if r.get("reach_err_cm") is not None else 1e9
                                                  for r in all_], y),
        "e_pred within row+cond": boot_auroc(residualize([r["e_pred"] for r in all_],
                                                         [[r["cond"] for r in all_], [r["row"] for r in all_]]), y),
    }
    res["n_fail"] = int(sum(y))
    return res


def gate(summ: dict) -> dict:
    out = {}
    for ax in ("tilt", "pan"):
        conds = sorted([c for c in summ if axis_of(c)[0] == ax], key=lambda c: (abs(axis_of(c)[1]), axis_of(c)[1]))
        broken = [c for c in conds if summ[c]["success"] <= 7 and summ[c]["n"] >= 12]
        out[ax] = broken[0] if broken else None
    return out


def fmt(t, d=4, sign=False):
    if t is None or t[0] is None or not np.isfinite(t[0]):
        return "—"
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]"


ORDER = ["d0", "tilt-12", "tilt-8", "tilt-4", "tilt+4", "tilt+8", "tilt+12", "pan-12", "pan-8", "pan-4", "pan+4",
         "pan+8", "pan+12"]


def report(by, summ, corr) -> str:
    conds = [c for c in ORDER if c in summ]
    L = ["# T012 stage A analysis", "", f"episodes: {sum(s['n'] for s in summ.values())}", ""]
    L += ["## Per condition", "",
          "| cond | success | pens | others | timeouts | Δsucc vs d0 | e_pred@1 | Δe_pred@1 | e_pred@2 | Δe_pred@2 | "
          "e_pred_static | Δstatic | motion (ep−static) | Δmotion | e_track ° | Δe_track | reach cm (median) | Δreach | "
          "no close | path ° |",
          "|" + "---|" * 20]
    for c in conds:
        s = summ[c]
        L.append("| " + " | ".join([
            c, f"{s['success']}/{s['n']}", f"{s['pens'][0]}/{s['pens'][1]}", f"{s['others'][0]}/{s['others'][1]}",
            str(s["timeouts"]), fmt(s.get("d_success"), 2, True), fmt(s["e_pred"]), fmt(s.get("d_e_pred"), 4, True),
            fmt(s["e_pred2"]), fmt(s.get("d_e_pred2"), 4, True), fmt(s["e_pred_static"]),
            fmt(s.get("d_e_pred_static"), 4, True), fmt(s["motion"]), fmt(s.get("d_motion"), 4, True),
            fmt(s["e_track_deg"], 2), fmt(s.get("d_e_track_deg"), 2, True),
            f"{fmt(s['reach_err_cm'], 2)} ({s['reach_median']:.2f})", fmt(s.get("d_reach_err_cm"), 2, True),
            str(s["no_close"]), f"{s['path_len_deg'][0]:.0f}"]) + " |")
    L += ["", "## Per object (S success, T timeout, F failure; rep 0, rep 1)", "",
          "| object | " + " | ".join(conds) + " |", "|" + "---|" * (len(conds) + 1)]
    for o in ROWS:
        L.append(f"| {o} | " + " | ".join(summ[c]["per_object"].get(o, "") for c in conds) + " |")
    L += ["", "## Overhead vs wrist halves", "", "| cond | e_pred overhead | Δ | e_pred wrist | Δ | static overhead | Δ |",
          "|---|---|---|---|---|---|---|"]
    for c in conds:
        s = summ[c]
        L.append(f"| {c} | {fmt(s['e_pred_overhead'])} | {fmt(s.get('d_e_pred_overhead'), 4, True)} | "
                 f"{fmt(s['e_pred_wrist'])} | {fmt(s.get('d_e_pred_wrist'), 4, True)} | "
                 f"{fmt(s['e_pred_static_overhead'])} | {fmt(s.get('d_e_pred_static_overhead'), 4, True)} |")
    L += ["", "## Checks", ""]
    for c in conds:
        s = summ[c]
        L.append(f"- {c}: probe-0 vs rollout first chunk max |diff| = {s['probe_check_max']}; "
                 f"static-probe chunk motion (max |a − state|, LeRobot deg) = {s['static_chunk_motion']:.2f}")
    L += ["", "## e_pred vs reach error (Spearman, 95% bootstrap)", "",
          f"- episodes with a close event: {corr['n']}",
          f"- pooled (all conditions, all objects): {fmt(corr['pooled_e_pred'], 3)}",
          f"- within condition and layout (both residualized on cond + row): e_pred {fmt(corr['within_e_pred'], 3)}, "
          f"motion part {fmt(corr['within_motion'], 3)}, static floor {fmt(corr['within_static'], 3)}",
          f"- paired deltas vs d0 (n={corr['n_paired']}): Δe_pred~Δreach {fmt(corr['paired_delta_e_pred'], 3)}, "
          f"Δmotion~Δreach {fmt(corr['paired_delta_motion'], 3)}, Δstatic~Δreach {fmt(corr['paired_delta_static'], 3)}",
          f"- condition means (n={len(summ)}): e_pred~reach {corr['cond_means_e_pred_vs_reach']:.3f}, "
          f"static~reach {corr['cond_means_static_vs_reach']:.3f}",
          "", f"## Failure AUROC (pooled, {corr['n_fail']} failures)", ""]
    for k, v in corr["auroc"].items():
        L.append(f"- {k}: {fmt(v, 3) if v else '—'}")
    gt = gate(summ)
    L += ["", "## Stage B gate (Δ* = smallest |Δ| with success ≤ 7/12)", ""]
    for ax, c in gt.items():
        L.append(f"- {ax}: {c or 'none up to the largest tested |Δ|'}")
    return "\n".join(L) + "\n"


def plot(by, summ, corr, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    col = {"tilt": "#3366aa", "pan": "#cc6633"}
    for a in ("tilt", "pan"):
        cs = sorted([c for c in summ if axis_of(c)[0] == a] + ["d0"], key=lambda c: axis_of(c)[1])
        x = [axis_of(c)[1] for c in cs]
        sc = [summ[c]["success"] / summ[c]["n"] for c in cs]
        ax[0].plot(x, sc, "o-", color=col[a], label=a)
        for k, ls, lab in (("e_pred", "-", "rollout"), ("e_pred_static", "--", "static floor")):
            y = [summ[c][k][0] for c in cs]
            lo = [summ[c][k][1] for c in cs]
            hi = [summ[c][k][2] for c in cs]
            ax[1].errorbar(x, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="o" + ls, color=col[a],
                           capsize=2, label=f"{a} {lab}", alpha=1 if ls == "-" else 0.6)
        y = [summ[c]["motion"][0] for c in cs]
        ax[2].plot(x, y, "o-", color=col[a], label=f"{a} motion (rollout − static)")
        y = [summ[c]["reach_err_cm"][0] for c in cs]
        ax[2].plot([], [])
        ax2 = ax[2]
    ax[0].axhline(7 / 12, color="gray", ls=":", lw=1)
    ax[0].set(xlabel="Δ (deg)", ylabel="success", title="success (12 ep / cond); dotted = gate 7/12", ylim=(0, 1.05))
    ax[0].legend()
    ax[1].set(xlabel="Δ (deg)", ylabel="e_pred@1", title="e_pred@1: rollout vs static floor")
    ax[1].legend(fontsize=7)
    ax2.set(xlabel="Δ (deg)", ylabel="e_pred − e_pred_static", title="motion part of e_pred")
    ax2.legend(fontsize=7)
    rs = [r for v in by.values() for r in v if r.get("reach_err_cm") is not None and r.get("e_pred") is not None]
    for r in rs:
        a = axis_of(r["cond"])[0]
        ax[3].scatter(r["reach_err_cm"], r["e_pred"], s=14, color=col.get(a, "black"),
                      marker="o" if r["success"] else "x", alpha=0.7)
    ax[3].set(xscale="log", xlabel="reach error at first close (cm, log)", ylabel="e_pred@1",
              title=f"per episode (o success, x fail); within ρ={corr['within_e_pred'][0]:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def contact_sheet(root: Path, out: Path):
    from PIL import Image, ImageDraw

    conds = ["tilt-5", "d0", "tilt+5", "pan-5", "pan+5"]
    tiles = []
    for c in conds:
        d = root / "episodes" / "frames" / f"t012sheet-{c}-home0-r0"
        fs = sorted(d.glob("overhead_*.png"))
        if not fs:
            continue
        im = Image.open(fs[0]).convert("RGB")
        dr = ImageDraw.Draw(im)
        w, h = im.size
        dr.line([(w // 2, 0), (w // 2, h)], fill=(0, 255, 0))
        dr.line([(0, h // 2), (w, h // 2)], fill=(0, 255, 0))
        dr.rectangle([0, 0, 150, 24], fill=(0, 0, 0))
        dr.text((6, 6), c, fill=(255, 255, 255))
        tiles.append(im)
    if not tiles:
        raise SystemExit("no frames")
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * 3, h * 2), (40, 40, 40))
    pos = {"tilt-5": (1, 0), "d0": (1, 1), "tilt+5": (2, 0), "pan-5": (0, 1), "pan+5": (2, 1)}
    for c, im in zip([c for c in conds if (root / "episodes/frames" / f"t012sheet-{c}-home0-r0").exists()], tiles):
        x, y = pos[c]
        sheet.paste(im, (x * w, y * h))
    sheet.save(out)


def write_tsv(path: Path, summ, corr):
    rows = []
    for c in [c for c in ORDER if c in summ]:
        s = summ[c]
        d = s.get("d_e_pred")
        rows.append([f"T012A-{c}", TASK, "internw0", f"camera {c}", str(s["n"]), f"{s['success'] / s['n']:.3f}",
                     f"{s['e_track_deg'][0]:.3f}", "", METRIC, f"{s['e_pred'][0]:.4f}", "", "", "",
                     "" if d is None else f"{d[0]:+.4f}", "" if d is None else f"[{d[1]:+.4f}, {d[2]:+.4f}]", "", "",
                     f"overhead camera delta {c} (tilt=roll field, pan=-yaw field), no control gap, no adapter; home bin "
                     f"6 layouts x 2 T010-eval jitters, paired vs d0; e_pred@2={s['e_pred2'][0]:.4f}; "
                     f"e_pred_static(3 docked probes)={s['e_pred_static'][0]:.4f}; motion={s['motion'][0]:.4f}; "
                     f"reach_err mean {s['reach_err_cm'][0]:.2f} cm (no close {s['no_close']}); pens "
                     f"{s['pens'][0]}/{s['pens'][1]}; timeouts {s['timeouts']}"])
    lines = path.read_text().splitlines() if path.exists() else []
    lines = [l for l in lines if not l.startswith("T012A-")]
    lines += ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--plot", type=Path)
    ap.add_argument("--tsv", type=Path)
    ap.add_argument("--sheet", type=Path)
    a = ap.parse_args()
    if a.sheet:
        contact_sheet(a.root, a.sheet)
        print(f"wrote {a.sheet}")
        return
    by = load(a.root)
    summ = summarize(by)
    corr = correlations(by)
    txt = report(by, summ, corr)
    (a.root / "analysis.md").write_text(txt)
    print(txt)
    (a.root / "summary.json").write_text(json.dumps({"summary": summ, "corr": corr}, default=str, indent=1))
    if a.plot:
        plot(by, summ, corr, a.plot)
    if a.tsv:
        write_tsv(a.tsv, summ, corr)


if __name__ == "__main__":
    main()
