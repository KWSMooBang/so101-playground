# T005: can raw e_pred guide a correction of the control gap? (oracle correction, sim, LaWAM)

Branch `exp/wam-oracle` (from `wam-control-gap`). Outputs `/outputs/so101-playground/runs/T005/`. Rows in
`/outputs/so101-playground/results.tsv` (`T005-none`, `T005-<gap>-a<α>`). Figure:
`docs/T005_oracle_alpha.png` (copy of `runs/T005/T005_oracle_alpha.png`). Per-setting numbers: `runs/T005/lawam/summary.json`, `analysis.txt`.

## Question

On a real robot the only reward available is raw `e_pred` = d(WAM prediction, actual observation). Before
training an RL adapter on it (H2-P1), we check which gaps it can guide. The future adapter is replaced by an
**oracle correction** that knows the injected gap and applies it at strength α ∈ {0, 0.5, 1, 1.5}. If raw e_pred
does not go back to the `none` level at α=1 and does not change monotonically with α, an adapter trained on
e_pred cannot learn that correction either.

## Setup

Same as T004 (see `docs/T004-wam-control-gap.md`): LaWAM full FT u45k (`lawam_probe_server.py`, GPU 1), Bin
1-object episodes `tasks/T004_bin1.jsonl` rows 0–7, bench val layouts, env seed 1984, 30 s limit, action
horizon 36. Metric: LAM DINOv3 token MSE between `h_t1_pred` and the observed overhead frame 35 ticks after the
query. **No replay** (raw e_pred only). Flow noise is seeded per (episode, query tick) as in T004.

**Where the oracle acts** (`src/so101_playground/oracle.py`): between the bench client's 15°/tick clamp and the
gap, i.e. where a residual adapter would act. It only changes the command sent; sim parameters are never
touched.

    policy chunk → client clamp → cmd → oracle(α) → sent → gap → applied → env.step

| gap | gap (from T004) | oracle, strength α | exact at α=1? |
|---|---|---|---|
| offset6 | applied = sent + 6° on Pitch, Elbow | sent = cmd − α·offset | yes, by construction (applied = cmd on 100% of ticks) |
| gain2 | arm PD stiffness ×0.25 (damping kept) | sent = q + (1 + α(1/s − 1))·(cmd − q) on the 5 arm joints, s = 0.25 (×4 at α=1), q = the measured position this tick | approximately: q moves during the 8 physics substeps while the target is held. Measured: e_track 0.589° vs none 0.593° |
| latency6 | applied(t) = sent(t − 6) | at chunk row i send clamp(q_t, row min(i + round(6α), 35)) of the current chunk | no, see below |

`e_track` = mean |cmd − q(t+1)| over the arm joints and all policy ticks, with **cmd = the policy's command before
oracle and gap**, so it measures how well the arm does what the policy wanted.

**Latency oracle, measured inexactness.** At α=1 only 46% of ticks have applied = cmd. Two sources, both
inherent to an adapter that can only send commands early:
- The first 6 rows of every chunk can never be delivered: they would have to be sent before the chunk exists.
  The previous chunk's last row is held instead. Rows < 6: |applied − cmd| 1.37° (excl. the first chunk).
- The clamp is evaluated 6 ticks early, against q(t − 6) instead of q(t). For rows ≥ 6 that is the only
  difference, so the remaining |applied − cmd| = 0.76° (excl. the first chunk; 1.02° incl.) is entirely clamp.
  In the first chunk the arm leaves the hold pose with large clamped steps.

The latency oracle therefore removes only ~38% of the excess e_track (2.16° → 1.57°, none 0.59°). α=1.5 is not
better (1.65°).

Runs: `none` + 3 gaps × 4 α = 13 settings × 8 episodes, 2 Isaac clients in parallel against one server
(`scripts/run_t005_sweep.sh`).

## α = 0 sanity check vs T004

α=0 is the identity, so these are the T004 conditions again without replay.

| setting | T005 (α=0, no replay) success / e_track / e_pred / Δ vs none | T004 (nominal replay, or first sweep for offset6) |
|---|---|---|
| none | 8/8, 0.59°, 0.0689 | 8/8, 0.66°, 0.0753 (first sweep 7/8, 0.67°, 0.0739) |
| offset6 | 0/8, 2.06°, 0.0772, +0.0083 [0.0002, 0.0180] | first sweep 1/8, 2.06°, 0.0798, +0.0059 [−0.0020, 0.0153] |
| gain2 | 6/8, 2.17°, 0.1247, +0.056 [0.008, 0.100] | 5/8, 2.23°, 0.0846, +0.009 [−0.004, 0.022] (first sweep 7/8, 2.31°, 0.0982, +0.024) |
| latency6 | 5/8, 2.16°, 0.0857, +0.017 [0.005, 0.030] | 7/8, 2.46°, 0.0969, +0.022 [0.011, 0.033] |

- Success and e_track reproduce within noise. none e_track is lower without replay (0.59° vs 0.66°): the
  snapshot/restore of the T004 replays perturbs the arm.
- gain2 e_pred is higher here because two failed episodes (3, 5) have per-episode e_pred ≈ 0.20. These are
  failure states the model predicts badly. The six successful episodes give ≈ 0.082 (+0.013).
- So raw e_pred at α=0 depends strongly on which episodes fail: compare 0.085 / 0.088 / 0.098 / 0.125 for gain2
  across four runs.

**Noise floor.** offset6 α=1 is an exact inverse (applied = cmd on every tick), i.e. a second `none` run that
differs only by physics nondeterminism. Its Δ vs none is +0.0038 [0.0003, 0.0080] and on changed tokens
+0.011 [0.000, 0.026]. The episode-paired bootstrap CI therefore does not cover run-to-run variation.
|Δ| ≲ 0.004 (raw) / ≲ 0.012 (changed tokens) should be read as noise.

## Results

> **Caution when reading Δ vs none:** offset6 α=1 is an exact inverse (applied = cmd on every tick), so in effect
> it is a second `none` run, yet it sits at +0.0038 [0.0003, 0.0080]. The episode-bootstrap CIs do not cover
> run-to-run (physics nondeterminism) variation. Treat |Δ| ≲ 0.004 in raw e_pred (≲ 0.012 on changed tokens) as
> noise, even when the CI excludes 0.

Means are query-weighted (e_pred) or tick-weighted (e_track). CIs: 95% bootstrap over episodes; Δ vs none
episode-paired. "changed tokens" = per query, the 25% of the 16×16 LAM tokens with the largest observed change
mse(h_t, h_obs) (exactly k = 64 tokens, ties broken by token index; an approximate arm + object mask; the rest is
"unchanged").

| setting | success | e_track ° | \|applied − cmd\| ° | e_pred | Δ vs none [95% CI] | changed-token Δ vs none | queries |
|---|---|---|---|---|---|---|---|
| none | 8/8 | 0.59 | 0 | 0.0689 | — | — | 93 |
| offset6 α0 | 0/8 | 2.06 | 2.40 | 0.0772 | +0.0083 [0.0002, 0.0180] | +0.045 [0.028, 0.068] | 181 |
| offset6 α0.5 | 3/8 | 1.40 | 1.20 | 0.0875 | +0.0186 [0.0032, 0.0343] | +0.066 [0.024, 0.114] | 128 |
| offset6 α1 | 8/8 | 0.59 | 0 | 0.0727 | +0.0038 [0.0003, 0.0080] | +0.011 [0.000, 0.026] | 91 |
| offset6 α1.5 | 0/8 | 1.20 | 1.20 | 0.0725 | +0.0036 [−0.0071, 0.0165] | +0.032 [0.003, 0.064] | 184 |
| gain2 α0 | 6/8 | 2.17 | 0 | 0.1247 | +0.0557 [0.0078, 0.0998] | +0.140 [0.021, 0.250] | 128 |
| gain2 α0.5 | 8/8 | 0.84 | 1.76 | 0.0706 | +0.0016 [−0.0037, 0.0071] | +0.006 [−0.010, 0.023] | 90 |
| gain2 α1 | 8/8 | 0.59 | 2.51 | 0.0696 | +0.0007 [−0.0020, 0.0030] | +0.003 [−0.004, 0.008] | 97 |
| gain2 α1.5 | 7/8 | 0.65 | 3.75 | 0.0783 | +0.0094 [0.0032, 0.0170] | +0.033 [0.014, 0.053] | 111 |
| latency6 α0 | 5/8 | 2.16 | 1.92 | 0.0857 | +0.0168 [0.0050, 0.0295] | +0.045 [0.017, 0.079] | 129 |
| latency6 α0.5 | 7/8 | 1.97 | 1.68 | 0.0915 | +0.0226 [0.0111, 0.0355] | +0.065 [0.032, 0.100] | 119 |
| latency6 α1 | 7/8 | 1.57 | 1.11 | 0.0846 | +0.0157 [0.0016, 0.0331] | +0.047 [0.004, 0.102] | 110 |
| latency6 α1.5 | 8/8 | 1.65 | 1.43 | 0.0809 | +0.0120 [0.0059, 0.0186] | +0.040 [0.018, 0.059] | 106 |

(For gain2, |applied − cmd| is non-zero by design: the oracle amplifies the target.)

### Reward discriminability: AUROC that e_pred ranks α=0 (uncorrected) above α=1 (corrected)

0.5 = no signal. Episode = 8 vs 8 episode means; query = single queries; window4 = means of 4 consecutive
queries (non-overlapping) within an episode. CIs: episode-paired bootstrap (the same resampled episode ids, i.e.
the same layouts and seeds, are used for both settings).

| gap | raw e_pred: episode | query | window4 | changed tokens: episode | query | window4 |
|---|---|---|---|---|---|---|
| offset6 | 0.59 [0.29, 0.86] | 0.59 [0.52, 0.66] | 0.51 [0.40, 0.61] | 0.77 [0.46, 0.95] | 0.65 [0.60, 0.71] | 0.59 [0.49, 0.68] |
| gain2 | 0.77 [0.54, 0.95] | 0.69 [0.55, 0.78] | 0.71 [0.58, 0.80] | **0.91 [0.68, 1.00]** | 0.69 [0.57, 0.80] | 0.70 [0.58, 0.81] |
| latency6 | 0.63 [0.44, 0.84] | 0.45 [0.40, 0.52] | 0.48 [0.40, 0.58] | 0.66 [0.45, 0.88] | 0.46 [0.42, 0.52] | 0.47 [0.38, 0.59] |

α=1 vs none (should be 0.5): offset6 query 0.51, gain2 0.50, latency6 0.57.

## Read

- **gain2 (stiffness): e_pred can guide it.**
  - Raw e_pred is U-shaped in α with its minimum at α=1: +0.056 → +0.002 → +0.001 → +0.009. At α=1 it is back on
    the none level (CI of the difference [−0.002, 0.003]).
  - Overcorrecting (α=1.5) is worse again and clears zero (+0.009 [0.003, 0.017]; changed tokens +0.033).
  - e_track follows the same shape (2.17 → 0.84 → 0.59 → 0.65; none 0.59).
  - SNR: per-query AUROC 0.69, 4-query window 0.71, episode mean 0.77 (changed tokens 0.91). Usable but noisy.
  - Caveat: the α=0 gap in e_pred is partly carried by two failed episodes (successful episodes only: ≈ +0.013).
    The α=0.5 → 1 → 1.5 part, where all episodes mostly succeed, is the cleaner evidence.
- **offset6 (calibration offset): raw e_pred cannot guide it.**
  - The correction is exact and success goes 0/8 → 3/8 → 8/8 → 0/8. But raw e_pred is not monotone: α=0.5
    (+0.019) is *worse* than α=0 (+0.008).
  - α=1.5 (net −3°, 0/8 success, e_track 1.20°) is indistinguishable from none (+0.004, CI ∋ 0).
  - Per-query AUROC α0 vs α1 is 0.59 and the window4 AUROC is 0.51.
  - The changed-token split helps somewhat (+0.045 / +0.066 / +0.011 / +0.032, minimum at α=1, query AUROC 0.65),
    but it is still not monotone on 0 → 0.5.
  - Likely reason: a constant offset changes *where* the arm ends up, not how the motion unfolds, and failed
    rollouts move into different states (re-grasps, hovering) with their own e_pred level. That is the
    state-distribution confound flagged in T004.
- **latency6 (delay): untested at full strength.** The best command-side oracle only removes ~38% of the excess
  tracking error: the first 6 rows of each chunk cannot be delivered early, and the clamp is evaluated against a
  stale q. Raw e_pred does not respond to that partial correction (+0.017 / +0.023 / +0.016 / +0.012; query AUROC
  0.45). With a 36-step chunk and a 6-tick delay, a command-only adapter cannot fully undo the gap. Correcting it
  needs a policy-side change, e.g. querying earlier or conditioning on the delay.
- **Noise.** The exact-inverse offset6 α=1 run sits +0.004 above none, so differences of that size are run-to-run
  noise. Across T004/T005, the raw e_pred level of one condition varies by more than its CI (gain2 α0:
  0.085–0.125), driven by which episodes fail.

**For P1:**
- Raw e_pred is a usable reward only for gaps that change the arm's *dynamics* within the predicted window (the
  gain2 example), and even then the per-query AUROC is ~0.7. An RL adapter would need episode- or
  window-aggregated rewards and a strong baseline.
- Restricting the distance to the changed (arm/object) tokens raises the episode-level separability (gain2 0.91,
  offset6 0.77) and costs nothing on a real robot. It is the first change to try in the reward.
- Constant offsets are not observable through raw e_pred in this setup.

## Compute

GPU 1, LaWAM server + Isaac clients, 02:45–04:48 (2026-11-22): **≈ 2.05 GPU-h** (cap 3). Includes the smoke test
and two aborted starts: 4 clients, then 3, which OOM'd or failed at Isaac startup. Each client takes ~6.5–7 GB
next to the 7.7 GB server, so only 2 fit on the card. No episode of the aborted starts was kept (logs in
`runs/T005/lawam/aborted/`).

## Reproduce

```bash
W=/workspace/so101-playground/.claude/worktrees/wam-control-gap   # lawam venv + weights (T004)
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 $W/envs/lawam/.venv/bin/python src/so101_playground/lawam_probe_server.py \
    --repo_id $W/envs/lawam/weights/so101_bench_LaWAM --port 18805 --keep 64 --log_dir /outputs/so101-playground/runs/T005/lawam
GPU=1 scripts/run_t005_sweep.sh 18805 /outputs/so101-playground/runs/T005 2 8 none offset6:0 offset6:1 gain2:0 gain2:1 \
    latency6:0 latency6:1 offset6:0.5 gain2:0.5 latency6:0.5 offset6:1.5 gain2:1.5 latency6:1.5
PYTHONPATH=src $W/envs/flux3/flux-action/.venv/bin/python src/so101_playground/t005_analysis.py \
    --root /outputs/so101-playground/runs/T005 --plot /outputs/so101-playground/runs/T005/T005_oracle_alpha.png \
    --tsv /outputs/so101-playground/results.tsv
```

The Isaac env is `ISAAC_ENV` (default `.claude/worktrees/scaffold-env`, as in T004).
