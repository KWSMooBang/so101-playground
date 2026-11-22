# T006: does a second, video-generating WAM (InternW0-Δ) show the H2 signal? (sim only)

Branch `exp/wam-internw0` (from `exp/wam-oracle`). Outputs `/outputs/so101-playground/runs/T006/` (`nom/`,
`oracle/`, `server/`). Rows in `/outputs/so101-playground/results.tsv`: `T006nom-internw0-<cond>` (T004nom
design) and `T006-internw0-<setting>` (T005 design). Figure `docs/T006_internw0_vs_lawam.png` (copy of
`runs/T006/T006_internw0_vs_lawam.png`). Per-setting numbers: `runs/T006/{nom,oracle}/internw0/summary.json`.

Two questions, the same as T004nom and T005 asked of LaWAM:
1. Under an injected control gap, does InternW0's predicted future sit on the fault-free side of the actual outcome?
2. Does its raw e_pred, the only reward available on a real robot, guide the stiffness correction the way LaWAM's did?

## Model and server

**Checkpoint.** `5hadytru/so101_bench_InternW0-Delta` @ 75e60ae, the official step-18200 `model.pt` at the
repo root. The bench reports 20/39 for it on its own val set (`real_gr00t_val_v2`); step 16000 scores 22/39.
The InternW0-Delta repo is pinned at 90801ba.

**Environment and weights.** `envs/internw0/setup_internw0.sh` installs the venv and only the weights
inference needs, 28 GB in total:
- the checkpoint, its config and its stats (12.4 GB);
- the converted T5 encoder and Wan2.2 VAE from ModelScope (12 GB);
- RynnBrain1.1-2B (4.4 GB);
- the UMT5 tokenizer and the Wan2.2 config JSONs.

No Wan2.2 DiT weights are downloaded: the task config sets `skip_dit_load_from_pretrain`, and the checkpoint
holds the whole video and action expert. The venv takes 6.8 GB more.

**Server.** `src/so101_playground/internw0_probe_server.py` is a copy of the bench `internw0_server.py`; the
submodule is not edited. Start it with `scripts/run_internw0_server.sh`. On GPU 0 with 3 Isaac clients:
- **/act: 0.60 s per query** (median; p90 0.83 s);
- **/iw_score: 0.23 s**;
- **peak 17.5 GiB allocated / 17.9 GiB reserved, 18.8 GB on the card.**

The UMT5-XXL text encoder (11 GB) stays on the CPU and each instruction is encoded once (2.6 s). Without that
the model's own `__init__` moves it to the GPU and OOMs. Sessions (anchor + recent memory frames) are keyed by
(client run dir, client attempt id, episode) since the review (see Review corrections #1), so parallel clients
never share memory.

### What InternW0-Δ predicts, and e_pred

- **Training window.** 33 action ticks. Video frames are taken every 4 ticks: ticks 0, 4, …, 32, 9 frames on
  the 384×256 overhead-over-wrist canvas.
- **VAE latents.** The causal Wan2.2 VAE (×16 spatial, ×4 temporal) turns them into 3 latent frames of
  48×24×16: **L0 = tick 0, L1 = ticks 4–16, L2 = ticks 20–32**.
- **`future_delta`.** 96 learned query tokens (one per 2×2 latent patch; `num_views: 1`) are appended to the
  video expert. They attend only to the clean recent + current frame: never the noisy future, never actions.
  A LayerNorm+Linear head maps them to the latent deltas (L1 − L0, L2 − L1), trained with MSE (weight 0.5).
- **The actions read the prediction.** The action tokens attend to these queries.
- **Online inference denoises no video.** The video expert runs a single prefill over (anchor, recent,
  current, delta queries), and only the actions are denoised (10 steps). The repo never evaluates the delta
  head at inference. The probe server captures the last video-layer hidden state during that prefill and runs
  `video_expert.predict_future_delta` on it: exactly the tensor the training loss scored. This adds no extra
  forward pass, and the actions are unchanged.

**e_pred@k** = mse(L0 + Σ_{i≤k} d̂_i, Lk_obs) in Wan2.2 VAE latent space. Lk_obs comes from encoding the clip
[query canvas, observed frames at +4, +8, …], built with the same canvas code and VAE call as training.

- **Primary horizon k = 1** (ticks 4–16), inside the executed 16-action chunk; scored at s+16.
- **Second horizon k = 2** (ticks 20–32); scored at s+32. These ticks already run the next chunk's actions,
  which the policy chose after seeing the first half.

Consistency checks:
- Re-encoding the query canvas reproduces L0 exactly (`l0_check` = 0 on every row).
- Latent 1 is identical whether a 5- or a 9-frame clip is encoded (causal VAE).
- Each region is also reported: overhead = top 12 latent rows, wrist = bottom 12.

## Harness (scripts/wam_gap_eval.py --model internw0)

Same env, episodes and seeds as T004/T005: Bin single-object rows 0–7 of `tasks/T004_bin1.jsonl`, bench val
layouts, seed 1984, 30 s. The client is the bench client with action horizon 16, history 1, LeRobot frame and
no init frames.

The nominal replay (T004nom design) snapshots the scene at every 2nd query. At **s+32** (the end of the next
chunk) it replays the 32 issued commands (after the clamp, before the gap) on the fault-free PD arm, scores
the replay clip (kind=nominal), restores the scene and re-renders it 3×. The replay only posts to
`/iw_score`, which never touches the policy session, so no replay frame enters the model's memory.

**Harness check: `none` × 8 → 8/8 success** (no replay; the same run is the oracle baseline). LaWAM got 8/8
here. The bench's own 20/39 is on a harder multi-task val set and is not comparable per episode, but the
harness clearly works for this model, so the sweep ran.

## Results

Means are query-weighted. CIs are 95% bootstrap over episodes. Δ vs none is episode-paired (same layouts and
seeds), and d_fault − d_nominal is paired per query.

### 1. Nominal replay (T004nom design), e_pred@1 (VAE latent MSE, ticks 4–16)

| cond | ep | success | e_track ° | e_pred (all queries) | Δ vs none | d_fault | d_nominal | **d_fault − d_nominal** | Δ of that vs none | d(fault obs, nominal obs) |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 8 | 8/8 | 0.53 | 0.1144 | — | 0.1109 | 0.1127 | −0.0018 [−0.0063, 0.0016] (floor) | — | 0.013 [0.007, 0.021] |
| gain | 8 | 8/8 | 0.98 | 0.1459 | +0.031 [0.021, 0.040] | 0.1512 | 0.1341 | **+0.0171 [0.0087, 0.0233]** | +0.019 [0.014, 0.024] | 0.119 [0.108, 0.131] |
| gain2 | 8 | 5/8 | 1.58 | 0.1895 | +0.075 [0.066, 0.084] | 0.1906 | 0.1341 | **+0.0565 [0.0486, 0.0634]** | +0.058 [0.048, 0.068] | 0.209 [0.190, 0.226] |
| latency6 | 8 | 7/8 | 2.05 | 0.2102 | +0.096 [0.080, 0.111] | 0.2128 | 0.1313 | **+0.0815 [0.0701, 0.0945]** | +0.083 [0.072, 0.095] | 0.245 [0.225, 0.266] |

Paired queries: 100 / 108 / 161 / 147. e_static@1 (no-motion baseline) is 0.36–0.37 in every condition, so
the prediction explains ~70% of the observed latent change in `none`.

- **Yes, and more clearly than LaWAM.** In `none` the paired difference is at the floor (−0.002, CI ∋ 0). For
  every fault it is positive and clears zero, with a dose-response: gain ×0.5 +0.017 → gain2 ×0.25 +0.057 →
  latency6 +0.082. Δ vs none clears zero for all three.
- **Share of the fault's own effect the prediction does not follow:** (d_fault − d_nominal) /
  d(fault obs, nominal obs) = **14% [7, 20] (gain), 27% [24, 31] (gain2), 33% [30, 37] (latency6)**; LaWAM
  6% / 10% / 17% (point estimates; LaWAM's per-episode data were not re-bootstrapped). The % CIs come from a
  bootstrap with the ratio computed inside it (numerator and denominator from the same episode resample).
- **The wrist half carries most of it.** diff@1 is +0.031 / +0.100 / +0.136 on the wrist vs +0.004 / +0.013 /
  +0.027 overhead; all CIs exclude 0. The wrist camera sees the arm's own lag directly (FLUX's wrist half was
  the sensitive one too).
- **Second horizon (ticks 20–32).** The signal is weaker: diff@2 +0.005 [−0.005, 0.013] (gain), +0.022
  [0.013, 0.030] (gain2), +0.037 [0.026, 0.044] (latency6); share of the fault's effect 4% / 9% / 11%. By then
  the next chunk, issued after the policy saw the faulty first half, is running in both branches, and the
  prediction made at s cannot know those commands.
- **Raw e_pred rises much more than LaWAM's.** Δ vs none is +27% [19, 35] / +66% [56, 75] / +84% [68, 99] of
  the none level (LaWAM +7% / +12% / +29%), and every CI clears zero, including the mild `gain`.

### 2. Oracle stiffness correction (T005 design, gain2, no replay), e_pred@1

The oracle sits between the client clamp and the gap: sent = q + (1 + α(1/s − 1))·(cmd − q), s = 0.25
(`oracle.py`, unchanged from T005).

| setting | success | e_track ° | e_pred@1 | Δ vs none [95% CI] | e_pred@2 Δ vs none | queries |
|---|---|---|---|---|---|---|
| none | 8/8 | 0.50 | 0.1072 | — | — | 203 |
| gain2 α0 | **0/8** | 1.26 | 0.1938 | +0.087 [0.076, 0.098] | +0.108 [0.095, 0.123] | 424 |
| gain2 α1 | 8/8 | 0.47 | 0.1062 | **−0.001 [−0.008, 0.005]** | +0.002 [−0.011, 0.015] | 215 |
| gain2 α1.5 | 8/8 | 0.54 | 0.1345 | +0.027 [0.017, 0.039] | +0.015 [0.005, 0.029] | 198 |

**AUROC that raw e_pred ranks α0 (uncorrected) above α1 (corrected).** 0.5 = no signal. CIs come from an
episode-paired bootstrap.

| e_pred | query | window4 | episode |
|---|---|---|---|
| @1 all | **0.89 [0.83, 0.96]** | 0.93 [0.86, 0.99] | 1.00 [1.00, 1.00] |
| @1 wrist | 0.89 [0.81, 0.96] | 0.94 [0.86, 0.99] | 1.00 |
| @1 overhead | 0.71 [0.66, 0.76] | 0.73 [0.66, 0.80] | 0.88 [0.75, 1.00] |
| @2 all | 0.87 [0.81, 0.93] | 0.90 [0.84, 0.98] | 1.00 |
| LaWAM (T005, raw) | 0.69 [0.55, 0.78] | 0.71 [0.58, 0.80] | 0.77 [0.54, 0.95] |

α1 vs none: AUROC 0.49 (query), 0.47 (window4), 0.47 (episode). The corrected arm cannot be told apart from
the fault-free one.

- **Raw e_pred guides the stiffness correction.** It is U-shaped in α with its minimum at α=1, exactly on the
  none level (−0.001, CI ∋ 0). Overcorrecting (α=1.5) is worse again and clears zero (+0.027) even though all
  8 episodes succeed, so that penalty is not a failure-state artifact. e_track follows the same shape (1.26 →
  0.47 → 0.54; none 0.50).
- **The reward is much cleaner than LaWAM's.** Per query 0.89 vs 0.69. A 4-query window gives 0.93; episode
  means separate perfectly.
- **Caveat on α0.** It failed 8/8 here but 5/8 in the nominal-replay gain2 run; physics nondeterminism and the
  replay's snapshot/restore change the closed loop, as seen for LaWAM. Part of the α0 gap is failure states (no
  successful α0 episode exists to split on). The α1 vs α1.5 contrast (both 8/8) is the clean evidence.
- **Noise floor.** The two `none` runs (with / without replay) give e_pred@1 0.114 vs 0.107: a run-to-run
  spread of ~0.007, smaller than every fault effect above except α1 vs none.

### Review corrections (2026-11-22, no new runs; numbers recomputed from the logs)

1. **Server keys and parallel clients.** Sessions, predictions and observed latents used to be keyed by
   (run dir, episode) only. Two clients on the same run dir could therefore have reset each other's memory or
   overwritten each other's predictions. Checked in the logs:
   - The two `oracle/none` logs (`none.w1.log`, `none.w2.log`) come from a deliberate manual split by
     `--episode_offset`: episodes 0–3 and 4–7, attempts 81eadd5649a6 and 772c050f1538. The episode sets are
     disjoint, so their keys could not collide.
   - Across all 4636 score rows, 0 keys carry more than one attempt, and no (run dir, episode) was scored by two
     attempts. **No contamination; no number changes.**
   - Fixed for future runs: the client sends `trace.dir = "<run dir>#<attempt>"` and the server keys everything
     by (run dir, attempt, episode[, frame]).
2. **Queries whose horizon ends on the terminating tick.** These are still excluded, on purpose. On that tick
   `env.step` has already auto-reset, so the returned frame is the reset scene, not the outcome (T004 review #5).
   Including it would score against a wrong frame. Counts, latent 1 and latent 2 each:
   - gain 1 (ep7);
   - gain2-a1 2 (ep1, ep7);
   - gain2-a1.5 2 (ep1, ep2);
   - 0 in every other setting.

   All are successful episodes, out of 198–424 scored queries per setting. Queries whose horizon runs past the
   end are unscorable in any case.
3. **Paired bootstraps.** Δ vs none and the AUROC bootstrap now use only episodes that have samples of the
   metric in both settings. Every episode had samples of every metric, so the episode sets did not change. All
   point estimates and AUROCs are identical; bootstrap CIs moved by ≤ 0.003 (RNG stream order).
4. **Normalised CIs.** The figure is now in each model's raw units with raw-unit CIs, one row per model; the %
   values are point labels only. The % CIs quoted above are computed inside the bootstrap.

## Comparison with LaWAM

| | LaWAM (T004nom / T005) | InternW0-Δ (T006) |
|---|---|---|
| prediction | LAM DINOv3 tokens of the overhead frame at +35 | Wan2.2 VAE latents, ticks 4–16 (+20–32), overhead + wrist |
| none success (nominal) | 8/8 | 8/8 |
| d_fault − d_nominal, % of none d_fault: gain / gain2 / latency6 | +3% / +7% / +15% | **+15% / +51% / +74%** |
| share of the fault's effect not followed | 6% / 10% / 17% | **14% / 27% / 33%** |
| raw e_pred Δ vs none clears 0 | latency6 only | all three |
| gain2 oracle: α1 back to none | yes (+0.0007) | yes (−0.001) |
| gain2 oracle: α1.5 worse than α1 | yes | yes |
| AUROC α0 vs α1 (query / window4 / episode) | 0.69 / 0.71 / 0.77 | **0.89 / 0.93 / 1.00** |

The two models use different metric spaces, so the figure gives each model its own row in raw units
(95% CIs). Percentages appear as point labels only.

## Read

InternW0-Δ reproduces both LaWAM findings, with a much stronger signal.

Its `future_delta` prediction of the next 4–16 ticks behaves as a model of a fault-free arm. Under reduced
stiffness or a 200 ms delay, the prediction stays closer to what the nominal PD arm would have done with the
same commands than to what the faulty arm did. That holds for 14–33% of the fault's own latent effect, roughly
double LaWAM's share. The effect sits mostly in the wrist view, which sees the arm's lag directly.

Raw e_pred, the only reward available on a real robot, rises clearly for every fault including the mild one.
In the oracle test it falls back exactly to the fault-free level when the stiffness is compensated, and it
rises again when overcompensated. As a reward it ranks uncorrected above corrected at AUROC 0.89 per query.

Three things probably explain the gain over LaWAM:
- a shorter, denser horizon inside the executed chunk (ticks 4–16, vs a single frame at +35);
- the wrist camera in the target;
- a prediction the actions are trained to depend on.

For P1, InternW0's e_pred@1 (or its wrist half) is the better reward candidate. Its 0.6 s query makes it the
slower model (LaWAM 0.15 s), and the latency6 oracle limitation from T005 (a command-only adapter cannot undo a
delay) still applies.

Limits:
- 8 episodes per setting, and a single run of each.
- Success under gain2 α0 varies between runs (0/8 vs 5/8).
- latency6 and offset6 oracles and gain2 α0.5 were not run for InternW0.
- The k = 2 horizon mixes in the policy's reaction to the fault.

## Compute and disk

| use | window (2026-11-22) | GPU-h |
|---|---|---|
| server, GPU 0 (load attempts, smoke, harness check, sweep) | 05:35–07:20 | 1.75 |
| Isaac clients, GPU 1: smoke (2 short eps) | 05:39–05:43 | 0.07 |
| harness check `none` × 8 (2 clients) | 05:44–05:54 | 0.17 |
| sweep: nominal (none, gain2, latency6, gain; 2 clients) + oracle (gain2 α0/1/1.5; 1 client) | 05:54–07:20 | 1.43 |
| **T006 total** | | **≈ 3.4** (cap 6) |

The sweep ended at 07:20. The session then stopped on an expired login; the analysis, figure and doc were done
offline afterwards, with no further GPU use. The server process was left on GPU 0 for the lead to stop. Disk:
28 GB weights + 6.8 GB venv under `envs/internw0/` (in the `wam-internw0` worktree), 11 MB of outputs.

## Reproduce

```bash
envs/internw0/setup_internw0.sh                       # venv + ~28 GB weights
GPU=0 scripts/run_internw0_server.sh 18950 /outputs/so101-playground/runs/T006/server &
# harness check / oracle baseline (no replay)
GPU=1 scripts/run_t006_sweep.sh 18950 /outputs/so101-playground/runs/T006/oracle 1 8 none
# nominal replay (T004nom design)
GPU=1 EXTRA="--replay_episodes 8 --replay_every 2 --replay_mode nominal" \
    scripts/run_t006_sweep.sh 18950 /outputs/so101-playground/runs/T006/nom 2 8 none gain2 latency6 gain
# oracle (T005 design)
GPU=1 scripts/run_t006_sweep.sh 18950 /outputs/so101-playground/runs/T006/oracle 1 8 gain2:0 gain2:1 gain2:1.5
# analysis, results.tsv rows, figure (run from src/)
envs/internw0/InternW0-Delta/.venv/bin/python -m so101_playground.t006_analysis --root /outputs/so101-playground/runs/T006/nom --design nom --tsv /outputs/so101-playground/results.tsv
envs/internw0/InternW0-Delta/.venv/bin/python -m so101_playground.t006_analysis --root /outputs/so101-playground/runs/T006/oracle --design oracle --tsv /outputs/so101-playground/results.tsv
<python with matplotlib> -m so101_playground.t006_plot --out docs/T006_internw0_vs_lawam.png
```
