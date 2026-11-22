# T007: home-object task set (sim2real pairs) and an H2 re-check beyond seen-object Bin

Branch `exp/home-objects` (from `wam-control-gap` f4acb2d). Outputs `/outputs/so101-playground/runs/T007/`
(`A/<model>/none`, `B/internw0/<tag>`, `server/`, `frames/`, `summary_{A,B}.json`, `gpu_time.log`). Rows in
`/outputs/so101-playground/results.tsv`: `T007A-<model>-none` and `T007B-internw0-<cond>` /
`T007B-internw0-gain2-a<α>`. There is one row per family plus pooled, sim-seen and real-only; the subset is
named in `notes`. Figure: `docs/T007_home_objects.png`.

## 1. Task set

`tasks/home_objects.jsonl` + `tasks/layouts/home_objects_layouts.jsonl` (30 episodes) come from
`scripts/t007_make_home_objects.py`. That script does the generation and runs `--check`.

| home object | bench asset | group | asset size L × W × H (cm) |
|---|---|---|---|
| black pen | `black pen` | sim-seen | 14.2 × 1.9 × 1.2 |
| red pen | `red pen` | real-only | 14.6 × 2.1 × 1.3 |
| toothbrush | `purple toothbrush` | real-only | 16.5 × 1.4 × 1.8 |
| razor | `razor` | real-only | 14.5 × 5.6 × 2.7 |
| remote | `black remote` | sim-seen | 4.4 × 19.9 × 2.2 |
| paintbrush | `blue paintbrush` | sim-seen | 14.4 × 1.6 × 1.1 |
| plastic bin | `plastic_bin` | — | 32.0 × 19.7 × 10.1 |

Sizes are boxes of the collision-mesh points in each asset's frame (`scripts/t007_asset_dims.py` →
`docs/home_objects/asset_dims.json`). L is along the asset's +x, the arrow drawn on the sheets. The remote's
long side is its y axis. Its authored USD extent is wrong (117 cm), so the script reads the points.

**Families.**
- bin: one 1-object layout per object (6).
- move: 8 layouts, directions balanced 2 each.
- next_to: 8 ordered pairs.
- between: 8 (target, two references).

**Balance.**
- Each object is the target of 4 spatial episodes, and each family has 4 sim-seen and 4 real-only targets.
- No ordered pair or triple repeats.
- Every object appears in 17 of the 30 scenes.

**Scenes and instructions.**
- Spatial scenes hold 4 objects, the bench val-set convention. The extra objects are also drawn from the 6
  home objects, so the human needs nothing else.
- Instructions use the bench template (`task_generation.instruction_for_row`). A pen keeps its colour only
  when both pens are in the scene, e.g. "Move the pen left" vs "Move the red pen right".
- Appearance is canonical, with no domain randomization.

**Layouts.**
- They come from the bench sampler `layouts.generate_episode_layout`, with robot keep-out, footprint
  clearances and spawn regions.
- The bin stays at canonical pose 0 in all 30 layouts. The bench always uses pose 0 for spatial families;
  fixing it for bin too means the human never moves the bin.
- Move layouts must also leave a 7 cm corridor in the move direction, inside TABLE_BOUNDS and clear of other
  objects and the bin. The bench needs 5.1 cm of progress to score a move. Layouts that failed this were
  resampled (2–75 tries for 7 of the 8 move layouts).
- The bench validators pass on every row: `episode_spec_from_json`, `row_with_canonical_instruction_metadata`
  and `validate_move_episode_footprints`.

**Robot-base coordinates.** Each layout row has a `robot_base_frame` block. It gives the bench robot root
(world (0.0521, 0.1806), yaw 6.195°) and, for the bin and every object:
- the pose in world and root frames;
- `pan_fwd_cm` / `pan_side_cm` / `heading_from_fwd_deg`, measured from the **shoulder-pan axis**: root
  (0.0208, −0.0231) m (USD joint `Rotation`).

Axis conventions:
- fwd points away from the robot (root −y).
- side points to the robot's left (root +x), which is your right when you face the robot.
- The bench move directions use the same frame: left = world +x, forward = world −y.

## 2. Real-world layout sheets

`docs/home_objects/home_objects_layouts.pdf` (31 pages) and `docs/home_objects/index.html`. Page 1 lists the
sizes above and defines the reference frame. Each of the following 30 pages (also
`docs/home_objects/sheets/layout_NN.png`) shows:
- the top view with the robot at the top;
- a 5 cm grid in cm from the pan axis;
- the object area (TABLE_BOUNDS, 39 × 25.5 cm), the robot keep-out polygon, the robot root and the pan axis;
- the bin;
- each object's true footprint outline with its name, role (target/ref) and an arrow along its +x;
- for move layouts, the move direction;
- a table of fwd / side / heading per item;
- the sim overhead camera frame at the robot start (`docs/home_objects/sim_frames/`), which shows the human
  which end of each object is which and what their overhead camera should see.

All poses are also in `docs/home_objects/home_objects_poses.csv`.

## 3. Phase A: baseline success (`none`, 30 layouts × 1 run)

| family | InternW0-Δ | LaWAM |
|---|---|---|
| bin (1 object) | **5/6** | **6/6** |
| move | 1/8 | 5/8 |
| next_to | 2/8 | 1/8 |
| between | 3/8 | 1/8 |
| pooled | 11/30 (37%) | 13/30 (43%) |
| sim-seen targets | 6/15 | 9/15 |
| real-only targets | 5/15 | 4/15 |
| *T004–T006 seen-object Bin (8 val_set rows)* | *8/8* | *8/8* |

By target object, all families, out of 5 each:

| object | InternW0-Δ | LaWAM |
|---|---|---|
| black pen | 1 | 3 |
| black remote | 3 | 4 |
| blue paintbrush | 2 | 2 |
| red pen | 2 | 2 |
| toothbrush | 2 | 1 |
| razor | 1 | 1 |

- **Bin transfers to the home objects for both models, including the 3 real-only ones.**
  - InternW0's single bin failure was the black pen (time-out).
  - The black pen succeeded in the Phase B `none` replay run, where the red pen timed out instead. So one bin
    failure in 6 is within run-to-run spread.
- **Spatial families are mostly failures.** In 4-object scenes the two models score 1–3/8 per family, except
  LaWAM move at 5/8. Most failures are `failure` terminations (wrong object moved, non-target displaced or
  boundary rules), not time-outs.
- LaWAM is better on sim-seen targets (9/15 vs 4/15 real-only). InternW0 shows no object-group gap (6 vs 5).
  With 15 episodes per cell, neither gap is reliable.

## 4. Phase B: H2 on this set (InternW0-Δ only)

**Families.** Only bin reached ≥ 50% for InternW0 in Phase A (bin 5/6; between 3/8, next_to 2/8, move 1/8),
so Phase B ran on the 6 bin layouts (3 sim-seen, 3 real-only), as the rule says.

**Settings.**
- Nominal replay (T004nom design, replay every 2nd query): `none`, `gain2`, `latency6`.
- Oracle `gain2` α ∈ {0, 1}, no replay. The oracle's `none` is the Phase A InternW0 run: the same setting T006
  used for its oracle baseline.

**Metrics.** As in T006, all computed by the T006 functions:
- e_pred@1: Wan2.2 VAE latent MSE, ticks 4–16.
- Query-weighted means; 95% bootstrap CIs over episodes.
- Δ vs none and AUROC are episode-paired.

### 4.1 Nominal replay

| cond | subset | success | e_pred@1 | Δ vs none [CI] | d_fault | d_nominal | **d_fault − d_nominal** [CI] | d(fault obs, nominal obs) | share of fault effect [CI] |
|---|---|---|---|---|---|---|---|---|---|
| none | bin (6) | 5/6 | 0.1359 | — | 0.1345 | 0.1337 | +0.0008 [−0.0006, 0.0017] (floor) | 0.006 | — |
| gain2 | bin (6) | 6/6 | 0.1978 | +0.062 [0.037, 0.084] | 0.1947 | 0.1387 | **+0.0560 [0.0405, 0.0751]** | 0.220 | **25% [19, 32]** |
| | sim-seen (3) | 3/3 | 0.1939 | +0.047 [0.004, 0.091] | 0.1905 | 0.1393 | +0.0513 [0.0300, 0.0696] | 0.210 | 24% [16, 31] |
| | real-only (3) | 3/3 | 0.2013 | +0.077 [0.076, 0.089] | 0.1983 | 0.1383 | +0.0600 [0.0348, 0.0982] | 0.229 | 26% [16, 40] |
| latency6 | bin (6) | 4/6 | 0.2151 | +0.079 [0.052, 0.108] | 0.2126 | 0.1419 | **+0.0707 [0.0449, 0.0880]** | 0.254 | **28% [20, 32]** |
| | sim-seen (3) | 1/3 | 0.2104 | +0.063 [0.024, 0.105] | 0.2085 | 0.1509 | +0.0575 [0.0149, 0.0856] | 0.235 | 25% [8, 33] |
| | real-only (3) | 3/3 | 0.2205 | +0.097 [0.077, 0.117] | 0.2173 | 0.1315 | +0.0858 [0.0793, 0.0940] | 0.276 | 31% [30, 33] |
| *T006 seen Bin (8)* | gain2 / latency6 | 5/8, 7/8 | | +0.075 / +0.096 | | | *+0.0565 / +0.0815* | 0.209 / 0.245 | *27% / 33%* |

Paired queries: 106 / 112 / 129. The `none` floor for the sim-seen and real-only splits is +0.0007 and
+0.0009.

- **The H2 signature reproduces on the home bin set at the same size as T006.**
  - In `none` the paired gap sits at the floor.
  - Under gain2 it is +0.056, against T006's +0.0565. Under latency6 it is +0.071, against +0.082.
  - The prediction misses 25% / 28% of the fault's own latent effect, against 27% / 33% in T006.
- **Real-only objects show it as strongly as sim-seen ones.** The paired gap under gain2 is +0.060 (real-only)
  vs +0.051 (sim-seen); under latency6 it is +0.086 vs +0.058. All CIs exclude 0, and the CIs overlap.
  - Part of the latency6 difference is outcome: the sim-seen latency6 episodes failed 2/3, the real-only ones
    succeeded 3/3.
- **The wrist half again carries most of it.** diff@1 is +0.097 wrist vs +0.015 overhead (gain2) and +0.116
  vs +0.025 (latency6).
- **The second horizon is weaker, as in T006.** diff@2 is +0.023 (gain2) and +0.028 (latency6).

### 4.2 Oracle gain2

| setting | subset | success | e_pred@1 | Δ vs none (Phase A) [CI] | queries |
|---|---|---|---|---|---|
| none (Phase A) | bin (6) | 5/6 | 0.1338 | — | 211 |
| gain2 α0 | bin (6) | 2/6 | 0.1869 | **+0.053 [0.014, 0.094]** | 289 |
| | sim-seen / real-only | 2/3 / 0/3 | 0.1969 / 0.1787 | +0.045 [−0.028, 0.094] / +0.063 [0.014, 0.095] | |
| gain2 α1 | bin (6) | 5/6 | 0.1350 | **+0.001 [−0.003, 0.004]** | 196 |
| | sim-seen / real-only | 2/3 / 3/3 | 0.1534 / 0.1141 | +0.002 [−0.001, 0.004] / −0.002 [−0.007, 0.007] | |

AUROC that raw e_pred@1 ranks α0 above α1, with episode-paired bootstrap CIs:

| subset | query | window4 | episode |
|---|---|---|---|
| bin (6) | **0.72 [0.52, 0.93]** | 0.75 [0.52, 0.98] | **0.94 [0.67, 1.00]** |
| sim-seen (3) | 0.67 [0.31, 0.92] | 0.73 [0.31, 1.00] | 0.89 [0.00, 1.00] |
| real-only (3) | 0.79 [0.51, 0.97] | 0.80 [0.51, 1.00] | 1.00 [1.00, 1.00] |
| *T006 seen Bin (8)* | *0.89 [0.83, 0.96]* | *0.93 [0.86, 0.99]* | *1.00* |

The wrist half gives the same values (query 0.71, window4 0.76, episode 0.94).

- **The correction result holds.** Compensating the stiffness (α1) brings e_pred back to the fault-free level
  (+0.001, CI ∋ 0), and α0 sits clearly above it. Both the real-only and sim-seen splits show this.
- **Per query, the reward is weaker than on T006's Bin set:** 0.72 vs 0.89. The per-episode ranking stays
  near perfect at 0.94. The CIs are wide because there are only 6 episodes.
- **The likely cause is spread in baseline level between objects.** In `none`, e_pred@1 is 0.152 for
  sim-seen and 0.116 for real-only. The pooled per-query AUROC mixes episodes with different baselines.
  Within each group it is 0.67 (sim-seen) and 0.79 (real-only). A per-episode or per-object baseline (already
  a P1 requirement from T005) addresses exactly this.
- **Caveat, as in T006:** success under α0 varies by run (2/6 here), so part of the α0 gap is failure states.
  α1 vs none, at the same success rate, is the clean comparison.

## 5. Read: do the T004–T006 conclusions hold beyond seen-object Bin?

**For the objects, yes; for the task families, the question could not be tested.**

On the home set's bin layouts, half of them objects the sim training never saw, InternW0-Δ reproduces the H2
signature at the T006 magnitude:
- the paired prediction gap sits at the floor without a fault;
- it is +0.056 under gain2 and +0.071 under latency6, a 25–28% share of the fault's own latent effect;
- the gap is at least as large for real-only objects as for sim-seen ones;
- raw e_pred falls back exactly to the fault-free level under the oracle correction.

What did not carry over cleanly is the reward's per-query separability. It drops from 0.89 to 0.72, and the
most likely reason is between-object baseline spread, so P1 needs per-episode (or per-object) baselines.

Beyond Bin, neither model is competent enough in sim on 4-object spatial tasks (InternW0 1–3/8 per family,
LaWAM 1–5/8) to run Phase B there under the ≥ 50% rule. Home-set H2 evidence therefore covers only
single-object bin placement. For the sim2real pairs, the 6 bin layouts are the ones that work in sim for both
models; the spatial layouts mainly measure policy failure.

Limits:
- 6 episodes per Phase B setting, 3 per object group;
- one run each;
- α0 success varies by run;
- `none` with replay vs `none` without (Phase A) differ in e_pred@1 by 0.002 (0.1359 vs 0.1338).

## 6. Code and changes

- `scripts/t007_make_home_objects.py`: task + layout generator and validator (`--check`).
- `scripts/t007_asset_dims.py`: asset boxes and robot joint frames from the bench USDs (Isaac app, no
  physics).
- `scripts/t007_layout_sheets.py`: sheets, PDF, HTML and pose CSV.
- `scripts/run_t007_sweep.sh`: job queue over episode ranges.
- `scripts/run_lawam_server.sh`: LaWAM server launcher.
- `scripts/run_t004_eval.sh`: new `EPISODES` / `LAYOUTS` overrides.
- `scripts/run_internw0_server.sh`: `IW_ENV` override.
- `scripts/wam_gap_eval.py`: `--initial_frames_dir` saves the start frames; nothing else changed.
- `src/so101_playground/t007_analysis.py`: reuses the T006/T005 loaders and statistics.
- `src/so101_playground/t007_plot.py`: the figure.

The bench submodule is not edited.

## 7. Compute

| use | window (2026-11-22, KST) | GPU-h |
|---|---|---|
| GPU 0: InternW0-Δ server (smoke, Phase A, Phase B) | 20:52–22:31 | 1.64 |
| GPU 0: LaWAM server (Phase A) | 22:31–22:55 | 0.41 |
| GPU 1: asset probes (3 short Isaac runs) | ~20:48 | ~0.05 |
| GPU 1: smoke (3 episodes) | 20:55–21:02 | 0.13 |
| GPU 1: Phase A InternW0 (3 clients) | 21:02–21:36 | 0.55 |
| GPU 1: Phase B (2 nominal + 1 oracle client) | 21:36–22:31 | 0.92 |
| GPU 1: Phase A LaWAM (3 clients) | 22:32–22:55 | 0.39 |
| **total** | | **≈ 4.1** (cap 8) |

Both servers were shut down after their phases. The envs were reused from the `wam-internw0`,
`wam-control-gap` and `scaffold-env` worktrees, so no new disk was used for them. Repo additions: 15 MB under
`docs/home_objects/`.

## 8. Reproduce

```bash
source <scaffold-env>/envs/isaac/env.sh
python scripts/t007_make_home_objects.py --check
CUDA_VISIBLE_DEVICES=1 python scripts/t007_asset_dims.py --out docs/home_objects/asset_dims.json
R=/outputs/so101-playground/runs/T007
GPU=0 IW_ENV=<wam-internw0>/envs/internw0 scripts/run_internw0_server.sh 18970 $R/server/internw0 &
GPU=1 EXTRA="--initial_frames_dir $R/frames" scripts/run_t007_sweep.sh internw0 18970 $R/A 3 none@0+10 none@10+10 none@20+10
GPU=1 EXTRA="--replay_episodes 99 --replay_every 2 --replay_mode nominal" \
    scripts/run_t007_sweep.sh internw0 18970 $R/B 2 none@0+3 none@3+3 gain2@0+3 gain2@3+3 latency6@0+3 latency6@3+3
GPU=1 scripts/run_t007_sweep.sh internw0 18970 $R/B 1 gain2:0@0+6 gain2:1@0+6
GPU=0 LAWAM_ENV=<wam-control-gap>/envs/lawam scripts/run_lawam_server.sh 18971 $R/server/lawam &
GPU=1 scripts/run_t007_sweep.sh lawam 18971 $R/A 3 none@0+10 none@10+10 none@20+10
python scripts/t007_layout_sheets.py
cd src && python -m so101_playground.t007_analysis --phase A --tsv /outputs/so101-playground/results.tsv
python -m so101_playground.t007_analysis --phase B --tsv /outputs/so101-playground/results.tsv
python -m so101_playground.t007_plot --out ../docs/T007_home_objects.png
```
