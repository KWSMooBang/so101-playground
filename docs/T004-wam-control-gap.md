# T004: does a WAM's own future prediction expose an injected control gap? (H2-P0, sim only)

Branch `exp/wam-control-gap`. Outputs `/outputs/so101-playground/runs/T004/`. Rows in
`/outputs/so101-playground/results.tsv` (`T004-<model>-<condition>`).

## Models

Two WAMs, same conditions, episodes, seeds and layouts. Policy servers live in our code; the bench submodule is
not edited.

**FLUX 3 Action** (decision 2, 2026-11-21). The bench LoRA (`5hadytru/so101_bench_flux-3-action-so101` @ 07a7632,
raw update 3000, 5/39 on the bench val set) is merged into `black-forest-labs/flux-3-action-so101` @ c9e13b2
(`src/so101_playground/merge_flux3_lora.py`, W += α/r·BA; heads and so101_bench_sim normalization from the
adapter). Served by `src/so101_playground/flux3_probe_server.py`, which wraps the bench `flux3_server.py
--save_predictions`. The Qwen3-VL text encoder stays on the CPU and captions are encoded there, once per
instruction; with flux-action's own offload the encoder visits the GPU and OOMs next to the DiT. Server on
GPU 0: **1.25 s per query, 1.57 s including the predicted-video decode, peak 16.1 GiB allocated / 16.9 GB on the
card**. Isaac clients ran on GPU 1. The predicted video is the scene|wrist canvas (256×512), 41 frames; frame j
is aligned with the observation after s+j env steps (bench convention). A ±2-tick lag sweep was flat on the
scene half, and on the wrist half L1 kept falling up to +2, so the prediction may lead by a tick or two. The
bench alignment is kept as is.

**LaWAM.** Server on GPU 1: 0.14–0.15 s per query, peak 6.8 GiB allocated, plus 44 ms per `/lam_score`.

LaWAM details: `5hadytru/so101_bench_LaWAM_full_fine_tune` @ 8df458f (update 45000, 21/39 on the bench val set),
served by `src/so101_playground/lawam_probe_server.py`. That is a copy of the bench `lawam_server.py` (the
submodule is not edited) that also exposes the latent future prediction:

- LAM vision encoder (DINOv3, layer -2) on the current overhead frame → `h_t` (256 patch tokens × 768).
- The VLM predicts a latent action `pred_action_emb`; the LAM decoder gives `h_t1_pred = dec(h_t, pred_action_emb)`.
- Training loss `mse(h_t1_pred, h_t1_gt)`, with `h_t1_gt` = the same encoder on the overhead frame
  `action_horizon - 1 = 35` ticks later (`num_video_frames=2` → `image_observation_delta_indices = [0, 35]`).
- The flow head is conditioned on `h_t1_pred`. LaWAM executes all 36 actions of a chunk, so frame s+35 always
  falls inside the chunk issued at s.

`/lam_score` re-encodes a later frame with the same preprocessor and encoder and logs the metrics below. The
flow noise is seeded per (episode, query tick), so every condition sees the same sampling noise for the same
inputs.

## Control-gap conditions (`src/so101_playground/control_gap.py`)

`cmd` is the policy's joint target in sim radians, after the bench client's 15°/tick clamp. The gap sits
between `cmd` and `env.step`; the bench env is otherwise unchanged.

| condition | what it does |
|---|---|
| none | nominal bench actuators, applied = cmd |
| gain | PD stiffness ×0.5 on the five arm joints (Jaw kept, so grip force is unchanged; damping unchanged), set in the env cfg before build |
| latency | applied(t) = cmd(t−3): 3 control ticks = 100 ms delay; the first 3 ticks repeat the hold pose |
| offset | applied = cmd + 3° on Pitch (shoulder_lift) and Elbow (elbow_flex); encoders report the true angle |
| backlash | ±1.5° dead band per arm joint: the applied target moves only once cmd leaves the band, then trails it by 1.5° |
| gain2 / latency6 / offset6 | stronger doses: stiffness ×0.25 / 6 ticks (200 ms) / +6° (dose-response check; FLUX ran gain2 and latency6 only, budget) |

## Episodes

`tasks/T004_bin1.jsonl`: the bench `val_set.jsonl` single-object Bin rows with rigid single-body objects
(28 rows; 2 shoe rows dropped as multi-body). Fixed bench layouts (`tasks/layouts/val_set_layouts.jsonl`), env
seed 1984, 30 s limit. The first 8 rows were run under every condition (identical layouts and seeds). Task
`So101Bench-Bin-v0`, overhead + wrist 640×480, LeRobot joint frame, no init frames, bench leaderboard settings:
LaWAM action horizon 36, history 1; FLUX action horizon 32 (of 42), history 8.

## Metrics (per query s; `src/so101_playground/t004_analysis.py`)

LaWAM (latent, its own training loss):

- **e_pred** = `mse(h_t1_pred, LAM(observed overhead frame after s+35 env steps))`, the training loss in
  rollout. Also logged: 1 − token cosine.
- **e_ideal / d_nominal** = the same distance against a **replay** of the chunk from a scene snapshot taken at
  query s (every 2nd query): `--replay_mode teleport` (perfect tracking) or `nominal` (fault-free PD arm). See
  "Review corrections" in Results for the exact procedure; the first sweep's replay is invalid.
  `e_pred@35_paired` (= d_fault) and the replay distance use exactly the same queries.
- **e_static** = `mse(h_t, LAM(observed s+35))`: how much the scene changed. This is the "predict no motion"
  baseline, for scale.
- **e_track** = mean |cmd(t) − q(t+1)| over the five arm joints and all policy ticks (deg), with cmd before the
  gap. `e_track@35` = the same at the predicted frame (cmd issued at s+34 vs q after s+35).
- **e_act_lat** = mse(pred_action_emb, LAM teacher latent action inferred from (o_s, observed o_{s+35})), the
  model's distill-loss target.
- rho = Spearman between per-query e_pred and the per-query chunk-mean e_track.

FLUX (pixels; `src/so101_playground/flux_pixel_metrics.py`): FLUX's decoded prediction (scene|wrist, 256×512) at
horizon j ∈ {8, 16, 31} vs the observed canvas after s+j steps (built with FLUX's own resize: each camera
bilinear+antialias to 256×256). Distances are computed on the scene and wrist halves separately (no other mask):
L1 on [0,1], LPIPS (AlexNet), and DINOv2-small patch-token cosine distance (224², averaged over patches).
e_ideal uses the replay canvas after j ticks. **e_exec** = observed vs ideal canvas, i.e. the control gap itself
in image space, with no model involved. Stored frames are JPEG q95 and predictions H.264 CRF 23; both set the
floors, not the comparisons. The primary metric for the TSV, DINO scene@16, was fixed before the sweep. The
wrist half turned out to be the sensitive one.

Both models: CIs are 95% bootstrap over episodes. `Δ vs none` is episode-paired (same layouts) and
`obs−ideal` is query-paired.

Checks: re-encoding the query frame reproduces `h_t` (mse ~2e-4 vs e_pred ~0.04–0.3). Replaying from a
restored snapshot reproduces e_ideal to 0.3% across runs, and the observed e_pred varies by ~1% across runs.
Skipping renders inside the replay changed the RTX image enough to move e_ideal by 3–10%, so the replay renders
every tick.


## Results

Figure: `/outputs/so101-playground/runs/T004/T004_pred_vs_track.png`. Left: LaWAM, nominal replay. Right: FLUX
e_pred only.

Videos (made with the pre-review replay; their "ideal" rows show the old teleport replay):
- `runs/T004/lawam_videos/T004-<cond>.mp4`
- `runs/T004/flux3_videos/T004-flux3-<cond>.mp4`

### Review corrections (2026-11-21)

Every mean below is query-weighted, every Δ vs none is the difference of query-weighted means, and every CI is a
95% bootstrap over episodes, episode-paired for Δ (review #7).

A code review found four problems in the first sweep's ideal replay:
- **#2:** the replay bypassed the client's 15°/tick clamp. The clamp is active on 8–62% of ticks, mostly
  triggered by a blocked gripper, which scales the whole arm command back.
- **#3:** in the gain conditions the replay kept the reduced stiffness.
- **#4:** the replay's frames stayed in the RTX temporal history. FLUX's first observed frames after a replay
  were worse (wrist LPIPS +0.03–0.06 at j=1, mostly gone by j=8). LaWAM is unaffected (it scores 35 renders later).
- **#5:** one post-termination frame per condition at most entered the FLUX scores.

So **e_ideal, observed−ideal and FLUX e_exec of the first sweep (`runs/T004`) are invalid** and are marked
`invalid` in results.tsv. e_pred, e_track, success and Δ vs none do not depend on the replay and stand. #5
(terminal frame) and @40 were also dropped from the FLUX metrics, which were then recomputed.

Two corrected baselines were then run for LaWAM. Both replay the commands the policy actually issued (after the
clamp, before the gap) from a scene snapshot taken at the query tick. The replay runs at the end of the chunk,
gains are restored afterwards, and the restored scene is re-rendered 3× before the rollout continues.
- **v2, teleport** (`runs/T004_v2`, none/gain/gain2/latency6 × 8 ep): the arm is placed on the command at every
  physics substep, i.e. perfect tracking. The jaw tracks through nominal PD.
- **nominal** (`runs/T004_nominal`, decision 2026-11-21): the same commands on a fault-free arm. All joints run
  through the nominal PD drives, as the training data was recorded, with no teleport, delay, offset or backlash.
  The run covers none and latency6 × 8 ep and gain2 × 4 ep. The 10.8 GPU-h cap was reached when GPU 1 slowed down
  next to an external process, so gain was not run.

The nominal comparison answers the H2 question directly. d_fault = d(prediction, the gap arm's actual
future), d_nominal = d(prediction, a fault-free arm given the same commands). A positive d_fault − d_nominal
means the prediction is closer to what a fault-free arm would have done.

### LaWAM, nominal replay (LAM token MSE at +35 ticks; paired queries = every 2nd query)

| cond | ep | success | e_track ° | e_pred (all queries) | Δ vs none | d_fault | d_nominal | **d_fault − d_nominal** | Δ of that vs none | d(fault obs, nominal obs) |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 8 | 8/8 | 0.66 | 0.0753 | — | 0.0754 | 0.0769 | −0.0015 [−0.0038, −0.0001] (noise floor) | — | 0.019 [0.012, 0.031] |
| gain2 | 4 | 1/4 | 2.20 | 0.0885 | +0.016 [−0.004, 0.030] | 0.0930 | 0.0853 | **+0.0077 [0.0022, 0.0169]** | +0.0094 [0.0035, 0.0170] | 0.054 [0.043, 0.061] |
| latency6 | 8 | 7/8 | 2.46 | 0.0969 | +0.022 [0.010, 0.034] | 0.0946 | 0.0832 | **+0.0114 [0.0066, 0.0153]** | +0.0129 [0.0089, 0.0164] | 0.067 [0.058, 0.076] |

- In `none` both observations should be the same, and they nearly are. d(fault obs, nominal obs) = 0.019 and
  d_fault − d_nominal = −0.0015 give the render/physics noise floor.
- Under the strong gaps the prediction is clearly closer to the fault-free arm's outcome: +0.008 (gain2) and
  +0.011 (latency6), 5–8× the noise floor.
- The gap itself moves the scene by 0.054–0.067 in LAM space, about 3× the floor.

### LaWAM, teleport replay v2 (same conditions, 8 ep each)

| cond | success | e_track ° | e_pred | Δ vs none | e_pred (paired) | e_ideal (teleport) | obs − ideal |
|---|---|---|---|---|---|---|---|
| none | 8/8 | 0.66 | 0.0720 | — | 0.0700 | 0.0788 | −0.0088 [−0.0117, −0.0059] |
| gain | 7/8 | 1.39 | 0.0828 | +0.011 [−0.002, 0.025] | 0.0790 | 0.0871 | −0.0080 [−0.0146, −0.0017] |
| gain2 | 4/8 | 2.23 | 0.0882 | +0.016 [0.000, 0.032] | 0.0896 | 0.0882 | +0.0013 [−0.0014, 0.0041] |
| latency6 | 6/8 | 2.25 | 0.0937 | +0.022 [0.017, 0.027] | 0.0854 | 0.0871 | −0.0017 [−0.0089, 0.0038] |

In `none` the prediction is closer to the PD-tracked arm than to a perfectly tracked one (−0.009): LaWAM has
learned the sim's tracking lag. So "perfect tracking" is the wrong reference. Against it, the gap conditions
only move obs − ideal from −0.009 up to ~0. The nominal replay removes that offset.

### LaWAM, first sweep (8 conditions × 8 ep; e_pred side only, valid)

| cond | success | e_track ° | e_pred | Δ vs none [95% CI] |
|---|---|---|---|---|
| none | 7/8 | 0.67 | 0.0739 | — |
| gain | 8/8 | 1.37 | 0.0782 | +0.0044 [−0.0011, 0.0091] |
| latency | 7/8 | 1.64 | 0.0833 | +0.0094 [−0.0005, 0.0204] |
| offset | 5/8 | 1.52 | 0.0889 | +0.0150 [−0.0054, 0.0363] |
| backlash | 7/8 | 1.24 | 0.0764 | +0.0025 [−0.0059, 0.0094] |
| gain2 | 7/8 | 2.31 | 0.0982 | +0.0243 [0.0042, 0.0460] |
| latency6 | 7/8 | 2.37 | 0.0984 | +0.0245 [0.0156, 0.0342] |
| offset6 | 1/8 | 2.06 | 0.0798 | +0.0059 [−0.0020, 0.0153] |

Across the 8 condition means, r(e_track, e_pred) = 0.83. The three runs differ in rollouts and success
(e.g. gain2 7/8, 4/8, 1/4) because the replay and physics nondeterminism change the closed loop; that is a
caveat on any single-condition number.

### FLUX 3 Action (first sweep, 8 ep; e_pred only, ideal/exec invalid, no re-run by decision)

| cond | success | e_track ° | DINO scene@16 e_pred | Δ vs none | LPIPS wrist@16 e_pred | Δ vs none |
|---|---|---|---|---|---|---|
| none | 2/8 | 1.68 | 0.138 | — | 0.189 | — |
| gain | 3/8 | 1.70 | 0.144 | +0.006 [−0.003, 0.013] | 0.231 | +0.043 [0.018, 0.064] |
| latency | 1/8 | 1.95 | 0.144 | +0.006 [−0.003, 0.013] | 0.236 | +0.047 [0.007, 0.086] |
| offset | 3/8 | 1.70 | 0.146 | +0.008 [−0.001, 0.019] | 0.244 | +0.055 [0.023, 0.091] |
| backlash | 1/8 | 2.58 | 0.143 | +0.005 [−0.001, 0.010] | 0.244 | +0.056 [0.029, 0.085] |
| gain2 | 1/8 | 2.61 | 0.146 | +0.007 [−0.000, 0.014] | 0.193 | +0.004 [−0.030, 0.041] |
| latency6 | 1/8 | 2.66 | 0.149 | +0.010 [0.000, 0.019] | 0.260 | +0.072 [0.026, 0.117] |

FLUX's own success in nominal sim (2/8) is low enough that the lead has put a check of the eval harness ahead of
any further FLUX runs. FLUX integrates its delta actions onto the last command, so its command stream leads
the arm (nominal e_track 1.68°) and e_track barely moves under gain/offset.

### Read

With the corrected baseline the H2-P0 signal is there for LaWAM. When the arm is faulty, LaWAM's prediction of
the future is closer to what a fault-free arm would have done with the same commands than to what the faulty arm
did: d_fault − d_nominal = +0.008 (gain2), +0.011 (latency6) vs a −0.0015 floor in `none`. The
gap's own visual effect (d(fault obs, nominal obs) 0.05–0.07) is clearly above the floor (0.019).

So "predicted future vs observed future" carries information about the control gap, and the right reward
for P1 is the paired difference against a nominal reference, not raw e_pred. Raw e_pred also rises (Δ +0.02 at
the strong doses), but it is confounded by the state distribution: the gap changes where the closed loop goes.

Limits:
- Only the strong doses were tested with the nominal baseline. The mild doses (3°, 100 ms, ×0.5) moved raw e_pred
  by amounts inside the CI with 8 episodes.
- gain2 has 4 episodes.
- The effect is ~10–15% of the prediction error itself, so a per-query reward will be noisy.
- FLUX is not re-evaluated.

## Compute

| use | window | GPU-h |
|---|---|---|
| first sweep (LaWAM GPU 1 16:43–20:50; FLUX GPU 0 19:33–22:28 + GPU 1 20:50–22:27) | | 8.65 |
| FLUX metric recompute | | ~0.05 |
| v2 teleport: smoke + 4 conditions | 22:40–23:30 | 0.83 |
| nominal: smoke + none, latency6, gain2 (4 ep) | 23:31–00:43 | 1.2 |
| **T004 total** | | **≈ 10.8** (cap raised to 10.8 by human decision) |

During the nominal run a 3.1 GB process outside T004 on GPU 1 caused one OOM at Isaac startup (gain; re-queued,
then cut at the cap) and slowed episodes about 3×.

## Reproduce

```bash
# LaWAM
LEROBOT_SRC=<checkout>/third_party/lerobot envs/lawam/setup_lawam.sh
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 envs/lawam/.venv/bin/python src/so101_playground/lawam_probe_server.py \
    --repo_id envs/lawam/weights/so101_bench_LaWAM --port 18800 --log_dir /outputs/so101-playground/runs/T004/lawam
GPU=1 scripts/run_t004_sweep.sh lawam 18800 /outputs/so101-playground/runs/T004 none,gain,latency,offset,backlash,gain2,latency6,offset6 8 8 --replay_every 2
GPU=1 scripts/run_t004_videos.sh 18800 /outputs/so101-playground/runs/T004 none,gain,latency,offset,latency6 360
# corrected baselines (server --log_dir runs/T004_v2/lawam or runs/T004_nominal/lawam)
GPU=1 scripts/run_t004_sweep.sh lawam 18800 /outputs/so101-playground/runs/T004_v2 none,gain,gain2,latency6 8 8 --replay_every 2 --replay_mode teleport
GPU=1 scripts/run_t004_sweep.sh lawam 18800 /outputs/so101-playground/runs/T004_nominal none,latency6,gain2,gain 8 8 --replay_every 2 --replay_mode nominal

# FLUX 3 Action (both black-forest-labs licenses must be accepted on the Hub)
envs/flux3/setup_flux3.sh
SO101_BENCH_ROOT=<bench> HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 envs/flux3/flux-action/.venv/bin/python \
    src/so101_playground/flux3_probe_server.py --package $PWD/envs/flux3/weights/so101_bench_raw3000 \
    --vae $PWD/envs/flux3/weights/base/video_vae.safetensors --text_encoder $PWD/envs/flux3/weights/base/text_encoder \
    --save_predictions --port 18900 --log_dir /outputs/so101-playground/runs/T004/flux3
GPU=1 scripts/run_t004_sweep.sh flux3 18900 /outputs/so101-playground/runs/T004 none,gain,latency,offset,backlash,latency6,gain2 8 8 --replay_every 2
CUDA_VISIBLE_DEVICES=0 envs/flux3/flux-action/.venv/bin/python src/so101_playground/flux_pixel_metrics.py /outputs/so101-playground/runs/T004/flux3/<cond>
envs/flux3/flux-action/.venv/bin/python src/so101_playground/t004_flux_videos.py <run dir> <out.mp4>

# analysis, results.tsv rows, figure
envs/flux3/flux-action/.venv/bin/python src/so101_playground/t004_analysis.py --root /outputs/so101-playground/runs/T004 \
    --model lawam --tsv /outputs/so101-playground/results.tsv      # and --model flux3
envs/flux3/flux-action/.venv/bin/python src/so101_playground/t004_plot.py /outputs/so101-playground/runs/T004 /outputs/so101-playground/runs/T004_nominal out.png
```
