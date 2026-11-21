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
- **e_ideal** = the same distance against a **kinematic replay**. At query s the whole scene is snapshotted
  (`scene.get_state`); the chunk is executed with perfect tracking (joint positions written each control tick,
  PD target = command, zero velocity, objects simulated, every tick rendered as `env.step` does); the frame after
  35 ticks is scored; the scene is restored. This runs on every 2nd query of every episode.
  `e_pred@35_paired` and `e_ideal@35_paired` use exactly the same queries.
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

Figure: `/outputs/so101-playground/runs/T004/T004_pred_vs_track.png`.

Videos:
- LaWAM: `runs/T004/lawam_videos/T004-<cond>.mp4` for none/gain/latency/offset/latency6. Observed | ideal
  canvases, plus PCA-RGB maps of h_t1_pred / LAM(observed) / LAM(ideal) with one PCA basis shared per episode.
  Episode 0, first 360 ticks, replay at every query.
- FLUX: `runs/T004/flux3_videos/T004-flux3-<cond>.mp4` for all 7 FLUX conditions. Rows are predicted / observed /
  ideal for j = 1..31. Episode 0, first 8 replayed queries.

### LaWAM (8 episodes per condition; distance = LAM token MSE at +35 ticks)

| cond | success | e_track ° | e_pred | Δ vs none [95% CI] | e_pred (paired) | e_ideal (paired) | obs−ideal [95% CI] | ρ(e_pred, e_track) |
|---|---|---|---|---|---|---|---|---|
| none | 7/8 | 0.67 | 0.0739 | — | 0.0658 | 0.0724 | −0.0066 [−0.0117, −0.0024] | 0.35 |
| gain | 8/8 | 1.37 | 0.0782 | +0.0034 [−0.0019, 0.0084] | 0.0692 | 0.0745 | −0.0053 [−0.0083, −0.0026] | 0.35 |
| latency | 7/8 | 1.64 | 0.0833 | +0.0069 [−0.0015, 0.0165] | 0.0794 | 0.0787 | +0.0006 [−0.0084, 0.0072] | 0.56 |
| offset | 5/8 | 1.52 | 0.0889 | +0.0147 [−0.0048, 0.0366] | 0.0904 | 0.0871 | +0.0033 [−0.0083, 0.0133] | 0.62 |
| backlash | 7/8 | 1.24 | 0.0764 | +0.0008 [−0.0065, 0.0076] | 0.0746 | 0.0781 | −0.0035 [−0.0069, −0.0002] | 0.39 |
| gain2 | 7/8 | 2.31 | 0.0982 | +0.0213 [0.0039, 0.0417] | 0.0949 | 0.0896 | +0.0053 [−0.0056, 0.0170] | 0.43 |
| latency6 | 7/8 | 2.37 | 0.0984 | +0.0246 [0.0171, 0.0327] | 0.0931 | 0.0836 | +0.0095 [0.0042, 0.0172] | 0.55 |
| offset6 | 1/8 | 2.06 | 0.0798 | +0.0061 [−0.0016, 0.0151] | 0.0790 | 0.0779 | +0.0011 [−0.0069, 0.0071] | 0.56 |

- e_static, the no-motion baseline, is 0.09–0.16 in every condition.
- Across the 8 condition means, r(e_track, e_pred) = 0.83 and r(e_track, obs−ideal) = 0.89. Without offset6 they
  are 0.95 and 0.91; offset6 failed 7/8 episodes, so its states are degenerate.
- e_exec from the video pass (LAM distance, observed vs ideal at +35; episode 0, 8 queries): none 0.014,
  gain 0.028, latency 0.032, offset 0.063, latency6 0.068.

So the gap is clearly visible in LAM space, but it is small next to the prediction error itself (~0.07–0.10).

### FLUX 3 Action (8 episodes per condition; offset6 not run)

| cond | success | e_track ° | DINO scene@16 e_pred | Δ vs none [CI] | e_ideal (paired) | e_exec | LPIPS wrist@16 e_pred | Δ vs none [CI] | e_ideal (paired) | e_exec |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 2/8 | 1.68 | 0.138 | — | 0.151 | 0.038 | 0.189 | — | 0.322 | 0.235 |
| gain | 3/8 | 1.70 | 0.144 | +0.005 [−0.003, 0.011] | 0.155 | 0.038 | 0.231 | +0.037 [0.011, 0.061] | 0.362 | 0.291 |
| latency | 1/8 | 1.95 | 0.144 | +0.005 [−0.002, 0.012] | 0.163 | 0.053 | 0.236 | +0.034 [−0.005, 0.074] | 0.433 | 0.361 |
| offset | 3/8 | 1.70 | 0.146 | +0.006 [−0.001, 0.016] | 0.158 | 0.044 | 0.244 | +0.048 [0.017, 0.086] | 0.434 | 0.396 |
| backlash | 1/8 | 2.58 | 0.143 | +0.004 [−0.000, 0.010] | 0.162 | 0.049 | 0.244 | +0.051 [0.028, 0.080] | 0.423 | 0.353 |
| gain2 | 1/8 | 2.61 | 0.146 | +0.009 [0.001, 0.016] | 0.157 | 0.040 | 0.193 | +0.008 [−0.029, 0.043] | 0.389 | 0.342 |
| latency6 | 1/8 | 2.66 | 0.149 | +0.011 [0.002, 0.023] | 0.170 | 0.061 | 0.260 | +0.060 [0.020, 0.100] | 0.455 | 0.405 |

- **e_pred < e_ideal in every condition, `none` included.** All paired obs−ideal CIs are below 0. The
  predicted video is closer to what the PD-tracked arm actually does than to a perfect execution of the same
  commands.
- **e_track is a poor x-axis for FLUX.** FLUX integrates its delta actions onto the last command, so its command
  stream leads the arm: nominal e_track is 1.68° (0.67° for LaWAM) and barely moves under gain or offset.
- **FLUX e_pred follows e_exec, not e_track.** Across conditions, r(e_exec, e_pred) is 0.75 for wrist LPIPS and
  0.81 for wrist L1; r(e_track, e_pred) is 0.21 and 0.10.

Full per-metric numbers (L1, LPIPS, DINO × scene/wrist × j = 8/16/31) are in `runs/T004/flux3/summary.json`.

### Read

The future prediction does respond to an injected control gap, in both models:
- **LaWAM:** its own training loss rises roughly in proportion to e_track across conditions (r = 0.83). The rise
  is clear at the strong doses: gain2 +29% and latency6 +33% vs none.
- **FLUX:** the wrist-view prediction error rises 18–32% under gain, offset, backlash and latency6.

The effect is small next to the prediction error the models carry anyway. At the mild doses (3°, 100 ms, ×0.5)
it cannot be separated from none with 8 episodes, except in the FLUX wrist view.

e_ideal does not stay at the none level; it rises with the gap too:
- the closed loop visits different states under a gap (offset and offset6 also cut success);
- for FLUX, e_ideal sits far above e_pred because the model has learned the sim's tracking lag.

The cleaner signal is LaWAM's paired obs−ideal difference. It moves from −0.007 (none) to +0.010 (latency6): under
a strong gap the observed outcome becomes worse-predicted than the perfectly executed one.

For P1 this suggests:
- reward on (e_pred − e_ideal), or on e_pred normalized per state, rather than raw e_pred;
- test at gap sizes at least as large as the strong doses here, or with more than 8 episodes.

Caveats:
- 8 episodes per cell.
- FLUX succeeds rarely (1–3/8), so most FLUX queries come from failing rollouts.
- e_ideal replays every 2nd query only.
- The ideal replay teleports joints. It is "perfect tracking of the command", not "the arm the model saw in
  training".
- FLUX frames are compared after JPEG/H.264 coding.
- FLUX's lag sweep hints the prediction may lead the bench alignment by 1–2 ticks on the wrist.

## Compute

| use | window | GPU-h |
|---|---|---|
| GPU 1: LaWAM server + Isaac | 16:43–20:50 | 4.1 |
| GPU 0: FLUX server | 19:33–22:28 | 2.9 |
| GPU 1: FLUX-only Isaac | 20:50–22:27 | 1.6 |
| **Total** | | **≈ 8.7** (cap 10) |

Up to three Isaac clients (~6.4 GB each) shared GPU 1 against one policy server. Two attempts OOM'd when a
fourth process started on GPU 1; those runs were redone.

## Reproduce

```bash
# LaWAM
LEROBOT_SRC=<checkout>/third_party/lerobot envs/lawam/setup_lawam.sh
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 envs/lawam/.venv/bin/python src/so101_playground/lawam_probe_server.py \
    --repo_id envs/lawam/weights/so101_bench_LaWAM --port 18800 --log_dir /outputs/so101-playground/runs/T004/lawam
GPU=1 scripts/run_t004_sweep.sh lawam 18800 /outputs/so101-playground/runs/T004 none,gain,latency,offset,backlash,gain2,latency6,offset6 8 8 --replay_every 2
GPU=1 scripts/run_t004_videos.sh 18800 /outputs/so101-playground/runs/T004 none,gain,latency,offset,latency6 360

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
envs/flux3/flux-action/.venv/bin/python src/so101_playground/t004_plot.py /outputs/so101-playground/runs/T004 out.png
```
