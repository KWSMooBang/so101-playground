# T004: does a WAM's own future prediction expose an injected control gap? (H2-P0, sim only)

Branch `exp/wam-control-gap`. Outputs `/outputs/so101-playground/runs/T004/`. Rows in
`/outputs/so101-playground/results.tsv` (`T004-<condition>`).

## Model

**LaWAM only** (decision L1, 2026-11-21): all runs on GPU 1, and the FLUX 3 Action arm is deferred. FLUX
needs the 13.9 GB bf16 DiT + 2.5 GB VAE + activations (est. 17–19 GB with the text encoder offloaded to CPU,
not measured); with Isaac (~6.2 GB) on the same 24 GB card that is borderline. The FLUX code (server wrapper with
text-encoder offload, LoRA merge, pixel/LPIPS/DINOv2 metrics) is committed but has not been run. Weights stay in
`envs/flux3/weights/` (base VAE + text encoder, SO-101 package, bench LoRA raw update 3000; not yet merged).

LaWAM = `5hadytru/so101_bench_LaWAM_full_fine_tune` @ 8df458f (update 45000, 21/39 on the bench val set),
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

## Episodes

`tasks/T004_bin1.jsonl`: the bench `val_set.jsonl` single-object Bin rows with rigid single-body objects
(28 rows; 2 shoe rows dropped as multi-body). Fixed bench layouts (`tasks/layouts/val_set_layouts.jsonl`), env
seed 1984, 30 s limit. The first 8 rows were run under every condition (identical layouts and seeds). Task
`So101Bench-Bin-v0`, overhead + wrist 640×480, LeRobot joint frame, action horizon 36, history 1, no init frames
(bench leaderboard settings for LaWAM).

## Metrics (per query s; `src/so101_playground/t004_analysis.py`)

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

Checks: re-encoding the query frame reproduces `h_t` (mse ~2e-4 vs e_pred ~0.04–0.3). Replaying from a
restored snapshot reproduces e_ideal to 0.3% across runs, and the observed e_pred varies by ~1% across runs.
Skipping renders inside the replay changed the RTX image enough to move e_ideal by 3–10%, so the replay renders
every tick.

## Reproduce

```bash
LEROBOT_SRC=<checkout>/third_party/lerobot envs/lawam/setup_lawam.sh    # venv + weights
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 envs/lawam/.venv/bin/python src/so101_playground/lawam_probe_server.py \
    --repo_id envs/lawam/weights/so101_bench_LaWAM --port 18800 --log_dir /outputs/so101-playground/runs/T004/lawam
GPU=1 scripts/run_t004_sweep.sh lawam 18800 /outputs/so101-playground/runs/T004 none,gain,latency,offset 8 8 --replay_every 2
envs/flux3/flux-action/.venv/bin/python src/so101_playground/t004_analysis.py --root /outputs/so101-playground/runs/T004 \
    --model lawam --tsv /outputs/so101-playground/results.tsv
```

## Results

(filled after the sweep)
