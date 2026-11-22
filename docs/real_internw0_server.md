# T008: InternW0-Δ server for the real SO-101 (GPU 0, 127.0.0.1:8004)

Branch `exp/real-internw0-server` (from `wam-control-gap`). Code: `src/so101_playground/internw0_probe_server.py
--real`, self-test `scripts/real_internw0_selftest.py`. Outputs `/outputs/so101-playground/runs/real_robot/internw0_server/`.

## Why not the bench server, and what `--real` changes

- The bench `internw0_server.py` (470e09e) keeps the UMT5-XXL text encoder on the GPU and OOMs on a 24 GB 4090
  (23.6 GiB used; `server_1122_2334.log`). The probe server keeps it on the CPU and fits on the card.
- The real runner (`scripts/policy_runner.py` on `exp/home-desktop-setup` → bench `so101_real_policy_runner.py` →
  `MolmoAct2RemotePolicy._query_server`) sends, once every 16 control ticks (30 Hz):
  `{images_png_base64: [overhead, wrist] (640×480 PNG), camera_names, instruction, state (6, LeRobot units),
  num_steps: 10, episode}`. `episode` is `int(time.time())`, set when the runner opens an episode recording
  (`EpisodeRecorder`). It sends **no `trace`**.
- Without `trace` the probe server's sim logic restarted the memory on every query (`None == None`).
  With `--real`, a request **without `trace`** uses one session. A new session (fresh memory and action history)
  starts only when `episode` differs from the current session's, when the body has `"reset": true`, or after
  `POST /reset`. This is the bench server's per-episode behaviour. Within a session the server
  acknowledges the 16 pending actions before each query (`update_obs(None)`), as the bench server does. So
  the session's tick counter advances by 16 per query, and the memory is the episode's anchor frame plus the
  frame 32 ticks back (replan 16, 10 denoising steps). Requests **with `trace`** keep the T006 sim behaviour,
  unchanged.
- `--host` now defaults to `127.0.0.1` (it was `0.0.0.0`). `GET /act` also reports `"real": true`. Its
  `"memory": false` is the config's top-level `model.memory.enabled`. The video memory
  (`model.memory.video`: 1 anchor + 1 recent) is on, and the session uses it (self-test below:
  34 frames kept).

## Real-mode log (for offline e_pred; nothing is scored at runtime)

Under `<log_dir>/real/`:
- `real_queries.jsonl`: one line per real /act, with these fields:
  - `req`: index since server start;
  - `episode`, and `q`: query index in that episode's session;
  - `restarted`: whether this query opened a new session;
  - `session_step` (= 16·q) and `memory_frames`;
  - `t_recv` / `t_reply`: server `time.time()` at receipt and before the reply;
  - `state`, `instruction`, `num_steps`, `camera_names`;
  - `actions`: the returned 16×6 chunk;
  - `npz`: path of the query's npz.
- `ep<episode>/q<NNNN>.npz`: `l0` [48,24,16], `d1`, `d2` (fp16; L̂1 = l0 + d1, L̂2 = l0 + d1 + d2, the
  Wan2.2 VAE latents of ticks 4–16 and 20–32 after the query), `actions`, `state`.
- `ep<episode>/q<NNNN>_{overhead,wrist}.png`: the exact PNG bytes the runner sent. About 220 KB per query
  (2.2 MB per 10 queries).
- `server_requests.jsonl`: per-request latency and peak GPU memory (as in T006).

### Alignment assumption (query ↔ runner frame)

Each runner episode directory (`outputs/real_robot/runs/<YYYYmmdd_HHMMSS>_<slug>/`) has `overhead.mp4`,
`wrist.mp4` (30 fps, one frame per control step) and `trajectory.npz`. `trajectory.npz` holds `t`
(monotonic s since the recording opened) and `policy_query` (bool per step).
- **Episode.** The server's `episode` (`int(time.time())`, home-desktop clock) is set right after the
  recorder's directory timestamp (`datetime.now()`, same clock), so they agree to within a second. Pair
  each server episode with the recording whose directory time is closest. Do not rely on the server's
  `t_recv`: it is the GPU server's clock, and the SSH tunnel adds latency.
- **Query.** The runner's control loop blocks during a query, and it records the same frames it sent at
  the step where `policy_query` is true. So **the q-th `policy_query == True` step of the recording is the
  q-th query (`q`) of that episode.** Steps where the camera freezes are not recorded and do not query.
  To verify, compare `q<NNNN>_overhead.png` with that video frame (mp4 compression only).
- **Observed future.** Ticks +4, +8, …, +32 after the query step are the frames that L1 (ticks 4–16) and L2
  (20–32) encode. Encode [query PNG canvas, video frames at +4…+32] with the T006 `/iw_score` recipe, then
  take e_pred@k = mse(L̂k, Lk). The real runner executes 16 actions per query, so ticks 17–32 already come
  from the next chunk (in sim, T006 also replanned every 16).
- **Caveats.**
  - The video frames are mp4-compressed, while the query canvas comes from the lossless PNG. Measure
    e_static at k = 0 (the L0 PNG vs the L0 mp4 frame) as the codec floor.
  - An instruction change mid-episode (runner key) clears the client's queue and queries early. The server
    still counts 16 ticks, so its `session_step` and the recording disagree from then on (the bench server
    behaves the same).
  - Pausing or docking without a new recording keeps the same `episode`, and with it the same memory.

## Self-test (GPU 0, `CUDA_VISIBLE_DEVICES=0`)

`scripts/real_internw0_selftest.py` sends runner-shaped requests. Frames are recorded sim frames (T004
`lawam_videos/{none,gain}/frames_ep000.npz`, ticks 0, 16, …, 144). The state is the sim trajectory mapped to
LeRobot units.

Episodes, 10 queries each:
- A, episode e1;
- B, episode e2, another recording;
- A again under a new episode e3;
- A with a sim `trace` (the T006 harness path).

Results (`/outputs/so101-playground/runs/real_robot/internw0_server/selftest_1122_2339/`):

| check | result |
|---|---|
| `GET /act` | `status: ok`, `real: true`, chunk 32, n_action_steps 16 |
| memory within an episode | `memory_frames` 1, 17, 33, 34, 34, … (anchor + 33 recent); `session_step` 0, 16, …, 144 |
| reset across episodes | `restarted` only at q = 0 of each episode; memory back to 1 frame |
| A again after the reset vs A | max \|Δaction\| **0.0** (the denoiser seed is fixed per call) |
| real mode vs T006 sim-trace path, same inputs | max \|Δaction\| **0.0** |
| A vs B at q = 0 | 0.30 (different inputs give different actions) |
| latency (server) | first query 1.63 s, q = 2 of the first episode 2.49 s (warm-up), then **median 0.57 s** |
| GPU memory | model 17.02 GiB; peak 17.24 GiB allocated / 17.63 GiB reserved; ~18.5 GB on the card |

## Running server

Started from a code snapshot so that it does not depend on this worktree:
`/outputs/so101-playground/runs/real_robot/internw0_server/code_<commit>/internw0_probe_server.py`.

```bash
# start (GPU server)
W=/workspace/so101-playground/.claude/worktrees/wam-internw0/envs/internw0
D=/outputs/so101-playground/runs/real_robot/internw0_server
SRC=$D/code_<commit>/internw0_probe_server.py   # or src/so101_playground/internw0_probe_server.py in a checkout
T=$(date +%m%d_%H%M)
CUDA_VISIBLE_DEVICES=0 WAM_CHECKPOINT_ROOT=$W/weights/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 \
  setsid nohup $W/InternW0-Delta/.venv/bin/python $SRC \
    --repo $W/InternW0-Delta --checkpoint $W/weights/so101_bench_InternW0-Delta/model.pt \
    --train_config $W/weights/so101_bench_InternW0-Delta/config.yaml \
    --stats $W/weights/so101_bench_InternW0-Delta/dataset_stats.json \
    --real --host 127.0.0.1 --port 8004 --device cuda:0 --log_dir $D/run_$T \
    > $D/server_$T.log 2>&1 < /dev/null &
# ready after ~70 s:
curl -s localhost:8004/act          # {"status":"ok", ..., "real":true}

# equivalent from a checkout of this branch
IW_ENV=$W GPU=0 scripts/run_internw0_server.sh 8004 $D/run_$T --real --host 127.0.0.1

# stop
pkill -f 'internw0_probe_server.py.*--port 8004'      # or: kill <PID>

# home desktop
ssh -N -L 8004:localhost:8004 <gpu-server>
envs/lerobot/.venv/bin/python scripts/policy_runner.py --policy internw0 --instruction "..."
```

Current instance: see "Current instance" below (PID, log).
