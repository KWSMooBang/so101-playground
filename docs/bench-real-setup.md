# SO-101 Bench 실기 셋업 감사 (T002)

SO-101 Bench 의 정책 서버·실기 runner·캘리브레이션 코드를 읽고, 그들의 정책을 우리 집 SO-101 에서 돌리고 sim↔real 을 비교하려면 무엇을 맞춰야 하는지 정리한 문서.

- 대상 코드: `github.com/5hadytru/so101_bench` @ `470e09e` (2026-11-21 clone). 아래 `file:line` 은 모두 이 커밋의 bench 레포 기준 경로.
- 논문: `so101-bench.github.io/assets/paper-9eb0bad2dd.pdf` (arXiv 2610.36420). 하드웨어 서술은 거의 없고(§3.3, §5, App. B), 카메라 모델 등은 코드에만 있다.
- 실행한 것 없음. 전부 코드/논문 읽기 결과이며, "추정" 이라고 적은 수치는 측정값이 아니다.

## 0. 한눈에 보기

- 정책 서버는 전부 `POST /act` (JSON, PNG base64 이미지, 6-D 상태) → `{"actions": [[6]×T], "dt_ms"}`. GR00T N1.7 만 예외로 ZMQ+msgpack (포트 5555).
- 실기 runner(`scripts/so101_real_policy_runner.py`)는 LeRobot 의 `FeetechMotorsBus` 를 **raw tick** 으로 직접 쓴다. LeRobot `SO101Follower` 의 정규화(`.pos`)를 쓰지 않는다.
- 정책이 보는 `.pos` 단위는 "우리 팔의 LeRobot 캘리브레이션" 이 아니라 **bench 가 코드에 박아 둔 고정 캘리브레이션**으로 정의된 sim 관절각의 bijection 이다. 우리 팔은 `so101_sim_alignment.py measure` 로 관절별 zero tick 을 재서 sim 각도에 맞춰야 한다. 이게 가장 중요한 항목.
- 제어는 30 Hz, 쿼리는 동기식: 쿼리 중에는 팔이 마지막 목표를 유지하고 스텝이 세지지 않는다. 그래서 원격 분리에 **hard deadline 은 없고**, 지연은 정지 시간으로만 나타난다.
- 원격 분리는 코드 수정 없이 가능: 터널이 먼저 떠 있으면 runner 가 "이미 응답하는 서버" 로 보고 재사용한다.

## 1. `/act` API 명세

### 1.1 공통 (HTTP 서버)

클라이언트는 하나(`MolmoAct2RemotePolicy`, `source/so101_bench/so101_bench/utils/molmoact2.py:118`)가 모든 `*_server.py` 를 상대한다.

요청 `POST http://<host>:<port>/act`, `Content-Type: application/json` (`utils/molmoact2.py:349-365`):

| 필드 | 타입 | 내용 |
|---|---|---|
| `images_png_base64` | `[str]` | 카메라별 PNG(RGB, uint8)의 base64. 순서 = `camera_names` |
| `camera_names` | `[str]` | `["overhead","wrist"]` + (init 프레임 사용 시) `"overhead_init"`, `"wrist_init"` 순서 (`molmoact2.py:203-210`) |
| `instruction` | `str` | 자연어 지시 |
| `state` | `[float×6]` | 관절 상태, 순서·단위는 1.3 |
| `num_steps` | `int` | flow/denoise 스텝 (기본 10). 안 쓰는 서버는 무시 |
| `episode` | `int` | 에피소드 id. stateful 서버가 바뀌면 메모리 리셋. runner 는 에피소드 시작 시 `int(time.time())` (`so101_real_policy_runner.py:1264-1266`) |
| `snapshot_images_png_base64`, `state_history`, `command_history` | | `history_ticks > 1` 일 때만 (FLUX 3). 창의 가장 오래된 틱 이미지 + 틱별 상태/명령 (`molmoact2.py:358-361`) |
| `trace` | `{dir,episode,frame}` | 선택, FLUX 예측 비디오 저장용 |

응답: `{"actions": [[6 floats] × T], "dt_ms": float}`. 실패 시 `{"error": "..."}` (서버에 따라 HTTP 400/500, G0.5 는 200 에 error 필드만: `g05_server.py:137`). 클라이언트는 `error` 키를 보고 예외 (`molmoact2.py:375`).

헬스체크: `GET /act` → `{"status":"ok","repo_id","norm_tag","num_images","camera_keys", ...}`. 클라이언트 `connect()` 가 `history_ticks`(기본 1)와 `num_images` 가 자기가 보낼 이미지 수와 같은지 검사한다 (`molmoact2.py:216-235`).

이미지:
- 해상도 **640×480** (W×H). runner 가 `FRAME_SIZE=(640,480)` 로 맞춰 보낸다 (`so101_real_policy_runner.py:89, 1177-1181, 799`). 모델 입력 해상도로의 resize 는 서버 쪽 전처리가 한다 (π0.5 224×224 pad, LaWAM 256×256, InternW0 192×256 타일 등).
- 인코딩은 PNG (`molmoact2.py:328-333`). 서버 디코드는 `PIL.Image.open(...).convert("RGB")` 라서 포맷을 자동 판별한다 (`pi05_server.py:46`, `g05_server.py:40`, `molmoact2_server.py:50`). 즉 JPEG 를 보내도 서버는 받지만, 그건 bench 프로토콜에서 벗어나는 것이다 (4.3).
- 색 순서 RGB (runner 가 BGR→RGB 변환, `so101_real_policy_runner.py:1178`).

### 1.2 action chunk / replan

- 서버는 체크포인트의 chunk 전체를 돌려주고(`--replan_steps` 플래그가 있는 G0.5·InternW0·ImageWAM 서버만 그 값을 쓴다. π0.5·MolmoAct2·LaWAM·FLUX 서버에는 없다), 클라이언트가 앞의 `action_horizon` 개만 큐에 넣어 한 틱에 하나씩 실행한다. 큐가 비면 다시 쿼리 (`molmoact2.py:394-409`). temporal ensembling 없음, 비동기 추론 없음.
- 액션은 **절대 관절 목표 위치**, 상태와 같은 단위·순서. gripper 도 절대값.
- 정책별 설정 (`README.md:61-69`, `so101_real_policy_runner.py:108-121`, 논문 Table 8):

| 정책 | 서버 / 기본 포트(runner) | 반환 / 실행 스텝 | init 프레임 | joint_frame | 이미지 수 |
|---|---|---|---|---|---|
| Galaxea 0.5 | `g05_server.py --mode ar --replan_steps 15` / 8001 | 27 / 15 | 없음 | lerobot | 2 |
| GR00T N1.7 | upstream GR00T ZMQ 서버 / 5555 | – / 16 | 있음 | lerobot | 4 |
| π0.5 | `pi05_server.py --num_steps 10` / 8002 | 30 / 15 | 있음 | lerobot | 4 |
| MolmoAct2 | `molmoact2_server.py --num_steps 10` / 8003 | 30 / 30 | 있음 | molmoact2 | 4 |
| InternW0-Δ, ImageWAM | `internw0_server.py`, `imagewam_server.py` | 16 / 16 | 없음 | lerobot | 2 |
| LaWAM | `lawam_server.py` | 36 / 36 | 없음 | lerobot | 2 |
| FLUX 3 Action | `flux3_server.py` | 42 / 32 | 없음 | lerobot | 2 (+snapshot 2), `history_ticks 8` |

실기 runner 에 배선된 것은 groot / g05 / pi05 / molmoact2 (+ LLM agent 2종) 뿐이다 (`so101_real_policy_runner.py:108-121`). 나머지 서버는 sim eval 클라이언트(`scripts/molmoact2_eval.py`)에서만 쓰였다. 프로토콜이 같으므로 `POLICY_DEFAULTS` 에 항목을 추가하면 실기에서도 돌 수 있지만 FLUX 는 runner 가 `history_ticks=1` 로 고정이라(`:809`) 그대로는 안 된다.

### 1.3 state / action 단위와 순서

순서: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (`utils/lerobot_calibration.py:23-30`). 데이터셋 feature 이름은 `<joint>.pos`.

`joint_frame=lerobot` (G0.5, π0.5, GR00T 등):
- body 5관절: `[-100, 100]`, gripper: `[0, 100]` (`pi05_server.py:12`).
- 정의 (`utils/groot.py:224-254`, `utils/lerobot_calibration.py:64-71,126-146`):
  - body: `tick = sim_deg / (360/4095) + 2047.5`, `pos = (tick - range_min)/(range_max - range_min) * 200 - 100`. `range_min/max` 는 **`REAL_SO101_CALIBRATION` 상수**(데이터 수집에 쓴 팔의 값, `lerobot_calibration.py:64-71`).
  - gripper: sim jaw 각도에 선형. `pos = (jaw_deg - (-12.75)) / (115.5 - (-12.75)) * 100`. `0` = 완전히 닫힘(jaw −12.75°, 끝이 닿는 −12° 보다 0.75° 더), `100` = 완전 개방(115.5°). tick 범위 기반이 아니다.
  - sim 관절 한계(USD): pan (−115, 122), lift (−105.5, 101.9), elbow (−102, 90.25), wrist_flex (−103.15, 101.25), wrist_roll (−168, 168), jaw (−12.75, 115.5) deg (`lerobot_calibration.py:46-53`).
- 즉 `.pos` 는 "sim 관절각의 고정 bijection" 이고 실기 팔의 캘리브레이션과 무관하다 (`utils/real_sim_alignment.py:3-5`).

`joint_frame=molmoact2` (MolmoAct2 만): 공개 SO-100/101 v2.1 관례의 degree 프레임. body 는 `sim_deg + offset`(offset 은 위 상수 범위 중심으로부터 계산) 에 부호 `(1,-1,1,1,1,1)`, 오프셋 `(0,90,90,0,0,0)` deg 적용, gripper 는 0–100 (`utils/molmoact2.py:24-90`).

GR00T: 같은 lerobot 단위지만 `state.single_arm`(5) + `state.gripper`(1) 로 나뉘고, 액션도 `single_arm`/`gripper` 키로 온다. arm 액션은 relative, gripper 는 absolute 표현으로 학습됐고 서버가 절대값으로 풀어 준다 (`scripts/GR00T_N1.7_fine_tuning/configs/so101_config.py:17-46`, `utils/groot.py:502-590`).

### 1.4 서버별 차이

| 서버 | 상태 사용 | 카메라 매칭 | stateful | 비고 |
|---|---|---|---|---|
| `pi05_server.py` | 사용 | **이름으로** 매칭, 누락/초과 시 에러 (`:70-73`) | 아니오 | `num_steps` 반영. `--repo_id` 는 로컬 export 디렉터리만 (`:142`) |
| `molmoact2_server.py` | 사용 (shape (6,) 강제 `:142`) | **순서로만** (camera_names 무시, `:194`) | 아니오 | `/healthz` 도 있음 |
| `g05_server.py` | 사용 | overhead→`exterior`, wrist→`wrist_right`, 없는 슬롯은 검은 프레임 (`:36,106-109`) | **예**: 에피소드별 프레임 히스토리, `episode` 바뀌면 리셋 (`:97-105`) | `--replan_steps` 가 클라이언트 horizon 과 같아야 학습 때의 프레임 간격 재현 (`:9-12`) |
| `lawam_server.py` | **무시** (받기만 함) | `--camera_map` | 아니오 | |
| `internw0_server.py` | 사용 | 이름 | **예** (에피소드 메모리, `episode`) | `--replan_steps 16` 전제 |
| `imagewam_server.py` | 사용 | 이름 | 아니오 | 지시문 임베딩 캐시 |
| `flux3_server.py` | `state_history`/`command_history` | `--client_cameras` | 아니오 (클라이언트가 창 전송) | `state` 대신 history 필드 필수 |
| GR00T | ZMQ REQ/REP, msgpack, 이미지는 `np.save` 바이트(무압축) (`utils/groot.py:81-86`) | `video.{overhead,wrist,overhead_init,wrist_init}` | 서버 reset 엔드포인트 | 언어 키 `annotation.human.task_description` |

stateful 서버(G0.5, InternW0)는 프로세스당 에피소드 하나만 추적한다. 서버 하나를 여러 클라이언트가 동시에 쓰면 안 된다.

## 2. 실기 runner

`scripts/so101_real_policy_runner.py` (1824 줄, tkinter GUI + 제어 스레드).

- **모터 통신**: LeRobot 의 `lerobot.motors.feetech.FeetechMotorsBus` 를 직접 생성, 모터 6개 `sts3215`, id 1–6 (`:326-339`). `SO101Follower` 클래스는 안 쓴다. 읽기 `sync_read("Present_Position", normalize=False)`, 쓰기 `sync_write("Goal_Position", ticks, normalize=False)` (`:356-377`). 변환은 `RealSimAlignment`: `ticks = zero_tick + sign * sim_deg * 4096/360` (`utils/real_sim_alignment.py:70-80`).
- 시작 시 모터 EEPROM 의 `homing_offset` 이 alignment 를 잰 때와 다르면 **종료**한다 (`:337-339`, `real_sim_alignment.py:91-102`). LeRobot 재캘리브레이션을 하면 alignment 도 다시 재야 한다.
- P/I/D gain, 가속도 등은 runner 가 쓰지 않는다 (repo 전체에 `P_Coefficient` 쓰기 없음). 모터에 남아 있는 값 그대로 쓰며, 주석은 P=16 (LeRobot 기본) 을 전제한다 (`:361-362`).
- **제어 주기**: `CONTROL_HZ = 30.0` (`:88`), 고정 주기 루프, 밀리면 따라잡지 않고 다음 틱 재설정 (`:1627-1656`). sim 도 physics 240 Hz / control 30 Hz (`so101_bench_env_cfg.py:117-118`).
- **action horizon / replan**: 1.2 표. 쿼리는 제어 스레드 안에서 **동기 호출** (`:1435-1438`). 쿼리 동안 팔은 마지막 `Goal_Position` 유지, 스텝 카운트·녹화 없음. sim 이 쿼리 중 physics 를 멈추는 것에 대응 (`:97-99, :1442`). 에피소드 제한시간은 제어 스텝 수로 센다: 30 s(900 스텝), 4-object bin 은 75 s (`:820-825`, `benchmark.py:114-123`). 제한을 넘어도 계속 돌고 표시만 한다 (`:1447-1450`).
- **시작 조건**: 에피소드 모드에서는 dock pose(= sim 초기 자세, lerobot 단위 `(-6.5, -98.8, 81.2, 71.2, -0.8, 0.9)`, `lerobot_calibration.py:55-62`) 에서 5° 이내여야 시작 (`:1219-1224`). init 프레임은 dock 에서 정지 상태로만 캡처 가능 (`:1190-1198`). sim eval 은 초기 자세를 1.5 s 유지 후 첫 관측 (`molmoact2_eval.py:125-127`).
- **안전 장치**:
  - 틱당 목표 변화 제한: sim 속도 한계 `[3.0,3.3,3.1,2.9,4.2,3.2] rad/s ÷ 30` ≈ 5.7–8.0°/틱 (`:91, :1443-1444`).
  - 클라이언트 측 `max_joint_step_deg=15`: 현재 상태 대비 최대 관절 변화 15°/틱으로 스케일 (`molmoact2.py:133, 385-392`).
  - 액션은 USD 관절 한계로 clamp (`groot.py:250-253`), tick 은 `range_min/max` 로 clamp (`real_sim_alignment.py:73-75`).
  - watchdog: 목표-측정 오차 > 25° 가 1 s 지속되면 pause (`:104-105, :1463-1468`).
  - gripper: jaw 하한 −12.75° 아래로 명령하지 않음, 시작 시 EEPROM `Min_Position_Limit` 을 그 tick 으로 기록 (`:341-354, :391-395, :1459-1460`). 더 조이면 프린트 jaw 가 깨진다는 경고가 두 번 적혀 있다 (`:216-226`). 닫을 때 측정된 stall 만큼(`gripper_stall_ticks`, 그들 팔은 44.5 tick ≈ 3.9°) 목표를 더 보낸다 (`:366-372`). `--gripper_squeeze_deg` 는 기본 0.
  - 제어 스텝 예외 → hold + action queue 비움 (`:1640-1648`). `q` → dock 후 torque off. dock 은 30°/s smoothstep (`:92, :398-423`).
  - 카메라 프레임이 0.5 s 넘게 안 오면 에피소드 freeze(팔 hold), 돌아오면 자동 재개 (`:319-323, :1472-1497`).
- **카메라**: OpenCV `VideoCapture(device, CAP_V4L2)` 백그라운드 스레드가 최신 프레임만 보관, buffer size 1, 640×480 요청 (`:266-323`).
  - wrist: `/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0`, FOURCC `MJPG`, `v4l2-ctl` 로 고정 컨트롤 재적용 (`:208, :285-289, :725-729`).
  - overhead: `/dev/video10` = DroidCam 의 v4l2loopback 장치(아이폰) (`:210, :6`).
  - 640×480 이 아니면 `INTER_AREA` 로 **crop 없이** resize → 4:3 이 아닌 소스는 찌그러진다 (`:1177-1181`).
- **세션 드리프트 검사**(에피소드 모드): 매 에피소드 시작 전 overhead 를 기준 프레임과 비교, 카메라 이동 > 3 px 또는 밝기 변화 > 8% 면 시작 거부 (`:184-185, :682-716, :1239-1248`). 논문: 225 실기 에피소드에서 최대 2.6 px / 7%.
- **레이아웃 재현**: 에피소드의 저장된 object/bin footprint 를 sim overhead 카메라 모델로 투영해 실시간 overhead 영상에 겹쳐 그린다. 사람이 그 윤곽에 물체를 놓는다 (`utils/layout_overlay.py:33-57`, runner `:1509-1512`). 실기 overhead 카메라가 sim 카메라와 정렬돼 있어야만 성립한다. blend/edges/diff 오버레이는 `outputs/real_robot/sim_camera_reference/overhead_*.jpg` (sim 렌더) 와 비교용 (`:186, :731`) — 이 참조 이미지는 레포에 없다.
- **채점**: 사람이 `y`/`f` 로 채점, 실패 유형 태그 (`:132-138`). 결과는 `<results_root>/<policy>/results.jsonl`.
- **기록**: 에피소드마다 `overhead.mp4`, `wrist.mp4`(mp4v, 30 fps), `trajectory.npz`(sim deg 상태/명령, 쿼리 여부), `episode.json` (`:426-467`). LeRobot 데이터셋 포맷이 아니다.

보조 스크립트:
- `scripts/so101_real_recording.py`: runner 의 `v` 녹화 모드용. wall-clock 클립(overhead|wrist, 30 fps, 640×480) + 주석 본. 하드웨어 로직 없음 (`:1-11, :32-33`).
- `scripts/so101_gripper_test.py`: gripper 모터만 torque on. `c/o/<deg>` 로 닫힘 확인, `t` 로 stall 측정 후 alignment JSON 의 `gripper_stall_ticks` 저장 (`:1-17, :53-83`).
- `scripts/so101_follower_teleop.py`: 이름과 달리 **sim teleop** 이다. 실기 follower 팔을 torque off 로 손으로 움직여 "리더" 로 쓰고(또는 gamepad), Isaac sim 안의 로봇을 조종해 LeRobot 데이터셋을 기록한다 (`:4, :1012-1184`). 여기서는 LeRobot `SO101FollowerConfig(use_degrees=False)` 를 쓴다 (`:1137-1154`). 속도 제한 `(110,140,150,125,110,120)` pos-units/s (`:33`). Isaac Sim 이 필요하므로 집 데스크탑용이 아니다.
- `scripts/so101_real_trajectory_replay.py`: 데이터셋 궤적 하나를 실기에서 30 Hz 로 재생하는 특성화 도구. LeRobot `SO101Follower` 사용, tracking error 지속 시 abort (`:1-7, :73-98, :206-230`). 이 스크립트는 팔의 LeRobot 캘리브레이션을 그대로 쓰므로 sim alignment 를 거치지 않는다.

## 3. 캘리브레이션과 정렬

### 3.1 관절 zero / offset

- sim zero pose 정의 (측정 절차가 곧 정의, `scripts/so101_sim_alignment.py:72-122`): 상완 수직(lift 0), 전완 수평 전방(elbow 0), 손목 일직선 수평(wrist_flex 0), gripper 수직·움직이는 jaw 가 위·wrist 카메라가 로봇 왼쪽(roll 0), 팔이 base 정면(pan 0), jaw 끝이 막 닿음 = −12°.
- `measure`: torque off 상태에서 손으로 각 자세를 잡고 폰 수평계(모터 케이스 면)로 맞춘 뒤 Present_Position 을 캡처해 관절별 `zero_tick` 을 푼다. 반복 2회, 불일치 1.5° 경고, check 자세 허용 2° (`:51-55, :168-248`). 결과는 `source/so101_bench/so101_bench/assets/real_robot/<robot_id>_sim_alignment.json`. `live` 는 읽기 전용 스트림, `verify` 만 torque on (`:9-11`).
- 그들 팔의 결과 (`assets/real_robot/follower_arm_1_sim_alignment.json`): zero tick 이 2027–2070 (gripper 1697), sign 전부 +1. 즉 LeRobot 표준 캘리브레이션(중립 자세에서 homing → 2047 근처) 과 수십 tick(≈1–4°) 차이. 이 차이를 재는 것이 목적이다.
- LeRobot SO-101 캘리브레이션과의 관계:
  - LeRobot: `homing_offset` 은 중간 자세에서 설정, `range_min/max` 는 관절을 끝까지 돌려 기록, `.pos` 는 그 범위로 정규화. 팔마다 다르다.
  - bench: `.pos` ↔ sim 각도는 `REAL_SO101_CALIBRATION` 상수로 고정, sim 각도 ↔ 우리 팔 tick 은 `zero_tick` 으로. 우리 팔의 `range_min/max` 는 clamp 에만, `homing_offset` 은 "바뀌었는지 검사" 에만 쓰인다.
  - 따라서 우리 팔을 LeRobot 으로 그냥 캘리브레이션해서 얻은 `.pos` 를 bench 정책에 넣으면 스케일·오프셋이 틀린다. bench 정책에는 반드시 `tick → sim_deg(zero_tick) → bench .pos` 경로를 써야 한다.
  - LeRobot teleop 을 캘리브레이션과 함께 다시 연결하면 gripper `range_min` 이 덮어써져서 runner 가 매번 다시 쓴다 (`so101_real_policy_runner.py:344-345`).

### 3.2 카메라 intrinsics / extrinsics (sim 기준값)

world frame: 테이블 상면 z=0 (`so101_bench_env_cfg.py:52`). 값은 `benchmark.py:393-417` (DomainRandomizationSpec 기본 = canonical).

| | overhead | wrist |
|---|---|---|
| 실기 장치 (코드상) | iPhone 17 Pro main camera, DroidCam (`so101_bench_env_cfg.py:152-158`) | Innomaker U20CAM-720P USB (`:160-166`) |
| 해상도 | 640×480 | 640×480 |
| 수평 FOV | 74.0° (focal ≈ 425 px) | 72.5° (focal ≈ 436 px) |
| 위치 | world `(0.0, −0.34, 0.45722)` m | gripper link 기준 `(−0.0025, 0.0595, −0.0605)` m |
| 자세 | rpy `(51.361, −0.169, 0.603)`°, OpenGL 관례(−Z 를 봄) → 수직 아래에서 +y 쪽으로 약 51° 기울임 | rpy `(−45, 0, −1)`° |
| principal point | 중심 (offset 0) | 중심 |
| 렌즈 왜곡 | 모델 없음 (pinhole) | 모델 없음 |

- 로봇 base: world `(0.05209, 0.18061, −0.03102)` m, yaw 6.195° (`assets/so101.py:59-60`). overhead 카메라는 로봇 **맞은편**(base 에서 y 방향으로 약 52 cm 앞, 테이블 위 약 46 cm)에서 로봇 쪽을 비스듬히 내려다본다. "천장에서 수직" 이 아니다.
- 실기 카메라의 intrinsics 캘리브레이션(체커보드)·왜곡 보정 코드는 없다. 정렬은 overlay 로 눈으로 맞추는 방식이다.
- `scripts/view_wrist_camera.py` 는 **sim** 카메라 관측을 저장/표시하는 Isaac 스크립트다 (`:4, :17-45`). 실기 wrist 뷰와 나란히 비교할 참조 이미지를 뽑는 용도. wrist 카메라 pose 슬라이더는 `so101_follower_teleop.py` 의 calibration UI 에 있다 (`:496, :2649-2656`).
- 학습 데이터는 조명, 카메라 intrinsics/extrinsics, 배경, 로봇 색을 랜덤화했고 canonical 도 포함한다 (논문 §3.5). 랜덤화 폭은 `tasks/so101_bench_sim.jsonl` 의 행별 값에 있다(이 감사에서는 분포를 집계하지 않았다).

### 3.3 wrist 카메라 화이트밸런스 / 노출

`scripts/so101_wrist_white_balance.py`: 흰 종이를 15 cm 앞에 두고 auto WB 를 끈 뒤 색온도 2800–6500 K 를 훑어 R≈B 인 값을 고르고 `<robot_id>_wrist_camera_controls.json` 에 저장 (`:1-12, :61-86`). 기본 컨트롤: brightness 5, contrast 32, saturation 64, gamma 100, gain 0, sharpness 3, auto_exposure 3 (`:32-35`). 목표는 "검은 테이블의 wrist 영상 median 밝기 ≈ 41 (sim 40)" (`:28-31`). 그들 팔의 controls JSON 은 레포에 없다. Innomaker 전용 수치이므로 다른 카메라에서는 목표(중립 WB, 테이블 밝기)만 따라간다.

### 3.4 작업공간 / 테이블 / 조명

- 물체 배치 영역 `TABLE_BOUNDS`: x ∈ (−0.14, 0.25), y ∈ (−0.10, 0.155) m → **39 × 25.5 cm** (`so101_bench_env_cfg.py:59`).
- bin: canonical 위치 `(−0.129, −0.163)` yaw −66°, 그 외 5개 후보 위치 (`:71-86`). 에피소드별 실제 위치는 layouts JSONL. bin 치수·제품명은 코드/논문에 없다 (USD 에셋에만 있음). 논문: 실물 bin 이 가벼워 잘 밀려서 real bin 과제 성공률이 낮았다 (§5).
- 테이블: 검은색 (`so101_wrist_white_balance.py:2, :28-31`). 치수·재질 서술 없음. 배경은 방 스캔(`background_1.usdc`, "bedroom tabletop").
- 조명(sim): dome + key/fill sphere light, 6500 K, 색 (1.0, 0.96, 0.88) (`benchmark.py:419-423`, `so101_bench_env_cfg.py:194-205`). 실기 조명 장비 서술 없음. 논문: "Lighting and camera placement are fixed for the whole study."
- 로봇 색: canonical 은 주황-빨강 `(0.95, 0.15, 0.02)` (`benchmark.py:425-427`), 학습에 6색, held-out 4색.
- 물체: 실기에 쓴 것은 benchmark 138개 중 실물로 보유한 40개(seen 18, unseen object 11, unseen class 11). 구매 목록/BOM 없음. 큰 물체는 sim 에서 0.7–0.9배로 축소돼 있다 (논문 §5).

## 4. 데이터 포맷

- `so101_bench_sim`, `so101_bench_real` 모두 LeRobot **v3.0**, `robot_type: so101_follower`, 30 fps (HF `meta/info.json`).
  - `action`, `observation.state`: float32 `(6,)`, names `shoulder_pan.pos … gripper.pos` (`utils/lerobot_dataset.py:250-263`).
  - `observation.images.overhead`, `observation.images.wrist`: video, `(480, 640, 3)`, AV1 yuv420p (`:264-269`).
  - `so101_bench_sim_WM` 은 `observation.images.overhead_init`, `observation.images.wrist_init` (에피소드 시작 프레임) 추가.
  - real: 4,367 에피소드 / 2.47M 프레임 / 약 35개 물체, sim: 3,629 에피소드 / 2.0M 프레임.
- `scripts/rename_lerobot_camera_keys.py`: `observation.images.front(_init)` → `observation.images.wrist(_init)` 로 재인코딩 없이 키만 변경 (`:15-18`). 초기 데이터는 wrist 를 `front` 로 기록했다는 뜻. 우리 LeRobot 기록도 카메라 키를 **`overhead`, `wrist`** 로 지어야 bench 정책/플러그인과 어댑터 없이 맞는다.
- `scripts/repair_lerobot_stats.py`: per-episode 메타에서 dataset-level `stats` 를 다시 계산 (병합·편집 후 깨진 stats 복구용).
- `scripts/so101_lerobot_replay.py`: 데이터셋의 action 스트림을 **sim** 에 재생 (episodes/layouts JSONL 필요). 실기 재생은 `so101_real_trajectory_replay.py`.
- 주의: `so101_bench_real` 의 `.pos` 는 그들 팔의 LeRobot 캘리브레이션(= `REAL_SO101_CALIBRATION`) 기준이다. 우리가 LeRobot 으로 수집한 데이터의 `.pos` 는 우리 팔 범위 기준이라 같은 숫자가 다른 각도다. 섞어 학습하거나 bench 정책에 넣으려면 `우리 .pos → tick → sim_deg(zero_tick) → bench .pos` 변환이 필요하다. 특히 gripper 는 bench 가 jaw 각도 선형, LeRobot 은 tick 범위 선형이다.
- LeRobot 플러그인(`integrations/lerobot`): `lerobot-eval --env.type=so101_bench`, 관측 `pixels`(overhead, wrist 480×640) + `agent_pos`, 같은 lerobot 단위, 15°/틱 clamp, 초기 1.5 s hold (`integrations/lerobot/README.md`). sim 전용.

## 5. 원격 분리 평가 (집 데스크탑 runner ↔ GPU 서버 정책)

### 5.1 가능 여부: 가능, 코드 수정 없음

- runner 는 `--policy_host`(기본 127.0.0.1) / `--policy_port` 로 접속한다 (`so101_real_policy_runner.py:193, :200`).
- 시작 시 `PolicyServerManager.ensure()` 가 먼저 해당 포트의 서버가 살아 있는지 확인하고(HTTP 정책은 `GET /act` 가 `status: ok`, GR00T 는 ZMQ REQ 로 msgpack `{"endpoint": "ping"}` 을 보내 2 s 안에 응답, `:484-510`), 응답하면 "reusing" 하고 로컬 서버를 띄우지 않는다 (`:553-557`). 터널이 안 떠 있으면 `~/workspace/...` 의 로컬 레시피를 실행하려다 "checkpoint missing" 으로 죽는다 (`:559-563`). → **runner 보다 터널을 먼저** 띄울 것.
- `stop_strays()` 는 로컬 `pgrep` 만 하므로 원격 서버에 영향 없음 (`:526-551`).
- 정책 전환 키(`p`, `1`–`6`)도 같은 `ensure()` 를 타므로, 정책별 포트(5555/8001/8002/8003)를 각각 터널링해 두면 전환도 된다. 단 GPU 서버에서 여러 정책을 동시에 올려 두어야 한다(그들은 3090 한 장이라 하나씩 교체, `:527-528`).
- 예: `ssh -N -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 -L 8003:127.0.0.1:8003 -L 5555:127.0.0.1:5555 <gpu-server>`. HTTP 는 요청마다 새 연결(urllib, keep-alive 없음)이지만 터널 안에서는 채널 하나 여는 비용뿐이다. GR00T 의 ZMQ REQ/REP 도 TCP 라 `-L` 로 된다.
- 집 데스크탑에 필요한 것: Python + `torch`(CPU 면 충분, 매핑용), `opencv`, `Pillow`, `numpy`, `tkinter`, `lerobot`(motors.feetech), `pyzmq`+`msgpack`(HTTP 정책만 써도 `utils/groot.py` 가 import 됨), GUI 디스플레이, `v4l2-ctl`(wrist 컨트롤 파일이 있을 때), 폰을 overhead 로 쓰면 DroidCam + v4l2loopback. bench 패키지는 Isaac 없이 `benchmark`, `utils/*` 만 namespace 로 import 한다 (`:72-85`). 그들은 Isaac 용 Python 3.11 venv 에 LeRobot 을 넣어 실행했다. LeRobot(>=3.12) env 에서 그대로 도는지는 미확인.

### 5.2 요청 payload (추정, 미측정)

640×480 RGB 한 장: raw 921,600 B. PNG 는 무손실이라 실제 카메라 영상(센서 노이즈)에서는 대략 raw 의 50–80% → 0.45–0.75 MB, base64 로 ×4/3 → **장당 약 0.6–1.0 MB** (압축 불가 상한 1.23 MB).

| 정책 | 요청당 이미지 | 요청 크기(추정) | 쿼리 간격(로봇 시간) |
|---|---|---|---|
| Galaxea 0.5 | 2 | 1.2–2.0 MB | 15 스텝 = 0.5 s |
| π0.5 | 4 (live 2 + init 2) | 2.4–4.0 MB | 0.5 s |
| MolmoAct2 | 4 | 2.4–4.0 MB | 30 스텝 = 1.0 s |
| GR00T N1.7 (ZMQ, 무압축 npy) | 4 | 3.7 MB 고정 | 16 스텝 = 0.53 s |
| FLUX 3 | 2 + snapshot 2 | 2.4–4.0 MB | 32 스텝 = 1.07 s |

- init 프레임은 에피소드 내내 같은 이미지인데 매 요청 다시 보낸다 (`molmoact2.py:335-345`).
- 응답은 T×6 float JSON, 수 KB. 병목은 **집 → 서버 업로드**.
- SSH 압축(`-C`)은 PNG/base64 에 거의 효과 없음(GR00T 의 무압축 npy 에는 효과 있음).

업로드 시간 = 크기 ÷ uplink:

| uplink | 1.6 MB (G0.5) | 3.2 MB (π0.5/MolmoAct2) | 3.7 MB (GR00T) |
|---|---|---|---|
| 20 Mbit/s | 0.64 s | 1.28 s | 1.48 s |
| 50 Mbit/s | 0.26 s | 0.51 s | 0.59 s |
| 100 Mbit/s | 0.13 s | 0.26 s | 0.30 s |
| 500 Mbit/s | 0.03 s | 0.05 s | 0.06 s |

### 5.3 지연 예산

- **정확성 측면의 deadline 은 없다.** 쿼리는 동기식이고 그동안 팔은 hold, 스텝·제한시간은 멈춘다. HTTP timeout 은 120 s (`molmoact2.py:131`). 지연이 커도 정책이 받는 관측-행동 정합은 깨지지 않는다(관측을 찍은 뒤 팔이 안 움직이므로).
- 지연의 비용은 (a) wall-clock: 쿼리마다 정지, (b) "0.5 s 움직이고 멈춤" 의 stop-and-go 가 길어지는 것. 물체가 구르거나 미끄러지는 중이면 정지 시간만큼 실제와 sim(쿼리 중 physics 정지)이 어긋난다.
- 그들 셋업의 기준선: 서버 추론 median 이 π0.5 337 ms, MolmoAct2 697 ms, G0.5 792 ms (RTX A5000, 논문 Table 8; 실기는 로컬 RTX 3090). 즉 원본도 0.5 s 움직이고 0.3–0.8 s 서는 방식이었다. 실기 rollout median 38–59 s (제한 30 s 로봇 시간 대비, App. B Table 5).
- 권장 목표: **전송 오버헤드(인코드+업로드+RTT) ≤ 0.2 s/쿼리** — 원본의 정지 시간 편차 범위 안. 2-이미지 정책은 uplink ≈ 100 Mbit/s, 4-이미지 정책은 ≈ 150–200 Mbit/s 가 필요하다. 20–50 Mbit/s 가정용 uplink 면 쿼리당 0.3–1.5 s 가 더해져 정지 시간이 원본의 2–3배가 된다. 동작은 하지만 wall-clock 이 늘고 비교 조건이 달라진다.
- 4090 이 A5000/3090 보다 빨라 추론 시간이 줄면 그만큼 전송 예산이 생긴다(미측정).
- uplink 가 부족할 때의 선택지(전부 bench 프로토콜 이탈, 채택 전 사람 결정 필요):
  1. 클라이언트에서 JPEG(q≈90–95)로 인코딩: 서버 수정 없이 받는다(PIL 자동 판별). 요청이 약 1/5–1/10. 단 입력 분포가 달라진다(학습 데이터는 AV1 영상이긴 하다).
  2. init 프레임을 서버에 캐시하도록 양쪽 수정.
  3. runner 를 GPU 서버 쪽에 두고 카메라/시리얼만 원격화 — 30 Hz 모터 루프가 네트워크를 타므로 비권장.
- 백로그의 "SSH 터널 연결 테스트" 에서 잴 것: 실제 카메라 프레임의 PNG 크기, uplink, 전송 오버헤드[s] = `last_query_s` − `dt_ms / 1000`. `last_query_s` 는 runner 가 잰 쿼리 왕복(초, `:1439-1440`), `dt_ms` 는 HTTP 서버가 응답에 넣는 추론 시간(밀리초, 클라이언트가 로그로 출력 `molmoact2.py:382`). GR00T 는 `dt_ms` 가 없으므로 서버 쪽에서 따로 재야 한다.

## 6. 맞춰야 할 것 체크리스트

must = 안 맞으면 정책이 틀린 입력/출력을 보거나 sim↔real 비교가 성립하지 않음. nice = 순위 비교의 잡음을 줄이지만 학습 랜덤화 범위 안일 가능성.

### 관절 / 제어

- [ ] **must** 우리 팔의 sim alignment 측정: `so101_sim_alignment.py measure --robot_id <ours>` → `zero_tick` 6개. 이후 LeRobot 재캘리브레이션 금지(하면 재측정). (3.1)
- [ ] **must** bench 정책의 state/action 은 bench 고정 `.pos` 프레임(1.3). 우리 팔의 LeRobot `.pos` 를 직접 넣지 않는다. runner 의 `RealArm`(raw tick) 경로를 쓰거나 같은 변환을 구현.
- [ ] **must** 관절 순서 `pan, lift, elbow, wrist_flex, wrist_roll, gripper`, 모터 id 1–6, 부호 관례가 USD 와 같은지 `live`/`verify` 로 확인.
- [ ] **must** gripper 관례: 0 = 닫힘(jaw −12.75°), 100 = 115.5° 개방, jaw 각도 선형. gripper zero 는 "끝이 막 닿음 = −12°" 로 측정. EEPROM `Min_Position_Limit` 하한 준수(jaw 파손 방지).
- [ ] **must** 제어 30 Hz, 쿼리 중 hold. 쿼리당 실행 스텝은 **runner 의 `--action_horizon`** 이 정한다 (G0.5 15, π0.5 15, GR00T 16, MolmoAct2 30; `POLICY_DEFAULTS`). 서버 쪽 `--replan_steps` 는 그 플래그가 있는 서버(G0.5 15, InternW0 16, ImageWAM 16)에서만 같은 값으로 맞춘다. π0.5·MolmoAct2 서버에는 그 플래그가 없고 chunk 전체(30)를 돌려준다.
- [ ] **must** 에피소드는 dock pose `(-6.5, -98.8, 81.2, 71.2, -0.8, 0.9)` 에서 시작. init 프레임 쓰는 정책(GR00T, π0.5, MolmoAct2)은 dock 정지 상태에서 캡처.
- [ ] **must** 안전 한계 유지: 틱당 속도 clamp, 15°/틱 clamp, watchdog 25°/1 s.
- [ ] nice gripper stall 보정 측정: `so101_gripper_test.py` 를 인자 없이(필요하면 `--port`, `--robot_id`) 실행한 뒤 인터랙티브 프롬프트에서 `t` 입력. `t` 는 CLI 인자가 아니다 (`so101_gripper_test.py:34-38, :106-110`).
- [ ] nice 모터 P gain 이 LeRobot 기본(P=16) 인지 확인 — runner 는 gain 을 쓰지 않는다.
- [ ] nice `verify` 로 tracking error 기록(그들 sim actuator 는 실기에 맞춰 튜닝됨).

### 카메라

- [ ] **must** 카메라 역할/이름 `overhead`, `wrist`, 각각 640×480 RGB, 4:3 소스(아니면 crop 을 직접 넣어야 함 — runner 는 찌그러뜨린다).
- [ ] **must** overhead pose: world `(0, −0.34, 0.457)` m, 수직에서 약 51° 기울여 로봇을 마주 봄, 수평 FOV 74°. runner 의 layout/blend overlay 로 sim 뷰와 겹치게 조정. 레이아웃 재현이 이 정렬에 의존한다.
- [ ] **must** overhead 카메라는 연구 기간 내내 고정(드리프트 ≤ 3 px), 조명 고정(밝기 변화 ≤ 8%).
- [ ] **must** wrist 카메라 장착: gripper link 기준 `(−0.0025, 0.0595, −0.0605)` m, rpy `(−45, 0, −1)`°, roll 0 에서 로봇 왼쪽. 수평 FOV 72.5°. sim wrist 렌더(`view_wrist_camera.py`)와 나란히 비교.
- [ ] nice FOV 가 다르면 crop/resize 로 74° / 72.5° 에 맞춤 (학습에 intrinsics 랜덤화가 있으나 폭은 미집계).
- [ ] nice wrist 카메라 auto WB 끄고 고정, 테이블이 어둡게(wrist median ≈ 40/255) 나오도록 노출 설정.
- [ ] nice 렌즈 왜곡이 큰 카메라면 undistort (sim 은 pinhole).
- [ ] nice 카메라 지연/FPS ≥ 30, 최신 프레임만 사용(buffer 1).

### 장면

- [ ] **must** world frame 재현: 테이블 상면 = z 0, 로봇 base `(0.052, 0.181)` yaw 6.2°, 물체 영역 x (−0.14, 0.25) × y (−0.10, 0.155) m 가 overhead 시야 안.
- [ ] **must** (MMRV 비교 시) `tasks/mmrv_eval.jsonl` 45 에피소드 + `tasks/layouts/mmrv_eval_layouts.jsonl`, overlay 로 물체/bin 배치, 사람이 benchmark 규칙으로 채점, 제한시간 30 s / 75 s(제어 스텝 기준).
- [ ] nice 검은 테이블 상판. canonical 배경과 다른 것은 허용(그들 실기도 "학습에 없는 환경").
- [ ] nice bin: 비슷한 크기/색의 플라스틱 bin. 가벼우면 밀려서 실패가 늘어난다(논문 §5) — 무게를 더하는 것은 프로토콜 이탈이므로 결정 필요.
- [ ] nice 로봇 색: 주황-빨강이 canonical. 다른 색도 학습 랜덤화(6색)에 포함될 수 있음.
- [ ] nice 물체: benchmark 물체와 같은 class/색. 동일 실물 목록은 공개돼 있지 않다 → H1 을 "같은 45 레이아웃" 으로 재현하는 것은 물체 확보에 달려 있다.

### 데이터 / 소프트웨어

- [ ] **must** 우리 LeRobot 기록의 카메라 키 `observation.images.overhead` / `observation.images.wrist`, 30 fps, 480×640.
- [ ] **must** 우리 데이터를 bench 데이터와 섞거나 bench sim 에서 평가할 정책을 학습하려면 `.pos` 를 bench 프레임으로 변환해 저장(또는 처음부터 그 프레임으로 기록).
- [ ] **must** 터널을 runner 보다 먼저, 정책별 포트 일치.
- [ ] nice stateful 서버(G0.5, InternW0)는 클라이언트 하나만.

## 7. 사람만 답할 수 있는 질문 (집 셋업)

1. overhead 카메라 기종과 수평 FOV, 출력 종횡비(4:3 가능?), 장착 방식. `(0, −0.34, 0.457)` m·51° 기울임 위치에 고정할 수 있는 거치대가 있는가? 지금 "overhead" 가 천장 수직 뷰라면 다시 달아야 한다.
2. wrist 카메라 기종/FOV 와 마운트(SO-101 표준 wrist cam 마운트인가, 어느 쪽에 달렸나). Innomaker U20CAM-720P 가 아니면 FOV 차이를 crop 으로 맞출지.
3. 팔: 프린트 색, 모터(STS3215 7.4 V / 12 V), gripper/jaw 가 표준 부품인가, 현재 LeRobot 캘리브레이션 파일의 robot id. alignment 측정(수평계로 6자세, 약 20–30분)을 직접 해 줄 수 있는가 — 원격으로는 불가능한 유일한 단계.
4. 테이블: 상판 색/재질, base 를 clamp 한 위치 기준으로 앞 52 cm·좌우 40 cm 의 빈 영역이 나오는가. 검은 매트를 깔 수 있는가.
5. 조명: 고정 조명(창문 햇빛 차단 가능?)과 색온도.
6. 집 인터넷 **uplink** 속도와 GPU 서버까지 RTT. 100 Mbit/s 미만이면 JPEG 전송(프로토콜 이탈)을 허용할지.
7. 집 데스크탑: OS, GUI 세션 유무(runner 는 tkinter 창 + 키 입력이 필요 → 원격 조작 시 VNC/X forwarding 중 무엇?), `v4l-utils`·v4l2loopback 설치 가능 여부(apt, 사람 승인 필요), USB 포트/허브 구성.
8. 실기 평가 때 물체를 놓고 리셋하고 채점할 사람이 현장에 있는가. "원격 조작" 이 teleop 만인지, 평가 에피소드까지 무인으로 돌리려는 것인지. bench 프로토콜은 사람이 매 에피소드 배치·채점한다.
9. 물체: benchmark 물체 중 무엇을 가지고 있거나 살 수 있는가, bin 은 무엇을 쓸 것인가. H1(순위 유지)을 MMRV 45 레이아웃 그대로 할지, 우리 물체로 축소 세트를 새로 만들지.
10. 비상 정지: 전원 차단 스위치가 손 닿는 곳에 있는가. 원격 실행 중 watchdog 외의 물리적 정지 수단.

## 8. 감사에서 확인하지 못한 것

- 실제 PNG 크기, 터널 지연, 4090 에서의 추론 시간: 전부 미측정(5절 수치는 추정).
- runner 가 LeRobot(Python ≥ 3.12) 환경에서 그대로 도는지: 미실행.
- bin 과 테이블의 실제 치수, 실기 조명, 실물 물체 목록: 코드·논문에 없음. USD 에셋을 열면 bin 치수는 알 수 있다(T001 의 에셋 다운로드 이후).
- 학습 데이터의 카메라/조명 랜덤화 폭: `tasks/so101_bench_sim.jsonl` 집계 필요.
- 그들 wrist 카메라의 고정 컨트롤 값과 sim 참조 이미지(`sim_camera_reference`)는 레포에 없다.
