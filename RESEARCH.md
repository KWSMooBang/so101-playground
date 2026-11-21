# so101-playground — RESEARCH

## 목표
- 여러 SO-101 benchmark·simulator·framework 를 한 레포에서 다루는 실험장.
- 집 SO-101 실기(overhead + wrist 카메라)를 원격으로 조작·평가하는 환경 구축.
- SO-101 Bench (https://so101-bench.github.io, github.com/5hadytru/so101_bench) 로 sim2real 검증.

## 구조 결정 (2026-11-21)
- 외부 코드는 `third_party/` 아래 git submodule, 커밋 pin. 업스트림 수정이 필요하면 fork 로 submodule 을 바꾼다 (vendoring 금지).
  - third_party/so101_bench (Isaac Lab ext, Isaac Sim 5.1, Python 3.11)
  - third_party/lerobot (Python >= 3.12)
- Python 버전이 달라 환경 2개: isaac(3.11) / lerobot(3.12). 우리 코드는 `src/`.
- 역할 분담: 실기 조작·데이터 수집·작은 정책(ACT·SmolVLA) = LeRobot. sim 평가·sim2real = SO-101 Bench (`POST /act` HTTP 정책 서버).
  우리 정책 → bench 의 LeRobot 플러그인(`lerobot-eval --env.type=so101_bench`)으로 sim 평가.
- 원격 조작: 팔(USB 시리얼)·카메라·robot client 는 집 데스크탑, 정책 추론은 GPU 서버. SSH -L 로 정책 포트(bench /act HTTP, LeRobot async gRPC) 터널링.

## 가설 / 질문
- H1: bench 의 "sim 학습 정책 순위가 real 에서 유지된다"(ρ=1.0, 5 policies) 가 그들의 digital twin 이 아닌 우리 집 셋업에서도 성립하는가.
- Q1: 공개 체크포인트(π0.5, G0.5, GR00T 등)의 우리 실기 zero-shot 성능.
- Q2: 카메라 배치·캘리브레이션을 bench twin 에 맞추면(so101_sim_alignment) 순위 상관이 얼마나 오르는가.

## 제약
- 클러스터 GPU: RTX 4090 24GB ×8 (공용). Isaac Sim 은 RT 코어 필요 → 4090 가능.
  bench 의 fine-tuning 레시피(4×A100-80GB, 36h)는 그대로 재현 불가 → LoRA·작은 정책 위주.
- 컨테이너: Ubuntu 22.04, driver 550.120, NVIDIA Vulkan ICD 없음 → Isaac Sim 동작 여부 T001 에서 확인. 안 되면 새 컨테이너.
- /workspace 디스크 여유 ~231G (공용).

## 백로그
- [부분] T001-scaffold-env: 골격·submodule·환경 2개 완료(exp/scaffold-env), Isaac 은 libGLU 없어 카메라 env 실패 → libglu1-mesa 설치
- [진행] T003-isaac-bench-smoke: T001 이어서 bench env headless step
- [완료] T002-bench-real-audit: docs/bench-real-setup.md (exp/bench-real-audit). 1순위 = so101_sim_alignment 로 우리 팔 관절 zero 측정(현장 작업). 집 셋업 질문 10개(§7) 답변 대기
- [ ] 집 데스크탑 robot client ↔ GPU 서버 정책 서버 SSH 터널 연결 테스트 (지연·대역폭 측정)
- [ ] bench sim 평가 baseline (공개 체크포인트 1~2개, 축소 split)
- [ ] 공개 체크포인트 실기 zero-shot (Q1)

## GPU 예산
- 미정. 셋업 단계(T001·T002)는 학습 없음, 확인용 GPU 1장 단시간.
