# so101-playground — EXPERIMENTS

run 별 결과와 keep/discard 판단. 숫자는 /outputs/so101-playground/results.tsv 원문 인용.

| run_id | task | 내용 | 결과 | 판단 |
|---|---|---|---|---|
| T004-lawam-{none,gain,latency,offset,backlash,gain2,latency6,offset6} | T004-wam-pred-gap-probe (H2-P0) | LaWAM (full FT u45k), sim Bin 1-obj 8 ep/조건, 주입 control gap. e_pred = LAM DINOv3 token MSE@35 | e_pred none 0.0739 → gain2 0.0982 (Δ+0.0213 [0.0039,0.0417]), latency6 0.0984 (Δ+0.0246 [0.0171,0.0327]); mild 조건 Δ CI 0 포함. 조건평균 r(e_track,e_pred)=0.83. paired observed−ideal: none −0.0066 [−0.0117,−0.0024] → latency6 +0.0095 [0.0042,0.0172]. e_static 0.13–0.16 (예측은 정보 있음). 0.15 s/query, 6.8 GiB | **keep** (약한 양성). nominal 에선 예측이 ideal 보다 실제 실행에 가까움 = WAM 이 sim 의 추종 지연까지 학습. 강한 gap 에서만 관측이 ideal 보다 덜 예측됨 → H2 서명이 보이나 효과 작음. P1 후보 모델 |
| T004-flux3-{none,gain,latency,offset,backlash,gain2,latency6} | T004-wam-pred-gap-probe (H2-P0) | FLUX3 bench LoRA raw u3000, 같은 조건 (offset6 미실행). e_pred = DINOv2 scene@16 | success 0.250/0.375/0.125/0.375/0.125/0.125/0.125. e_pred none 0.1384 → latency6 0.1486 (Δ+0.0112 [0.0016,0.0230]), gain2 Δ+0.0085 [0.0007,0.0164], 나머지 CI 0 포함. e_ideal_paired > e_pred_paired 전 조건 (none 0.1508 vs 0.1375). LPIPS wrist@16 가 더 민감(L2 보고 +0.037~0.060, CI>0). 1.25 s/query, 16.9 GB | **discard as baseline (보류)**. nominal 성공 2/8 로 너무 낮음 → eval harness 불일치 의심, 확인 전엔 해석 보류. 예측 비디오가 명령이 아니라 지연된 실제 실행을 따라감 → e_ideal 기준으로 부적합 |

T004 종합 (2026-11-21, 8.7 GPU-h / cap 10): **잠정** — codex 리뷰에서 e_ideal·pairing 편향 가능 blocker 6건 (ideal replay 가 clamp 우회·gain 조건 actuator 유지, RTX history, 재실행 행 혼입, FLUX terminal frame, LaWAM 캐시 충돌), L2 검증·재계산 중. H2 는 sim 에서 **부분 지지(잠정)**. WAM 예측은 학습 때의 dynamics(추종 지연 포함)를 따라가므로, control gap 은 "학습 분포 대비 추가 gap" 일 때만 e_pred 증가로 나타남. 효과는 모델 자체 예측 오차 대비 작고(LaWAM 강한 gap 에서 ~+30%), mild gap 은 8 ep 로 분리 불가.
next: (1) FLUX 낮은 nominal 성공 원인 확인 (bench 보고치 대비) (2) P1 은 LaWAM + 강한 gap(latency6·gain2) 에서, reward = −(e_pred − e_ideal) 또는 state 별 정규화, 에피소드 수 늘려 per-query reward SNR 먼저 확인 (3) T005 InternW0-Δ 는 T004 병합 후.
