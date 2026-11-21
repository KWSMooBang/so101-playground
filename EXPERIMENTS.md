# so101-playground — EXPERIMENTS

run 별 결과와 keep/discard 판단. 숫자는 /outputs/so101-playground/results.tsv 원문 인용.

| run_id | task | 내용 | 결과 | 판단 |
|---|---|---|---|---|
| T004-lawam-{none,gain,latency,offset,backlash,gain2,latency6,offset6} | T004 | LaWAM (bench full FT u45k), sim control gap 주입, Bin 1-obj 8 ep/조건, e_pred = LAM token MSE@35 | success 7/8,8/8,7/8,5/8,7/8,7/8,7/8,1/8; e_pred 0.0739→gain2 0.0982 (Δ+0.0213 [0.0039,0.0417]), latency6 0.0984 (Δ+0.0246 [0.0171,0.0327]); 약한 gap 은 CI 가 0 포함; 조건 평균 r(e_track,e_pred)=0.83 | keep: e_pred 는 gap 에 비례해 오르나 효과 작음. P1 reward 는 e_pred−e_ideal 쪽 권장 (docs/T004-wam-control-gap.md) |
| T004-flux3-{none,gain,latency,offset,backlash,gain2,latency6} | T004 | FLUX 3 Action (bench LoRA raw u3000), 같은 조건·에피소드, e_pred = DINO scene@16 (주), LPIPS/L1/DINO × scene/wrist | success 2/8,3/8,1/8,3/8,1/8,1/8,1/8; DINO scene@16 Δ +0.004~+0.011 (gain2·latency6 만 CI>0); LPIPS wrist@16 Δ +0.037/+0.034/+0.048/+0.051/+0.008/+0.060 (gain·offset·backlash·latency6 CI>0); e_ideal > e_pred 전 조건 | keep: FLUX 예측은 PD 추종 지연까지 학습 → e_ideal 기준선으로 부적합, e_exec 와 상관(r 0.75~0.81) |
