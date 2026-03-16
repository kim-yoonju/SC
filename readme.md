파이프라인 A: Online PPO (메인 학습)

1. bash models/run_ppo_online_agi.sh    # PPO iteration 1 실행
                                         # → models/ppo_online/checkpoint-1 저장

2. bash models/run_ppo_iter2.sh          # checkpoint-1에서 iteration 2 재시작
                                         # → models/ppo_online/checkpoint-2 저장

3. bash models/eval_iter2_parallel.sh    # checkpoint-2 최종 평가
                                         # → data/eval_results/summary_iter2.json
run_ppo_online_agi.sh와 scripts/run_ppo_online.sh는 동일한 역할, conda 경로만 다름 (agi 서버용 vs seoyoon 로컬용)

파이프라인 B: Offline REINFORCE (PPO 이후 추가 학습)

1. [파이프라인 A 완료 후]

2. bash models/run_train_offline.sh      # PPO rollout 데이터로 REINFORCE 학습
                                         # + 학습 후 자동으로 평가까지 수행
                                         # → models/reinforce_offline/ 저장
이 스크립트는 models/ppo_online/rollouts/ 에 있는 PPO 롤아웃 데이터를 입력으로 쓰기 때문에 반드시 파이프라인 A 이후에 실행해야 합니다.