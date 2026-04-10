### 모델 구조
# action
* solve
* rethink
* end

### 학습 순서
# SFT
* run_sft.sh # sft 진행
* generate_sft_sovle.py # 처음부터 끝까지 잘 푸는 데이터 생성
* generate_sft_rethink.py # 추론 중 1~3개의 스텝에서 실수하고 rethink하는 데이터 생성
* evaluate_sft.sh 

# PPO
* generate_trajectory.py # ppo할 때 쓰는 리워드 딸린 trajectory 생성

# PRM
* 무조건 api 호출해서 리워드 측정하는게 아니라 일단 PRM 쓰고, 애매하다 싶으면 api 호출


SFT
sft 할 때는 리워드가 없어도 되니까 patcher 모델에게 처음부터 끝까지 trajectory를 생성해 달라고 함



### trajectory

문제는 여러 스텝으로 나눠서 풀 수 있음
매 스텝마다 state, gold_action, pred_action이 존재함
state_list: [solve, correct_gen, correct_pat, end_max, end_answer]
action_list: [<|solve|>, <|correct|>, <|end|>]


evaluate_single_reasoning.py
모델한테 그냥 추론해~ 시켰을 때의 성능
Format 정확도: boxed{} 같이 정해진 형식으로, 룰기반으로 추출하는 성능
LLM 평가 정확도: llm한테 정답 추출해줘. 정답이랑 추출한거랑 맞나 봐봐 시켰을때 (ex. 2.5와 5/2 비교)



evaluate_step_reasoninig.py
end 또는 solve, boxed{} → 종료



generate_trajectory.py
solve, llm_reward>0.5 → solve
          llm_reward<=0.5 → correct_gen
correct_gen, llm_reward>0.5 → solve
                  llm_reward<=0.5 → correct_pat
correct_pat, llm_reward>0.5 → solve
                  llm_rewardr<=0.5 → end_pat
end or have_boxed{}, llm_reward>0.5 → end_answer
        llm_reward<=0.5 → correct_gen

if step_idx>max_steps: state=end_max


이 때
solve -> <|solve|>
correct_gen, correct_pat -> <|correct|>
end_max, end_answer, end_pat -> <|end|>

Rule 채점 + LLM 채점
boxed{}는 스텝 상관없이 마지막으로 나온게 정답
gen은 generator로 문제를 푸는 모델
pat은 patcher로 대신 한 스텝만 풀어주는 API 모델
매 스텝마다 모델의 추론에서 다음 액션을 추출하거나 없으면 세개의 action 중에 가장 확률이 높은걸 생성, 해당 액션을 따라 다음 스텝 진행
end_answer로 끝난 trajectory만 ppo tuning에 사용함

R_PRM (잘 풀었는지): 0~1.0
R_format (boxed{} 형식을 잘 지켰는지): 0.0 or 0.1
하나의 스텝에서 512 토큰 이후로 매 토큰마다 0.0002씩 패널티를 주기

여기서 마지막 special token에 한해서만 cross entropy loss를 쓸 지 그냥 ppo 를 쓸지 -> 우선 ppo 로 하되 나중에 수정하자


B. inp_ids 왼쪽 truncate	update 시 inp_ids[:, -MAX_INP_LEN:]

SFT를 제대로 하기 위해서 generate_trajectory -> filtering
