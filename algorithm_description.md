# Trajectory SFT Data Generation Algorithm

## 목적

Generator(SFT 학습 대상 모델)가 수학 문제를 틀렸을 때, Patcher(강력한 API 모델)가 첫 번째 오류 스텝 하나만 교정해준다. 이 교정-재시도 과정을 반복하여 최종 정답에 이르는 trajectory를 SFT 학습 데이터로 수집한다.

---

## 변수 정의

| 변수 | 설명 |
|------|------|
| $Q$ | 문제 (problem) |
| $A^*$ | 정답 (gold answer) |
| $H_t$ | 라운드 $t$ 시작 시 generator에게 주어지는 history 스텝 목록 |
| $n_t = \|H_t\|$ | history 스텝 수 (= step offset) |
| $G_t$ | 라운드 $t$에서 generator가 새로 생성한 스텝 목록 |
| $A_t$ | $G_t$의 마지막 스텝에서 추출한 generator의 예측 답 |
| $e_t$ | 라운드 $t$에서 patcher가 찾은 첫 번째 오류 스텝의 전체 인덱스 (1-based) |
| $C_t$ | $G_t$ 중 오류 이전까지의 정답 스텝: $G_t[\ :\ e_t - 1 - n_t]$ |
| $s_t^{\text{err}}$ | 오류 스텝: $(H_t \cup G_t)[e_t - 1]$, `is_error=True` 로 마킹 |
| $s_t^{\text{pat}}$ | patcher가 $e_t$ 위치를 교정한 단일 스텝, `is_first_pat=True` 로 마킹 |
| $B$ | mix_buf: 학습 데이터로 저장할 전체 스텝 누적 버퍼 |

---

## 알고리즘

```
초기화:
  H_1 ← []          (history, generator input context)
  B   ← []          (mix_buf, training data buffer)
  t   ← 1

라운드 루프 (t = 1, 2, ...):

  1. Generator 실행
     G_t ← Generator(Q, H_t)
     A_t ← extract_boxed(G_t[-1])

  2. 정답 확인
     if check_solved(A_t, A*):
       if t == 1:
         저장: traj_gen  ←  B(=∅) + G_t          # generator 단독 정답
       else:
         저장: traj_mix  ←  B + G_t               # gen+patcher 혼합 정답
       종료

  3. 오류 수정 (A_t ≠ A*)
     e_t, s_t^pat ← Patcher(Q, G_t, A*, step_offset=n_t)
       # patcher는 G_t만 검토, e_t는 전체 기준 1-based 인덱스로 반환

     C_t ← G_t[ : e_t - 1 - n_t ]
     s_t^err ← (H_t ∪ G_t)[e_t - 1]  with is_error=True

     # 학습 데이터 버퍼 갱신 (오류 스텝 포함)
     B ← B + C_t + [s_t^err, s_t^pat]

     # Generator context 갱신 (오류 스텝 제외, 교정본만)
     H_{t+1} ← H_t + C_t + [s_t^pat]

     # patcher 1스텝이 바로 정답인 경우
     if check_solved(s_t^pat, A*):
       저장: traj_mix ← B
       종료

     t ← t + 1
```

---

## 핵심 설계 원칙

### mix_buf B vs history H의 분리

| | mix_buf $B$ | history $H_{t+1}$ |
|---|---|---|
| 역할 | SFT 학습 데이터 | Generator의 다음 라운드 input |
| 오류 스텝 $s_t^{\text{err}}$ | **포함** | **제외** |
| 이유 | `<\|rethink\|>` 레이블 학습에 필요 | 오류를 보여주면 generator가 같은 실수 반복 |

→ Generator는 항상 "지금까지 올바르게 풀어온 흐름"만 봄

### SFT 레이블 (next_gold_action)

| 스텝 종류 | state | next_gold_action |
|-----------|-------|-----------------|
| 일반 gen 스텝 | `solve` | `<\|solve\|>` |
| 오류 gen 스텝 ($s_t^{\text{err}}$) | `solve` | `<\|rethink\|>` |
| patcher 교정 스텝 ($s_t^{\text{pat}}$), 비마지막 | `rethink_pat` | `<\|solve\|>` |
| 마지막 스텝 (어떤 종류든) | (위와 동일) | `<\|end\|>` |

### 스텝 레이블 (step 필드)

- Generator 스텝: `G_{pos:02d}` — 전체 trajectory 내 순서 위치
- Patcher 스텝: `P_{pos:02d}` — 오류가 발생한 위치($e_t$)와 동일 번호 부여

---

## 출력 trajectory 종류

| traj_type | 조건 | 내용 |
|-----------|------|------|
| `gen` | patcher 없이 $t=1$에서 정답 | $G_1$ 전체 |
| `mix` | patcher 개입 후 정답 | $B + G_t$ (최종 라운드 gen 포함) |
| `mix_intermediate` | 매 patcher 라운드 후 저장 | 현재 $B$ 상태 스냅샷 (`is_right=False`) |

---

## 파라미터

| 파라미터 | 값 | 의미 |
|----------|----|------|
| `MAX_STEPS` | 30 | 최대 라운드 수 |
| `MAX_API` | 10 | patcher API 최대 호출 횟수 |
| `TRAJ_MAX_NEW_TOKENS` | 4096 | generator 최대 생성 토큰 |
