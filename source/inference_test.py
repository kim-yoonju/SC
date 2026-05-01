import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import time

# 1. 모델 저장 경로 및 GPU 설정
os.environ["HF_HOME"] = "/mnt/.cache/huggingface"
# GPU 2개를 사용하기 위해 사용할 GPU 번호 2개를 지정합니다. (예: 6번, 7번)
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7" 

# 원본 70B 모델 ID (AWQ가 아닌 경우)
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
data_path = "/mnt/yoonju/SC/datasets/deepmath_16k/base_multi_reasoning_wrong.parquet"

# 2. 데이터 로드
df = pd.read_parquet(data_path)
test_problems = df['problem'].head(10).tolist()

print(f"\n모델 로딩 중: {model_id}")
print(f"저장 경로: {os.environ['HF_HOME']}")

load_start = time.time()
# 3. 모델 로드 (지피유 2개 분산 설정)
llm = LLM(
    model=model_id,
    tensor_parallel_size=2,      # GPU 2개에 모델을 쪼개서 올림
    max_model_len=8192,
    gpu_memory_utilization=0.9,  # 각 GPU당 메모리 점유율
    trust_remote_code=True,
    dtype="bfloat16",            # Llama-3/R1 계열은 bfloat16이 성능에 가장 좋습니다.
)
load_elapsed = time.time() - load_start
print(f"모델 로딩 완료: {load_elapsed:.1f}초")

# 4. 샘플링 설정 (수학 문제이므로 정답 고정을 위해 temperature=0 권장)
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=0.95,
    max_tokens=4096
)

# 5. 프롬프트 생성 (Chat Template 적용)
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for p in test_problems
]

print("\n--- 추론 시작 (Batch Inference) ---")

# 6. 배치 추론 실행
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
total_elapsed = time.time() - t0

# 7. 결과 출력
for i, output in enumerate(outputs):
    # vLLM의 output.outputs[0].text는 이미 디코딩된 텍스트를 포함합니다.
    print(f"\n[문제 {i+1} 결과]\n{output.outputs[0].text}")

print(f"\n=== 추론 요약 ===")
print(f"모델 로딩: {load_elapsed:.1f}초")
print(f"전체 추론 시간 (배치): {total_elapsed:.1f}초")
print(f"문제당 평균 시간: {total_elapsed / len(prompts):.1f}초")
print("\n--- 모든 추론 완료 ---")