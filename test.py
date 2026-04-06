import os
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model_only(config_path="config/config.yaml", model_id="Qwen/Qwen2.5-Math-72B-Instruct"):
    # 1. 설정 파일에서 캐시 경로 읽기
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cache_dir = config.get('checkpoint', {}).get('cache_dir', "/tmp")
    hf_token = os.getenv("HF_TOKEN")

    print(f"모델 다운로드를 시작합니다. (경로: {cache_dir})")
    print("이 작업은 모델 크기에 따라 수십 분이 소요될 수 있습니다.")

    # 2. 토크나이저 다운로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        token=hf_token, 
        cache_dir=cache_dir
    )

    # 3. 모델 다운로드 (device_map을 지정하지 않아 CPU에서 다운로드만 수행)
    # 가중치 파일만 로컬로 가져오기 위해 low_cpu_mem_usage를 활성화합니다.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype="auto",  # 메타데이터 확인용
        device_map="cpu"     # 의도적으로 CPU 할당
    )

    print(f"다운로드가 완료되었습니다: {cache_dir}")

if __name__ == "__main__":
    download_model_only()