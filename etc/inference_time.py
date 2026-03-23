import os
import time
import statistics
from pathlib import Path

os.environ["HF_HOME"] = "/mnt/.cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-Math-7B"
MAX_TOKENS_LIST = [2048, 512, 256]
NUM_WARMUP = 2
NUM_RUNS = 5

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"

USER_PROMPTS = [
    "What is the sum of all integers from 1 to 100?",
    "Find all prime numbers less than 50.",
    "A train travels 120 km in 2 hours. What is its average speed?",
    "Simplify: (x^2 + 5x + 6) / (x + 2)",
    "Calculate the area of a circle with radius 7.",
]


def measure_inference_time(llm: LLM, prompts: list[str], sampling_params: SamplingParams) -> list[float]:
    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        llm.generate(prompts, sampling_params)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times


def build_prompts(system_prompt: str) -> list[str]:
    return [f"{system_prompt}\n\nProblem: {p}" for p in USER_PROMPTS]


def main():
    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()
    prompts = build_prompts(system_prompt)

    print(f"Loading model: {MODEL}")
    llm = LLM(model=MODEL, download_dir="/mnt/.cache", tensor_parallel_size=1)

    results = {}

    for max_tokens in MAX_TOKENS_LIST:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        print(f"\n[max_tokens={max_tokens}] Warming up ({NUM_WARMUP} runs)...")
        for _ in range(NUM_WARMUP):
            llm.generate(prompts, sampling_params)

        print(f"[max_tokens={max_tokens}] Measuring ({NUM_RUNS} runs)...")
        times = measure_inference_time(llm, prompts, sampling_params)
        results[max_tokens] = times

    print("\n" + "=" * 60)
    print(f"{'max_tokens':>12} | {'mean (s)':>10} | {'std (s)':>10} | {'min (s)':>10} | {'max (s)':>10}")
    print("-" * 60)
    for max_tokens, times in results.items():
        mean = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"{max_tokens:>12} | {mean:>10.3f} | {std:>10.3f} | {min(times):>10.3f} | {max(times):>10.3f}")
    print("=" * 60)

    baseline = statistics.mean(results[MAX_TOKENS_LIST[0]])
    print(f"\nSpeedup relative to max_tokens={MAX_TOKENS_LIST[0]}:")
    for max_tokens, times in results.items():
        mean = statistics.mean(times)
        speedup = baseline / mean
        print(f"  max_tokens={max_tokens}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
