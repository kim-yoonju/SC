"""
deepseek-chat 5000개 동시 호출 rate limit 테스트 (asyncio)
"""
import sys
import time
import asyncio
from collections import Counter

import httpx
sys.path.insert(0, "/mnt/yoonju/SC/source")
from utils import DEEPSEEK_API_KEY

from openai import AsyncOpenAI

MODEL     = "deepseek-chat"
N_CALLS   = 5000
TIMEOUT   = 30  # 요청당 최대 대기 초
LOG_EVERY = 500

# 연결 풀을 5000개로 확대
client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    http_client=httpx.AsyncClient(
        limits=httpx.Limits(max_connections=5000, max_keepalive_connections=5000),
        timeout=TIMEOUT,
    ),
)

async def call_one(i: int) -> str:
    try:
        await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "1+1="}],
            max_tokens=8,
        )
        return "ok"
    except Exception as e:
        err = str(e)
        if "429" in err or "rate" in err.lower():
            return "rate_limit"
        elif "timeout" in err.lower() or "timed out" in err.lower():
            return "timeout"
        else:
            return f"error:{err[:80]}"

async def main():
    print(f"[*] {N_CALLS}개 동시 호출  model={MODEL}  timeout={TIMEOUT}s")
    t0 = time.time()
    done = 0
    results = []

    # as_completed처럼 끝나는 순서대로 결과 수집
    tasks = [asyncio.create_task(call_one(i)) for i in range(N_CALLS)]
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        done += 1
        if done % LOG_EVERY == 0 or done == N_CALLS:
            counts = Counter(results)
            print(f"  [{done}/{N_CALLS}]  {dict(counts)}  ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    counts = Counter(results)
    print(f"\n[완료] {elapsed:.1f}s")
    print(f"  ok:         {counts.get('ok', 0)}")
    print(f"  rate_limit: {counts.get('rate_limit', 0)}")
    print(f"  timeout:    {counts.get('timeout', 0)}")
    for k, v in counts.items():
        if k not in ("ok", "rate_limit", "timeout"):
            print(f"  {k}: {v}")

asyncio.run(main())
