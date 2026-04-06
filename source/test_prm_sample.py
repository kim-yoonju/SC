"""PRM 10개 샘플 테스트 - 점수 범위 및 동작 확인"""

import json, sys
from pathlib import Path
from collections import defaultdict

ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "output" / "train_ppo_data.jsonl"
CONFIG_PATH = ROOT / "config" / "config.yaml"
GPU_ID      = 4

sys.path.insert(0, str(ROOT / "source"))

# ── 데이터 로드 (처음 10개 유효 스텝)
def infer_state(step):
    if "state" in step:
        return step["state"]
    for v in (step.get("action",""), step.get("gold_next_action","") or "",
              step.get("predicted_next_action","") or ""):
        if "correct" in v:
            return "correct"
    return step.get("action", "")

samples = []
steps_by_problem = defaultdict(list)

with open(DATA_PATH) as f:
    for line in f:
        if len(samples) >= 50:   # 넉넉히 로드해서 히스토리 구성
            break
        d = json.loads(line)
        gold = str(d.get("gold_answer", ""))
        for step in d["steps"]:
            if step["text"] == "...":
                continue
            entry = {
                "problem_id":  d["problem_id"],
                "problem":     d["problem"],
                "gold_answer": gold,
                "step_idx":    step["step_idx"],
                "state":       infer_state(step),
                "text":        step["text"],
                "llm_reward":  step.get("llm_reward", 0.0),
            }
            steps_by_problem[d["problem_id"]].append(entry)
            if len(samples) < 10:
                samples.append(entry)

# ── PRM 로드
from inference_prm import get_prm_inference

OUT_PATH = ROOT / "output" / "prm_analysis" / "test_prm_sample.jsonl"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── 10개 채점
print(f"\n{'='*70}")
print(f"{'idx':>3}  {'problem_id':>10}  {'step':>4}  {'state':>12}  "
      f"{'prm':>6}  {'llm':>6}  {'diff':>6}")
print(f"{'-'*70}")

with open(OUT_PATH, "w") as out_f:
    for i, s in enumerate(samples):
        hist = [e["text"] for e in steps_by_problem[s["problem_id"]]
                if e["step_idx"] < s["step_idx"]]
        score = get_prm_inference(
            problem=s["problem"],
            current_step=s["text"],
            gold_answer=s["gold_answer"],
            history=hist,
            gpu_num=GPU_ID,
            config_file=str(CONFIG_PATH),
        )
        diff = score - s["llm_reward"]
        flag = "  ← LARGE" if abs(diff) > 0.4 else ""
        print(f"{i:>3}  {s['problem_id']:>10}  {s['step_idx']:>4}  {s['state']:>12}  "
              f"{score:>6.3f}  {s['llm_reward']:>6.3f}  {diff:>+6.3f}{flag}")
        print(f"     problem : {s['problem'][:80]!r}")
        print(f"     step    : {s['text'][:100]!r}")
        print()

        row = {
            "idx":         i,
            "problem_id":  s["problem_id"],
            "problem":     s["problem"],
            "gold_answer": s["gold_answer"],
            "step_idx":    s["step_idx"],
            "state":       s["state"],
            "history":     hist,
            "current_step": s["text"],
            "llm_reward":  s["llm_reward"],
            "prm_reward":  round(score, 4),
            "diff":        round(diff, 4),
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"{'='*70}")
print(f"저장 완료: {OUT_PATH}")
