"""
PRM reward vs LLM reward 비교 분석 스크립트

- train_ppo_data.jsonl에서 1000개 샘플 추출
- text == '...' 스텝 제외
- 각 스텝에 MathShepherdPRM (Qwen 72B) 실행 → prm_reward
  · action_prompts.jsonl의 llm_score_sft 프롬프트 사용
  · 문제 + 이전 스텝 히스토리 + 현재 스텝 + gold_answer 입력
- llm_reward (GPT/Gemini 채점)와 비교
- scatter plot + 최적 threshold 탐색
- 결과를 prm_llm_pairs.jsonl로 저장
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_PATH   = ROOT / "output" / "train_ppo_data.jsonl"
PAIRS_PATH  = ROOT / "output" / "prm_analysis" / "prm_llm_pairs.jsonl"
OUT_DIR     = ROOT / "output" / "prm_analysis"
CONFIG_PATH = ROOT / "config" / "config.yaml"

DEFAULT_GPUS = [4, 5, 6, 7]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. 데이터 로드 및 스텝 추출
# ─────────────────────────────────────────────────────────────

def infer_state(step: dict) -> str:
    """state 필드가 없는 구버전 데이터는 action/gold_next_action으로 추론.
    'correct'가 포함된 필드가 하나라도 있으면 무조건 'correct'."""
    if "state" in step:
        return step["state"]
    action       = step.get("action", "") or ""
    gold_next    = step.get("gold_next_action", "") or ""
    pred_next    = step.get("predicted_next_action", "") or ""
    if any("correct" in v for v in (action, gold_next, pred_next)):
        return "correct"
    return action  # "solve" or "end"


def load_steps(n_samples: int = 1000):
    """JSONL에서 n_samples개 로드, '...' 제외하고 step 단위로 flatten.
    gold_answer도 함께 저장."""
    steps = []
    with open(DATA_PATH) as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            d = json.loads(line)
            gold_answer = str(d.get("gold_answer", ""))
            for step in d["steps"]:
                if step["text"] == "...":
                    continue
                llm_reward = step.get("llm_reward")
                if llm_reward is None:
                    continue
                steps.append({
                    "problem_id":  d["problem_id"],
                    "problem":     d["problem"],
                    "gold_answer": gold_answer,
                    "step_idx":    step["step_idx"],
                    "state":       infer_state(step),
                    "text":        step["text"],
                    "llm_reward":  llm_reward,
                    "is_right":    d.get("is_right", False),
                })
    return steps


def build_history_list(steps_by_problem: dict, problem_id: str, step_idx: int) -> list[str]:
    """problem_id의 step_idx 이전 스텝 텍스트 목록 반환."""
    traj = steps_by_problem.get(problem_id, [])
    return [s["text"] for s in traj if s["step_idx"] < step_idx]


# ─────────────────────────────────────────────────────────────
# 2. PRM 추론 (멀티 GPU 병렬, 스텝별 즉시 큐 전송)
# ─────────────────────────────────────────────────────────────

def _worker(gpu_id: int, all_steps: list, my_indices: list, result_queue) -> None:
    """GPU 워커: my_indices에 해당하는 스텝만 채점, 히스토리는 all_steps 전체 참조.
    채점 완료한 스텝을 즉시 큐에 넣고, 마지막에 None(완료 신호)을 전송."""
    import sys
    sys.path.insert(0, str(ROOT / "source"))
    from inference_prm import MathShepherdPRM
    from collections import defaultdict

    prm = MathShepherdPRM(config_path=str(CONFIG_PATH), gpu_id=gpu_id)

    # 전체 스텝으로 히스토리 맵 구성 (문제별 전체 스텝 순서 보존)
    steps_by_problem = defaultdict(list)
    for s in all_steps:
        steps_by_problem[s["problem_id"]].append(s)

    n = len(my_indices)
    for rank, idx in enumerate(my_indices):
        s = all_steps[idx]
        history = build_history_list(steps_by_problem, s["problem_id"], s["step_idx"])
        score = prm.get_step_score(
            problem=s["problem"],
            current_step=s["text"],
            gold_answer=s["gold_answer"],
            history=history,
        )
        result_queue.put((idx, score))   # 즉시 전송
        if (rank + 1) % 20 == 0:
            print(f"  [GPU {gpu_id}] {rank+1}/{n} 완료", flush=True)

    result_queue.put(None)  # 완료 신호


def run_prm_inference(steps: list, gpu_ids: list[int] = None) -> list:
    """steps를 gpu_ids에 균등 분배해 병렬 채점.
    메인 프로세스는 결과를 받는 즉시 PAIRS_PATH에 기록."""
    import multiprocessing as mp

    if gpu_ids is None:
        gpu_ids = DEFAULT_GPUS

    n_gpu   = len(gpu_ids)
    # 각 GPU가 담당할 steps 인덱스 (라운드로빈)
    index_chunks = [list(range(i, len(steps), n_gpu)) for i in range(n_gpu)]
    print(f"PRM 추론: {len(steps)}개 스텝 → GPU {gpu_ids} 분배 "
          f"(청크: {[len(c) for c in index_chunks]})")

    ctx = mp.get_context("spawn")
    q   = ctx.Queue()
    procs = []
    for gpu_id, indices in zip(gpu_ids, index_chunks):
        if not indices:
            continue
        p = ctx.Process(target=_worker, args=(gpu_id, steps, indices, q))
        p.start()
        procs.append(p)

    # 결과 수집 + 즉시 파일 기록
    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results    = [None] * len(steps)
    done_gpus  = 0
    total_done = 0

    with open(PAIRS_PATH, "w") as out_f:
        while done_gpus < len(procs):
            item = q.get()
            if item is None:
                done_gpus += 1
                continue
            idx, score = item
            s = steps[idx]
            row = {
                "problem_id": s["problem_id"],
                "step_idx":   s["step_idx"],
                "state":      s["state"],
                "llm_reward": s["llm_reward"],
                "prm_reward": float(score),
                "diff":       round(abs(float(score) - s["llm_reward"]), 4),
                "llm_good":   bool(s["llm_reward"] > 0.5),
                "is_right":   bool(s["is_right"]),
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            results[idx] = {**s, "prm_reward": float(score)}
            total_done += 1

    for p in procs:
        p.join()

    results = [r for r in results if r is not None]
    print(f"PRM 추론 완료: {len(results)}개 → {PAIRS_PATH}")
    return results


# ─────────────────────────────────────────────────────────────
# 3. 분석 및 시각화
# ─────────────────────────────────────────────────────────────

def analyze(results):
    llm = np.array([r["llm_reward"] for r in results])
    prm = np.array([r["prm_reward"]  for r in results])
    n   = len(results)

    llm_label = (llm > 0.5).astype(int)

    # ── 최적 PRM threshold 탐색
    thresholds = np.round(np.arange(0.0, 1.01, 0.01), 2)
    accuracies, precisions, recalls = [], [], []
    for t in thresholds:
        pred = (prm > t).astype(int)
        acc  = (pred == llm_label).mean()
        tp   = ((pred == 1) & (llm_label == 1)).sum()
        fp   = ((pred == 1) & (llm_label == 0)).sum()
        fn   = ((pred == 0) & (llm_label == 1)).sum()
        accuracies.append(acc)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(   tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    best_idx  = int(np.argmax(accuracies))
    best_t    = thresholds[best_idx]
    best_acc  = accuracies[best_idx]
    best_prec = precisions[best_idx]
    best_rec  = recalls[best_idx]

    pred_opt   = (prm > best_t).astype(int)
    agree_mask = pred_opt == llm_label

    # ── PRM score 분포 (0.0 / (0,1) / 1.0 비율)
    n_zero  = int((prm == 0.0).sum())
    n_one   = int((prm == 1.0).sum())
    n_mid   = n - n_zero - n_one

    # ── bin별 일치율
    bins       = np.arange(0, 1.1, 0.1)
    bin_labels = [f"[{bins[i]:.1f},{bins[i+1]:.1f})" for i in range(len(bins)-1)]
    bin_agree, bin_counts, bin_llm_pos = [], [], []
    for i in range(len(bins)-1):
        mask = (prm >= bins[i]) & (prm < bins[i+1])
        c = int(mask.sum())
        if c > 0:
            ag = float(agree_mask[mask].mean())
            lp = float(llm_label[mask].mean())
        else:
            ag, lp = float("nan"), float("nan")
        bin_agree.append(ag);  bin_counts.append(c);  bin_llm_pos.append(lp)

    ambiguous = []
    tp = int(((pred_opt==1) & (llm_label==1)).sum())
    fp = int(((pred_opt==1) & (llm_label==0)).sum())
    fn = int(((pred_opt==0) & (llm_label==1)).sum())
    tn = int(((pred_opt==0) & (llm_label==0)).sum())

    # ── 수치 요약
    L = []
    L += [f"{'='*58}",
          f" PRM vs LLM Reward Analysis  (n={n})",
          f"{'='*58}", ""]

    L += ["[Basic Statistics]",
          f"  llm_reward : mean={llm.mean():.3f}  std={llm.std():.3f}"
          f"  (>0.5: {llm_label.mean():.1%})",
          f"  prm_reward : mean={prm.mean():.3f}  std={prm.std():.3f}",
          f"  |prm-llm|  : mean={np.abs(prm-llm).mean():.3f}  max={np.abs(prm-llm).max():.3f}",
          ""]

    L += ["[PRM Score Distribution]",
          f"  prm == 0.0 : {n_zero:5d}  ({n_zero/n:.1%})",
          f"  0 < prm < 1: {n_mid:5d}  ({n_mid/n:.1%})",
          f"  prm == 1.0 : {n_one:5d}  ({n_one/n:.1%})",
          f"  → When prm==0: LLM>0.5 rate = "
          f"{llm_label[prm==0.0].mean():.1%}  (expected low)",
          f"  → When prm==1: LLM>0.5 rate = "
          f"{llm_label[prm==1.0].mean():.1%}  (expected high)" if n_one > 0 else
          f"  → When prm==1: n/a",
          ""]

    L += [f"[Optimal PRM Threshold]",
          f"  threshold : {best_t:.2f}  (sweep 0.00~1.00 step 0.01)",
          f"  accuracy  : {best_acc:.1%}",
          f"  precision : {best_prec:.1%}",
          f"  recall    : {best_rec:.1%}",
          ""]

    L += [f"[Confusion Matrix  (PRM>{best_t:.2f} vs LLM>0.5)]",
          f"              LLM<=0.5  |  LLM>0.5",
          f"  PRM<={best_t:.2f}    {tn:5d}    |    {fn:5d}   (TN / FN)",
          f"  PRM> {best_t:.2f}    {fp:5d}    |    {tp:5d}   (FP / TP)",
          ""]

    L += ["[Agreement Rate by PRM Bin  (* = ambiguous)]",
          f"  {'PRM bin':14s} {'n':>5s}  {'agree':>6s}  {'LLM>0.5':>7s}"]
    L += [f"  {'-'*40}"]
    for lb, cnt, ag, lp in zip(bin_labels, bin_counts, bin_agree, bin_llm_pos):
        flag = ""
        if not np.isnan(ag) and ag < 0.70 and cnt >= 5:
            flag = "  *"
            ambiguous.append(lb)
        ag_s = f"{ag:.1%}" if not np.isnan(ag) else "   n/a"
        lp_s = f"{lp:.1%}" if not np.isnan(lp) else "   n/a"
        L.append(f"  {lb:14s} {cnt:5d}  {ag_s:>6s}  {lp_s:>7s}{flag}")

    if ambiguous:
        L += ["", f"  Ambiguous PRM range: {', '.join(ambiguous)}"]
        L += [f"  → Use LLM recheck when prm in {ambiguous}"]

    # ── 2D count matrix (PRM bin x LLM bin)
    edges      = np.arange(0, 1.1, 0.1)          # 0.0, 0.1, ..., 1.0
    n_bins     = len(edges) - 1                   # 10
    count_mat  = np.zeros((n_bins, n_bins), dtype=int)
    for pv, lv in zip(prm, llm):
        pi = min(int(pv * 10), n_bins - 1)
        li = min(int(lv * 10), n_bins - 1)
        count_mat[li, pi] += 1                    # row=LLM, col=PRM

    bin_ticks  = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(n_bins)]

    # text summary of 2D matrix
    L += ["", "[2D Count Matrix  (row=LLM bin, col=PRM bin)]"]
    header = "  LLM\\PRM  " + "  ".join(f"{b:>7s}" for b in bin_ticks)
    L.append(header)
    L.append("  " + "-" * (len(header) - 2))
    for ri, row_label in enumerate(bin_ticks):
        row_vals = "  ".join(f"{count_mat[ri, ci]:>7d}" for ci in range(n_bins))
        L.append(f"  {row_label:8s}  {row_vals}")
    L.append(f"{'='*58}")

    summary = "\n".join(L)
    print(summary)
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write(summary)

    # ── 시각화 (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("PRM reward vs LLM reward", fontsize=14, fontweight="bold")

    # Panel (0,0): Scatter
    ax = axes[0, 0]
    ax.scatter(prm[ agree_mask], llm[ agree_mask], s=8, alpha=0.4,
               c="steelblue", label="agree")
    ax.scatter(prm[~agree_mask], llm[~agree_mask], s=8, alpha=0.4,
               c="tomato", label="disagree")
    ax.axvline(best_t, color="green",  ls="--", lw=1.2,
               label=f"PRM thr={best_t:.2f}")
    ax.axhline(0.5,    color="orange", ls="--", lw=1.2,
               label="LLM thr=0.5")
    ax.set_xlabel("prm_reward"); ax.set_ylabel("llm_reward")
    ax.set_title(f"Scatter  (agree {agree_mask.mean():.1%})")
    ax.legend(fontsize=8)

    # Panel (0,1): threshold sweep
    ax = axes[0, 1]
    ax.plot(thresholds, accuracies,  label="accuracy",  lw=1.5)
    ax.plot(thresholds, precisions,  label="precision", lw=1.2, ls="--")
    ax.plot(thresholds, recalls,     label="recall",    lw=1.2, ls=":")
    ax.axvline(best_t, color="green", ls="--", lw=1.2,
               label=f"best={best_t:.2f} ({best_acc:.1%})")
    ax.set_xlabel("PRM threshold"); ax.set_ylabel("score")
    ax.set_title("Threshold Sweep  (vs LLM>0.5)")
    ax.legend(fontsize=8); ax.set_ylim(0, 1)

    # Panel (0,2): PRM score 분포 (LLM label별)
    ax = axes[0, 2]
    bins20 = np.linspace(0, 1, 21)
    ax.hist(prm[llm_label == 0], bins=bins20, alpha=0.6,
            color="tomato",    label="LLM<=0.5 (bad)",  density=True)
    ax.hist(prm[llm_label == 1], bins=bins20, alpha=0.6,
            color="steelblue", label="LLM>0.5  (good)", density=True)
    ax.axvline(best_t, color="green", ls="--", lw=1.2,
               label=f"PRM thr={best_t:.2f}")
    ax.set_xlabel("prm_reward"); ax.set_ylabel("density")
    ax.set_title("PRM Score Dist by LLM Label")
    ax.legend(fontsize=8)

    # Panel (1,0): bin별 LLM>0.5 비율
    ax = axes[1, 0]
    bin_centers = (edges[:-1] + edges[1:]) / 2
    valid = [not np.isnan(lp) for lp in bin_llm_pos]
    vc = np.array(bin_centers)[valid]
    vl = np.array(bin_llm_pos)[valid]
    vn = np.array(bin_counts)[valid]
    bars = ax.bar(vc, vl, width=0.08, alpha=0.75,
                  color=["tomato" if l < 0.50 else "steelblue" for l in vl])
    ax.axhline(0.5, color="gray", ls=":", lw=1.0, label="50% baseline")
    ax.axvline(best_t, color="green", ls="--", lw=1.2,
               label=f"PRM thr={best_t:.2f}")
    for bar, cnt in zip(bars, vn):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                str(int(cnt)), ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("prm_reward bin"); ax.set_ylabel("LLM>0.5 rate")
    ax.set_title("LLM Good Rate by PRM Bin")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.1)

    # Panel (1,1)+(1,2): 2D count heatmap (큰 영역)
    ax = fig.add_subplot(2, 3, (5, 6))  # 5번, 6번 subplot 합침
    axes[1, 1].remove()
    axes[1, 2].remove()

    import matplotlib.colors as mcolors
    # 0인 셀은 흰색, 나머지는 Blues
    masked = np.ma.masked_where(count_mat == 0, count_mat)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("white")
    im = ax.imshow(masked, cmap=cmap, aspect="auto",
                   norm=mcolors.LogNorm(vmin=1, vmax=max(count_mat.max(), 1)))

    ax.set_xticks(range(n_bins)); ax.set_xticklabels(
        [f"{edges[i]:.1f}" for i in range(n_bins)], fontsize=8)
    ax.set_yticks(range(n_bins)); ax.set_yticklabels(
        [f"{edges[i]:.1f}" for i in range(n_bins)], fontsize=8)
    ax.set_xlabel("prm_reward  (bin left edge)")
    ax.set_ylabel("llm_reward  (bin left edge)")
    ax.set_title("2D Count Matrix  (row=LLM, col=PRM)  [log scale]")

    # 각 셀에 숫자 표기
    for ri in range(n_bins):
        for ci in range(n_bins):
            v = count_mat[ri, ci]
            if v > 0:
                color = "white" if v > count_mat.max() * 0.5 else "black"
                ax.text(ci, ri, str(v), ha="center", va="center",
                        fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="count (log scale)", shrink=0.8)

    plt.tight_layout()
    out_png = OUT_DIR / "prm_vs_llm.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nGraph saved: {out_png}")

    return float(best_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS,
                        help="GPU ID list (default: 4 5 6 7)")
    args = parser.parse_args()

    print(f"[1] Load data ({args.n_samples} samples)...")
    steps = load_steps(args.n_samples)
    print(f"    Valid steps: {len(steps)}  (excluded '...')")

    print(f"[2] PRM inference (GPU {args.gpus})...")
    print(f"    Writing results live to: {PAIRS_PATH}")
    results = run_prm_inference(steps, gpu_ids=args.gpus)

    print(f"[3] Analysis ({len(results)} steps)...")
    analyze(results)
