"""
analyze_trajectory.py
trajectory jsonl 파일의 rethink / patcher 사용 분포를 시각화.

사용법:
    python source/analyze_trajectory.py /mnt/yoonju/SC/output/sft_trajectory/20260513_040011/traj_all.jsonl
    python source/analyze_trajectory.py output/traj_sft_right_1241.jsonl --out output/analysis.png
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# True면 is_right=True인 trajectory만 분석에 사용
CORRECT_FILTERING = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_stats(path: str) -> list[dict]:
    stats = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if CORRECT_FILTERING and not d.get("is_right", False):
                continue
            steps   = d.get("steps", [])
            sources = [s["source"] for s in steps]
            states  = [s["state"]  for s in steps]

            n_rethink    = sum(1 for st in states  if st == "rethink")
            n_rethink_ok = n_rethink  # state=="rethink"이 모든 시도 포함
            n_patcher    = sum(1 for src in sources if src == "patcher")
            n_gen        = sum(1 for src, st in zip(sources, states)
                               if src == "gen" and st != "rethink")
            n_total   = len(steps)
            fail      = d.get("fail_reason")
            is_right  = d.get("is_right", False)

            if fail == "max_steps":
                category = "max_steps"
            elif fail == "patcher_fail":
                category = "patcher_fail"
            elif n_rethink == 0 and n_patcher == 0:
                category = "gen_only"
            elif n_patcher == 0:
                category = "rethink_only"
            else:
                category = "rethink+patcher"

            stats.append({
                "n_gen":          n_gen,
                "n_rethink":      n_rethink,      # state=="rethink"인 모든 시도
                "n_rethink_ok":   n_rethink_ok,   # (= n_rethink, state 기반)
                "n_patcher":      n_patcher,
                "n_total":        n_total,
                "category":       category,
                "is_right":       is_right,
            })
    return stats


def _bar(ax, counter: Counter, title: str, xlabel: str, color: str, max_x: int | None = None):
    if not counter:
        return
    max_val = max_x or max(counter)
    xs = list(range(max_val + 1))
    ys = [counter.get(x, 0) for x in xs]
    bars = ax.bar(xs, ys, color=color, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("trajectories", fontsize=10)
    ax.set_xticks(xs)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for bar, y in zip(bars, ys):
        if y > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.3, str(y),
                    ha="center", va="bottom", fontsize=8)


def plot(stats: list[dict], input_path: str, out_path: str | None):
    n = len(stats)
    cat_counter = Counter(s["category"]  for s in stats)
    rethink_cnt = Counter(s["n_rethink"] for s in stats)
    patcher_cnt = Counter(s["n_patcher"] for s in stats)
    total_cnt   = Counter(s["n_total"]   for s in stats)

    CLIP = 10
    def clip(cnt: Counter) -> Counter:
        out = Counter()
        for k, v in cnt.items():
            out[min(k, CLIP)] += v
        return out

    fig = plt.figure(figsize=(22, 10))
    filter_tag = "  [correct only]" if CORRECT_FILTERING else ""
    fig.suptitle(f"{Path(input_path).name}  (n={n}){filter_tag}", fontsize=14, fontweight="bold", y=0.98)

    # ── 1. step-type pie ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 3, 1)
    total_patcher = int(sum(s["n_patcher"] for s in stats))
    total_rethink = int(sum(s["n_rethink"] for s in stats))
    total_gen     = int(sum(s["n_gen"]     for s in stats))
    step_labels = ["gen", "rethink", "patcher"]
    step_sizes  = [total_gen, total_rethink, total_patcher]
    step_colors = ["#4CAF50", "#2196F3", "#FF9800"]
    step_labels, step_sizes, step_colors = zip(
        *[(l, v, c) for l, v, c in zip(step_labels, step_sizes, step_colors) if v > 0]
    )
    wedges, _, autotexts = ax1.pie(
        step_sizes, labels=None, autopct="%1.1f%%",
        colors=step_colors, startangle=140, pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax1.legend(wedges, [f"{l}  ({v})" for l, v in zip(step_labels, step_sizes)],
               loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=9, ncol=1)
    ax1.set_title("step type distribution (all steps)", fontsize=12, fontweight="bold")

    # ── 2. trajectory category pie ────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 3, 2)
    cat_order  = ["gen_only", "rethink_only", "rethink+patcher", "max_steps", "patcher_fail"]
    cat_labels = [c for c in cat_order if c in cat_counter]
    cat_sizes  = [cat_counter[c] for c in cat_labels]
    cat_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"][:len(cat_labels)]
    wedges2, _, autotexts2 = ax2.pie(
        cat_sizes, labels=None, autopct="%1.1f%%",
        colors=cat_colors, startangle=140, pctdistance=0.75,
    )
    for at in autotexts2:
        at.set_fontsize(8)
    ax2.legend(wedges2, [f"{c}  ({v})" for c, v in zip(cat_labels, cat_sizes)],
               loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=2)
    ax2.set_title("trajectory category", fontsize=12, fontweight="bold")

    # ── 3. bubble chart (clip=10) ─────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 3, 3)
    bubble_cnt: Counter = Counter((s["n_rethink"], s["n_patcher"]) for s in stats)
    bx      = [k[0] for k in bubble_cnt]
    by      = [k[1] for k in bubble_cnt]
    bsize   = [bubble_cnt[k] for k in bubble_cnt]
    max_sz  = max(bsize)
    scaled  = [v / max_sz * 2000 for v in bsize]
    ax3.scatter(bx, by, s=scaled, alpha=0.5, color="#7B68EE", edgecolors="white", linewidths=0.5)
    for x, y, v in zip(bx, by, bsize):
        if v >= max(2, max_sz * 0.02):
            ax3.text(x, y, str(v), ha="center", va="center", fontsize=7, fontweight="bold")
    ax3.set_xlabel("# rethink attempts (success + failed)", fontsize=10)
    ax3.set_ylabel("# patch steps (P+)", fontsize=10)
    ax3.set_title("Rethink attempts vs Patch  (bubble size = count)", fontsize=11, fontweight="bold")
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax3.grid(True, linestyle="--", alpha=0.4)

    # ── 4. total steps distribution ───────────────────────────────────────────
    ax5 = fig.add_subplot(2, 3, 4)
    STEP_CLIP   = 20
    step_clip   = Counter({min(k, STEP_CLIP): v for k, v in total_cnt.items()})
    step_xs     = list(range(1, STEP_CLIP + 1))
    step_ys     = [step_clip.get(x, 0) for x in step_xs]
    step_labels = [str(x) if x < STEP_CLIP else f"{STEP_CLIP}+" for x in step_xs]
    ax5.bar(step_labels, step_ys, color="#4CAF50", edgecolor="white", linewidth=0.5)
    ax5.set_title("total steps distribution", fontsize=12, fontweight="bold")
    ax5.set_xlabel("total steps per trajectory", fontsize=10)
    ax5.set_ylabel("trajectories", fontsize=10)
    ax5.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax5.tick_params(axis="x", labelsize=7)

    # ── 5. step-level stats table ─────────────────────────────────────────────
    ax6 = fig.add_subplot(2, 3, 5)
    ax6.axis("off")

    gv  = np.array([s["n_gen"]        for s in stats])
    rv  = np.array([s["n_rethink"]    for s in stats])
    rok = np.array([s["n_rethink_ok"] for s in stats])
    pv  = np.array([s["n_patcher"]    for s in stats])
    tv  = np.array([s["n_total"]      for s in stats])

    col_labels = ["min", "mean", "median", "max"]
    row_labels = ["gen", "rethink(try)", "rethink(ok)", "patcher", "total"]
    rows = []
    for arr in [gv, rv, rok, pv, tv]:
        rows.append([
            f"{arr.min()}",
            f"{arr.mean():.2f}",
            f"{np.median(arr):.1f}",
            f"{arr.max()}",
        ])

    tbl = ax6.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 2.0)

    # 헤더 / row label 색상
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#DDEBF7")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 == 1 else "#F5F5F5")

    ax6.set_title("steps per trajectory  (gen / rethink / patcher)", fontsize=10, fontweight="bold")

    plt.tight_layout()

    if not out_path:
        out_path = str(Path(input_path).with_suffix(".png"))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="output/traj_sft_right_1241.jsonl",
                        help="traj jsonl 파일 경로 (기본: output/traj_sft_right_1241.jsonl)")
    parser.add_argument("--out", default=None, help="출력 이미지 경로 (없으면 화면 출력)")
    args = parser.parse_args()

    stats = load_stats(args.input)
    filter_msg = " (correct only)" if CORRECT_FILTERING else ""
    print(f"로드: {len(stats)}개 trajectory{filter_msg}")
    plot(stats, args.input, args.out)


if __name__ == "__main__":
    main()
