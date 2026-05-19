"""
analyze_trajectory.py
trajectory jsonl 파일의 state 분포를 시각화.

사용법:
    python source/analyze_trajectory.py /mnt/yoonju/SC/output/sft_trajectory/20260513_040011/traj_all.jsonl
    python source/analyze_trajectory.py output/traj_sft_right_1241.jsonl --out output/analysis.png
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

STATE_ORDER = ["gen_solve", "gen_rethink", "pat_rethink", "gen_end", "pat_end",
               "rethink_solve", "rethink_rethink", "rethink_end"]
STATE_COLORS = {
    "gen_solve":      "#4CAF50",  # green (solve)
    "gen_rethink":    "#1565C0",  # dark blue (rethink)
    "pat_rethink":    "#64B5F6",  # light blue (rethink)
    "gen_end":        "#E65100",  # dark orange (end)
    "pat_end":        "#FFAB40",  # light orange (end)
    "rethink_solve":  "#66BB6A",  # light green
    "rethink_rethink": "#42A5F5", # blue
    "rethink_end":    "#FF7043",  # orange-red
}


def load_stats(path: str) -> list[dict]:
    stats = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            steps = d.get("steps", [])
            if steps and "source" in steps[0]:
                sources = [s["source"] for s in steps]
                states  = [s["state"]  for s in steps]
            else:
                sources = [s.get("role", "") for s in steps]
                states  = [f"{s.get('role', '')}_{s.get('next_pred_action', '')}" for s in steps]

            state_counts = Counter(states)

            n_patcher     = sum(1 for src in sources if src in ("patcher", "rethink"))
            n_gen         = sum(1 for src in sources if src == "gen")
            n_rethink     = sum(v for k, v in state_counts.items() if "rethink" in k)

            n_gen_solve   = state_counts.get("gen_solve", 0)
            n_gen_rethink = state_counts.get("gen_rethink", 0)
            n_gen_end     = state_counts.get("gen_end", 0)
            n_pat_rethink = state_counts.get("pat_rethink", 0)
            n_pat_end     = state_counts.get("pat_end", 0)
            n_total       = len(steps)
            fail          = d.get("fail_reason")
            is_right      = d.get("is_right", False)

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

            next_gold_actions = [
                s.get("next_gold_action") for s in steps
                if s.get("next_gold_action")
            ]
            gold_fail_rubrics = []
            for s in steps:
                gold_fail_rubrics.extend(s.get("gold_fail_rubrics") or [])

            stats.append({
                "n_gen":              n_gen,
                "n_patcher":          n_patcher,
                "n_rethink":          n_rethink,
                "n_total":            n_total,
                "category":           category,
                "is_right":           is_right,
                "state_counts":       dict(state_counts),
                "n_gen_solve":        n_gen_solve,
                "n_gen_rethink":      n_gen_rethink,
                "n_gen_end":          n_gen_end,
                "n_pat_rethink":      n_pat_rethink,
                "n_pat_end":          n_pat_end,
                "next_gold_actions":  next_gold_actions,
                "gold_fail_rubrics":  gold_fail_rubrics,
            })
    return stats


def _state_pie(ax, group: list[dict], title: str):
    totals = Counter()
    for s in group:
        for st, cnt in s["state_counts"].items():
            totals[st] += cnt

    ordered = [(st, totals[st]) for st in STATE_ORDER if totals.get(st, 0) > 0]
    for st, cnt in totals.items():
        if st not in STATE_ORDER and cnt > 0:
            ordered.append((st, cnt))

    if not ordered:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    labels = [l for l, _ in ordered]
    sizes  = [v for _, v in ordered]
    colors = [STATE_COLORS.get(l, "#9E9E9E") for l in labels]

    wedges, _, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=140, pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, [f"{l}  ({v})" for l, v in zip(labels, sizes)],
              loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=2)
    n = len(group)
    ax.set_title(f"{title}  (n={n})", fontsize=12, fontweight="bold")


def _cat_pie(ax, group: list[dict], title: str):
    cat_order  = ["gen_only", "rethink_only", "rethink+patcher", "max_steps", "patcher_fail"]
    cat_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]
    cat_counter = Counter(s["category"] for s in group)
    cat_labels  = [c for c in cat_order if c in cat_counter]
    cat_sizes   = [cat_counter[c] for c in cat_labels]
    color_map   = dict(zip(cat_order, cat_colors))
    colors      = [color_map[c] for c in cat_labels]

    n = len(group)
    ax.set_title(f"{title}  (n={n})", fontsize=12, fontweight="bold")
    if not cat_labels:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    wedges, _, autotexts = ax.pie(
        cat_sizes, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=140, pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, [f"{c}  ({v})" for c, v in zip(cat_labels, cat_sizes)],
              loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=2)


ACTION_COLORS = {
    "solve":   "#4CAF50",
    "rethink": "#1565C0",
    "end":     "#E65100",
    "patcher": "#FF9800",
}


def _action_pie(ax, stats: list[dict], title: str):
    totals: Counter = Counter()
    for s in stats:
        for a in s.get("next_gold_actions", []):
            if a:
                label = a.replace("<|", "").replace("|>", "")
                totals[label] += 1

    if not totals:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    labels = list(totals.keys())
    sizes  = list(totals.values())
    colors = [ACTION_COLORS.get(l, "#9E9E9E") for l in labels]

    wedges, _, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=140, pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, [f"{l}  ({v})" for l, v in zip(labels, sizes)],
              loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=2)
    ax.set_title(f"{title}  (n={sum(sizes)})", fontsize=12, fontweight="bold")


def _rubric_pie(ax, stats: list[dict], title: str):
    totals: Counter = Counter()
    for s in stats:
        for r in s.get("gold_fail_rubrics", []):
            totals[r] += 1

    if not totals:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    labels = [l for l, _ in totals.most_common()]
    sizes  = [totals[l] for l in labels]
    cmap   = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(labels))]

    wedges, _, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=140, pctdistance=0.8,
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax.legend(wedges, [f"{l}  ({v})" for l, v in zip(labels, sizes)],
              loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=7, ncol=2)
    ax.set_title(f"{title}  (n={sum(sizes)})", fontsize=12, fontweight="bold")


def plot(stats: list[dict], input_path: str, out_path: str | None):
    n = len(stats)
    correct   = [s for s in stats if s["is_right"]]
    incorrect = [s for s in stats if not s["is_right"]]

    total_cnt = Counter(s["n_total"] for s in stats)

    fig = plt.figure(figsize=(22, 15))
    fig.suptitle(f"{Path(input_path).name}  (n={n})", fontsize=14, fontweight="bold", y=0.99)

    # ── 1. state pie (correct) ────────────────────────────────────────────────
    ax1 = fig.add_subplot(3, 3, 1)
    _state_pie(ax1, correct, "state distribution  [correct]")

    # ── 2. state pie (incorrect) ──────────────────────────────────────────────
    ax2 = fig.add_subplot(3, 3, 2)
    _state_pie(ax2, incorrect, "state distribution  [incorrect]")

    # ── 3. trajectory category pie (correct) ─────────────────────────────────
    ax3 = fig.add_subplot(3, 3, 3)
    _cat_pie(ax3, correct, "trajectory category  [correct]")

    # ── 4. bubble chart (rethink vs patcher) ─────────────────────────────────
    ax4 = fig.add_subplot(3, 3, 4)
    bubble_cnt: Counter = Counter((s["n_rethink"], s["n_patcher"]) for s in stats)
    bx     = [k[0] for k in bubble_cnt]
    by     = [k[1] for k in bubble_cnt]
    bsize  = [bubble_cnt[k] for k in bubble_cnt]
    max_sz = max(bsize)
    scaled = [v / max_sz * 2000 for v in bsize]
    ax4.scatter(bx, by, s=scaled, alpha=0.5, color="#7B68EE", edgecolors="white", linewidths=0.5)
    for x, y, v in zip(bx, by, bsize):
        if v >= max(2, max_sz * 0.02):
            ax4.text(x, y, str(v), ha="center", va="center", fontsize=7, fontweight="bold")
    ax4.set_xlabel("# rethink steps", fontsize=10)
    ax4.set_ylabel("# patcher steps", fontsize=10)
    ax4.set_title("Rethink vs Patcher  (bubble size = count)", fontsize=11, fontweight="bold")
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax4.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax4.grid(True, linestyle="--", alpha=0.4)

    # ── 5. trajectory category pie (incorrect) ───────────────────────────────
    ax5 = fig.add_subplot(3, 3, 5)
    _cat_pie(ax5, incorrect, "trajectory category  [incorrect]")

    # ── 6. total steps distribution ───────────────────────────────────────────
    ax6 = fig.add_subplot(3, 3, 6)
    STEP_CLIP  = 20
    step_clip  = Counter({min(k, STEP_CLIP): v for k, v in total_cnt.items()})
    step_xs    = list(range(1, STEP_CLIP + 1))
    step_ys    = [step_clip.get(x, 0) for x in step_xs]
    step_xlbls = [str(x) if x < STEP_CLIP else f"{STEP_CLIP}+" for x in step_xs]
    ax6.bar(step_xlbls, step_ys, color="#4CAF50", edgecolor="white", linewidth=0.5)
    ax6.set_title("total steps distribution", fontsize=12, fontweight="bold")
    ax6.set_xlabel("total steps per trajectory", fontsize=10)
    ax6.set_ylabel("trajectories", fontsize=10)
    ax6.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax6.tick_params(axis="x", labelsize=7)

    # ── 7. next gold action pie (all steps) ──────────────────────────────────
    ax7 = fig.add_subplot(3, 3, 7)
    _action_pie(ax7, stats, "next gold action  [all steps]")

    # ── 8. fail rubrics pie (all steps) ──────────────────────────────────────
    ax8 = fig.add_subplot(3, 3, 8)
    _rubric_pie(ax8, stats, "gold fail rubrics  [all steps]")

    plt.tight_layout()

    if not out_path:
        out_path = str(Path(input_path).with_suffix(".png"))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="output/traj_sft_right_1241.jsonl",
                        help="traj jsonl 파일 경로")
    parser.add_argument("--out", default=None, help="출력 이미지 경로")
    args = parser.parse_args()

    stats = load_stats(args.input)
    n_correct = sum(1 for s in stats if s["is_right"])
    print(f"로드: {len(stats)}개 trajectory  (correct={n_correct}, incorrect={len(stats)-n_correct})")
    plot(stats, args.input, args.out)


if __name__ == "__main__":
    main()
