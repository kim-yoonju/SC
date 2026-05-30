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
                "fail_reason":        fail,
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


FAIL_REASON_ORDER  = ["gen_wrong_answer", "patcher_wrong_answer", "patcher_fail", "max_steps", None]
FAIL_REASON_COLORS = {
    "gen_wrong_answer":     "#E53935",
    "patcher_wrong_answer": "#FF7043",
    "patcher_fail":         "#F44336",
    "max_steps":            "#9C27B0",
    None:                   "#78909C",
}
FAIL_REASON_LABELS = {
    "gen_wrong_answer":     "gen_wrong_answer",
    "patcher_wrong_answer": "patcher_wrong_answer",
    "patcher_fail":         "patcher_fail",
    "max_steps":            "max_steps",
    None:                   "no fail_reason",
}


def _fail_reason_pie(ax, group: list[dict], title: str):
    counter = Counter(s.get("fail_reason") for s in group)
    ordered = [(r, counter[r]) for r in FAIL_REASON_ORDER if counter.get(r, 0) > 0]
    for r, cnt in counter.items():
        if r not in FAIL_REASON_ORDER and cnt > 0:
            ordered.append((r, cnt))

    n = len(group)
    ax.set_title(f"{title}  (n={n})", fontsize=12, fontweight="bold")
    if not ordered:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    labels = [r for r, _ in ordered]
    sizes  = [v for _, v in ordered]
    colors = [FAIL_REASON_COLORS.get(l, "#9E9E9E") for l in labels]
    disp   = [FAIL_REASON_LABELS.get(l, str(l)) for l in labels]

    wedges, _, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=colors, startangle=140, pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.legend(wedges, [f"{d}  ({v})" for d, v in zip(disp, sizes)],
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


def plot(stats: list[dict], input_path: str, out_path: str | None,
         rethink_stats: tuple[dict, dict, dict] | None = None):
    n = len(stats)
    correct   = [s for s in stats if s["is_right"]]
    incorrect = [s for s in stats if not s["is_right"]]

    total_cnt = Counter(s["n_total"] for s in stats)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(f"{Path(input_path).name}  (n={n})", fontsize=14, fontweight="bold", y=0.99)
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. state pie (correct) ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _state_pie(ax1, correct, "state distribution  [correct]")

    # ── 2. state pie (incorrect) ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _state_pie(ax2, incorrect, "state distribution  [incorrect]")

    # ── 3. trajectory category pie (correct) ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _cat_pie(ax3, correct, "trajectory category  [correct]")

    # ── 4. bubble chart (rethink vs patcher) ─────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 2])
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
    ax4.set_xlabel("# rethink (G+P) steps", fontsize=10)
    ax4.set_ylabel("# patcher steps", fontsize=10)
    ax4.set_title("Rethink vs Patcher  (bubble size = count)", fontsize=11, fontweight="bold")
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax4.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax4.grid(True, linestyle="--", alpha=0.4)

    # ── 5. fail reason pie (incorrect) ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _fail_reason_pie(ax5, incorrect, "fail reason  [incorrect]")

    # ── 6. total steps distribution ───────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
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
    ax7 = fig.add_subplot(gs[2, 0])
    _action_pie(ax7, stats, "next gold action  [all steps]")

    # ── 8. fail rubrics pie (all steps) ──────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    _rubric_pie(ax8, stats, "gold fail rubrics  [all steps]")

    # ── 9. rethink success rate by rubric ────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    gen_succ, pat_succ, total_att = rethink_stats if rethink_stats else ({}, {}, {})
    _rethink_success_bar(ax9, gen_succ, pat_succ, total_att, "Rethink Success Rate by Rubric")

    if not out_path:
        out_path = str(Path(input_path).with_suffix(".png"))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")


RUBRICS = [
    "Progress and Non-Repetition", "Atomicity", "Algebraic Manipulation",
    "Differential Equations", "Function and Limit Analysis", "Calculus Computation",
    "Logical and Discrete Reasoning", "Abstract and Linear Algebra Operations",
    "Number Theoretic Reasoning", "Geometric Reasoning", "Counting and Probability",
]
RUBRIC_SHORT = {
    "Progress and Non-Repetition":          "P+NR",
    "Atomicity":                            "Atom",
    "Algebraic Manipulation":               "Alg",
    "Differential Equations":               "DE",
    "Function and Limit Analysis":          "F+L",
    "Calculus Computation":                 "Calc",
    "Logical and Discrete Reasoning":       "Logic",
    "Abstract and Linear Algebra Operations": "Abst",
    "Number Theoretic Reasoning":           "NTR",
    "Geometric Reasoning":                  "Geo",
    "Counting and Probability":             "C+P",
}
TRANSITION_COLORS = {
    "inc→inc": "#1565C0",
    "inc→cor": "#E53935",
    "inc→N/A": "#FB8C00",
    "cor→cor": "#43A047",
    "cor→N/A": "#B0BEC5",
    "cor→inc": "#6A1B9A",
}

# 수학 루브릭 a-z, 중복/비반복(10번째), atomicity(11번째)
RETHINK_RUBRIC_ORDER = [
    "Abstract and Linear Algebra Operations",
    "Algebraic Manipulation",
    "Calculus Computation",
    "Counting and Probability",
    "Differential Equations",
    "Function and Limit Analysis",
    "Geometric Reasoning",
    "Logical and Discrete Reasoning",
    "Number Theoretic Reasoning",
    "Progress and Non-Repetition",
    "Atomicity",
]


def _compute_critique_transitions(path: str) -> dict:
    """루브릭별 fast→deep 전환 카운트 반환.
    반환: {rubric: Counter({transition: count})}
    transition = "inc→inc" | "inc→cor" | "inc→N/A" | "cor→N/A"
    """
    def verd(raw): return "inc" if (raw or "").lower() in ("incorrect", "fail") else "cor"
    from collections import defaultdict
    counts = defaultdict(Counter)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            for step in traj.get("steps", []):
                fast = step.get("prm_fast_critique") or {}
                deep_list = step.get("prm_deep_critique") or []
                if not fast and not deep_list:
                    continue
                deep = {d["rubric"]: d for d in deep_list if d.get("rubric")}
                for r in RUBRICS:
                    if r not in fast:
                        continue
                    fv = verd(fast[r].get("verdict"))
                    dv_raw = deep[r].get("verdict") if r in deep else None
                    dv = verd(dv_raw) if dv_raw is not None else "N/A"
                    counts[r][f"{fv}→{dv}"] += 1
    return dict(counts)


def _critique_table(ax, counts: dict):
    """루브릭별 fast→deep 전환 분포 테이블."""
    rubrics = [r for r in RUBRICS if r in counts]
    ax.axis("off")
    ax.set_title("Fast→Deep Critique Transition", fontsize=11, fontweight="bold", pad=8)

    if not rubrics:
        ax.text(0.5, 0.5, "no critique data", ha="center", va="center", transform=ax.transAxes)
        return

    keys = ["inc→inc", "inc→cor", "inc→N/A", "cor→N/A"]
    col_labels = ["Rubric"] + keys

    rows = []
    for r in rubrics:
        c = counts[r]
        row = [RUBRIC_SHORT[r]] + [str(c.get(k, 0)) for k in keys]
        rows.append(row)

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="right",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_height(0.08)

    # 헤더 스타일
    HDR_COLORS = {"Rubric": "#37474F",
                  "inc→inc": "#1565C0", "inc→cor": "#C62828",
                  "inc→N/A": "#E65100", "cor→N/A": "#546E7A"}
    for j, lbl in enumerate(col_labels):
        cell = tbl[0, j]
        cell.set_facecolor(HDR_COLORS.get(lbl, "#37474F"))
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    # 행 교차 색상 + inc→cor 비율 높으면 강조
    for i, r in enumerate(rubrics):
        bg = "#F5F5F5" if i % 2 == 0 else "white"
        c = counts[r]
        total = sum(c.values())
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)
            tbl[i + 1, j].set_edgecolor("#E0E0E0")
        if total > 0 and c.get("inc→cor", 0) / total > 0.3:
            tbl[i + 1, keys.index("inc→cor") + 1].set_facecolor("#FFCDD2")


def _rethink_success_bar(ax, gen_success: dict, pat_success: dict,
                          total_attempts: dict, title: str):
    """루브릭별 rethink 성공률 누적 수평 막대 그래프.

    연주황색: generator가 rethink해서 직접 해결한 비율
    파란색: generator rethink 실패 후 patcher가 해결한 비율
    나머지: 실패 (미표시)
    """
    rubrics = [r for r in RETHINK_RUBRIC_ORDER if r in total_attempts]
    if not rubrics:
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.5, 0.5, "no rethink data", ha="center", va="center", transform=ax.transAxes)
        return

    total_att = [total_attempts.get(r, 0) for r in rubrics]
    gen_succ  = [gen_success.get(r, 0)    for r in rubrics]
    pat_succ  = [pat_success.get(r, 0)    for r in rubrics]
    gen_rates = [gs / ta if ta > 0 else 0 for gs, ta in zip(gen_succ, total_att)]
    pat_rates = [ps / ta if ta > 0 else 0 for ps, ta in zip(pat_succ, total_att)]
    total_rates = [g + p for g, p in zip(gen_rates, pat_rates)]

    short_labels = [RUBRIC_SHORT.get(r, r[:12]) for r in rubrics]
    n = len(rubrics)

    # generator 성공 (연주황) 먼저, patcher 성공 (파란) 누적
    ax.barh(range(n), gen_rates, color="#FFCC80", edgecolor="white", height=0.65, label="generator")
    ax.barh(range(n), pat_rates, left=gen_rates, color="#42A5F5",
            edgecolor="white", height=0.65, label="patcher")

    for i, (tr, ta, gs, ps) in enumerate(zip(total_rates, total_att, gen_succ, pat_succ)):
        ax.text(
            tr + 0.02, i,
            f"{tr:.0%}  (gen:{gs} pat:{ps} / {ta})",
            va="center", ha="left", fontsize=7.5,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Success Rate", fontsize=10)
    ax.set_xlim(0, 1.6)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()


def _get_fail_rubrics(step: dict) -> set[str]:
    """스텝의 gold_fail_rubrics를 정규화해서 반환."""
    rubrics = step.get("gold_fail_rubrics") or []
    if isinstance(rubrics, str):
        rubrics = [rubrics]
    return {r for r in rubrics if isinstance(r, str) and r not in ("<|none|>", "none", "")}


def _compute_rethink_rubric_stats(path: str) -> tuple[dict, dict, dict]:
    """루브릭별 gen_rethink / pat_rethink 시도·성공 횟수 집계.

    반환: (gen_success, pat_success, total_attempts)

    각 gen_rethink / pat_rethink 스텝 i에서:
        - 이전 평가 스텝의 gold_fail_rubrics = 이 rethink를 유발한 루브릭들 (시도)
        - 현재 스텝의 gold_fail_rubrics에서 사라진 루브릭 = rethink 성공 (is_fail=False)
        - 여전히 남아있는 루브릭 = rethink 실패 (is_fail=True)

    total_attempts = gen_attempts + pat_attempts
    """
    from collections import defaultdict
    gen_succ:  dict = defaultdict(int)
    pat_succ:  dict = defaultdict(int)
    total_att: dict = defaultdict(int)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj = json.loads(line)
            steps = traj.get("steps", [])
            n = len(steps)

            for i, step in enumerate(steps):
                state_i = step.get("state", "")
                if state_i not in ("gen_rethink", "pat_rethink"):
                    continue

                is_pat       = state_i == "pat_rethink"
                current_fail = _get_fail_rubrics(step)

                # 이전 평가 스텝 탐색 (source=rethink 중간 스텝 건너뜀)
                j = i - 1
                while j >= 0 and steps[j].get("source", steps[j].get("role", "")) == "rethink":
                    j -= 1
                if j < 0:
                    continue

                prev_fail = _get_fail_rubrics(steps[j])

                for rubric in prev_fail:
                    total_att[rubric] += 1
                    if rubric not in current_fail:
                        # rethink가 이 루브릭을 고침 → 성공 (is_fail=False)
                        if is_pat:
                            pat_succ[rubric] += 1
                        else:
                            gen_succ[rubric] += 1

    return dict(gen_succ), dict(pat_succ), dict(total_att)


def plot_rethink_failure_by_rubric(path: str, out_path: str | None = None):
    """루브릭별 rethink 실패율 수평 막대 그래프를 생성하고 저장."""
    gen_succ, pat_succ, total_att = _compute_rethink_rubric_stats(path)
    attempts = total_att
    failures = {r: total_att[r] - gen_succ.get(r, 0) - pat_succ.get(r, 0) for r in total_att}

    if not attempts:
        print("rethink 데이터가 없습니다 (next_gold_action=rethink 스텝 없음).")
        return

    rubrics = [r for r in RUBRICS if r in attempts]
    if not rubrics:
        rubrics = sorted(attempts.keys(), key=lambda r: -attempts[r])

    fail_rates     = [failures.get(r, 0) / attempts[r] for r in rubrics]
    attempt_counts = [attempts[r] for r in rubrics]
    fail_counts    = [failures.get(r, 0) for r in rubrics]
    short_labels   = [RUBRIC_SHORT.get(r, r[:12]) for r in rubrics]

    # 실패율 높은 순 정렬
    order = sorted(range(len(rubrics)), key=lambda i: -fail_rates[i])
    rubrics        = [rubrics[i]        for i in order]
    fail_rates     = [fail_rates[i]     for i in order]
    attempt_counts = [attempt_counts[i] for i in order]
    fail_counts    = [fail_counts[i]    for i in order]
    short_labels   = [short_labels[i]   for i in order]

    n = len(rubrics)
    cmap   = plt.get_cmap("RdYlGn_r")
    colors = [cmap(r) for r in fail_rates]

    fig, ax = plt.subplots(figsize=(13, max(5, n * 0.7 + 2)))
    bars = ax.barh(range(n), fail_rates, color=colors, edgecolor="white", height=0.65)

    for i, (bar, rate, fcnt, acnt) in enumerate(zip(bars, fail_rates, fail_counts, attempt_counts)):
        ax.text(
            min(bar.get_width() + 0.015, 1.02), i,
            f"{rate:.1%}  ({fcnt}/{acnt})",
            va="center", ha="left", fontsize=9,
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlabel("Rethink Failure Rate", fontsize=11)
    ax.set_xlim(0, 1.35)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_title(
        f"Rethink Failure Rate by Rubric\n{Path(path).name}",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Failure rate", pad=0.01)

    if not out_path:
        out_path = str(Path(path).with_suffix("")) + "_rethink_failure.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="output/traj_sft_right_1241.jsonl",
                        help="traj jsonl 파일 경로")
    parser.add_argument("--out", default=None, help="출력 이미지 경로")
    args = parser.parse_args()

    stats = load_stats(args.input)
    n_correct = sum(1 for s in stats if s["is_right"])
    print(f"로드: {len(stats)}개 trajectory  (correct={n_correct}, incorrect={len(stats)-n_correct})")
    rethink_stats = _compute_rethink_rubric_stats(args.input)
    plot(stats, args.input, args.out, rethink_stats=rethink_stats)


if __name__ == "__main__":
    main()
