"""
Analyze and visualize the distribution of steps and token counts
from model reasoning results.

Usage:
python etc/plot_reasoning_distribution.py --input output/eval_results/iter_0003_20260327_142449/math500/worker_0.jsonl
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(path: str):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_stats(records):
    n_steps = []
    total_tokens = []
    tokens_per_step = []  # flattened per-step token counts
    mean_tokens_per_step = []  # per-problem mean tokens per step
    correct = []

    for r in records:
        n_steps.append(r["n_steps"])
        total = sum(r["token_counts"])
        total_tokens.append(total)
        tokens_per_step.extend(r["token_counts"])
        mean_tokens_per_step.append(np.mean(r["token_counts"]))
        correct.append(r.get("correct", r.get("llm_correct", False)))

    return {
        "n_steps": np.array(n_steps),
        "total_tokens": np.array(total_tokens),
        "tokens_per_step": np.array(tokens_per_step),
        "mean_tokens_per_step": np.array(mean_tokens_per_step),
        "correct": np.array(correct, dtype=bool),
    }



def plot_hist(arr, label, ax, bins=40, color="#4C72B0", log_y=False):
    ax.hist(arr, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    ax.axvline(arr.mean(), color="#DD4444", linewidth=1.5, linestyle="--", label=f"mean={arr.mean():.1f}")
    ax.axvline(np.median(arr), color="#44AA44", linewidth=1.5, linestyle=":", label=f"median={np.median(arr):.1f}")
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=9)
    if log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_step_bar(n_steps, ax):
    values, counts = np.unique(n_steps, return_counts=True)
    bars = ax.bar(values, counts, color="#5599CC", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Number of steps", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xticks(values)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center", va="bottom", fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to worker .jsonl file")
    parser.add_argument("--output", "-o", default=None, help="Save figure to this path (default: same dir as input)")
    args = parser.parse_args()

    records = load_data(args.input)
    stats = extract_stats(records)

    n = len(records)
    n_steps = stats["n_steps"]
    total_tokens = stats["total_tokens"]
    tokens_per_step = stats["tokens_per_step"]
    mean_tps = stats["mean_tokens_per_step"]
    correct = stats["correct"]

    # ── Layout: 2 rows × 3 cols ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Reasoning Distribution  (n={n} problems)\n{Path(args.input).parent.name}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.subplots_adjust(wspace=0.35, hspace=0.45)

    ax_steps, ax_tok_h, ax_stp_h = axes[0]
    ax_sc_cor, ax_sc_wrg, ax_empty = axes[1]

    # ── Row 0: distribution plots ─────────────────────────────────────────────
    plot_step_bar(n_steps, ax_steps)
    ax_steps.set_title("Steps per problem", fontsize=11)

    plot_hist(total_tokens, "Total tokens / problem", ax_tok_h, bins=50, color="#4C72B0")
    ax_tok_h.set_title("Total tokens per problem", fontsize=11)

    plot_hist(tokens_per_step, "Tokens per step", ax_stp_h, bins=50, color="#DD8833")
    ax_stp_h.set_title("Tokens per step (all steps)", fontsize=11)

    # ── Row 1: scatter — n_steps vs mean tokens/step (correct / wrong) ────────
    jitter = 0.15
    for ax, mask, label, color in [
        (ax_sc_cor, correct,  "Correct",   "#44AA44"),
        (ax_sc_wrg, ~correct, "Incorrect", "#DD4444"),
    ]:
        xs = n_steps[mask] + np.random.default_rng(0).uniform(-jitter, jitter, mask.sum())
        ys = mean_tps[mask]
        ax.scatter(xs, ys, alpha=0.45, s=25, color=color, edgecolors="none")
        # per-step-count mean line
        for s in np.unique(n_steps[mask]):
            grp = mean_tps[mask][n_steps[mask] == s]
            ax.plot(s, grp.mean(), marker="D", markersize=7,
                    color="black", zorder=5)
        ax.set_xlabel("Number of steps", fontsize=11)
        ax.set_ylabel("Mean tokens per step", fontsize=11)
        ax.set_xticks(np.unique(n_steps))
        ax.set_title(f"Steps vs Mean tokens/step  ({label})", fontsize=11)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n_cor = correct.sum()
    n_wrg = (~correct).sum()
    ax_empty.text(
        0.5, 0.5,
        f"Correct  : {n_cor} ({100*n_cor/n:.1f}%)\nIncorrect: {n_wrg} ({100*n_wrg/n:.1f}%)",
        transform=ax_empty.transAxes, fontsize=12, va="center", ha="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#F5F5F5", edgecolor="#AAAAAA"),
    )
    ax_empty.axis("off")

    # ── Save / show ──────────────────────────────────────────────────────────
    if args.output is None:
        out_path = Path(args.input).parent / "reasoning_distribution.png"
    else:
        out_path = Path(args.output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
