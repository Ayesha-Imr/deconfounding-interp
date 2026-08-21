"""Figure 6: Stability ranking dot plot — consistent across models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.theme import (
    CAPTION_GRAY,
    CORAL,
    MEDIUM_GRAY,
    MODEL_DISPLAY,
    MODEL_MARKERS,
    SLATE_BLUE,
    TRAIT_DISPLAY,
    add_caption,
    apply_theme,
    savefig,
)


def plot(data_dir: Path, output_dir: Path) -> None:
    apply_theme()

    df = pd.read_csv(data_dir / "phase2" / "stability_summary.csv")
    df = df[df["status"] == "completed"].copy()

    # Sort by average stability across models
    avg_stab = df.groupby("trait_id")["mean_cosine"].mean().sort_values()
    traits = avg_stab.index.tolist()
    y_pos = np.arange(len(traits))

    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["qwen2_5_7b_instruct", "llama_3_1_8b_instruct"]
    model_colors = {
        "qwen2_5_7b_instruct": CORAL,
        "llama_3_1_8b_instruct": SLATE_BLUE,
    }

    for i, trait in enumerate(traits):
        trait_data = df[df["trait_id"] == trait]

        # Connect the two model dots
        means = []
        for model in models:
            row = trait_data[trait_data["model_id"] == model]
            if len(row) > 0:
                means.append(row["mean_cosine"].values[0])
        if len(means) == 2:
            ax.plot(means, [i, i], color=MEDIUM_GRAY, linewidth=1.5, zorder=1)

        # Plot each model's dot with min-max range
        for model in models:
            row = trait_data[trait_data["model_id"] == model]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            mean_val = r["mean_cosine"]
            min_val = r["min_cosine"]
            max_val = r["max_cosine"]

            # Min-max range bar
            ax.plot(
                [min_val, max_val],
                [i, i],
                color=model_colors[model],
                linewidth=1.2,
                alpha=0.35,
                zorder=2,
            )

            # Mean dot
            ax.scatter(
                mean_val,
                i,
                color=model_colors[model],
                marker=MODEL_MARKERS[model],
                s=90,
                edgecolors="white",
                linewidth=0.6,
                zorder=4,
            )

    # Stability threshold lines
    ax.axvline(0.9, color=CAPTION_GRAY, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0.7, color=CAPTION_GRAY, linewidth=0.8, linestyle="--", alpha=0.5)

    ax.text(
        0.92,
        len(traits) - 0.3,
        "high\nstability",
        fontsize=7,
        color=CAPTION_GRAY,
        ha="left",
        va="top",
        alpha=0.7,
    )
    ax.text(
        0.72,
        len(traits) - 0.3,
        "moderate",
        fontsize=7,
        color=CAPTION_GRAY,
        ha="left",
        va="top",
        alpha=0.7,
    )
    ax.text(
        0.05,
        len(traits) - 0.3,
        "unstable",
        fontsize=7,
        color=CAPTION_GRAY,
        ha="left",
        va="top",
        alpha=0.7,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([TRAIT_DISPLAY[t] for t in traits], fontsize=10)
    ax.set_xlabel("Mean pairwise cosine similarity across prompt variants")
    ax.set_title("Direction Stability: Consistent Rankings Across Models")
    ax.set_xlim(0.0, 1.02)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CORAL,
            markersize=7,
            label=MODEL_DISPLAY["qwen2_5_7b_instruct"],
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=SLATE_BLUE,
            markersize=7,
            label=MODEL_DISPLAY["llama_3_1_8b_instruct"],
        ),
        Line2D(
            [0],
            [0],
            color=CAPTION_GRAY,
            linewidth=0.8,
            linestyle="--",
            label="Stability thresholds",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5)

    add_caption(
        fig,
        "Mean pairwise cosine between 10 prompt-variant directions per trait. "
        "Thin lines show the min–max range. "
        "Sycophancy is the clear outlier — some variant pairs are nearly "
        "orthogonal (min cos 0.07 on Llama). "
        "Rankings are consistent across both models.",
        y=0.01,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    savefig(fig, output_dir / "fig6_stability_ranking.png")
