"""Figure 2: Surface overlap horizontal bar chart — how much is just wording?"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.theme import (
    CORAL,
    SLATE_BLUE,
    TRAIT_COLORS,
    TRAIT_DISPLAY,
    TRAIT_TYPE,
    MEDIUM_GRAY,
    CAPTION_GRAY,
    DEEP_CHARCOAL,
    apply_theme,
    add_caption,
    savefig,
)


def plot(data_dir: Path, output_dir: Path) -> None:
    apply_theme()

    df = pd.read_csv(data_dir / "phase2" / "surface_overlap_summary.csv")
    df = df[df["status"] == "completed"].copy()

    # Compute per-trait average and individual model values
    avg = df.groupby("trait_id").agg(
        mean_overlap=("overlap_fraction", "mean"),
        random_baseline=("random_baseline_mean", "mean"),
    ).reset_index()
    avg = avg.sort_values("mean_overlap", ascending=True)

    traits = avg["trait_id"].tolist()
    y_pos = np.arange(len(traits))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bars colored by trait
    bar_colors = [TRAIT_COLORS[t] for t in traits]
    bars = ax.barh(y_pos, avg["mean_overlap"].values, height=0.52,
                   color=bar_colors, edgecolor="white", linewidth=0.5, zorder=2)

    # Individual model dots (placed above/below bar center to avoid label clash)
    for _, row in df.iterrows():
        trait = row["trait_id"]
        if trait in traits:
            idx = traits.index(trait)
            model = row["model_id"]
            nudge = 0.13 if "qwen" in model else -0.13
            ax.scatter(row["overlap_fraction"], idx + nudge, color="white", s=22,
                       zorder=3, edgecolors=DEEP_CHARCOAL, linewidth=0.6)

    # Random baseline
    random_mean = avg["random_baseline"].mean()
    ax.axvline(random_mean, color=MEDIUM_GRAY, linewidth=1.0, linestyle=":",
               zorder=1, label=f"Random baseline ({random_mean:.1%})")

    # Percentage labels — placed past the furthest model dot
    for i, trait in enumerate(traits):
        trait_rows = df[df["trait_id"] == trait]["overlap_fraction"]
        rightmost = max(avg.loc[avg["trait_id"] == trait, "mean_overlap"].values[0],
                        trait_rows.max())
        val = avg.loc[avg["trait_id"] == trait, "mean_overlap"].values[0]
        ax.text(rightmost + 0.025, i, f"{val:.0%}", va="center", fontsize=9.5,
                color=DEEP_CHARCOAL, fontweight="medium")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{TRAIT_DISPLAY[t]}  ({TRAIT_TYPE[t][:1].upper()})" for t in traits],
                       fontsize=10)
    ax.set_xlabel("Fraction of concept direction in surface-form subspace")
    ax.set_title("How Much of Your Concept Direction Is Just Wording?")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8.5)

    add_caption(
        fig,
        "Fraction of the standard DiM direction explained by surface-form variation. "
        "White dots show individual models (Qwen, Llama). Random baseline ≈ 0.1%. "
        "Even behavioral traits like sycophancy have ~44% contamination.",
        y=0.01,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    savefig(fig, output_dir / "fig2_surface_overlap.png")
