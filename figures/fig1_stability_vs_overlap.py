"""Figure 1: Stability vs Surface Overlap scatter — the key dissociation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figures.theme import (
    TRAIT_COLORS,
    TRAIT_DISPLAY,
    TRAIT_TYPE,
    MODEL_DISPLAY,
    MODEL_MARKERS,
    CAPTION_GRAY,
    MEDIUM_GRAY,
    apply_theme,
    add_caption,
    savefig,
)


def plot(data_dir: Path, output_dir: Path) -> None:
    apply_theme()

    stab = pd.read_csv(data_dir / "phase2" / "stability_summary.csv")
    surf = pd.read_csv(data_dir / "phase2" / "surface_overlap_summary.csv")

    stab = stab[stab["status"] == "completed"]
    surf = surf[surf["status"] == "completed"]

    df = stab.merge(surf, on=["model_id", "trait_id"], suffixes=("_stab", "_surf"))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for _, row in df.iterrows():
        trait = row["trait_id"]
        model = row["model_id"]
        ax.scatter(
            row["mean_cosine"],
            row["overlap_fraction"],
            c=TRAIT_COLORS[trait],
            marker=MODEL_MARKERS[model],
            s=120,
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )

    # Label each point — offset to avoid overlap
    offsets = {
        ("formality", "qwen2_5_7b_instruct"): (8, 6),
        ("formality", "llama_3_1_8b_instruct"): (-60, -12),
        ("sycophancy", "qwen2_5_7b_instruct"): (8, 8),
        ("sycophancy", "llama_3_1_8b_instruct"): (8, -14),
        ("toxicity", "qwen2_5_7b_instruct"): (8, -12),
        ("toxicity", "llama_3_1_8b_instruct"): (8, 8),
        ("hallucination", "qwen2_5_7b_instruct"): (8, 6),
        ("hallucination", "llama_3_1_8b_instruct"): (-70, -6),
        ("dramatic", "qwen2_5_7b_instruct"): (8, 8),
        ("dramatic", "llama_3_1_8b_instruct"): (8, -14),
        ("verbosity", "qwen2_5_7b_instruct"): (-52, 8),
        ("verbosity", "llama_3_1_8b_instruct"): (8, -6),
    }
    for _, row in df.iterrows():
        trait = row["trait_id"]
        model = row["model_id"]
        dx, dy = offsets.get((trait, model), (8, 4))
        ax.annotate(
            TRAIT_DISPLAY[trait],
            (row["mean_cosine"], row["overlap_fraction"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            color=TRAIT_COLORS[trait],
            fontweight="medium",
        )

    # Quadrant guide lines
    ax.axhline(0.5, color=MEDIUM_GRAY, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0.9, color=MEDIUM_GRAY, linewidth=0.6, linestyle="--", alpha=0.5)

    # Quadrant labels
    ax.text(0.62, 0.88, "unstable\nhigh confound", fontsize=7.5, color=CAPTION_GRAY,
            ha="center", va="center", alpha=0.7)
    ax.text(0.96, 0.88, "stable but\nconfounded", fontsize=7.5, color=CAPTION_GRAY,
            ha="center", va="center", alpha=0.7)
    ax.text(0.62, 0.12, "unstable\nbut cleaner", fontsize=7.5, color=CAPTION_GRAY,
            ha="center", va="center", alpha=0.7)
    ax.text(0.96, 0.12, "stable\nand clean", fontsize=7.5, color=CAPTION_GRAY,
            ha="center", va="center", alpha=0.7)

    ax.set_xlabel("Direction stability (mean pairwise cosine)")
    ax.set_ylabel("Surface-form overlap fraction")
    ax.set_title("Stability ≠ Purity: The Dissociation")

    ax.set_xlim(0.55, 1.0)
    ax.set_ylim(0.0, 0.95)

    # Legend for model markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CAPTION_GRAY,
               markersize=7, label="Qwen2.5-7B"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=CAPTION_GRAY,
               markersize=7, label="Llama-3.1-8B"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TRAIT_COLORS["sycophancy"],
               markersize=7, label="Behavioral"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TRAIT_COLORS["toxicity"],
               markersize=7, label="Stylistic"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8.5)

    add_caption(
        fig,
        "Stable ≠ clean. Formality produces a consistent direction every time (cos 0.94) "
        "— but 83% of it is surface-form artifact. Sycophancy is both inconsistent and contaminated.",
        y=0.01,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    savefig(fig, output_dir / "fig1_stability_vs_overlap.png")
