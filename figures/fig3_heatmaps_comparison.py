"""Figure 3: Side-by-side heatmaps — sycophancy (chaotic) vs toxicity (uniform)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from figures.theme import (
    CORAL,
    DEEP_CHARCOAL,
    DIVERGING_CMAP,
    SLATE_BLUE,
    add_caption,
    apply_theme,
    savefig,
)


def _load_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).values


def plot(data_dir: Path, output_dir: Path) -> None:
    apply_theme()

    mat_dir = data_dir / "phase2" / "stability_matrices"
    syco = _load_matrix(mat_dir / "qwen2_5_7b_instruct__sycophancy.csv")
    toxi = _load_matrix(mat_dir / "qwen2_5_7b_instruct__toxicity.csv")

    # Shared color normalization — center at 0.7 to highlight the spread
    vmin = min(syco.min(), toxi.min()) - 0.02
    vmax = 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.7, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    variant_labels = [f"v{i}" for i in range(10)]

    for ax, mat, title, color in [
        (axes[0], syco, "Sycophancy (unstable)", CORAL),
        (axes[1], toxi, "Toxicity (stable)", SLATE_BLUE),
    ]:
        im = ax.imshow(mat, cmap=DIVERGING_CMAP, norm=norm, aspect="equal")

        # Annotate cells
        for i in range(10):
            for j in range(10):
                if i != j:
                    val = mat[i, j]
                    text_color = "white" if val < 0.5 else DEEP_CHARCOAL
                    ax.text(
                        j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=text_color
                    )

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(variant_labels, fontsize=8)
        ax.set_yticklabels(variant_labels, fontsize=8)
        ax.set_title(title, fontsize=13, fontweight="semibold", color=color, pad=10)
        ax.set_xlabel("Prompt variant", fontsize=9)
        ax.set_ylabel("Prompt variant", fontsize=9)

        # Subtle border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#E0E0E0")
            spine.set_linewidth(0.5)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03, shrink=0.82)
    cbar.set_label("Cosine similarity", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Direction Stability: Sycophancy vs. Toxicity (Qwen2.5-7B)",
        fontsize=14,
        fontweight="semibold",
    )

    add_caption(
        fig,
        "Each cell = cosine similarity between DiM directions from two different prompt variants. "
        "Sycophancy variants 0 and 1 are nearly orthogonal (cos 0.25). "
        "All toxicity pairs exceed 0.89.",
        y=0.01,
    )
    plt.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.12, wspace=0.3)
    savefig(fig, output_dir / "fig3_heatmaps_comparison.png")
