"""Figure 4: Cross-trait cosine heatmap — are these different concepts?"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.theme import (
    DIVERGING_CMAP,
    TRAIT_DISPLAY,
    DEEP_CHARCOAL,
    CAPTION_GRAY,
    apply_theme,
    add_caption,
    savefig,
)

TRAIT_ORDER = ["sycophancy", "hallucination", "toxicity", "dramatic", "formality", "verbosity"]


def plot(data_dir: Path, output_dir: Path) -> None:
    apply_theme()

    df = pd.read_csv(data_dir / "phase2" / "cross_trait_cosines.csv")

    # Average across models
    avg = df.groupby(["left_trait_id", "right_trait_id"])["cosine"].mean().reset_index()

    n = len(TRAIT_ORDER)
    mat = np.zeros((n, n))
    np.fill_diagonal(mat, np.nan)

    for _, row in avg.iterrows():
        left = row["left_trait_id"]
        right = row["right_trait_id"]
        if left in TRAIT_ORDER and right in TRAIT_ORDER:
            i = TRAIT_ORDER.index(left)
            j = TRAIT_ORDER.index(right)
            mat[i, j] = row["cosine"]
            mat[j, i] = row["cosine"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    mask = np.eye(n, dtype=bool)
    masked_mat = np.ma.masked_where(mask, mat)

    im = ax.imshow(masked_mat, cmap=DIVERGING_CMAP, vmin=-0.5, vmax=0.5, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i != j:
                val = mat[i, j]
                fontweight = "bold" if abs(val) > 0.3 else "normal"
                text_color = "white" if abs(val) > 0.4 else DEEP_CHARCOAL
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color=text_color, fontweight=fontweight)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color=CAPTION_GRAY)

    labels = [TRAIT_DISPLAY[t] for t in TRAIT_ORDER]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Diagonal shading
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=True, facecolor="#F5F5F5", edgecolor="none", zorder=0))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#E0E0E0")
        spine.set_linewidth(0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Cosine similarity", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title("Cross-Trait Direction Cosines\n(averaged across Qwen & Llama)",
                 fontsize=13, fontweight="semibold", pad=12)

    add_caption(
        fig,
        "Cosine similarity between standard DiM directions for different traits. "
        "Blue = anti-aligned, white = independent, warm = correlated. "
        "Toxicity–formality anti-correlation (−0.39) reflects that toxic language tends to be informal.",
        y=0.01,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    savefig(fig, output_dir / "fig4_cross_trait_heatmap.png")
