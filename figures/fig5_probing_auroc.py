"""Figure 5: Probing AUROC by direction type — does cleaning help?"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.theme import (
    DIRECTION_COLORS,
    DIRECTION_DISPLAY,
    MEDIUM_GRAY,
    TRAIT_DISPLAY,
    add_caption,
    apply_theme,
    savefig,
)

DIRECTION_ORDER = ["standard", "averaged", "subtracted", "single_variant"]
TRAIT_ORDER = ["sycophancy", "hallucination", "toxicity", "dramatic", "formality", "verbosity"]


def plot(data_dir: Path, output_dir: Path) -> None:
    apply_theme()

    df = pd.read_csv(data_dir / "phase3" / "summary" / "probing_summary.csv")

    # Create grouped labels: trait / model
    models = ["qwen2_5_7b_instruct", "llama_3_1_8b_instruct"]
    groups = []
    for trait in TRAIT_ORDER:
        for model in models:
            groups.append((trait, model))

    n_groups = len(groups)
    n_dirs = len(DIRECTION_ORDER)
    bar_width = 0.18
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for k, dtype in enumerate(DIRECTION_ORDER):
        aurocs = []
        for trait, model in groups:
            row = df[
                (df["trait_id"] == trait)
                & (df["model_id"] == model)
                & (df["direction_type"] == dtype)
            ]
            aurocs.append(row["auroc"].values[0] if len(row) > 0 else 0)
        offset = (k - n_dirs / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            aurocs,
            bar_width * 0.88,
            color=DIRECTION_COLORS[dtype],
            edgecolor="white",
            linewidth=0.3,
            label=DIRECTION_DISPLAY[dtype],
            zorder=2,
        )

    # Group labels — trait name on first model only, model letter on both
    tick_labels = []
    for trait, model in groups:
        short_model = "Qwen" if "qwen" in model else "Llama"
        tick_labels.append(f"{TRAIT_DISPLAY[trait]}\n({short_model})")

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.7, 1.03)
    ax.set_title("Probing AUROC by Direction Type")

    # Trait group separators
    for i in range(1, len(TRAIT_ORDER)):
        sep_x = i * 2 - 0.5
        ax.axvline(sep_x, color=MEDIUM_GRAY, linewidth=0.4, linestyle="-", alpha=0.4)

    ax.legend(loc="upper right", ncol=4, fontsize=9, framealpha=0.9, edgecolor="#E0E0E0")

    add_caption(
        fig,
        "AUROC of each direction type as a linear probe on held-out activations. "
        "Standard and averaged perform similarly. Subtracted sometimes degrades "
        "(Llama formality: 1.00 to 0.84) -- surface-form signal may carry "
        "useful discriminative information.",
        y=0.01,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    savefig(fig, output_dir / "fig5_probing_auroc.png")
