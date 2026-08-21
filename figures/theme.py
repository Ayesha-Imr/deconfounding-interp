"""Anthropic-inspired visual theme for publication figures."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Core palette ──────────────────────────────────────────────────────────────

CORAL = "#D97757"
SLATE_BLUE = "#5B7B9A"
SAGE = "#7A9A6D"
WARM_GRAY = "#8B8680"
SOFT_GOLD = "#C4A35A"
DUSTY_ROSE = "#B07A8F"
DEEP_CHARCOAL = "#2D2D2D"
LIGHT_GRAY = "#F5F5F5"
MEDIUM_GRAY = "#CCCCCC"
CAPTION_GRAY = "#777777"

# Trait-type palettes
BEHAVIORAL_COLORS = {
    "sycophancy": CORAL,
    "hallucination": SOFT_GOLD,
}
STYLISTIC_COLORS = {
    "toxicity": SLATE_BLUE,
    "dramatic": DUSTY_ROSE,
    "formality": SAGE,
    "verbosity": WARM_GRAY,
}
TRAIT_COLORS = {**BEHAVIORAL_COLORS, **STYLISTIC_COLORS}

TRAIT_DISPLAY = {
    "sycophancy": "Sycophancy",
    "hallucination": "Hallucination",
    "toxicity": "Toxicity",
    "dramatic": "Dramatic",
    "formality": "Formality",
    "verbosity": "Verbosity",
}

TRAIT_TYPE = {
    "sycophancy": "Behavioral",
    "hallucination": "Behavioral",
    "toxicity": "Stylistic",
    "dramatic": "Stylistic",
    "formality": "Stylistic",
    "verbosity": "Stylistic",
}

MODEL_DISPLAY = {
    "qwen2_5_7b_instruct": "Qwen2.5-7B",
    "llama_3_1_8b_instruct": "Llama-3.1-8B",
}

MODEL_MARKERS = {
    "qwen2_5_7b_instruct": "o",
    "llama_3_1_8b_instruct": "^",
}

DIRECTION_COLORS = {
    "standard": SLATE_BLUE,
    "averaged": SAGE,
    "subtracted": CORAL,
    "single_variant": WARM_GRAY,
}

DIRECTION_DISPLAY = {
    "standard": "Standard",
    "averaged": "Averaged",
    "subtracted": "Subtracted",
    "single_variant": "Single variant",
}

# ── Colormaps ─────────────────────────────────────────────────────────────────

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "anthropic_diverging",
    [SLATE_BLUE, "#FFFFFF", CORAL],
    N=256,
)

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "anthropic_sequential",
    ["#FDF0EB", "#E8A98A", CORAL, "#A0502F"],
    N=256,
)


# ── Theme application ────────────────────────────────────────────────────────


def apply_theme():
    """Set global matplotlib rcParams for the Anthropic aesthetic."""
    font_families = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"]
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": MEDIUM_GRAY,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#ECECEC",
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            "font.family": "sans-serif",
            "font.sans-serif": font_families,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": DEEP_CHARCOAL,
            "ytick.color": DEEP_CHARCOAL,
            "axes.labelcolor": DEEP_CHARCOAL,
            "text.color": DEEP_CHARCOAL,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )


def add_caption(fig, text, y=-0.02):
    """Add an italic interpretation caption below the figure."""
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="top",
        fontsize=9,
        fontstyle="italic",
        color=CAPTION_GRAY,
        wrap=True,
        transform=fig.transFigure,
    )


def trait_sort_key(trait_id):
    """Sort traits: behavioral first, then stylistic, alphabetical within."""
    type_order = {"Behavioral": 0, "Stylistic": 1}
    return (type_order.get(TRAIT_TYPE.get(trait_id, ""), 2), trait_id)


def savefig(fig, path):
    """Save figure with tight layout and close."""
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {path}")
