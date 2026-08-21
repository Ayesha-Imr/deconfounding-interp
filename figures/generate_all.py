#!/usr/bin/env python3
"""Generate all publication figures from committed result CSVs."""

import sys
from pathlib import Path

# Add project root so `figures.theme` is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from figures import (  # noqa: E402
    fig1_stability_vs_overlap,
    fig2_surface_overlap,
    fig3_heatmaps_comparison,
    fig4_cross_trait_heatmap,
    fig5_probing_auroc,
    fig6_stability_ranking,
)

DATA_DIR = project_root / "outputs" / "reports"
OUTPUT_DIR = project_root / "figures" / "output"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data:   {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    figures = [
        ("Fig 1: Stability vs Overlap scatter", fig1_stability_vs_overlap),
        ("Fig 2: Surface overlap bars", fig2_surface_overlap),
        ("Fig 3: Heatmap comparison", fig3_heatmaps_comparison),
        ("Fig 4: Cross-trait cosines", fig4_cross_trait_heatmap),
        ("Fig 5: Probing AUROC", fig5_probing_auroc),
        ("Fig 6: Stability ranking", fig6_stability_ranking),
    ]

    for name, module in figures:
        print(f"Generating {name}...")
        module.plot(DATA_DIR, OUTPUT_DIR)

    print(f"\nDone — {len(figures)} figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
