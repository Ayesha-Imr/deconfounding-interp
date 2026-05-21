"""Pipeline stage: aggregate Phase 3 steering + probing results into reports and figures."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

from deconfounding_interp import io as dio
from deconfounding_interp.pipelines.base import StageContext

logger = logging.getLogger(__name__)

DIRECTION_TYPES = ("standard", "averaged", "subtracted", "single_variant")


class Phase3SummaryStage:
    name = "phase3_summary"

    def run(self, job: dict[str, Any], context: StageContext) -> dict[str, Any]:
        if context.dry_run:
            logger.info("[DRY RUN] Would build Phase 3 summary")
            return {"status": "dry_run"}

        t0 = time.time()
        bundle = context.bundle
        report_root = dio.resolve_paths(bundle)["report_dir"] / "phase3"
        out_dir = report_root / "summary"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect steering results
        steering_rows = _collect_steering(bundle, report_root)
        logger.info("Collected %d steering aggregate rows", len(steering_rows))

        # Collect probing results
        probing_rows = _collect_probing(bundle, report_root)
        logger.info("Collected %d probing rows", len(probing_rows))

        # Write CSVs
        if steering_rows:
            _write_csv(out_dir / "steering_summary.csv", steering_rows)
            dio.save_results_json(out_dir / "steering_summary.json", {"rows": steering_rows})

        if probing_rows:
            _write_csv(out_dir / "probing_summary.csv", probing_rows)
            dio.save_results_json(out_dir / "probing_summary.json", {"rows": probing_rows})

        # Write figures
        write_figures = job.get("payload", {}).get("write_figures", True)
        figures = []
        if write_figures and (steering_rows or probing_rows):
            fig_dir = out_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            figures = _write_figures(fig_dir, steering_rows, probing_rows)
            logger.info("Wrote %d figures", len(figures))

        # Write markdown report
        md = _render_report(steering_rows, probing_rows, figures)
        (out_dir / "summary.md").write_text(md)

        elapsed = time.time() - t0
        logger.info("Phase 3 summary complete (%.1fs)", elapsed)
        return {
            "status": "completed",
            "n_steering_rows": len(steering_rows),
            "n_probing_rows": len(probing_rows),
            "n_figures": len(figures),
        }


def _collect_steering(bundle, report_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model_id in bundle.models:
        for trait_id in bundle.traits:
            for dt in DIRECTION_TYPES:
                agg_path = report_root / trait_id / model_id / f"steering_{dt}_aggregates.json"
                if not agg_path.exists():
                    continue
                data = dio.load_results_json(agg_path)
                for alpha_str, agg in data.get("per_alpha", {}).items():
                    row = {
                        "model_id": model_id,
                        "trait_id": trait_id,
                        "direction_type": dt,
                        "alpha": float(alpha_str),
                        "trait_score_mean": agg.get("trait_score_mean"),
                        "trait_score_std": agg.get("trait_score_std"),
                        "coherence_score_mean": agg.get("coherence_score_mean"),
                        "coherence_score_std": agg.get("coherence_score_std"),
                        "n_responses": agg.get("n_responses"),
                    }
                    leakage = agg.get("cross_trait_leakage", {})
                    leak_means = [v["mean"] for v in leakage.values() if v.get("mean") is not None]
                    row["mean_cross_trait_leakage"] = (
                        sum(leak_means) / len(leak_means) if leak_means else None
                    )
                    rows.append(row)
    return rows


def _collect_probing(bundle, report_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model_id in bundle.models:
        for trait_id in bundle.traits:
            probe_path = report_root / trait_id / model_id / "probing_results.json"
            if not probe_path.exists():
                continue
            data = dio.load_results_json(probe_path)
            for dt, result in data.items():
                rows.append({
                    "model_id": model_id,
                    "trait_id": trait_id,
                    "direction_type": dt,
                    "auroc": result.get("auroc"),
                    "accuracy": result.get("accuracy"),
                    "n_pos": result.get("n_pos"),
                    "n_neg": result.get("n_neg"),
                })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figures(
    fig_dir: Path,
    steering_rows: list[dict],
    probing_rows: list[dict],
) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        logger.warning("matplotlib not available, skipping figures")
        return []

    figures = []

    # Figure 1: Trait expression vs alpha
    if steering_rows:
        fig_path = _plot_metric_vs_alpha(
            fig_dir, steering_rows, "trait_score_mean",
            "Trait Expression Score", "trait_expression_vs_alpha.png",
        )
        if fig_path:
            figures.append(fig_path)

    # Figure 2: Coherence vs alpha
    if steering_rows:
        fig_path = _plot_metric_vs_alpha(
            fig_dir, steering_rows, "coherence_score_mean",
            "Coherence Score", "coherence_vs_alpha.png",
        )
        if fig_path:
            figures.append(fig_path)

    # Figure 3: Cross-trait leakage vs alpha
    if steering_rows:
        fig_path = _plot_metric_vs_alpha(
            fig_dir, steering_rows, "mean_cross_trait_leakage",
            "Mean Cross-Trait Leakage", "leakage_vs_alpha.png",
        )
        if fig_path:
            figures.append(fig_path)

    # Figure 4: Probing AUROC bar chart
    if probing_rows:
        fig_path = _plot_probing_auroc(fig_dir, probing_rows)
        if fig_path:
            figures.append(fig_path)

    return figures


def _plot_metric_vs_alpha(
    fig_dir: Path,
    rows: list[dict],
    metric_key: str,
    ylabel: str,
    filename: str,
) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = sorted({(r["model_id"], r["trait_id"]) for r in rows})
    if not pairs:
        return None

    n_cols = min(len(pairs), 4)
    n_rows_grid = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows_grid, n_cols,
        figsize=(4 * n_cols, 3 * n_rows_grid), squeeze=False,
    )

    colors = {
        "standard": "#1f77b4", "averaged": "#2ca02c",
        "subtracted": "#d62728", "single_variant": "#9467bd",
    }

    for idx, (mid, tid) in enumerate(pairs):
        ax = axes[idx // n_cols][idx % n_cols]
        for dt in DIRECTION_TYPES:
            dt_rows = sorted(
                [r for r in rows
                 if r["model_id"] == mid
                 and r["trait_id"] == tid
                 and r["direction_type"] == dt],
                key=lambda r: r["alpha"],
            )
            alphas = [r["alpha"] for r in dt_rows]
            vals = [r.get(metric_key) for r in dt_rows]
            if any(v is not None for v in vals):
                ax.plot(alphas, vals, "o-", label=dt, color=colors.get(dt), markersize=3)
        ax.set_title(f"{tid} / {mid.split('_')[0]}", fontsize=8)
        ax.set_xlabel("alpha", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)

    # Legend on first axes
    if pairs:
        axes[0][0].legend(fontsize=6)

    # Hide unused axes
    for idx in range(len(pairs), n_rows_grid * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    path = fig_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return filename


def _plot_probing_auroc(fig_dir: Path, rows: list[dict]) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pairs = sorted({(r["model_id"], r["trait_id"]) for r in rows})
    if not pairs:
        return None

    pair_labels = [f"{tid[:4]}/{mid.split('_')[0]}" for mid, tid in pairs]
    dt_list = [dt for dt in DIRECTION_TYPES if any(r["direction_type"] == dt for r in rows)]

    x = np.arange(len(pairs))
    width = 0.8 / max(len(dt_list), 1)
    colors = {
        "standard": "#1f77b4", "averaged": "#2ca02c",
        "subtracted": "#d62728", "single_variant": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(max(len(pairs) * 1.2, 6), 4))
    for i, dt in enumerate(dt_list):
        vals = []
        for mid, tid in pairs:
            match = [
                r for r in rows
                if r["model_id"] == mid
                and r["trait_id"] == tid
                and r["direction_type"] == dt
            ]
            auroc = match[0].get("auroc") if match else None
            vals.append(auroc if auroc is not None else 0)
        ax.bar(x + i * width, vals, width, label=dt, color=colors.get(dt))

    ax.set_xticks(x + width * (len(dt_list) - 1) / 2)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_title("Probing AUROC by Direction Type")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    path = fig_dir / "probing_auroc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return "probing_auroc.png"


def _render_report(
    steering_rows: list[dict],
    probing_rows: list[dict],
    figures: list[str],
) -> str:
    lines = ["# Phase 3: Downstream Evaluation Results\n"]

    if steering_rows:
        lines.append("## Steering Comparison\n")
        lines.append(f"Total conditions evaluated: {len(steering_rows)}\n")
        for fig in figures:
            if "trait_expression" in fig or "coherence" in fig or "leakage" in fig:
                lines.append(f"![{fig}](figures/{fig})\n")

    if probing_rows:
        lines.append("## Probing Comparison\n")
        lines.append(f"Total probing results: {len(probing_rows)}\n")
        # Summary table
        lines.append("| Model | Trait | Direction | AUROC | Accuracy |")
        lines.append("|-------|-------|-----------|-------|----------|")
        def _sort_key(x):
            return (x["model_id"], x["trait_id"], x["direction_type"])
        for r in sorted(probing_rows, key=_sort_key):
            auroc = f"{r['auroc']:.4f}" if r.get("auroc") is not None else "—"
            acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "—"
            mid = r["model_id"]
            tid = r["trait_id"]
            dt = r["direction_type"]
            lines.append(f"| {mid} | {tid} | {dt} | {auroc} | {acc} |")
        lines.append("")
        for fig in figures:
            if "probing" in fig:
                lines.append(f"![{fig}](figures/{fig})\n")

    return "\n".join(lines)
