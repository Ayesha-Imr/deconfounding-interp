from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from deconfounding_interp import io as dio
from deconfounding_interp.analysis.stability import summarize_stability
from deconfounding_interp.analysis.surface_overlap import compute_surface_overlap
from deconfounding_interp.directions import (
    cosine_similarity,
    orthonormal_basis,
    pairwise_cosine,
    subspace_overlap_fraction,
)


def configured_variant_count(bundle) -> int:
    generation = bundle.experiment.generation
    n_pairs = int(generation.get("system_prompt_pairs", 5))
    paraphrases_per_prompt = int(generation.get("paraphrases_per_prompt", 1))
    return n_pairs * (1 + paraphrases_per_prompt)


def stability_label(mean_cosine: float, high: float = 0.90, moderate: float = 0.70) -> str:
    if mean_cosine >= high:
        return "high"
    if mean_cosine >= moderate:
        return "moderate"
    return "unstable"


def resolve_selected_layer(
    bundle,
    model_id: str,
    trait_id: str,
    variant_count: int | None = None,
) -> tuple[int | None, str | None]:
    model_config = bundle.models[model_id]
    configured = model_config.trait_layers.get(trait_id)
    fallback: tuple[int | None, str | None] = (None, None)
    if configured is not None:
        configured = int(configured)
        if variant_count is None or _layer_is_complete(
            bundle,
            trait_id,
            model_id,
            configured,
            variant_count,
        ):
            return configured, "model_config"
        fallback = (configured, "model_config_incomplete")

    sweep_path = dio.trait_interim_dir(bundle, trait_id, model_id) / "selected_layer.json"
    if sweep_path.exists():
        data = dio.load_results_json(sweep_path)
        if "best_layer" in data:
            selected = int(data["best_layer"])
            if variant_count is None or _layer_is_complete(
                bundle,
                trait_id,
                model_id,
                selected,
                variant_count,
            ):
                return selected, str(sweep_path)
            fallback = (
                fallback
                if fallback[0] is not None
                else (selected, f"{sweep_path}:incomplete")
            )

    direction_layer_path = dio.direction_dir(bundle, trait_id, model_id) / "selected_layer.json"
    if direction_layer_path.exists():
        data = dio.load_results_json(direction_layer_path)
        if "best_layer" in data:
            selected = int(data["best_layer"])
            if variant_count is None or _layer_is_complete(
                bundle,
                trait_id,
                model_id,
                selected,
                variant_count,
            ):
                return selected, str(direction_layer_path)
            fallback = (
                fallback
                if fallback[0] is not None
                else (selected, f"{direction_layer_path}:incomplete")
            )

    if variant_count is not None:
        inferred = infer_complete_activation_layer(bundle, trait_id, model_id, variant_count)
        if inferred is not None:
            return inferred, "inferred_complete_activation_layer"

    if fallback[0] is not None:
        return fallback

    return None, str(sweep_path)


def validate_trait_model_readiness(
    bundle,
    trait_id: str,
    model_id: str,
    variant_count: int | None = None,
    selected_layer: int | None = None,
) -> dict[str, Any]:
    variant_count = (
        configured_variant_count(bundle) if variant_count is None else int(variant_count)
    )
    layer_source = None
    if selected_layer is None:
        selected_layer, layer_source = resolve_selected_layer(
            bundle,
            model_id,
            trait_id,
            variant_count,
        )
    else:
        layer_source = "payload"

    interim = dio.trait_interim_dir(bundle, trait_id, model_id)
    activation_root = interim / "activations"
    variants: list[dict[str, Any]] = []
    hidden_dims: set[int] = set()

    for vi in range(variant_count):
        act_dir = activation_root / f"variant_{vi:02d}"
        entry: dict[str, Any] = {
            "variant_index": vi,
            "path": str(act_dir),
            "status": "missing",
            "missing_sides": ["pos", "neg"],
            "shapes": {},
        }
        if selected_layer is None:
            entry["status"] = "missing_layer_selection"
            variants.append(entry)
            continue

        acts = dio.load_activations(act_dir, layer=selected_layer)
        sides = acts.get(selected_layer, {})
        missing_sides = [side for side in ("pos", "neg") if side not in sides]
        shapes = {}
        for side, arr in sides.items():
            shapes[side] = list(arr.shape)
            if arr.ndim == 2:
                hidden_dims.add(int(arr.shape[1]))

        if missing_sides:
            entry["status"] = "incomplete" if act_dir.exists() else "missing"
            entry["missing_sides"] = missing_sides
        else:
            invalid = [
                side for side in ("pos", "neg")
                if sides[side].ndim != 2 or sides[side].shape[0] == 0
            ]
            entry["missing_sides"] = []
            entry["status"] = "invalid_shape" if invalid else "ready"
            if invalid:
                entry["invalid_sides"] = invalid
        entry["shapes"] = shapes
        variants.append(entry)

    usable = [v["variant_index"] for v in variants if v["status"] == "ready"]
    problems = [v for v in variants if v["status"] != "ready"]
    hidden_dim_problem = len(hidden_dims) > 1
    status = (
        "ready"
        if not problems and not hidden_dim_problem and selected_layer is not None
        else "blocked"
    )

    return {
        "trait_id": trait_id,
        "model_id": model_id,
        "status": status,
        "selected_layer": selected_layer,
        "selected_layer_source": layer_source,
        "variant_count_expected": variant_count,
        "variant_count_ready": len(usable),
        "usable_variant_indices": usable,
        "missing_variant_indices": [
            v["variant_index"]
            for v in variants
            if v["status"] in {"missing", "incomplete", "missing_layer_selection"}
        ],
        "problem_count": len(problems) + int(hidden_dim_problem),
        "hidden_dims": sorted(hidden_dims),
        "hidden_dim_consistent": not hidden_dim_problem,
        "variants": variants,
    }


def infer_complete_activation_layer(
    bundle,
    trait_id: str,
    model_id: str,
    variant_count: int,
) -> int | None:
    """Find a saved layer with both sides for every configured variant."""
    activation_root = dio.trait_interim_dir(bundle, trait_id, model_id) / "activations"
    layer_ready_counts: dict[int, int] = {}
    for vi in range(int(variant_count)):
        act_dir = activation_root / f"variant_{vi:02d}"
        acts = dio.load_activations(act_dir)
        for layer, sides in acts.items():
            if "pos" in sides and "neg" in sides:
                layer_ready_counts[layer] = layer_ready_counts.get(layer, 0) + 1

    complete_layers = [
        layer for layer, count in layer_ready_counts.items() if count == int(variant_count)
    ]
    if not complete_layers:
        return None
    return min(complete_layers)


def _layer_is_complete(
    bundle,
    trait_id: str,
    model_id: str,
    layer: int,
    variant_count: int,
) -> bool:
    activation_root = dio.trait_interim_dir(bundle, trait_id, model_id) / "activations"
    for vi in range(int(variant_count)):
        acts = dio.load_activations(activation_root / f"variant_{vi:02d}", layer=layer)
        sides = acts.get(layer, {})
        if "pos" not in sides or "neg" not in sides:
            return False
    return True


def validate_direction_readiness(bundle, variant_count: int | None = None) -> dict[str, Any]:
    variant_count = (
        configured_variant_count(bundle) if variant_count is None else int(variant_count)
    )
    entries = [
        validate_trait_model_readiness(bundle, trait_id, model_id, variant_count)
        for model_id in bundle.models
        for trait_id in bundle.traits
    ]
    blocked = [entry for entry in entries if entry["status"] != "ready"]
    return {
        "status": "ready" if not blocked else "blocked",
        "variant_count_expected": variant_count,
        "n_trait_model_pairs": len(entries),
        "n_ready": len(entries) - len(blocked),
        "n_blocked": len(blocked),
        "entries": entries,
    }


def write_direction_readiness_report(bundle, variant_count: int | None = None) -> dict[str, Any]:
    report = validate_direction_readiness(bundle, variant_count)
    out_dir = dio.resolve_paths(bundle)["report_dir"] / "phase2"
    dio.save_results_json(out_dir / "readiness.json", report)
    return report


def build_direction_summary(
    bundle,
    variant_count: int | None = None,
    random_seed: int | None = None,
    random_baseline_n: int = 100,
    write_figures: bool = True,
) -> dict[str, Any]:
    variant_count = (
        configured_variant_count(bundle) if variant_count is None else int(variant_count)
    )
    random_seed = bundle.experiment.random_seed if random_seed is None else int(random_seed)
    out_dir = dio.resolve_paths(bundle)["report_dir"] / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    readiness = write_direction_readiness_report(bundle, variant_count)
    stability_rows, matrices = _build_stability_rows(bundle)
    cross_trait_rows = _build_cross_trait_rows(bundle)
    surface_rows = _build_surface_rows(bundle, random_seed, random_baseline_n)
    control_checks = _build_control_checks(surface_rows)

    _write_csv(out_dir / "stability_summary.csv", stability_rows)
    _write_csv(out_dir / "cross_trait_cosines.csv", cross_trait_rows)
    _write_csv(out_dir / "surface_overlap_summary.csv", surface_rows)
    dio.save_results_json(out_dir / "stability_summary.json", {"rows": stability_rows})
    dio.save_results_json(out_dir / "cross_trait_cosines.json", {"rows": cross_trait_rows})
    dio.save_results_json(out_dir / "surface_overlap_summary.json", {"rows": surface_rows})
    dio.save_results_json(out_dir / "control_checks.json", control_checks)

    matrix_dir = out_dir / "stability_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        np.savetxt(matrix_dir / f"{name}.csv", matrix, delimiter=",", fmt="%.10f")

    report_md = _render_direction_report(
        readiness,
        stability_rows,
        cross_trait_rows,
        surface_rows,
        control_checks,
    )
    (out_dir / "summary.md").write_text(report_md)

    figures = []
    if write_figures:
        figures = _write_figures(out_dir, stability_rows, cross_trait_rows, surface_rows, matrices)

    summary = {
        "status": "completed",
        "readiness_status": readiness["status"],
        "n_stability_rows": len(stability_rows),
        "n_cross_trait_rows": len(cross_trait_rows),
        "n_surface_rows": len(surface_rows),
        "control_checks": control_checks,
        "figures": [str(path) for path in figures],
        "output_dir": str(out_dir),
    }
    dio.save_results_json(out_dir / "direction_summary.json", summary)
    return summary


def _build_stability_rows(bundle) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    analysis_cfg = bundle.experiment.analysis.get("stability", {})
    thresholds = analysis_cfg.get("within_trait_thresholds", {})
    high = float(thresholds.get("high", 0.90))
    moderate = float(thresholds.get("moderate", 0.70))

    rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    for model_id in bundle.models:
        for trait_id, trait in bundle.traits.items():
            d_dir = dio.direction_dir(bundle, trait_id, model_id)
            variant_dirs = _load_prefixed_directions(d_dir, "variant_")
            if len(variant_dirs) < 2:
                rows.append({
                    "model_id": model_id,
                    "trait_id": trait_id,
                    "expected_surface_confound": trait.expected_surface_confound,
                    "status": "missing",
                    "n_variant_directions": len(variant_dirs),
                })
                continue

            arr = np.array([direction for _, direction in variant_dirs])
            matrix = pairwise_cosine(arr)
            summary = summarize_stability(arr)
            matrix_name = f"{model_id}__{trait_id}"
            matrices[matrix_name] = matrix
            rows.append({
                "model_id": model_id,
                "trait_id": trait_id,
                "expected_surface_confound": trait.expected_surface_confound,
                "status": "completed",
                "n_variant_directions": len(variant_dirs),
                "mean_cosine": summary.mean_cosine,
                "std_cosine": summary.std_cosine,
                "min_cosine": summary.min_cosine,
                "max_cosine": summary.max_cosine,
                "n_pairs": summary.n_pairs,
                "label": stability_label(summary.mean_cosine, high=high, moderate=moderate),
            })
    return rows, matrices


def _build_cross_trait_rows(bundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in bundle.models:
        standard_dirs = []
        for trait_id in bundle.traits:
            path = dio.direction_dir(bundle, trait_id, model_id) / "standard.npy"
            if path.exists():
                standard_dirs.append((trait_id, np.load(path)))

        for i, (left_trait, left_dir) in enumerate(standard_dirs):
            for right_trait, right_dir in standard_dirs[i + 1:]:
                rows.append({
                    "model_id": model_id,
                    "left_trait_id": left_trait,
                    "right_trait_id": right_trait,
                    "cosine": cosine_similarity(left_dir, right_dir),
                })
    return rows


def _build_surface_rows(
    bundle,
    random_seed: int,
    random_baseline_n: int,
) -> list[dict[str, Any]]:
    surface_basis_cfg = bundle.experiment.corrections.get("surface_basis", {})
    max_rank = surface_basis_cfg.get("max_rank", 5)
    variance_threshold = surface_basis_cfg.get("variance_threshold", 0.90)

    rows: list[dict[str, Any]] = []
    for model_id in bundle.models:
        for trait_id, trait in bundle.traits.items():
            d_dir = dio.direction_dir(bundle, trait_id, model_id)
            standard_path = d_dir / "standard.npy"
            surface_dirs = _load_prefixed_directions(d_dir, "surface_")
            if not standard_path.exists() or len(surface_dirs) < 2:
                rows.append({
                    "model_id": model_id,
                    "trait_id": trait_id,
                    "expected_surface_confound": trait.expected_surface_confound,
                    "status": "missing",
                    "n_surface_directions": len(surface_dirs),
                })
                continue

            standard = np.load(standard_path)
            surface = np.array([direction for _, direction in surface_dirs])
            overlap = compute_surface_overlap(
                standard,
                surface,
                max_rank=max_rank,
                variance_threshold=variance_threshold,
            )
            baseline = random_surface_overlap_baseline(
                surface,
                hidden_dim=int(standard.shape[0]),
                seed=_stable_seed(random_seed, model_id, trait_id),
                n_samples=random_baseline_n,
                max_rank=max_rank,
                variance_threshold=variance_threshold,
            )
            rows.append({
                "model_id": model_id,
                "trait_id": trait_id,
                "expected_surface_confound": trait.expected_surface_confound,
                "status": "completed",
                "n_surface_directions": len(surface_dirs),
                **asdict(overlap),
                "random_baseline_mean": baseline["mean"],
                "random_baseline_std": baseline["std"],
                "random_baseline_n": baseline["n"],
                "overlap_minus_random_mean": overlap.overlap_fraction - baseline["mean"],
            })
    return rows


def random_surface_overlap_baseline(
    surface_directions,
    hidden_dim: int,
    seed: int,
    n_samples: int = 100,
    max_rank: int | None = 5,
    variance_threshold: float | None = 0.90,
) -> dict[str, float | int]:
    surface = np.asarray(surface_directions, dtype=np.float64)
    if surface.ndim != 2:
        raise ValueError("surface_directions must have shape (n_directions, hidden_dim)")
    if int(hidden_dim) != surface.shape[1]:
        raise ValueError("hidden_dim must match surface direction width")
    basis = orthonormal_basis(surface, max_rank=max_rank, variance_threshold=variance_threshold)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_samples)):
        direction = rng.normal(size=hidden_dim)
        values.append(subspace_overlap_fraction(direction, basis))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "n": int(n_samples),
    }


def _load_prefixed_directions(path: Path, prefix: str) -> list[tuple[str, np.ndarray]]:
    return [
        (fpath.stem, np.load(fpath))
        for fpath in sorted(path.glob(f"{prefix}*.npy"))
    ]


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join((str(base_seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_control_checks(surface_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "toxicity_positive_control": _control_check(
            surface_rows,
            trait_id="toxicity",
            expectation="high_surface_confound",
        ),
        "power_seeking_negative_control": _control_check(
            surface_rows,
            trait_id="power_seeking",
            expectation="low_surface_confound",
        ),
    }


def _control_check(
    surface_rows: list[dict[str, Any]],
    trait_id: str,
    expectation: str,
) -> dict[str, Any]:
    rows = [row for row in surface_rows if row.get("trait_id") == trait_id]
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return {
            "trait_id": trait_id,
            "expectation": expectation,
            "status": "unavailable",
            "n_completed": 0,
            "n_total": len(rows),
            "reason": "No completed Phase 2 surface-overlap rows for this control trait",
        }

    overlaps = [float(row["overlap_fraction"]) for row in completed]
    baseline_deltas = [float(row["overlap_minus_random_mean"]) for row in completed]
    return {
        "trait_id": trait_id,
        "expectation": expectation,
        "status": "completed",
        "n_completed": len(completed),
        "n_total": len(rows),
        "mean_overlap_fraction": float(np.mean(overlaps)),
        "min_overlap_fraction": float(np.min(overlaps)),
        "max_overlap_fraction": float(np.max(overlaps)),
        "mean_overlap_minus_random": float(np.mean(baseline_deltas)),
    }


def _render_direction_report(
    readiness: dict[str, Any],
    stability_rows: list[dict[str, Any]],
    cross_trait_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    control_checks: dict[str, Any],
) -> str:
    completed_stability = [row for row in stability_rows if row.get("status") == "completed"]
    completed_surface = [row for row in surface_rows if row.get("status") == "completed"]

    lines = [
        "# Phase 2 Summary",
        "",
        (
            f"- Readiness: {readiness['status']} "
            f"({readiness['n_ready']}/{readiness['n_trait_model_pairs']} "
            "trait/model pairs ready)"
        ),
        f"- Stability rows: {len(completed_stability)} completed",
        f"- Cross-trait cosine rows: {len(cross_trait_rows)}",
        f"- Surface-overlap rows: {len(completed_surface)} completed",
        "",
        "## Control Checks",
        "",
        _format_control_check(control_checks["toxicity_positive_control"]),
        _format_control_check(control_checks["power_seeking_negative_control"]),
        "",
        "## Stability Labels",
        "",
    ]
    for row in completed_stability:
        lines.append(
            f"- {row['model_id']} / {row['trait_id']}: "
            f"{row['label']} (mean cosine {row['mean_cosine']:.3f})"
        )

    blocked = [entry for entry in readiness["entries"] if entry.get("status") != "ready"]
    if blocked:
        lines.extend(["", "## Excluded Or Blocked Pairs", ""])
        for entry in blocked:
            source = entry.get("selected_layer_source") or "none"
            lines.append(
                f"- {entry['model_id']} / {entry['trait_id']}: "
                f"{entry['variant_count_ready']}/{entry['variant_count_expected']} variants ready; "
                f"layer source {source}"
            )
    lines.append("")
    return "\n".join(lines)


def _format_control_check(check: dict[str, Any]) -> str:
    trait_id = check["trait_id"]
    expectation = check["expectation"]
    if check["status"] != "completed":
        return f"- {trait_id}: unavailable ({expectation})"
    return (
        f"- {trait_id}: mean overlap {check['mean_overlap_fraction']:.3f}, "
        f"mean above random {check['mean_overlap_minus_random']:.3f} ({expectation})"
    )


def _write_figures(
    out_dir: Path,
    stability_rows: list[dict[str, Any]],
    cross_trait_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    matrices: dict[str, np.ndarray],
) -> list[Path]:
    _prepare_figure_outputs(out_dir)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return _write_svg_figures(
            out_dir,
            stability_rows,
            cross_trait_rows,
            surface_rows,
            matrices,
        )

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    within = [
        float(row["mean_cosine"])
        for row in stability_rows
        if row.get("status") == "completed" and "mean_cosine" in row
    ]
    cross = [
        float(row["cosine"])
        for row in cross_trait_rows
        if "cosine" in row
    ]
    if within or cross:
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [values for values in (within, cross) if values]
        labels = [
            label
            for label, values in (("within-trait", within), ("cross-trait", cross))
            if values
        ]
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("cosine")
        ax.set_title("Phase 2 Stability")
        path = figure_dir / "stability_within_vs_cross.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    for name, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(name.replace("__", " / "))
        fig.colorbar(im, ax=ax, label="cosine")
        path = figure_dir / f"heatmap__{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    surface_completed = [row for row in surface_rows if row.get("status") == "completed"]
    if surface_completed:
        labels = [f"{row['model_id']}\n{row['trait_id']}" for row in surface_completed]
        overlaps = [float(row["overlap_fraction"]) for row in surface_completed]
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.7), 4))
        ax.bar(range(len(labels)), overlaps)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_ylabel("overlap fraction")
        ax.set_title("Surface Overlap")
        path = figure_dir / "surface_overlap_bars.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(
            [float(row["cosine_with_mean_surface"]) for row in surface_completed],
            overlaps,
        )
        ax.set_xlabel("cosine with mean surface direction")
        ax.set_ylabel("overlap fraction")
        ax.set_title("Surface Cosine vs Overlap")
        path = figure_dir / "surface_cosine_scatter.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    return written


def _prepare_figure_outputs(out_dir: Path) -> None:
    for stale_notice in ("figures_fallback.json", "figures_skipped.json"):
        path = out_dir / stale_notice
        if path.exists():
            path.unlink()

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.svg"):
        for path in figure_dir.glob(pattern):
            path.unlink()


def _write_svg_figures(
    out_dir: Path,
    stability_rows: list[dict[str, Any]],
    cross_trait_rows: list[dict[str, Any]],
    surface_rows: list[dict[str, Any]],
    matrices: dict[str, np.ndarray],
) -> list[Path]:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    within = [
        float(row["mean_cosine"])
        for row in stability_rows
        if row.get("status") == "completed" and "mean_cosine" in row
    ]
    cross = [float(row["cosine"]) for row in cross_trait_rows if "cosine" in row]
    if within or cross:
        path = figure_dir / "stability_within_vs_cross.svg"
        path.write_text(_boxplot_svg({"within-trait": within, "cross-trait": cross}, "cosine"))
        written.append(path)

    for name, matrix in matrices.items():
        path = figure_dir / f"heatmap__{name}.svg"
        path.write_text(_heatmap_svg(matrix, name.replace("__", " / ")))
        written.append(path)

    surface_completed = [row for row in surface_rows if row.get("status") == "completed"]
    if surface_completed:
        path = figure_dir / "surface_overlap_bars.svg"
        labels = [f"{row['model_id']} / {row['trait_id']}" for row in surface_completed]
        values = [float(row["overlap_fraction"]) for row in surface_completed]
        path.write_text(_bar_svg(labels, values, "overlap fraction"))
        written.append(path)

        path = figure_dir / "surface_cosine_scatter.svg"
        points = [
            (
                float(row["cosine_with_mean_surface"]),
                float(row["overlap_fraction"]),
                f"{row['model_id']} / {row['trait_id']}",
            )
            for row in surface_completed
        ]
        path.write_text(_scatter_svg(points, "cosine with mean surface", "overlap fraction"))
        written.append(path)

    dio.save_results_json(
        out_dir / "figures_fallback.json",
        {"reason": "matplotlib is not installed; wrote SVG fallback figures"},
    )
    return written


def _svg_frame(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def _boxplot_svg(groups: dict[str, list[float]], y_label: str) -> str:
    width, height = 640, 420
    plot_top, plot_bottom = 40, 340
    y_min, y_max = -1.0, 1.0

    def y(value: float) -> float:
        return plot_bottom - ((value - y_min) / (y_max - y_min)) * (plot_bottom - plot_top)

    body = [
        '<line x1="80" y1="40" x2="80" y2="340" stroke="#222"/>',
        '<line x1="80" y1="340" x2="600" y2="340" stroke="#222"/>',
        f'<text x="20" y="205" transform="rotate(-90 20 205)" font-size="14">{y_label}</text>',
    ]
    populated = [(label, values) for label, values in groups.items() if values]
    for idx, (label, values) in enumerate(populated):
        x = 210 + idx * 220
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        low, high = min(values), max(values)
        body.extend([
            f'<line x1="{x}" y1="{y(low):.1f}" x2="{x}" y2="{y(high):.1f}" stroke="#444"/>',
            f'<rect x="{x - 38}" y="{y(q3):.1f}" width="76" '
            f'height="{max(1, y(q1) - y(q3)):.1f}" fill="#cfe3ff" stroke="#24527a"/>',
            f'<line x1="{x - 38}" y1="{y(median):.1f}" '
            f'x2="{x + 38}" y2="{y(median):.1f}" stroke="#9b1c31" stroke-width="2"/>',
            f'<text x="{x}" y="376" text-anchor="middle" font-size="13">{label}</text>',
        ])
    return _svg_frame(width, height, "\n".join(body))


def _heatmap_svg(matrix: np.ndarray, title: str) -> str:
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    cell = max(12, min(34, 300 // max(1, n)))
    width = 120 + n * cell
    height = 100 + n * cell
    body = [f'<text x="40" y="28" font-size="16">{title}</text>']
    for i in range(n):
        for j in range(n):
            color = _diverging_color(float(matrix[i, j]))
            body.append(
                f'<rect x="{60 + j * cell}" y="{50 + i * cell}" width="{cell}" '
                f'height="{cell}" fill="{color}" stroke="#fff" stroke-width="0.5"/>'
            )
    return _svg_frame(width, height, "\n".join(body))


def _bar_svg(labels: list[str], values: list[float], y_label: str) -> str:
    width, height = max(720, 160 + len(values) * 90), 440
    baseline = 330
    max_value = max([1.0, *values])
    body = [
        '<line x1="80" y1="40" x2="80" y2="330" stroke="#222"/>',
        '<line x1="80" y1="330" x2="680" y2="330" stroke="#222"/>',
        f'<text x="20" y="205" transform="rotate(-90 20 205)" font-size="14">{y_label}</text>',
    ]
    bar_width = 44
    step = max(90, (width - 140) // max(1, len(values)))
    for idx, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = 100 + idx * step
        bar_h = (value / max_value) * 260
        body.extend([
            f'<rect x="{x}" y="{baseline - bar_h:.1f}" width="{bar_width}" '
            f'height="{bar_h:.1f}" fill="#78a6d6"/>',
            f'<text x="{x + bar_width / 2}" y="352" text-anchor="end" '
            f'transform="rotate(-35 {x + bar_width / 2} 352)" font-size="11">{label}</text>',
        ])
    return _svg_frame(width, height, "\n".join(body))


def _scatter_svg(points: list[tuple[float, float, str]], x_label: str, y_label: str) -> str:
    width, height = 640, 420
    x_min = min([-1.0, *(p[0] for p in points)])
    x_max = max([1.0, *(p[0] for p in points)])
    y_min = 0.0
    y_max = max([1.0, *(p[1] for p in points)])

    def xp(value: float) -> float:
        return 80 + ((value - x_min) / (x_max - x_min)) * 520

    def yp(value: float) -> float:
        return 340 - ((value - y_min) / (y_max - y_min)) * 300

    body = [
        '<line x1="80" y1="40" x2="80" y2="340" stroke="#222"/>',
        '<line x1="80" y1="340" x2="600" y2="340" stroke="#222"/>',
        f'<text x="340" y="390" text-anchor="middle" font-size="14">{x_label}</text>',
        f'<text x="20" y="205" transform="rotate(-90 20 205)" font-size="14">{y_label}</text>',
    ]
    for x_value, y_value, label in points:
        body.extend([
            f'<circle cx="{xp(x_value):.1f}" cy="{yp(y_value):.1f}" r="5" fill="#9b1c31"/>',
            f'<text x="{xp(x_value) + 8:.1f}" y="{yp(y_value) - 8:.1f}" '
            f'font-size="10">{label}</text>',
        ])
    return _svg_frame(width, height, "\n".join(body))


def _diverging_color(value: float) -> str:
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        t = value
        r = int(255 * (1 - t) + 180 * t)
        g = int(255 * (1 - t) + 40 * t)
        b = int(255 * (1 - t) + 40 * t)
    else:
        t = -value
        r = int(255 * (1 - t) + 50 * t)
        g = int(255 * (1 - t) + 100 * t)
        b = int(255 * (1 - t) + 180 * t)
    return f"#{r:02x}{g:02x}{b:02x}"
