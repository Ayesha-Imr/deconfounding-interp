"""Data I/O: path resolution, save/load for activations, directions, responses, and CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_paths(bundle) -> dict[str, Path]:
    root = bundle.project_root
    p = bundle.experiment.paths
    return {
        "data_dir": root / p.get("data_dir", "data"),
        "raw_dir": root / p.get("raw_dir", "data/raw"),
        "interim_dir": root / p.get("interim_dir", "data/interim"),
        "processed_dir": root / p.get("processed_dir", "data/processed"),
        "manifest_dir": root / p.get("manifest_dir", "outputs/manifests"),
        "direction_dir": root / p.get("direction_dir", "outputs/directions"),
        "report_dir": root / p.get("report_dir", "outputs/reports"),
    }


def trait_raw_dir(bundle, trait_id: str) -> Path:
    return resolve_paths(bundle)["raw_dir"] / trait_id


def trait_interim_dir(bundle, trait_id: str, model_id: str) -> Path:
    return resolve_paths(bundle)["interim_dir"] / trait_id / model_id


def direction_dir(bundle, trait_id: str, model_id: str) -> Path:
    return resolve_paths(bundle)["direction_dir"] / trait_id / model_id


def report_dir(bundle, trait_id: str, model_id: str) -> Path:
    return resolve_paths(bundle)["report_dir"] / trait_id / model_id


# ---------------------------------------------------------------------------
# Activation I/O  —  .npy per (layer, side)
# ---------------------------------------------------------------------------

def save_activations(
    path: Path,
    layer_acts: dict[int, dict[str, np.ndarray]],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for layer_idx, sides in layer_acts.items():
        for side, arr in sides.items():
            np.save(path / f"layer_{layer_idx:02d}_{side}.npy", arr)


def load_activations(
    path: Path,
    layer: int | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    if layer is not None:
        for side in ("pos", "neg"):
            fpath = path / f"layer_{layer:02d}_{side}.npy"
            if fpath.exists():
                result.setdefault(layer, {})[side] = np.load(fpath)
        return result

    for fpath in sorted(path.glob("layer_*_*.npy")):
        parts = fpath.stem.split("_")  # layer_NN_side
        layer_idx = int(parts[1])
        side = parts[2]
        result.setdefault(layer_idx, {})[side] = np.load(fpath)
    return result


# ---------------------------------------------------------------------------
# Direction I/O  —  unit-norm vectors as .npy
# ---------------------------------------------------------------------------

def save_direction(path: Path, name: str, direction: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f"{name}.npy", direction)


def load_direction(path: Path, name: str) -> np.ndarray:
    return np.load(path / f"{name}.npy")


# ---------------------------------------------------------------------------
# Response / result JSON I/O
# ---------------------------------------------------------------------------

def save_responses_json(path: Path, responses: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(responses, f, indent=2, default=str)


def load_responses_json(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def save_results_json(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV audit trail for LLM calls
# ---------------------------------------------------------------------------

_AUDIT_COLUMNS = [
    "timestamp", "model", "prompt_hash", "prompt_preview",
    "response_preview", "tokens_used", "latency_ms", "metadata_json",
]


def save_llm_audit_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_AUDIT_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(records)


def load_llm_audit_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
