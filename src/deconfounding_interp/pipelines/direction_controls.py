"""Resolve fitted directions and deterministic causal-control directions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from deconfounding_interp.directions import normalize

CONTROL_DIRECTION_TYPES = frozenset({"random", "sign_reversed"})


def stable_control_seed(base_seed: int, model_id: str, trait_id: str) -> int:
    """Derive a cross-process seed without relying on Python's randomized hash."""

    material = f"{int(base_seed)}|{model_id}|{trait_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32)


def resolve_direction(
    direction_dir: Path,
    direction_type: str,
    *,
    base_seed: int,
    model_id: str,
    trait_id: str,
    fit_excluded_variant: int | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load a fitted direction or synthesize a deterministic control.

    ``random`` uses only the fitted standard direction to obtain the hidden
    dimension; ``sign_reversed`` negates that same fitted direction.  The
    returned metadata is persisted with downstream results so controls cannot
    be mistaken for learned directions.
    """

    suffix = ""
    if fit_excluded_variant is not None:
        suffix = f"_fit_excluding_variant_{int(fit_excluded_variant):02d}"

    source_type = "standard" if direction_type in CONTROL_DIRECTION_TYPES else direction_type
    candidates = [
        direction_dir / f"{source_type}{suffix}.npy",
        direction_dir / f"{source_type}.npy",
    ]
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        return None, {"direction_source": str(candidates[0]), "control_type": direction_type}

    fitted = normalize(np.load(source_path))
    metadata: dict[str, Any] = {
        "direction_source": source_path.name,
        "control_type": direction_type if direction_type in CONTROL_DIRECTION_TYPES else None,
    }
    if direction_type == "sign_reversed":
        return -fitted, metadata
    if direction_type == "random":
        seed = stable_control_seed(base_seed, model_id, trait_id)
        direction = normalize(np.random.default_rng(seed).normal(size=fitted.shape))
        metadata["control_seed"] = seed
        return direction, metadata
    return fitted, metadata
