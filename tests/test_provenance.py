"""Tests for run metadata and array checksum manifests."""

import json

import numpy as np

from deconfounding_interp import io
from deconfounding_interp.provenance import refresh_checksum_manifest, sha256_file


def test_checksum_manifest_tracks_numpy_outputs(tmp_path):
    path = tmp_path / "activations"
    io.save_activations(path, {0: {"pos": np.ones((2, 3)), "neg": np.zeros((2, 3))}})

    manifest = json.loads((path / ".checksums.json").read_text())

    assert set(manifest) == {"layer_00_neg.npy", "layer_00_pos.npy"}
    assert manifest["layer_00_pos.npy"]["sha256"] == sha256_file(
        path / "layer_00_pos.npy",
    )
    assert manifest["layer_00_pos.npy"]["size_bytes"] > 0


def test_direction_checksum_refreshes_after_multiple_writes(tmp_path):
    path = tmp_path / "directions"
    io.save_direction(path, "standard", np.ones(4))
    io.save_direction(path, "averaged", np.zeros(4) + 2)

    manifest = json.loads((path / ".checksums.json").read_text())

    assert set(manifest) == {"averaged.npy", "standard.npy"}
    assert manifest["standard.npy"]["sha256"]
    assert refresh_checksum_manifest(path) == path / ".checksums.json"
