#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set for private HF materialization/upload}"

source /lambda/nfs/ic-fs-2/envs/deconfounding-interp.env
cd /lambda/nfs/ic-fs-2/repos/deconfounding-interp
git pull --ff-only

python - <<'PY'
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

repo = "ic-org/deconfounding-interp-shared"
prefix = "experiments/geometry_null_repair_8b_20260821"
pairs = [
    ("data/raw/sycophancy/assets.json", "data/raw/geometry_null_repair_8b/sycophancy/assets.json"),
    ("data/raw/hallucination/assets.json", "data/raw/geometry_null_repair_8b/hallucination/assets.json"),
    (
        "data/interim/sycophancy/llama_3_1_8b_instruct_geometry/activations/variant_00/layer_16_neg.npy",
        "data/interim/geometry_null_repair_8b/sycophancy/llama_3_1_8b_instruct_geometry/activations/variant_00/layer_16_neg.npy",
    ),
    (
        "data/interim/sycophancy/llama_3_1_8b_instruct_geometry/activations/variant_00/layer_16_pos.npy",
        "data/interim/geometry_null_repair_8b/sycophancy/llama_3_1_8b_instruct_geometry/activations/variant_00/layer_16_pos.npy",
    ),
    (
        "data/interim/hallucination/llama_3_1_8b_instruct_geometry/activations/variant_00/layer_16_neg.npy",
        "data/interim/geometry_null_repair_8b/hallucination/llama_3_1_8b_instruct_geometry/activations/variant_00/layer_16_neg.npy",
    ),
    (
        "data/interim/hallucination/llama_3_1_8b_instruct_geometry/activations/variant_00/layer_16_pos.npy",
        "data/interim/geometry_null_repair_8b/hallucination/llama_3_1_8b_instruct_geometry/activations/variant_00/layer_16_pos.npy",
    ),
    (
        "outputs/directions/sycophancy/llama_3_1_8b_instruct_geometry/selected_layer.json",
        "outputs/directions/geometry_null_repair_8b/sycophancy/llama_3_1_8b_instruct_geometry/selected_layer.json",
    ),
    (
        "outputs/directions/sycophancy/llama_3_1_8b_instruct_geometry/standard.npy",
        "outputs/directions/geometry_null_repair_8b/sycophancy/llama_3_1_8b_instruct_geometry/standard.npy",
    ),
    (
        "outputs/directions/hallucination/llama_3_1_8b_instruct_geometry/selected_layer.json",
        "outputs/directions/geometry_null_repair_8b/hallucination/llama_3_1_8b_instruct_geometry/selected_layer.json",
    ),
    (
        "outputs/directions/hallucination/llama_3_1_8b_instruct_geometry/standard.npy",
        "outputs/directions/geometry_null_repair_8b/hallucination/llama_3_1_8b_instruct_geometry/standard.npy",
    ),
]

for source, target in pairs:
    cached = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=f"{prefix}/{source}",
        token=True,
    )
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, destination)

print(f"MATERIALIZATION_GATE_PASSED {len(pairs)}")
PY

export DECONFOUND_CODE_COMMIT="$(git rev-parse HEAD)"
RUN_ID=objective_data_causal_pilot_llama_8b_v1_20260821
CONFIG=configs/experiments/objective_data_causal_pilot_llama_8b_v1.yaml
MANIFEST=outputs/manifests/objective_data_causal_pilot_llama_8b_v1_20260821/objective_data_causal_pilot_llama_8b_v1_20260821.json
RUN_DIR=outputs/runs/objective_data_causal_pilot_llama_8b_v1_20260821

mkdir -p "$(dirname "$MANIFEST")"
deconfound make-manifest --config "$CONFIG" --output "$MANIFEST"
deconfound run-stage --config "$CONFIG" --manifest "$MANIFEST" --run-dir "$RUN_DIR" --phase downstream_evaluation
deconfound run-stage --config "$CONFIG" --manifest "$MANIFEST" --run-dir "$RUN_DIR" --phase phase3_summary
mkdir -p outputs/audits
deconfound audit-run \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --run-dir "$RUN_DIR" \
  --phase downstream_evaluation \
  --phase phase3_summary \
  --include-responses > outputs/audits/objective_data_causal_pilot_llama_8b_v1_audit.json

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/reports/objective_data_causal_pilot_llama_8b_v1/phase3")
for path in sorted(root.glob("*/llama_3_1_8b_instruct_geometry/steering_*_aggregates.json")):
    data = json.loads(path.read_text()).get("per_alpha", {})
    values = []
    for alpha, row in data.items():
        values.append(
            f"{alpha}:{row.get('trait_score_mean')}/{row.get('objective_score_mean')}"
        )
    print(path.parent.parent.name, path.name, " ".join(values))
PY

rm -rf /tmp/objective_llama_v1_stage
stage=/tmp/objective_llama_v1_stage
mkdir -p "$stage"/{manifests,runs,reports,configs,models,data,audits}
cp "$MANIFEST" "$stage/manifests/"
cp -r "$RUN_DIR" "$stage/runs/"
cp -r outputs/reports/objective_data_causal_pilot_llama_8b_v1 "$stage/reports/"
cp "$CONFIG" "$stage/configs/"
cp configs/models/llama_3_1_8b_instruct_objective_harder_pilot.yaml "$stage/models/"
cp configs/traits/sycophancy.yaml configs/traits/hallucination.yaml "$stage/configs/"
cp data/objective_tasks_8b_data_candidate_v1.json "$stage/data/"
cp outputs/audits/objective_data_causal_pilot_llama_8b_v1_audit.json "$stage/audits/"

python - <<'PY'
from pathlib import Path

from huggingface_hub import HfApi

api = HfApi(token=True)
stage = Path("/tmp/objective_llama_v1_stage")
prefix = "experiments/objective_data_causal_pilot_llama_8b_v1_20260821"
repo = "ic-org/deconfounding-interp-shared"
api.upload_folder(
    folder_path=str(stage),
    path_in_repo=prefix,
    repo_id=repo,
    repo_type="dataset",
    commit_message="Add Llama v1 objective replication pilot 2026-08-21",
    token=True,
)
entries = list(
    api.list_repo_tree(
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=prefix,
        recursive=True,
    )
)
print("HF_VERIFIED", len(entries), sum(getattr(entry, "size", 0) or 0 for entry in entries))
PY

rm -rf /tmp/objective_llama_v1_stage
for path in \
  data/raw/geometry_null_repair_8b \
  data/interim/geometry_null_repair_8b \
  outputs/directions/geometry_null_repair_8b \
  outputs/manifests/objective_data_causal_pilot_llama_8b_v1_20260821 \
  outputs/runs/objective_data_causal_pilot_llama_8b_v1_20260821 \
  outputs/reports/objective_data_causal_pilot_llama_8b_v1 \
  outputs/audits/objective_data_causal_pilot_llama_8b_v1_audit.json; do
  rm -rf -- "$path"
done

echo LLAMA_V1_PILOT_DONE
