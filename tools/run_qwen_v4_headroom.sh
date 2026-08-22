#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set for private HF materialization/upload}"

source /lambda/nfs/ic-fs-2/envs/deconfounding-interp.env
cd /lambda/nfs/ic-fs-2/repos/deconfounding-interp
git pull --ff-only

RUN_ID=objective_data_headroom_pilot_qwen_8b_v4_20260822
CONFIG=configs/experiments/objective_data_headroom_pilot_qwen_8b_v4.yaml
MANIFEST=outputs/manifests/objective_data_headroom_pilot_qwen_8b_v4_20260822/objective_data_headroom_pilot_qwen_8b_v4_20260822.json
RUN_DIR=outputs/runs/objective_data_headroom_pilot_qwen_8b_v4_20260822
REPORT_DIR=outputs/reports/objective_data_headroom_pilot_qwen_8b_v4
AUDIT_PATH=outputs/audits/objective_data_headroom_pilot_qwen_8b_v4_audit.json

# The prior fail-fast attempt reached the runner before discovering a path
# mismatch. Remove only this run's exact, unpromoted artifacts so a retry
# cannot inherit its checkpoint or empty phase-3 summary.
for path in \
  data/raw/geometry_null_repair_8b \
  data/interim/geometry_null_repair_8b \
  outputs/directions/geometry_null_repair_8b \
  "$(dirname "$MANIFEST")" \
  "$RUN_DIR" \
  "$REPORT_DIR" \
  "$AUDIT_PATH"; do
  rm -rf -- "$path"
done

python - <<'PY'
import json
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

repo = "ic-org/deconfounding-interp-shared"
prefix = "experiments/geometry_null_repair_8b_20260821"
static_pairs = [
    ("data/raw/sycophancy/assets.json", "data/raw/geometry_null_repair_8b/sycophancy/assets.json"),
    ("data/raw/hallucination/assets.json", "data/raw/geometry_null_repair_8b/hallucination/assets.json"),
]

def materialize(source, target):
    cached = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=f"{prefix}/{source}",
        token=True,
    )
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, destination)
    return cached


count = 0
for source, target in static_pairs:
    materialize(source, target)
    count += 1

# Read each archived direction's selected layer and fetch exactly that layer;
# this keeps the materialization contract coupled to the direction metadata
# instead of silently assuming every trait uses layer 16.
for trait in ("sycophancy", "hallucination"):
    model = "qwen2_5_7b_instruct_geometry"
    selected_source = f"outputs/directions/{trait}/{model}/selected_layer.json"
    selected_target = f"outputs/directions/geometry_null_repair_8b/{trait}/{model}/selected_layer.json"
    cached_selected = materialize(selected_source, selected_target)
    selected_layer = int(json.loads(Path(cached_selected).read_text())["best_layer"])
    materialize(
        f"outputs/directions/{trait}/{model}/standard.npy",
        f"outputs/directions/geometry_null_repair_8b/{trait}/{model}/standard.npy",
    )
    count += 2
    for side in ("neg", "pos"):
        filename = f"layer_{selected_layer:02d}_{side}.npy"
        materialize(
            f"data/interim/{trait}/{model}/activations/variant_00/{filename}",
            f"data/interim/geometry_null_repair_8b/{trait}/{model}/activations/variant_00/{filename}",
        )
        count += 1

print(f"MATERIALIZATION_GATE_PASSED {count}")
PY

python - "$CONFIG" <<'PY'
from pathlib import Path
import sys

from deconfounding_interp import io as dio
from deconfounding_interp.config import load_config_bundle

config_path = Path(sys.argv[1])
bundle = load_config_bundle(config_path)
missing = []
for trait_id in bundle.trait_ids:
    raw_assets = dio.trait_raw_dir(bundle, trait_id) / "assets.json"
    if not raw_assets.exists():
        missing.append(str(raw_assets))
    for model_id in bundle.model_ids:
        model = bundle.models[model_id]
        layer = model.trait_layers.get(trait_id)
        interim = dio.trait_interim_dir(bundle, trait_id, model_id)
        direction = dio.direction_dir(bundle, trait_id, model_id)
        if layer is None:
            missing.append(f"{model_id}/{trait_id}: selected layer is unset")
            continue
        for side in ("pos", "neg"):
            path = interim / "activations" / "variant_00" / f"layer_{layer:02d}_{side}.npy"
            if not path.exists():
                missing.append(str(path))
        for path in (direction / "selected_layer.json", direction / "standard.npy"):
            if not path.exists():
                missing.append(str(path))
if missing:
    raise SystemExit("CONFIG_ARTIFACT_GATE_FAILED\n" + "\n".join(missing))
print(f"CONFIG_ARTIFACT_GATE_PASSED {len(bundle.model_ids) * len(bundle.trait_ids)}")
PY

export DECONFOUND_CODE_COMMIT="$(git rev-parse HEAD)"

mkdir -p "$(dirname "$MANIFEST")"
deconfound make-manifest --config "$CONFIG" --output "$MANIFEST"
python - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.loads(open(sys.argv[1]).read())
downstream = [job for job in manifest["jobs"] if job["phase"] == "downstream_evaluation"]
expected = len(manifest["model_ids"]) * len(manifest["trait_ids"])
if len(downstream) != expected or any(
    not job["payload"].get("alpha_values") for job in downstream
):
    raise SystemExit(
        f"MANIFEST_GATE_FAILED downstream={len(downstream)} expected={expected}"
    )
print(f"MANIFEST_GATE_PASSED downstream={len(downstream)}")
PY
deconfound run-stage --config "$CONFIG" --manifest "$MANIFEST" --run-dir "$RUN_DIR" --phase downstream_evaluation
python - "$MANIFEST" "$RUN_DIR" "$REPORT_DIR" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
run_dir = Path(sys.argv[2])
report_dir = Path(sys.argv[3])
checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
completed = checkpoint.get("completed", {})
downstream = [job for job in manifest["jobs"] if job["phase"] == "downstream_evaluation"]
failures = []
for job in downstream:
    result = completed.get(job["job_id"], {}).get("result", {})
    if result.get("status") != "completed" or int(result.get("n_responses", 0)) <= 0:
        failures.append(f"{job['job_id']}: {result}")
    response_path = (
        report_dir / "phase3" / job["trait_id"] / job["model_id"]
        / f"steering_{job['payload']['direction_type']}_responses.json"
    )
    if not response_path.exists() or not json.loads(response_path.read_text()):
        failures.append(f"missing_or_empty_responses: {response_path}")
if failures:
    raise SystemExit("RESPONSE_GATE_FAILED\n" + "\n".join(failures))
print(f"RESPONSE_GATE_PASSED {len(downstream)}")
PY
deconfound run-stage --config "$CONFIG" --manifest "$MANIFEST" --run-dir "$RUN_DIR" --phase phase3_summary
mkdir -p outputs/audits
deconfound audit-run \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --run-dir "$RUN_DIR" \
  --phase downstream_evaluation \
  --phase phase3_summary \
  --include-responses > "$AUDIT_PATH"
python - "$AUDIT_PATH" <<'PY'
import json
import sys

audit = json.loads(open(sys.argv[1]).read())
if audit.get("status") != "passed" or int(audit.get("facts", {}).get("response_file_count", 0)) < 2:
    raise SystemExit(f"AUDIT_GATE_FAILED: {audit}")
print("AUDIT_GATE_PASSED", audit["facts"]["response_file_count"])
PY

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/reports/objective_data_headroom_pilot_qwen_8b_v4/phase3")
for path in sorted(root.glob("*/qwen2_5_7b_instruct_geometry/steering_*_aggregates.json")):
    data = json.loads(path.read_text()).get("per_alpha", {})
    values = []
    for alpha, row in data.items():
        values.append(
            f"{alpha}:{row.get('trait_score_mean')}/{row.get('objective_score_mean')}"
        )
    print(path.parent.parent.name, path.name, " ".join(values))
PY

rm -rf /tmp/objective_qwen_v4_headroom_stage
stage=/tmp/objective_qwen_v4_headroom_stage
mkdir -p "$stage"/{manifests,runs,reports,configs,models,data,audits}
cp "$MANIFEST" "$stage/manifests/"
cp -r "$RUN_DIR" "$stage/runs/"
cp -r outputs/reports/objective_data_headroom_pilot_qwen_8b_v4 "$stage/reports/"
cp "$CONFIG" "$stage/configs/"
cp configs/models/qwen2_5_7b_instruct_objective_harder_pilot.yaml "$stage/models/"
cp configs/traits/sycophancy.yaml configs/traits/hallucination.yaml "$stage/configs/"
cp data/objective_tasks_8b_data_candidate_v4.json "$stage/data/"
cp outputs/audits/objective_data_headroom_pilot_qwen_8b_v4_audit.json "$stage/audits/"

python - <<'PY'
from pathlib import Path

from huggingface_hub import HfApi

api = HfApi(token=True)
stage = Path("/tmp/objective_qwen_v4_headroom_stage")
prefix = "experiments/objective_data_headroom_pilot_qwen_8b_v4_20260822"
repo = "ic-org/deconfounding-interp-shared"
api.upload_folder(
    folder_path=str(stage),
    path_in_repo=prefix,
    repo_id=repo,
    repo_type="dataset",
    commit_message="Add Qwen v4 objective headroom pilot 2026-08-22",
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

rm -rf /tmp/objective_qwen_v4_headroom_stage
for path in \
  data/raw/geometry_null_repair_8b \
  data/interim/geometry_null_repair_8b \
  outputs/directions/geometry_null_repair_8b \
  outputs/manifests/objective_data_headroom_pilot_qwen_8b_v4_20260822 \
  outputs/runs/objective_data_headroom_pilot_qwen_8b_v4_20260822 \
  outputs/reports/objective_data_headroom_pilot_qwen_8b_v4 \
  outputs/audits/objective_data_headroom_pilot_qwen_8b_v4_audit.json; do
  rm -rf -- "$path"
done

echo QWEN_V4_HEADROOM_DONE
