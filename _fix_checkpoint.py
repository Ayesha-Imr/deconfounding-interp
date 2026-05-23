#!/usr/bin/env python3
"""Remove all llama_3_1_8b_instruct entries from the Phase 1 checkpoint."""
import json
import shutil
from pathlib import Path

cp_path = Path("outputs/runs/latest/checkpoint.json")

# Backup
shutil.copy2(cp_path, cp_path.with_suffix(".json.bak"))

with open(cp_path) as f:
    data = json.load(f)

before = len(data["completed"])
data["completed"] = {
    k: v for k, v in data["completed"].items()
    if "llama_3_1_8b_instruct" not in k
}
after = len(data["completed"])

with open(cp_path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"Removed {before - after} Llama entries, kept {after} entries")
print("Backup saved to checkpoint.json.bak")
