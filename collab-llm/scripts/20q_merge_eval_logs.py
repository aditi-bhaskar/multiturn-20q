
# python scripts/20q_merge_eval_logs.py

import os
import json
import glob

BASE_DIR = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER_ACTUAL"
# MODEL_PATTERN = "Llama-3.2-1B-Instruct_temp=0.7*/log.json"
MODEL_PATTERN = "Llama-3.2-1B-Instruct-20q_temp=0.7*/log.json"
# MODEL_PATTERN = "Llama-3.2-1B-Instruct-20q-reward_temp=0.7*/log.json"

json_paths = glob.glob(os.path.join(BASE_DIR, MODEL_PATTERN))

if not json_paths:
    raise FileNotFoundError("No matching log.json files found")

# Assume all matched paths have the same base model prefix
base_model_dir = os.path.basename(os.path.dirname(json_paths[0])).split("*")[0].rstrip("/")
merged_dir_name = base_model_dir + "-MERGED_LOGS"
merged_dir_path = os.path.join(BASE_DIR, merged_dir_name)
os.makedirs(merged_dir_path, exist_ok=True)

output_path = os.path.join(merged_dir_path, "log.json")

merged_data = {}
for path in json_paths:
    with open(path, "r") as f:
        data = json.load(f)
        for key in data:
            if key in merged_data:
                raise ValueError(f"Duplicate key '{key}' found in {path}")
            merged_data[key] = data[key]

with open(output_path, "w") as f:
    json.dump(merged_data, f, indent=2)

print(f"Merged {len(json_paths)} files into {output_path}")

