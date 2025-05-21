import json
import glob
import os

log_dir = "logs/saved_runs"
combined = {
    'idx': [],
    'prompt': [],
    'chosen': [],
    'rejected': [],
    'chosen_eval': [],
    'rejected_eval': [],
    'metadata': [],
    'prompt_item': []
}


for file in glob.glob(os.path.join(log_dir, "*args_dataset_generated_convs*.json")):
    print(f"\nfilename: {file}")
    with open(file, 'r') as f:
        data = json.load(f)
        for key in combined:
            combined[key].extend(data.get(key, []))

output_path = os.path.join(log_dir, "combined_generated_conversations.json")
with open(output_path, "w") as f:
    json.dump(combined, f, indent=2)


