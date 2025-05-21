import os
import glob
import json
from datasets import Dataset, DatasetDict

LOG_DIR = "logs/saved_runs"
OUTPUT_FILENAME = "combined_generated_conversations.json"
HF_REPO_ID = "aditijb/collabllm-20q"

def combine_jsons(log_dir):
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

    pattern = os.path.join(log_dir, "*args_dataset_generated_convs*.json")
    files = glob.glob(pattern)
    print("\nCombining files:")
    for file in files:
        print(f"  {file}")
        with open(file, 'r') as f:
            data = json.load(f)
            for key in combined:
                combined[key].extend(data.get(key, []))

    output_path = os.path.join(log_dir, OUTPUT_FILENAME)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    return output_path

def push_to_hf(json_path, repo_id):
    with open(json_path, "r") as f:
        saved_data = json.load(f)
    dataset = Dataset.from_dict(saved_data)
    DatasetDict({"train": dataset}).push_to_hub(repo_id=repo_id, private=True)

def main():
    print("\nrunning: 20q_combine_jsons_and_push_to_hf\n")
    output_path = combine_jsons(LOG_DIR)
    push_to_hf(output_path, HF_REPO_ID)

if __name__ == "__main__":
    main()
