

########################################################################
################### FOR SINGLE FILE (dont need to combine files)#########
########################################################################


import os
import json
from datasets import Dataset, DatasetDict
import random

HF_REPO_ID = "aditijb/collabllm-20q-filtered-reward"
OUTPUT_TRAIN = "train.json"
OUTPUT_TEST = "test.json"

filepath = "/Users/aditi/Documents/multiturn-20q/collab-llm/logs/new_reward/combined_generated_conversations_filtered.json"

def sanitize_example(example):
    """Clean 'rs' section of chosen_eval and rejected_eval for consistent float scores."""
    def clean_eval(eval_dict):
        if not eval_dict:
            return eval_dict
        for judge in eval_dict.get("rs", {}).values():
            for metric in ["accuracy", "information_gain", "interactivity"]:
                if isinstance(judge.get(metric), dict):
                    allowed_keys = {"score", "thought"}
                    judge[metric] = {
                        k: (float(v) if k == "score" else v)
                        for k, v in judge[metric].items()
                        if k in allowed_keys
                    }
        return eval_dict

    example["chosen_eval"] = clean_eval(example.get("chosen_eval", {}))
    example["rejected_eval"] = clean_eval(example.get("rejected_eval", {}))
    return example

def sanitize_data(data):
    return [sanitize_example(example) for example in data]

def split_and_save(data, output_dir, train_ratio=0.7):
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_path = os.path.join(output_dir, OUTPUT_TRAIN)
    test_path = os.path.join(output_dir, OUTPUT_TEST)

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)

    return train_path, test_path

def push_split_to_hf(train_path, test_path, repo_id):
    with open(train_path) as f:
        train_data = sanitize_data(json.load(f))
    with open(test_path) as f:
        test_data = sanitize_data(json.load(f))

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    })
    dataset.push_to_hub(repo_id=repo_id, private=True)

def main():
    print(f"Loading data from {filepath}")
    with open(filepath) as f:
        data = json.load(f)

    print("Converting dict-of-lists to list-of-dicts...")
    if isinstance(data, dict):
        keys = list(data.keys())
        length = len(data[keys[0]])
        data_list = [ {k: data[k][i] for k in keys} for i in range(length) ]
    else:
        data_list = data

    print("Sanitizing data and splitting into train/test...")
    train_path, test_path = split_and_save(data_list, os.path.dirname(filepath))
    push_split_to_hf(train_path, test_path, HF_REPO_ID)
    print("Done.")

if __name__ == "__main__":
    main()
