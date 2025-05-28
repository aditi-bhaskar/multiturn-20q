import os
import glob
import json
from datasets import Dataset, DatasetDict

# LOG_DIR = "logs/saved_runs_20q_dataset"
LOG_DIR = "logs/saved_runs_20q_2v_dataset"
OUTPUT_FILENAME = "combined_generated_conversations.json"
OUTPUT_TRAIN = "train.json"
OUTPUT_TEST = "test.json"

# HF_REPO_ID = "aditijb/collabllm-20q"
# HF_REPO_ID = "aditijb/collabllm-20q-2v"
HF_REPO_ID = "aditijb/collabllm-20q-tests"



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


def combine_jsons_split(log_dir, train_ratio=0.7):
    combined = {
        'idx': [], 'prompt': [], 'chosen': [], 'rejected': [],
        'chosen_eval': [], 'rejected_eval': [], 'metadata': [], 'prompt_item': []
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

    combined_list = [dict(zip(combined.keys(), values)) for values in zip(*combined.values())]

    split_idx = int(train_ratio * len(combined_list))
    train_data = sanitize_data(combined_list[:split_idx])
    test_data = sanitize_data(combined_list[split_idx:])

    train_path = os.path.join(log_dir, OUTPUT_TRAIN)
    test_path = os.path.join(log_dir, OUTPUT_TEST)
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=2)
# debug
    # print(f"train size: {len(train_data)}, test size: {len(test_data)}")
    # print(f"sample train: {train_data[0]}")

#  Todo: maybe preserve topics/run numbers in the train vs test ??

    return train_path, test_path


def sanitize_example(example):
    """Cleans up the 'rs' section of chosen_eval by removing extra keys from metric dicts."""
    for judge in example.get("chosen_eval", {}).get("rs", {}).values():
        for metric in ["accuracy", "information_gain", "interactivity"]:
            if isinstance(judge.get(metric), dict):
                allowed_keys = {"score", "thought"}
                judge[metric] = {k: v for k, v in judge[metric].items() if k in allowed_keys}
    return example


def sanitize_data(data):
    return [sanitize_example(example) for example in data]


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
    print("\nrunning: 20q_combine_jsons_and_push_to_hf\n\n")

    #  for a single dataset push
    # output_path = combine_jsons(LOG_DIR)
    # push_to_hf(output_path, HF_REPO_ID)

    #  for multiple datasets push
    output_path = combine_jsons_split(LOG_DIR)
    train_path = os.path.join(LOG_DIR, OUTPUT_TRAIN)
    test_path = os.path.join(LOG_DIR, OUTPUT_TEST)
    push_split_to_hf(train_path, test_path, HF_REPO_ID)


if __name__ == "__main__":
    main()



