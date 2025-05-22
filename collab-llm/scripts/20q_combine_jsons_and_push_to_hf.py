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





# TODO: (make sure to set random seed to 0 or smth)
# Yes, since your local JSON doesn’t have splits, you can manually split it. Here's how:

# python
# Copy
# Edit
# import json
# import random

# with open("your_data.json") as f:
#     data = json.load(f)

# random.shuffle(data)
# split_idx = int(0.7 * len(data))
# train_data = data[:split_idx]
# test_data = data[split_idx:]

# with open("train.json", "w") as f:
#     json.dump(train_data, f, indent=2)

# with open("test.json", "w") as f:
#     json.dump(test_data, f, indent=2)
# Then upload both files to Hugging Face with:

# css
# Copy
# Edit
# datasets.DatasetDict({
#     "train": datasets.load_dataset("json", data_files="train.json")["train"],
#     "test": datasets.load_dataset("json", data_files="test.json")["train"],
# })
# Let me know if you want to script the upload too.








# You said:
# idk how to make that split on hf when i push it tho. do i tag the data as trian vs split?


# ChatGPT said:
# On Hugging Face, the dataset split info is usually part of the dataset loading script or the data files' structure.

# If you’re just uploading raw JSON files, the common practice is:

# Upload separate JSON files named e.g. train.json and test.json.

# When loading, use load_dataset("json", data_files={"train": "train.json", "test": "test.json"}).

# The split tags aren’t in the data itself but inferred from which file you load as train/test.

# So, to create splits on HF:

# Locally split your data into files train.json, test.json (and maybe validation.json).

# Upload all these files to your HF dataset repo.

# In your dataset loading code, specify which file corresponds to which split as above.

# If you want the splits embedded inside a single JSON file, you’d have to add a split field to each entry yourself, but that’s less common for HF datasets. The recommended way is separate files.

# Want me to help generate the split files?







