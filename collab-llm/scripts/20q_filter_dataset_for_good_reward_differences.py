import json

src_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/logs/saved_runs_20q_dataset/combined_generated_conversations.json"
tgt_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/logs/saved_runs_20q_dataset/combined_generated_conversations_filtered.json"


# def print_structure(d, indent=0):
#     prefix = "  " * indent
#     if isinstance(d, dict):
#         print(f"{prefix}dict with keys:")
#         for k, v in d.items():
#             print(f"{prefix}- {k}: {type(v).__name__}")
#             print_structure(v, indent + 1)
#     elif isinstance(d, list):
#         print(f"{prefix}list of length {len(d)}")
#         if d:
#             # Just print first element's structure to avoid flooding output
#             print_structure(d[0], indent + 1)
#     else:
#         print(f"{prefix}{repr(d)} ({type(d).__name__})")

# Usage example:
# Assuming you have a variable `entry` holding one example dict from your dataset:


min_reward_diff = 0.01  # adjust as needed


def get_accuracy_diff(entry):
    try:
        # Make sure entry is a dict
        if not isinstance(entry, dict):
            return 0.0

        ce_list = entry["chosen_eval"]
        re_list = entry["rejected_eval"]

        # Defensive checks
        if not (isinstance(ce_list, list) and isinstance(re_list, list)):
            return 0.0

        if len(ce_list) != len(re_list):
            return 0.0

        for i in range(len(ce_list)):
            ce_item = ce_list[i]
            re_item = re_list[i]

            # Check if each item is a dict before accessing keys
            if not (isinstance(ce_item, dict) and isinstance(re_item, dict)):
                continue

            # Access accuracy dict safely using indexing
            if "accuracy" in ce_item and "accuracy" in re_item:
                ce_acc = ce_item.get("accuracy", {}).get("score")
                re_acc = re_item.get("accuracy", {}).get("score")
            else:
                ce_acc, re_acc = None, None

            if ce_acc is None or re_acc is None:
                continue

            print(f"Entry diff at index {i}: chosen={ce_acc}, rejected={re_acc}, diff={diff}")


            diff = abs(float(ce_acc) - float(re_acc))
            if diff >= min_reward_diff:
                return diff

        return 0.0

    except Exception as e:
        print("failed to get accuracy diff for entry:", e)


with open(src_path, "r") as infile:
    data = json.load(infile)  # data is a dict

filtered = {
    k: v
    for k, v in data.items()
    if get_accuracy_diff(v) >= min_reward_diff
}

with open(tgt_path, "w") as outfile:
    json.dump(filtered, outfile, indent=2)

print(f"Kept {len(filtered)} out of {len(data)} entries.")


# print_structure(entry)
