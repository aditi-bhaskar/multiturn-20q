import json

src_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/logs/new_reward/combined_generated_conversations.json"
tgt_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/logs/new_reward/combined_generated_conversations_filtered.json"

min_reward_diff = 0.01

with open(src_path) as f:
    data = json.load(f)

num_turns = len(data["idx"])
print(f"Total turns: {num_turns}")

filtered_data = {k: [] for k in data.keys()}

print("check 0")

# def get_accuracy_score(eval_obj):
#     if not isinstance(eval_obj, dict) or "rs" not in eval_obj:
#         print(f"Skipping invalid eval_obj: {eval_obj}")
#         return None
    
#     rs_dict = eval_obj["rs"]
#     scores = []
#     for turn_eval in rs_dict.values():
#         if isinstance(turn_eval, dict):
#             acc = turn_eval.get("accuracy", {}).get("score")
#             if acc is not None:
#                 scores.append(acc)
#         else:
#             print("Skipping non-dict turn_eval:", turn_eval)
#     return sum(scores) / len(scores) if scores else None


def get_combined_score(eval_obj):
    if not isinstance(eval_obj, dict) or "rs" not in eval_obj:
        return None
    rs_dict = eval_obj["rs"]
    combined_scores = []
    for turn_eval in rs_dict.values():
        if isinstance(turn_eval, dict):
            acc = turn_eval.get("accuracy", {}).get("score")
            inter = turn_eval.get("interactivity", {}).get("score")
            info = turn_eval.get("information_gain", {}).get("score")
            scores = [s for s in [acc, inter, info] if s is not None]
            if scores:
                combined_scores.append(sum(scores) / len(scores))
    return sum(combined_scores) / len(combined_scores) if combined_scores else None


print("check 1")


kept = 0
for i in range(num_turns):
    ce = data["chosen_eval"][i]
    re = data["rejected_eval"][i]
    ce_score = get_combined_score(ce)
    re_score = get_combined_score(re)
    if ce_score is None or re_score is None:
        print("check 2")
        continue
    diff = abs(ce_score - re_score)
    print(f"[{i}] ce_score = {ce_score}, re_score = {re_score}, diff = {diff}")
    if diff >= min_reward_diff:
        for k in data:
            filtered_data[k].append(data[k][i])
        kept += 1

print(f"Kept {kept} out of {num_turns} entries.")

with open(tgt_path, "w") as f:
    json.dump(filtered_data, f, indent=2)