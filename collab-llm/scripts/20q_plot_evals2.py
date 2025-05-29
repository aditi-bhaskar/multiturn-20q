# note: plots are not automatically saved
# plots the outcomes of the base llama 3.2 1b instruct model at playing 20q

########################################################
# just copy the file path from the folder of interest
########################################################
### for base model:
# folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER/Llama-3.2-1B-Instruct_prompt=none_temp=0.5_2025-05-25-22-26"
# folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER/Llama-3.2-1B-Instruct_prompt=none_temp=0.7_2025-05-25-23-21"
### for 1 epoch model:
# folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER/Llama-3.2-1B-Instruct-20q_temp=0.5_2025-05-27-19-08"
# folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER/Llama-3.2-1B-Instruct-20q_temp=0.7_2025-05-27-19-10"

folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/Llama-3.2-1B-Instruct-20q-test_temp=0.5_2025-05-29-12-04-56"

json_path = folder_path + "/log.json"

import json
import matplotlib.pyplot as plt
import collections

# Load your data
with open(json_path, "r") as f:
    data = json.load(f)

# Filter only numeric game keys
games = sorted([k for k in data.keys() if k.isdigit()], key=int)

# Prepare lists
num_turns = []
interactivity_scores = []
accuracy_scores = []
info_gain_scores = []
targets = []

for game_id in games:
    game = data[game_id]
    num_turns.append(game["token_amount"]["num_turns"])
    interactivity_scores.append(game["llm_judge"]["interactivity"]["score"])
    accuracy_scores.append(game["llm_judge"]["accuracy"]["score"])
    info_gain_scores.append(game["llm_judge"]["information_gain"]["score"])

    # Extract guessed target object
    guess = game["qa"][-1]["content"]
    if "The answer is:" in guess:
        target = guess.split("The answer is:")[-1].strip().strip(".")
    else:
        target = ""
    targets.append(target)

# Plotting helper
def plot_with_label_stacking(ax, x_vals, y_vals, labels, title, ylabel):
    points = list(zip(x_vals, y_vals))
    point_counts = collections.Counter(points)
    sizes = [30 + 20 * (point_counts[p] - 1) for p in points]

    ax.scatter(x_vals, y_vals, s=sizes)
    ax.set_title(title)
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)

    labels_by_point = collections.defaultdict(list)
    for x, y, label in zip(x_vals, y_vals, labels):
        if label:
            labels_by_point[(x, y)].append(label)

    for (x, y), lbls in labels_by_point.items():
        n = len(lbls)
        for i, lbl in enumerate(lbls):
            vertical_offset = (i - (n - 1) / 2) * 12  # center the stack vertically
            ax.annotate(
                lbl,
                (x, y),
                xytext=(-10, vertical_offset),  # full stack is placed to the left
                textcoords='offset points',
                fontsize=7,
                ha='right',
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
            )

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

plot_with_label_stacking(axs[0], num_turns, interactivity_scores, targets, "Interactivity vs Turns", "Interactivity Score")
plot_with_label_stacking(axs[1], num_turns, accuracy_scores, targets, "Accuracy vs Turns", "Accuracy Score")
plot_with_label_stacking(axs[2], num_turns, info_gain_scores, targets, "Information Gain vs Turns", "Information Gain Score")

plt.tight_layout()
plt.show()
