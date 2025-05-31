import json
import matplotlib.pyplot as plt
import collections

folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER/Llama-3.2-1B-Instruct_temp=0.5_2025-05-30-18-11-18"

json_path = folder_path + "/log.json"

with open(json_path, "r") as f:
    data = json.load(f)

title = folder_path.split("/")[-1].split("_2025")[0]

games = sorted([k for k in data.keys() if k.isdigit()], key=int)

num_turns = []
interactivity_scores = []
accuracy_scores = []
info_gain_scores = []
targets = []

# Track which indices have accuracy = 1.0
accuracy_is_one = []

for game_id in games:
    game = data[game_id]
    num_turns.append(game["token_amount"]["num_turns"])
    interactivity_scores.append(game["llm_judge"]["interactivity"]["score"])
    acc_score = game["llm_judge"]["accuracy"]["score"]
    accuracy_scores.append(acc_score)
    info_gain_scores.append(game["llm_judge"]["information_gain"]["score"])

    guess = game["qa"][-1]["content"]
    if "The answer is:" in guess:
        target = guess.split("The answer is:")[-1].strip().strip(".")
    else:
        target = ""
    targets.append(target)
    accuracy_is_one.append(acc_score == 1.0)

# Only use label if accuracy = 1.0
def filtered_labels():
    return [label if is_one else "" for label, is_one in zip(targets, accuracy_is_one)]

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
            vertical_offset = (i - (n - 1) / 2) * 12
            ax.annotate(
                lbl,
                (x, y),
                xytext=(-10, vertical_offset),
                textcoords='offset points',
                fontsize=7,
                ha='right',
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
            )

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
plt.suptitle(title, fontsize=14)

labels = filtered_labels()
plot_with_label_stacking(axs[0], num_turns, interactivity_scores, labels, "Interactivity vs Turns", "Interactivity Score")
plot_with_label_stacking(axs[1], num_turns, accuracy_scores, labels, "Accuracy vs Turns", "Accuracy Score")
plot_with_label_stacking(axs[2], num_turns, info_gain_scores, labels, "Information Gain vs Turns", "Information Gain Score")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(folder_path + "/plot_normal.png", dpi=300)
# plt.show()
