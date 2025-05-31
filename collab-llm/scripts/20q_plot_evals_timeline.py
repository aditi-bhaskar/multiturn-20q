import json
import matplotlib.pyplot as plt
import collections

folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER/Llama-3.2-1B-Instruct-20q-reward_temp=0.7_2025-COMBINED_LOGS"
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
label_mask = []  # True if accuracy == 1.0

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
    label_mask.append(acc_score == 1.0)

# Only use label if corresponding accuracy == 1.0
def filter_labels(labels, mask):
    return [label if show else "" for label, show in zip(labels, mask)]

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

plot_with_label_stacking(axs[0], num_turns, interactivity_scores, filter_labels(targets, label_mask), "Interactivity vs Turns", "Interactivity Score")
plot_with_label_stacking(axs[1], num_turns, accuracy_scores, filter_labels(targets, label_mask), "Accuracy vs Turns", "Accuracy Score")
plot_with_label_stacking(axs[2], num_turns, info_gain_scores, filter_labels(targets, label_mask), "Information Gain vs Turns", "Information Gain Score")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(folder_path + "/plot.png", dpi=300)
# plt.show()
