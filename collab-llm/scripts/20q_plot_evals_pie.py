import json
import matplotlib.pyplot as plt
import collections
import numpy as np

filepath = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER/Llama-3.2-1B-Instruct_temp=0.5_2025-05-30-18-11-18"

# === Load your data ===
json_path = filepath + "/log.json"  
with open(json_path, "r") as f:
    data = json.load(f)

games = sorted([k for k in data.keys() if k.isdigit()], key=int)

num_turns = []
interactivity_scores = []
accuracy_scores = []
info_gain_scores = []
targets = []

for game_id in games:
    game = data[game_id]
    num_turns.append(game["token_amount"]["num_turns"])
    interactivity_scores.append(round(game["llm_judge"]["interactivity"]["score"], 2))
    accuracy_scores.append(round(game["llm_judge"]["accuracy"]["score"], 2))
    info_gain_scores.append(round(game["llm_judge"]["information_gain"]["score"], 2))
    
    guess = game["qa"][-1]["content"]
    if "The answer is:" in guess:
        target = guess.split("The answer is:")[-1].strip().strip(".")
    else:
        target = ""
    targets.append(target)

# === Helper: exact score groupings ===
def group_by_exact(scores, labels):
    grouped = collections.defaultdict(list)
    for score, label in zip(scores, labels):
        grouped[score].append(label)
    return grouped

# === Plot setup ===
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
timeline_ax = fig.add_subplot(gs[1, :])

# === Pie charts with word arrows ===
score_data = [
    ("Interactivity", interactivity_scores),
    ("Accuracy", accuracy_scores),
    ("Information Gain", info_gain_scores)
]

for ax, (metric_name, scores) in zip(axes, score_data):
    grouped = group_by_exact(scores, targets)
    score_keys = sorted(grouped.keys())
    sizes = [len(grouped[k]) for k in score_keys]
    labels = [str(k) for k in score_keys]

    wedges, texts = ax.pie(sizes, startangle=90, wedgeprops=dict(width=0.4), labels=labels, labeldistance=1.05)
    ax.set_title(f"{metric_name} Distribution")

    # Annotate words around each wedge
    for i, wedge in enumerate(wedges):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        wordlist = grouped[score_keys[i]]
        ax.annotate(", ".join(wordlist), 
                    xy=(x*0.9, y*0.9), xytext=(x*1.5, y*1.5),
                    ha='center', va='center',
                    fontsize=6, arrowprops=dict(arrowstyle='->', lw=0.5))

# === Timeline ===
# timeline_ax.scatter(num_turns, np.zeros_like(num_turns), marker='|', s=100)
# for i, (x, label) in enumerate(zip(num_turns, targets)):
#     offset = 10 + (i % 3) * 10  # stagger vertically
#     timeline_ax.annotate(label, (x, 0), xytext=(0, offset), textcoords='offset points',
#                          fontsize=7, rotation=45, ha='center')

# timeline_ax.set_title("Number of Turns per Word")
# timeline_ax.set_xlabel("Number of Turns")
# timeline_ax.set_yticks([])
# timeline_ax.grid(True, axis='x', linestyle='--', alpha=0.3)

# === Timeline ===
timeline_ax.set_xlim(0, 20.5)
timeline_ax.set_ylim(-1, len(targets))
timeline_ax.set_yticks(range(len(targets)))
timeline_ax.set_yticklabels(targets, fontsize=7)
timeline_ax.set_xlabel("Number of Turns")
timeline_ax.set_title("Number of Turns per Word")
timeline_ax.grid(True, axis='x', linestyle='--', alpha=0.3)

# Horizontal markers at each word's turn count
timeline_ax.scatter(num_turns, range(len(targets)), marker='|', s=100, color='red')


plt.tight_layout()
plt.savefig(filepath + "/pie_plot.png", dpi=300)
# plt.show()
