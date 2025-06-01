import os
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

base_folder = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER_FR"


def plot_corr(x, y, labels, x_label, y_label, title, save_path):
    from collections import defaultdict

    # Fit regression
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y)
    model = LinearRegression().fit(X, Y)
    Y_pred = model.predict(X)
    r2 = r2_score(Y, Y_pred)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue')
    plt.plot(x, Y_pred, color='red', linewidth=2, label=f"Regression line (R² = {r2:.3f})")

    # Only label points with accuracy score 1
    point_to_labels = defaultdict(list)
    for xi, yi, label in zip(x, y, labels):
        if xi == 1.0:
            point_to_labels[(xi, yi)].append(label)

    # Stack labels vertically to the left of the point
    for (xi, yi), label_list in point_to_labels.items():
        for i, label in enumerate(label_list):
            offset = (i - (len(label_list) - 1) / 2) * 10  # spacing
            plt.annotate(
                label,
                (xi, yi),
                xytext=(-8, offset),
                textcoords="offset points",
                ha='right',
                va='center',
                fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}\nRegression R² = {r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def process_folder(folder_path):
    json_path = os.path.join(folder_path, "log.json")
    if not os.path.exists(json_path):
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    title = os.path.basename(folder_path).split("_2025")[0]
    games = sorted([k for k in data.keys() if k.isdigit()], key=int)

    accuracy_scores, info_gain_scores, interactivity_scores, targets = [], [], [], []

    for game_id in games:
        game = data[game_id]
        accuracy_scores.append(game["llm_judge"]["accuracy"]["score"])
        info_gain_scores.append(game["llm_judge"]["information_gain"]["score"])
        interactivity_scores.append(game["llm_judge"]["interactivity"]["score"])

        guess = game["qa"][-1]["content"]
        if "The answer is:" in guess:
            target = guess.split("The answer is:")[-1].strip().strip(".")
        else:
            target = ""
        targets.append(target)

    folder = folder_path
    plot_corr(accuracy_scores, info_gain_scores, targets,
              "Accuracy Score", "Information Gain Score",
              f"{title} — Accuracy vs Info Gain",
              os.path.join(folder, "plot_corr_accuracy_infogain.png"))

    plot_corr(accuracy_scores, interactivity_scores, targets,
              "Accuracy Score", "Interactivity Score",
              f"{title} — Accuracy vs Interactivity",
              os.path.join(folder, "plot_corr_accuracy_interactivity.png"))

    plot_corr(info_gain_scores, interactivity_scores, targets,
              "Information Gain Score", "Interactivity Score",
              f"{title} — Info Gain vs Interactivity",
              os.path.join(folder, "plot_corr_infogain_interactivity.png"))

# Run for each subdirectory
for entry in os.scandir(base_folder):
    if entry.is_dir():
        process_folder(entry.path)
