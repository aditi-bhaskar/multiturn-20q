
import os
import json
import matplotlib.pyplot as plt

# === Define folder and file ===
folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER_FR/Llama-3.2-1B-Instruct-20q_temp=0.7_2025-05-31-18-57-14"
json_path = os.path.join(folder_path, "log.json")

# === Manual label offsets ===
manual_offsets = {
    ("interactivity", 36): (0, -0.1),
    ("interactivity", 38): (0, -0.1),
    ("interactivity", 22): (1, -0.15),
    ("interactivity", 25): (0, -0.1),
    ("interactivity", 43): (1, -0.1),
    ("accuracy", 36): (1, -0.1),
    ("accuracy", 38): (0, -0.1),
    ("accuracy", 22): (1, -0.15),
    ("accuracy", 25): (0, -0.1),
    ("accuracy", 43): (1, -0.1),
    ("information_gain", 36): (1, -0.1),
    ("information_gain", 38): (0, -0.1),
    ("information_gain", 22): (1, -0.17),
    ("information_gain", 25): (0, -0.1),
    ("information_gain", 43): (1, -0.1),
    # Add more offsets as needed for more game IDs
}

# === Load Data ===
with open(json_path, "r") as f:
    data = json.load(f)

title = os.path.basename(folder_path).split("_2025")[0]
games = sorted([k for k in data.keys() if k.isdigit()], key=int)

num_turns, inter_scores, acc_scores, info_scores = [], [], [], []
labels, acc_values, game_ids = [], [], []

for gid in games:
    game = data[gid]
    turns = game["token_amount"]["num_turns"]
    inter = game["llm_judge"]["interactivity"]["score"]
    acc = game["llm_judge"]["accuracy"]["score"]
    info = game["llm_judge"]["information_gain"]["score"]

    guess = game["qa"][-1]["content"]
    target = guess.split("The answer is:")[-1].strip().strip(".") if "The answer is:" in guess else "N/A"
    label = f"{gid}: {target}"

    num_turns.append(turns)
    inter_scores.append(inter)
    acc_scores.append(acc)
    info_scores.append(info)
    labels.append(label)
    acc_values.append(acc)
    game_ids.append(int(gid))


# === Plotting Function ===
def plot_one(ax, x_vals, y_vals, labels, accs, title, ylabel, plot_type, gids):
    ax.scatter(x_vals, y_vals, s=40, color='tab:blue')
    ax.set_title(title)
    ax.set_xlabel("Number of Turns")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.3)

    print(f"\n--- Plotting: {plot_type.upper()} ---")
    for i, (x, y, label, acc, gid) in enumerate(zip(x_vals, y_vals, labels, accs, gids)):
        if acc == 1.0:
            dx, dy = manual_offsets.get((plot_type, gid), (0, 0))
            label_x = x + dx
            label_y = y + dy

            # Print label placement info in Spyder kernel
            print(f"Label: {label} | Game ID: {gid} | Offset: ({dx}, {dy}) | Final Pos: ({label_x:.2f}, {label_y:.2f})")

            ax.annotate(
                "", xy=(x, y), xytext=(label_x, label_y),
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
                annotation_clip=False
            )
            ax.text(
                label_x, label_y, label,
                ha='right', va='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8)
            )


# === Plot each graph one at a time ===

# 1. Interactivity
fig, ax = plt.subplots(figsize=(8, 6))
plot_one(ax, num_turns, inter_scores, labels, acc_values,
         f"{title} — Interactivity vs Turns", "Interactivity Score",
         "interactivity", game_ids)
plt.show()

# 2. Accuracy
fig, ax = plt.subplots(figsize=(8, 6))
plot_one(ax, num_turns, acc_scores, labels, acc_values,
         f"{title} — Accuracy vs Turns", "Accuracy Score",
         "accuracy", game_ids)
plt.show()

# 3. Information Gain
fig, ax = plt.subplots(figsize=(8, 6))
plot_one(ax, num_turns, info_scores, labels, acc_values,
         f"{title} — Information Gain vs Turns", "Information Gain Score",
         "information_gain", game_ids)
plt.show()




################################################################################################################################################################################
################################################################################################################################################################################

# import os
# import json
# import matplotlib.pyplot as plt

# # === Define folder and file ===
# folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER_FR/Llama-3.2-1B-Instruct-20q-reward_temp=0.7_2025-COMBINED_LOGS"
# json_path = os.path.join(folder_path, "log.json")

# # === Manual label offsets ===
# manual_offsets = {
#     ("interactivity", 35): (0, -0.2),
#     ("interactivity", 38): (0, -0.1),
#     ("interactivity", 5): (0, -0.05),
#     ("interactivity", 25): (-0.5, -0.1),
#     ("interactivity", 21): (1.5, -0.15),
#     ("interactivity", 20): (2, 0),
#     ("interactivity", 41): (-.5, -0.05),
#     ("accuracy", 35): (1.5, -0.2),
#     ("accuracy", 38): (0, -0.1),
#     ("accuracy", 20): (2, 0),
#     ("accuracy", 5): (-1, -0.15),
#     ("accuracy", 21): (2.5, -0.25),    
#     ("accuracy", 25): (0.5, -0.15),
#     ("accuracy", 41): (0, -0.1),
#     ("information_gain", 35): (0, -0.2),
#     ("information_gain", 38): (0, -0.1),
#     ("information_gain", 5): (0, -0.05),
#     ("information_gain", 25): (-0.5, -0.1),
#     ("information_gain", 21): (1.5, -0.15),
#     ("information_gain", 20): (2, 0),
#     ("information_gain", 41): (-.5, -0.05),
#     # Add more offsets as needed for more game IDs
# }

# # === Load Data ===
# with open(json_path, "r") as f:
#     data = json.load(f)

# title = os.path.basename(folder_path).split("_2025")[0]
# games = sorted([k for k in data.keys() if k.isdigit()], key=int)

# num_turns, inter_scores, acc_scores, info_scores = [], [], [], []
# labels, acc_values, game_ids = [], [], []

# for gid in games:
#     game = data[gid]
#     turns = game["token_amount"]["num_turns"]
#     inter = game["llm_judge"]["interactivity"]["score"]
#     acc = game["llm_judge"]["accuracy"]["score"]
#     info = game["llm_judge"]["information_gain"]["score"]

#     guess = game["qa"][-1]["content"]
#     target = guess.split("The answer is:")[-1].strip().strip(".") if "The answer is:" in guess else "N/A"
#     label = f"{gid}: {target}"

#     num_turns.append(turns)
#     inter_scores.append(inter)
#     acc_scores.append(acc)
#     info_scores.append(info)
#     labels.append(label)
#     acc_values.append(acc)
#     game_ids.append(int(gid))


# # === Plotting Function ===
# def plot_one(ax, x_vals, y_vals, labels, accs, title, ylabel, plot_type, gids):
#     ax.scatter(x_vals, y_vals, s=40, color='tab:blue')
#     ax.set_title(title)
#     ax.set_xlabel("Number of Turns")
#     ax.set_ylabel(ylabel)
#     ax.grid(True, linestyle='--', alpha=0.3)

#     print(f"\n--- Plotting: {plot_type.upper()} ---")
#     for i, (x, y, label, acc, gid) in enumerate(zip(x_vals, y_vals, labels, accs, gids)):
#         if acc == 1.0:
#             dx, dy = manual_offsets.get((plot_type, gid), (0, 0))
#             label_x = x + dx
#             label_y = y + dy

#             # Print label placement info in Spyder kernel
#             print(f"Label: {label} | Game ID: {gid} | Offset: ({dx}, {dy}) | Final Pos: ({label_x:.2f}, {label_y:.2f})")

#             ax.annotate(
#                 "", xy=(x, y), xytext=(label_x, label_y),
#                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
#                 annotation_clip=False
#             )
#             ax.text(
#                 label_x, label_y, label,
#                 ha='right', va='center',
#                 fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8)
#             )


# # === Plot each graph one at a time ===

# # 1. Interactivity
# fig, ax = plt.subplots(figsize=(8, 6))
# plot_one(ax, num_turns, inter_scores, labels, acc_values,
#          f"{title} — Interactivity vs Turns", "Interactivity Score",
#          "interactivity", game_ids)
# plt.show()

# # 2. Accuracy
# fig, ax = plt.subplots(figsize=(8, 6))
# plot_one(ax, num_turns, acc_scores, labels, acc_values,
#          f"{title} — Accuracy vs Turns", "Accuracy Score",
#          "accuracy", game_ids)
# plt.show()

# # 3. Information Gain
# fig, ax = plt.subplots(figsize=(8, 6))
# plot_one(ax, num_turns, info_scores, labels, acc_values,
#          f"{title} — Information Gain vs Turns", "Information Gain Score",
#          "information_gain", game_ids)
# plt.show()

############################################################################################################################################################################################
############################################################################################################################################################################################


# import os
# import json
# import matplotlib.pyplot as plt

# # === Define folder and file ===
# folder_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/PAPER_FR/Llama-3.2-1B-Instruct_temp=0.7_2025-COMBINED_LOGS"
# json_path = os.path.join(folder_path, "log.json")

# # === Manual label offsets ===
# manual_offsets = {
#     ("interactivity", 38): (0, -0.1),
#     ("interactivity", 43): (1, -0.1),
#     ("accuracy", 38): (0, -0.1),
#     ("accuracy", 43): (1, -0.1),
#     ("information_gain", 38): (0, -0.1),
#     ("information_gain", 43): (1, -0.1),
#     # Add more offsets as needed for more game IDs
# }

# # === Load Data ===
# with open(json_path, "r") as f:
#     data = json.load(f)

# title = os.path.basename(folder_path).split("_2025")[0]
# games = sorted([k for k in data.keys() if k.isdigit()], key=int)

# num_turns, inter_scores, acc_scores, info_scores = [], [], [], []
# labels, acc_values, game_ids = [], [], []

# for gid in games:
#     game = data[gid]
#     turns = game["token_amount"]["num_turns"]
#     inter = game["llm_judge"]["interactivity"]["score"]
#     acc = game["llm_judge"]["accuracy"]["score"]
#     info = game["llm_judge"]["information_gain"]["score"]

#     guess = game["qa"][-1]["content"]
#     target = guess.split("The answer is:")[-1].strip().strip(".") if "The answer is:" in guess else "N/A"
#     label = f"{gid}: {target}"

#     num_turns.append(turns)
#     inter_scores.append(inter)
#     acc_scores.append(acc)
#     info_scores.append(info)
#     labels.append(label)
#     acc_values.append(acc)
#     game_ids.append(int(gid))


# # === Plotting Function ===
# def plot_one(ax, x_vals, y_vals, labels, accs, title, ylabel, plot_type, gids):
#     ax.scatter(x_vals, y_vals, s=40, color='tab:blue')
#     ax.set_title(title)
#     ax.set_xlabel("Number of Turns")
#     ax.set_ylabel(ylabel)
#     ax.grid(True, linestyle='--', alpha=0.3)

#     print(f"\n--- Plotting: {plot_type.upper()} ---")
#     for i, (x, y, label, acc, gid) in enumerate(zip(x_vals, y_vals, labels, accs, gids)):
#         if acc == 1.0:
#             dx, dy = manual_offsets.get((plot_type, gid), (0, 0))
#             label_x = x + dx
#             label_y = y + dy

#             # Print label placement info in Spyder kernel
#             print(f"Label: {label} | Game ID: {gid} | Offset: ({dx}, {dy}) | Final Pos: ({label_x:.2f}, {label_y:.2f})")

#             ax.annotate(
#                 "", xy=(x, y), xytext=(label_x, label_y),
#                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.6),
#                 annotation_clip=False
#             )
#             ax.text(
#                 label_x, label_y, label,
#                 ha='right', va='center',
#                 fontsize=8,
#                 bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8)
#             )


# # === Plot each graph one at a time ===

# # 1. Interactivity
# fig, ax = plt.subplots(figsize=(8, 6))
# plot_one(ax, num_turns, inter_scores, labels, acc_values,
#          f"{title} — Interactivity vs Turns", "Interactivity Score",
#          "interactivity", game_ids)
# plt.show()

# # 2. Accuracy
# fig, ax = plt.subplots(figsize=(8, 6))
# plot_one(ax, num_turns, acc_scores, labels, acc_values,
#          f"{title} — Accuracy vs Turns", "Accuracy Score",
#          "accuracy", game_ids)
# plt.show()

# # 3. Information Gain
# fig, ax = plt.subplots(figsize=(8, 6))
# plot_one(ax, num_turns, info_scores, labels, acc_values,
#          f"{title} — Information Gain vs Turns", "Information Gain Score",
#          "information_gain", game_ids)
# plt.show()