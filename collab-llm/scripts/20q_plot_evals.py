# note: plots are not automatically saved
# plots the outcomes of the base llama 3.2 1b instruct model at playing 20q
json_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/LLAMA_BASE_EVALUATED_Llama-3.2-1B-Instruct_prompt=none_2025-05-22-19-27/log.json"


#  OBSOLETE << DO NOT USE

#  OBSOLETE << DO NOT USE

#  OBSOLETE << DO NOT USE

#  OBSOLETE << DO NOT USE

#  OBSOLETE << DO NOT USE

#  OBSOLETE << DO NOT USE


import json
import matplotlib.pyplot as plt

# Load your data (adjust path accordingly)
with open(json_path, "r") as f:
    data = json.load(f)


print(f"data = {data}")
# Filter only numeric game keys
games = sorted([k for k in data.keys() if k.isdigit()], key=int)

# Prepare lists for plotting
num_turns = []
interactivity_scores = []
accuracy_scores = []
info_gain_scores = []

print("gonna enter for loop")
print(f"games = {games}")

for game_id in games:
    game = data[game_id]
    # debug
    turns = game["token_amount"]["num_turns"]
    print(f"Game {game_id} num_turns:", turns, type(turns))

    num_turns.append(game["token_amount"]["num_turns"])
    llm_judge = game["llm_judge"]
    interactivity_scores.append(llm_judge["interactivity"]["score"])
    accuracy_scores.append(llm_judge["accuracy"]["score"])
    info_gain_scores.append(llm_judge["information_gain"]["score"])



# Extract target answers from the "qa" assistant content for each game
targets = []
for game_id in games:
    qa = data[game_id]["qa"]
    # Find the assistant's answer message, assume it contains "The answer is: XYZ."
    answer_text = None
    for turn in qa:
        if turn["role"] == "assistant" and "The answer is:" in turn["content"]:
            answer_text = turn["content"].split("The answer is:")[-1].strip().strip(".")
            break
    targets.append(answer_text or "")  # fallback empty string if not found

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

# Scatter plots
axs[0].scatter(num_turns, interactivity_scores)
axs[1].scatter(num_turns, accuracy_scores)
axs[2].scatter(num_turns, info_gain_scores)

# Titles and labels
axs[0].set_title("Turns vs. Interactivity")
axs[1].set_title("Turns vs. Accuracy")
axs[2].set_title("Turns vs. Information Gain")
for ax in axs:
    ax.set_xlabel("Number of Turns")
axs[0].set_ylabel("Interactivity Score")
axs[1].set_ylabel("Accuracy Score")
axs[2].set_ylabel("Information Gain Score")

# Annotate each point with target answer, below each dot on each plot
offset = -0.02  # vertical offset for label placement
for i, target in enumerate(targets):
    if target:
        axs[0].annotate(target, (num_turns[i], interactivity_scores[i] + offset), fontsize=7, ha='center', va='top', rotation=45)
        axs[1].annotate(target, (num_turns[i], accuracy_scores[i] + offset), fontsize=7, ha='center', va='top', rotation=45)
        axs[2].annotate(target, (num_turns[i], info_gain_scores[i] + offset), fontsize=7, ha='center', va='top', rotation=45)

plt.tight_layout()
plt.show()



# axs[0].scatter(num_turns, interactivity_scores)
# axs[0].set_title("Interactivity vs Turns")
# axs[0].set_xlabel("Number of Turns")
# axs[0].set_ylabel("Interactivity Score")

# axs[1].scatter(num_turns, accuracy_scores)
# axs[1].set_title("Accuracy vs Turns")
# axs[1].set_xlabel("Number of Turns")
# axs[1].set_ylabel("Accuracy Score")

# axs[2].scatter(num_turns, info_gain_scores)
# axs[2].set_title("Information Gain vs Turns")
# axs[2].set_xlabel("Number of Turns")
# axs[2].set_ylabel("Information Gain Score")

# plt.tight_layout()
# plt.show()
