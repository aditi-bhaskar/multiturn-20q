# note: plots are not automatically saved
# plots the outcomes of the base llama 3.2 1b instruct model at playing 20q
json_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/outputs/eval/20q/local20q/test/LLAMA_BASE_EVALUATED_Llama-3.2-1B-Instruct_prompt=none_2025-05-22-19-27/log.json"


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

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

axs[0].scatter(num_turns, interactivity_scores)
axs[0].set_title("Interactivity vs Turns")
axs[0].set_xlabel("Number of Turns")
axs[0].set_ylabel("Interactivity Score")

axs[1].scatter(num_turns, accuracy_scores)
axs[1].set_title("Accuracy vs Turns")
axs[1].set_xlabel("Number of Turns")
axs[1].set_ylabel("Accuracy Score")

axs[2].scatter(num_turns, info_gain_scores)
axs[2].set_title("Information Gain vs Turns")
axs[2].set_xlabel("Number of Turns")
axs[2].set_ylabel("Information Gain Score")

plt.tight_layout()
plt.show()
