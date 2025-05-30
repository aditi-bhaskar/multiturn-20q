import json
from random import choice

input_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data/eval_processed.json"
output_path = "/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data/eval_mtpo.json"

with open(input_path, "r") as f:
    data = json.load(f)

good_convos = []
bad_convos = []

def convo_to_text(chat):
    # Convert the chat turns to a single string context
    return "\n".join(f'{turn["role"]}: {turn["content"]}' for turn in chat)

# Separate good and bad conversations by looking at last assistant's final guess line
for convo in data:
    chat = convo["chat"]
    if len(chat) < 2:
        continue

    # Look for final assistant guess of the form "The answer is: ..."
    final_index = None
    for i in range(len(chat) - 1, -1, -1):
        if chat[i]["role"] == "assistant" and "the answer is" in chat[i]["content"].lower():
            final_index = i
            break

    if final_index is None or final_index == 0:
        continue

    # Check if previous user message confirms or denies the guess
    user_reply = chat[final_index - 1]
    if user_reply["role"] != "user":
        continue

    reply_text = user_reply["content"].strip().lower()
    flattened = convo_to_text(chat[:final_index + 1])  # up to and including final guess

    if reply_text == "yes." or reply_text == "yes":
        good_convos.append(flattened)
    elif reply_text == "no." or reply_text == "no":
        bad_convos.append(flattened)


# Now create MTPO pairs by pairing each good convo with a random bad convo
mtpo_data = []
for good in good_convos:
    if bad_convos:
        bad = choice(bad_convos)
        mtpo_data.append({
            "context": good,
            "good_response": "Final guess was correct.",
            "bad_response": "Final guess was incorrect."
        })

with open(output_path, "w") as f:
    json.dump(mtpo_data, f, indent=2)

print(f"Created {len(mtpo_data)} MTPO training samples.")
