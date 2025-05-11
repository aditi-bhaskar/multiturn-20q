import json
from collections import OrderedDict
from datasets import load_dataset
from collabllm.datasets.dataset import ChatDataset


# **
# {“role”: “user”, “content”: f“Given an object {object}, provide hints or answer to an assistant’s question. Forbid directly giving out the {object} so that at the end the assistant should figure out the object”}
# **

class TwentyQ(ChatDataset):

    def __init__(self, repo_id='lighteval/TwentyQ'):
        """
        Initializes the TwentyQ dataset with raw data.
        """
        print("in init")

        path = "/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data"
        with open(f"{path}/train.json", 'r') as f:  # from local json file
            raw_data = json.load(f)

        processed_data = self.preprocess(raw_data)

        print("going to process data!!")
        with open(f"{path}/train_processed.json", "w") as out_f:
            json.dump(processed_data, out_f, indent=2)

        super().__init__(processed_data)

        
    def preprocess(self, raw_data):
        """
        Processes the raw TwentyQ data into the format expected by ChatDataset.

        Returns:
        list: A list of processed chats with metadata.
        """
        processed_data = []
        for split in raw_data.keys():
            for entry in raw_data[split]:
                word = entry.get('word')
                lines = entry.get('lines', [])

                turns = []
                for line in lines:
                    if '?' in line:
                        q, a = line.split('?', 1)
                        q = q.strip() + '?'
                        a = a.strip()
                        turns.append({'role': 'user', 'content': a})
                        turns.append({'role': 'assistant', 'content': q})
                    else:
                        # fallback if any non-QA line is found
                        turns.append({'role': 'assistant', 'content': line})

                # Add final answer as the last assistant message
                final_answer = word[0] if isinstance(word, list) else word
                turns.append({'role': 'assistant', 'content': f'The answer is: {final_answer}.'})

                metadata = {
                    'split': split,  # ← moved from entry to here
                    'level': None,
                    'type': 'twenty_questions'
                }

                processed_data.append({'metadata': metadata, 'chat': turns})

        return processed_data
