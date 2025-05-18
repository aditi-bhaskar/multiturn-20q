# import json
# from collections import OrderedDict
# from datasets import load_dataset
# from collabllm.datasets.dataset import ChatDataset


# # **
# # {“role”: “user”, “content”: f“Given an object {object}, provide hints or answer to an assistant’s question. Forbid directly giving out the {object} so that at the end the assistant should figure out the object”}
# # **

# class TwentyQ(ChatDataset):

#     def __init__(self, repo_id='lighteval/TwentyQ'):
#         """
#         Initializes the TwentyQ dataset with raw data.
#         """
#         print("in init")

#         path = "/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data"
#         with open(f"{path}/train.json", 'r') as f:  # from local json file
#             raw_data = json.load(f)

#         processed_data = self.preprocess(raw_data)

#         print("going to process data!!")
#         with open(f"{path}/train_processed.json", "w") as out_f:
#             json.dump(processed_data, out_f, indent=2)

#         super().__init__(processed_data)

        
#     def preprocess(self, raw_data):
#         """
#         Processes the raw TwentyQ data into the format expected by ChatDataset.

#         Returns:
#         list: A list of processed chats with metadata.
#         """
#         processed_data = []
#         for split in raw_data.keys():
#             for entry in raw_data[split]:
#                 word = entry.get('word')
#                 lines = entry.get('lines', [])

#                 turns = []
#                 for line in lines:
#                     if '?' in line:
#                         q, a = line.split('?', 1)
#                         q = q.strip() + '?'
#                         a = a.strip()
#                         turns.append({'role': 'user', 'content': a})
#                         turns.append({'role': 'assistant', 'content': q})
#                     else:
#                         # fallback if any non-QA line is found
#                         turns.append({'role': 'assistant', 'content': line})

#                 # Add final answer as the last assistant message
#                 final_answer = word[0] if isinstance(word, list) else word
#                 turns.append({'role': 'assistant', 'content': f'The answer is: {final_answer}.'})

#                 metadata = {
#                     'split': split,  # ← moved from entry to here
#                     'level': None,
#                     'type': 'twenty_questions'
#                 }

#                 processed_data.append({'metadata': metadata, 'chat': turns})

#         return processed_data



import json
from collabllm.datasets.dataset import ChatDataset

class TwentyQ(ChatDataset):
    def __init__(self, path="/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data/eval.json"):
        print("Loading raw TwentyQ data from", path)
        with open(path, 'r') as f:
            raw_data = json.load(f)  # expecting a list of entries (no splits key)

        processed_data = self.preprocess(raw_data)

        # Optionally save processed data to file for inspection
        processed_path = path.replace(".json", "_processed.json")
        print("Saving processed data to", processed_path)
        with open(processed_path, "w") as out_f:
            json.dump(processed_data, out_f, indent=2)

        super().__init__(processed_data)

    def preprocess(self, raw_data):
        processed_data = []
        for entry in raw_data:
            lines = entry.get('lines', [])
            word = entry.get('word')
            turns = []

            for line in lines:
                if '?' in line:
                    # Split on the first '?'
                    question, answer = line.split('?', 1)
                    question = question.strip() + '?'
                    answer = answer.strip()
                    turns.append({'role': 'assistant', 'content': question})
                    if answer:
                        turns.append({'role': 'user', 'content': answer})
                else:
                    # Lines without '?' just become assistant turns (like the final answer)
                    turns.append({'role': 'assistant', 'content': line})

            # Append final assistant message giving the answer
            final_answer = word[0] if isinstance(word, list) else word
            turns.append({'role': 'assistant', 'content': f'The answer is: {final_answer}.'})

            metadata = {
                'type': 'twenty_questions',
                'source': 'local_file'
            }

            processed_data.append({'metadata': metadata, 'chat': turns})

        return processed_data
    
   

def main():
    dataset = TwentyQ()
    print(f"Loaded and processed {len(dataset)} conversations.")

if __name__ == "__main__":
    main()