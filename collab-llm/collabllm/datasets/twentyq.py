import json
from collabllm.datasets.dataset import ChatDataset


# NOTE THAT WE ALWAYS SHOULD USE THE EVAL SET GOING FORWARDS!!

class TwentyQ(ChatDataset):
    def __init__(self, path="/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data/eval.json"):
    # def __init__(self, path="/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data/train.json"):
        print("Loading raw TwentyQ data from", path)
        with open(path, 'r') as f:
            raw_data = json.load(f)  # expecting a list of entries (no splits key)


        # normal process of converting format
        processed_data = self.preprocess(raw_data)
        # Optionally save processed data to file for inspection
        processed_path = path.replace(".json", "_processed.json")
        print("Saving processed data to", processed_path)
        with open(processed_path, "w") as out_f:
            json.dump(processed_data, out_f, indent=2)


        # single turn process
        processed_data = self.preprocess_single_turn(raw_data)
        # Optionally save processed data to file for inspection
        processed_path = path.replace(".json", "_single_turn.json")
        print("Saving single turn processed data to", processed_path)
        with open(processed_path, "w") as out_f:
            json.dump(processed_data, out_f, indent=2)

        super().__init__(processed_data) # use the single turn data for now


    def preprocess(self, raw_data):
        processed_data = []
        intro_message = "Let's play 20 questions! I'll answer yes or no to help you guess the object."

        for entry in raw_data:
            lines = entry.get('lines', [])
            word = entry.get('word')
            turns = []

            # Start chat with user intro message
            turns.append({'role': 'user', 'content': intro_message})

            for line in lines:
                if '?' in line:
                    question, answer = line.split('?', 1)
                    question = question.strip() + '?'
                    answer = answer.strip()
                    turns.append({'role': 'assistant', 'content': question})
                    if answer:
                        turns.append({'role': 'user', 'content': answer})
                else:
                    turns.append({'role': 'assistant', 'content': line})

            final_answer = word[0] if isinstance(word, list) else word
            turns.append({'role': 'assistant', 'content': f'The answer is: {final_answer}.'})

            metadata = {
                'type': 'twenty_questions',
                'source': 'local_file'
            }

            processed_data.append({'metadata': metadata, 'chat': turns})

        return processed_data

    def preprocess_single_turn(self, raw_data):
        processed_data = []
        intro_message = "Let's play 20 questions! I will answer yes or no to help you guess the object I'm thinking of."

        for entry in raw_data:
            # Extract the final answer (handle list or string)
            word = entry.get('word')
            final_answer = word[0] if isinstance(word, list) else word

            turns = [
                {'role': 'user', 'content': intro_message},
                {'role': 'assistant', 'content': f'The answer is: {final_answer}.'}
            ]

            metadata = {
                'type': 'twenty_questions_single_turn',
                'source': 'local_file'
            }

            # Add target_object field
            processed_data.append({
                'metadata': metadata,
                'target_object': final_answer,
                'chat': turns
            })

        return processed_data


def main():
    dataset = TwentyQ()
    print(f"Loaded and processed {len(dataset)} conversations.")

if __name__ == "__main__":
    main()