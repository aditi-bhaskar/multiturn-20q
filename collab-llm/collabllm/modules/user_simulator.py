from typing import List
from collabllm.utils.api import get_llm_output
from collabllm.prompts import USER_SIMULATOR_PRONPTS
from collabllm.utils.template import chat_template
from collabllm.models import is_api_model_auto
from transformers import AutoTokenizer
from rich import print
import json
import argparse
import copy



class UserSimulator(object):
    def __init__(self, task_name, single_turn_data=None, **llm_kwargs):
        """
        Initialize the UserSimulator model.
        """
        super().__init__()
        self.task_name = task_name
        self.prompt_handler = USER_SIMULATOR_PRONPTS[task_name]
        self.llm_kwargs = llm_kwargs

        if task_name == '20q':
            self.object = self.llm_kwargs.get('target_object')
            if self.object is None:
                raise ValueError("Expected 'target_object' in llm_kwargs for 20q task.")
            self.question = self.object
            self.answer = self.object
        else:
            if single_turn_data is None:
                raise ValueError("Expected 'single_turn_data' for non-20q task.")
            self.question = [reply['content'] for reply in single_turn_data if reply['role'] == 'user'][0]
            self.answer = [reply['content'] for reply in single_turn_data if reply['role'] == 'assistant'][0]

        if is_api_model_auto(llm_kwargs['model']):
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_kwargs['model'])

    def __call__(self, messages: List[dict]):
        if len(messages) and messages[0]['role'] == 'system':
            messages = messages[1:]

        prompt = self.prompt_handler(
            question=self.question,
            answer=self.answer,
            chat_history=chat_template(messages)
        )

        if self.tokenizer is not None:
            chat = [{'content': prompt, 'role': 'system'}]
            prompt = self.tokenizer.encode(chat[0]['content'], return_tensors='pt')

        cnt = 0
        while True:
            cnt += 1
            response = get_llm_output(prompt, **self.llm_kwargs)

            # TODO
            # print('\n\n\nDEBUG:USERSIM response type:', type(response))
            # print('\n\n\nDEBUG:USERSIM response content:', response)
            print(prompt)
            breakpoint

            if isinstance(response, dict):
                try:
                    response = response['response']
                    break
                except Exception as e:
                    print(f'[UserSimulator] {e}')
            else:
                break
            if cnt > 5:
                import pdb; pdb.set_trace()


        if not isinstance(response, str):
            print(f'[UserSimulator] Unexpected response type: {type(response)}')
            response = ''
        return response.strip()

        # return response.strip()


# Function to write conversation data to a JSON file
def write_to_json(convs, pos_responses, neg_responses, idx):
    file_path = f"conversation_{idx}.json"
    
    conversation_data = {
        "conversation_id": idx,
        "conversation": [],
        "positive_response": pos_responses,
        "negative_response": neg_responses
    }

    for turn in convs:
        conversation_data["conversation"].append({
            "role": turn['role'],
            "content": turn['content']
        })

    with open(file_path, 'a') as f:
        json.dump(conversation_data, f, indent=4)
        f.write("\n")  # Add a newline to separate each conversation

# Process conversation function
def process_conversation(i, dataset, args, assistant_collabllm, assistant_vanilla):
    qa = dataset['chat'][i]
    question, answer = qa[-2]['content'], qa[-1]['content']

    print('*************** answer ****************\n', answer)
    
    conv = []
    exit_flag = False

    # This block is specific to the 20Q task
    if args.task == '20q':
        user_kwargs = {'target_object': answer}  # Only pass 'target_object' for 20Q tasks
    else:
        user_kwargs = {'single_turn_data': qa}  # This should be for non-20Q tasks, but we're ignoring that

    user = UserSimulator(task_name=args.task, **user_kwargs)

    convs = []
    pos_responses, neg_responses = [], []

    for _ in range(args.max_new_turns):
        # Initialize user_response to an empty string to prevent the UnboundLocalError
        user_response = ""

        try:
            # Only call user() for the 20Q task, since that's the only one you're running
            user_response = user(conv)
            if user_response is None or user_response.strip() == "":  # Ensure it's not empty
                print(f"[Turn {len(conv)}] **User**: Empty or None response, skipping turn.")
                continue  # Skip the empty response and continue with the next iteration
        except Exception as e:
            print(f"[Turn {len(conv)}] **Error**: {str(e)} - skipping turn.")
            continue

        # Check if termination condition is met
        if '[[TERMINATE CHAT]]' in user_response: 
            exit_flag = True 
            user_response = user_response.replace('[[TERMINATE CHAT]]', '')
        
        if not user_response.strip():  # Skip empty or whitespace-only responses
            print(f"[Turn {len(conv)}] **User**: Empty response received, skipping turn.")
            continue  # Skip the empty response and continue with the next iteration
        
        conv.append({'role': 'user', 'content': user_response})
        print(f"[Turn {len(conv)}] **User**: {user_response}")
        
        if exit_flag:
            break
        
        # Get assistant responses
        responses = [assistant_collabllm(conv, question=question, answer=answer), assistant_vanilla(conv)]
        
        # Simple response selection logic
        pos_response = responses[0]  # You can add reward logic here if needed
        neg_response = responses[1]

        # Append the responses and conversation turns
        convs.append(copy.deepcopy(conv))
        pos_responses.append(pos_response)
        neg_responses.append(neg_response)

        # Write the data to a JSON file
        write_to_json(convs, pos_responses, neg_responses, i)

        # Print the chosen and rejected responses
        print(f"[Turn {len(conv)}] Chosen: {pos_response}\n")
        print(f"[Turn {len(conv)}] Rejected: {neg_response}\n\n")

    return i, convs, pos_responses, neg_responses







# Main function to handle argument parsing and process each conversation
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_object', type=str, required=True, help="The object for the 20Q task")
    args = parser.parse_args()

    # Dummy dataset (Replace with actual dataset loading logic)
    dataset = {'train': {'chat': []}}  # This should be replaced with the actual dataset loading logic

    assistant_collabllm = lambda conv, question, answer: "Assistant response (collabllm)"
    assistant_vanilla = lambda conv: "Assistant response (vanilla)"

    # Iterate through each conversation in the dataset
    for i in range(len(dataset['train']['chat'])):
        i, convs, pos_responses, neg_responses = process_conversation(i, dataset['train'], args, assistant_collabllm, assistant_vanilla)