# from typing import List
# from collabllm.utils.api import get_llm_output
# from collabllm.prompts import USER_SIMULATOR_PRONPTS
# from collabllm.utils.template import chat_template
# from collabllm.models import get_meta_info_from_model_name, is_api_model_auto
# from transformers import AutoTokenizer
# from rich import print

# # note

# # if task_name == '20q':
# #     self.object = kwargs['target_object']
# #     self.question = self.object  # to fit existing prompt template
# #     self.answer = self.object

# class UserSimulator(object):
#     def __init__(self, task_name, single_turn_data, **llm_kwargs):
#         """
#         Initialize the UserSimulator model.
#         """
#         super().__init__()
#         self.task_name = task_name
#         self.prompt_handler = USER_SIMULATOR_PRONPTS[task_name]
#         self.llm_kwargs = llm_kwargs


# #  TODO INTEGRASTE TYHIS THIS
# # user_simulator = UserSimulator(
# #     task_name='20q',
# #     single_turn_data=some_data,
# #     target_object=TARGET_OBJECT,  # Make sure 'target_object' is in kwargs
# #     model='gpt-4o-mini',
# #     # other args
# # )

#         if task_name == '20q':  # aditi addition
#             # self.object = self.llm_kwargs['target_object']
#             self.object = self.llm_kwargs.get('target_object')
#             if self.object is None:
#                 raise ValueError("Expected 'target_object' in llm_kwargs for 20q task.")
#             self.question = self.object
#             # use this for question, instead?? f"Play a game of 20 questions: Given an object {self.object}, provide yes/no answers to the assistant’s question about the object. Forbid directly giving out the {self.object} so that at the end the assistant should figure out the object"
#             self.answer = self.object
#         else: 
#             # single_turn_data = llm_kwargs.pop('single_turn_data')
#             self.question = [reply['content'] for reply in single_turn_data if reply['role'] == 'user'][0]
#             self.answer = [reply['content'] for reply in single_turn_data if reply['role'] == 'assistant'][0]
#             if 'single_turn_data' in llm_kwargs:
#                 single_turn_data = llm_kwargs.pop('single_turn_data')
#             else:
#                 raise ValueError("Expected 'single_turn_data' in llm_kwargs for non-20q task.")
       
#         if is_api_model_auto(llm_kwargs['model']):
#             self.tokenizer = None
#         else:
#             self.tokenizer = AutoTokenizer.from_pretrained(llm_kwargs['model'])

#     def __call__(self, messages: List[dict]):
#         if len(messages) and messages[0]['role'] == 'system':
#             messages = messages[1:]
        
#         prompt = self.prompt_handler(question=self.question,
#                                      answer=self.answer,
#                                      chat_history=chat_template(messages)
#                                      )
#         if self.tokenizer is not None:
#             chat = [{'content': prompt, 'role': 'system'}]
#             meta_info = get_meta_info_from_model_name(self.llm_kwargs['model'])
#             prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
#             prompt = prompt + meta_info['response_template'] + "\n\n**user**: "
        
#         # TODO double check that prompt makes sense

#         cnt = 0
#         while True:
#             cnt += 1
#             response = get_llm_output(prompt, **self.llm_kwargs)
#             # TODO see if response ^ seems ok from the user simulator

#             if isinstance(response, dict):
#                 try:
#                     keys = response.keys()
#                     current_answer = response.pop('current_answer')
#                     thought = response.pop('thought')
#                     response = response['response']
#                     with open('logs/user_simulator.txt', 'a+') as f:
#                         f.write(f'\n\n[UserSimulator] `current_answer`={current_answer} | `thought`={thought}\n\n')
#                     break
#                 except Exception as e:
#                     print(f'[UserSimulator] {e}')
#             else:
#                 break
#             if cnt > 5:
#                 import pdb; pdb.set_trace()
#         return response.strip()



# # from typing import List
# # from collabllm.utils.api import get_llm_output
# # from collabllm.prompts import USER_SIMULATOR_PRONPTS
# # from collabllm.utils.template import chat_template
# # from collabllm.models import get_meta_info_from_model_name, is_api_model_auto
# # from transformers import AutoTokenizer
# # from rich import print

# # class UserSimulator(object):
# #     def __init__(self, task_name, single_turn_data, **llm_kwargs):
# #         """
# #         Initialize the UserSimulator model.
# #         """
# #         super().__init__()
# #         self.task_name = task_name
# #         self.prompt_handler = USER_SIMULATOR_PRONPTS[task_name]
# #         self.llm_kwargs = llm_kwargs

# #         if task_name == '20q':  # Aditi addition for 20Q task
# #             # Expecting 'target_object' in the llm_kwargs
# #             self.object = self.llm_kwargs.get('target_object')
# #             if self.object is None:
# #                 raise ValueError("Expected 'target_object' in llm_kwargs for 20q task.")
            
# #             # Set the question and answer as the target object for the game
# #             self.question = self.object
# #             self.answer = self.object
# #         else:
# #             # For non-20q tasks, extract question and answer from single_turn_data
# #             self.question = [reply['content'] for reply in single_turn_data if reply['role'] == 'user'][0]
# #             self.answer = [reply['content'] for reply in single_turn_data if reply['role'] == 'assistant'][0]
# #             if 'single_turn_data' in llm_kwargs:
# #                 single_turn_data = llm_kwargs.pop('single_turn_data')
# #             else:
# #                 raise ValueError("Expected 'single_turn_data' in llm_kwargs for non-20q task.")

# #         if is_api_model_auto(llm_kwargs['model']):
# #             self.tokenizer = None
# #         else:
# #             self.tokenizer = AutoTokenizer.from_pretrained(llm_kwargs['model'])

# #     def __call__(self, messages: List[dict]):
# #         if len(messages) and messages[0]['role'] == 'system':
# #             messages = messages[1:]
        
# #         prompt = self.prompt_handler(question=self.question,
# #                                      answer=self.answer,
# #                                      chat_history=chat_template(messages)
# #                                      )
# #         if self.tokenizer is not None:
# #             chat = [{'content': prompt, 'role': 'system'}]
# #             meta_info = get_meta_info_from_model_name(self.llm_kwargs['model'])
# #             prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
# #             prompt = prompt + meta_info['response_template'] + "\n\n**user**: "
        
# #         # TODO: double check that prompt makes sense

# #         cnt = 0
# #         while True:
# #             cnt += 1
# #             response = get_llm_output(prompt, **self.llm_kwargs)

# #             if isinstance(response, dict):
# #                 try:
# #                     keys = response.keys()
# #                     current_answer = response.pop('current_answer')
# #                     thought = response.pop('thought')
# #                     response = response['response']
# #                     with open('logs/user_simulator.txt', 'a+') as f:
# #                         f.write(f'\n\n[UserSimulator] `current_answer`={current_answer} | `thought`={thought}\n\n')
# #                     break
# #                 except Exception as e:
# #                     print(f'[UserSimulator] {e}')
# #             else:
# #                 break
# #             if cnt > 5:
# #                 import pdb; pdb.set_trace()
# #         return response.strip()


# #  uses the obj from the sh file

# import argparse
# from typing import List
# from collabllm.utils.api import get_llm_output
# from collabllm.prompts import USER_SIMULATOR_PRONPTS
# from collabllm.utils.template import chat_template
# from collabllm.models import get_meta_info_from_model_name, is_api_model_auto
# from transformers import AutoTokenizer
# from rich import print

# class UserSimulator(object):
#     def __init__(self, task_name, single_turn_data, **llm_kwargs):
#         """
#         Initialize the UserSimulator model.
#         """
#         super().__init__()
#         self.task_name = task_name
#         self.prompt_handler = USER_SIMULATOR_PRONPTS[task_name]
#         self.llm_kwargs = llm_kwargs

#         if task_name == '20q':  # Aditi addition for 20Q task
#             # Expecting 'target_object' in the llm_kwargs
#             self.object = self.llm_kwargs.get('target_object')
#             if self.object is None:
#                 raise ValueError("Expected 'target_object' in llm_kwargs for 20q task.")
            
#             # Set the question and answer as the target object for the game
#             self.question = self.object
#             self.answer = self.object
#         else:
#             # For non-20q tasks, extract question and answer from single_turn_data
#             self.question = [reply['content'] for reply in single_turn_data if reply['role'] == 'user'][0]
#             self.answer = [reply['content'] for reply in single_turn_data if reply['role'] == 'assistant'][0]
#             if 'single_turn_data' in llm_kwargs:
#                 single_turn_data = llm_kwargs.pop('single_turn_data')
#             else:
#                 raise ValueError("Expected 'single_turn_data' in llm_kwargs for non-20q task.")

#         if is_api_model_auto(llm_kwargs['model']):
#             self.tokenizer = None
#         else:
#             self.tokenizer = AutoTokenizer.from_pretrained(llm_kwargs['model'])

#     def __call__(self, messages: List[dict]):
#         if len(messages) and messages[0]['role'] == 'system':
#             messages = messages[1:]
        
#         prompt = self.prompt_handler(question=self.question,
#                                      answer=self.answer,
#                                      chat_history=chat_template(messages)
#                                      )
#         if self.tokenizer is not None:
#             chat = [{'content': prompt, 'role': 'system'}]
#             meta_info = get_meta_info_from_model_name(self.llm_kwargs['model'])
#             prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
#             prompt = prompt + meta_info['response_template'] + "\n\n**user**: "
        
#         # TODO: double check that prompt makes sense

#         cnt = 0
#         while True:
#             cnt += 1
#             response = get_llm_output(prompt, **self.llm_kwargs)

#             if isinstance(response, dict):
#                 try:
#                     keys = response.keys()
#                     current_answer = response.pop('current_answer')
#                     thought = response.pop('thought')
#                     response = response['response']
#                     with open('logs/user_simulator.txt', 'a+') as f:
#                         f.write(f'\n\n[UserSimulator] `current_answer`={current_answer} | `thought`={thought}\n\n')
#                     break
#                 except Exception as e:
#                     print(f'[UserSimulator] {e}')
#             else:
#                 break
#             if cnt > 5:
#                 import pdb; pdb.set_trace()
#         return response.strip()

# # Command-line argument parsing (integrating argparse)
# if __name__ == "__main__":
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--target_object', type=str, required=True, help="The object for the 20Q task")
#     args = parser.parse_args()

#     # Assuming `some_data` is already loaded or passed for non-20Q tasks
#     some_data = []  # This would come from your data or pre-processing

#     # Instantiate the UserSimulator with the target_object passed from the command line
#     user_simulator = UserSimulator(
#         task_name='20q',
#         single_turn_data=some_data,  # Provide any other necessary data for non-20q tasks
#         target_object=args.target_object,  # Pass the target_object received from the command line
#         model='gpt-4o-mini',  # Specify the model name
#     )

#     # Use the UserSimulator to run the task
#     messages = [
#         {'role': 'system', 'content': 'Start the 20Q game.'},
#         {'role': 'user', 'content': 'Is it an electronic device?'},
#     ]

#     # Get the simulated response
#     response = user_simulator(messages)

#     # Print the response or use it for further processing
#     print(response)



# # writes output to a json file

# import argparse
# import random
# import json
# import copy
# import concurrent.futures
# from typing import List
# from tqdm import tqdm
# from collabllm.utils.api import get_llm_output
# from collabllm.prompts import USER_SIMULATOR_PRONPTS
# from collabllm.utils.template import chat_template
# from collabllm.models import get_meta_info_from_model_name, is_api_model_auto
# from transformers import AutoTokenizer
# from rich import print


# # Write conversation data to a JSON file
# def write_to_json(convs, pos_responses, neg_responses, idx):
#     # Define the file path (it will create or append to a JSON file)
#     file_path = f"conversation_{idx}.json"
    
#     # Prepare the data to be written as a dictionary
#     conversation_data = {
#         "conversation_id": idx,
#         "conversation": [],
#         "positive_response": pos_responses,
#         "negative_response": neg_responses
#     }

#     for turn in convs:
#         conversation_data["conversation"].append({
#             "role": turn['role'],
#             "content": turn['content']
#         })

#     # Open the JSON file and write the conversation data
#     with open(file_path, 'a') as f:  # Open in append mode to add each conversation
#         json.dump(conversation_data, f, indent=4)
#         f.write("\n")  # Add a newline to separate each conversation


# # User Simulator Class
# class UserSimulator(object):
#     def __init__(self, task_name, single_turn_data, **llm_kwargs):
#         """
#         Initialize the UserSimulator model.
#         """
#         super().__init__()
#         self.task_name = task_name
#         self.prompt_handler = USER_SIMULATOR_PRONPTS[task_name]
#         self.llm_kwargs = llm_kwargs

#         if task_name == '20q':  # Aditi addition for 20Q task
#             # Expecting 'target_object' in the llm_kwargs
#             self.object = self.llm_kwargs.get('target_object')
#             if self.object is None:
#                 raise ValueError("Expected 'target_object' in llm_kwargs for 20q task.")
            
#             # Set the question and answer as the target object for the game
#             self.question = self.object
#             self.answer = self.object
#         else:
#             # For non-20q tasks, extract question and answer from single_turn_data
#             self.question = [reply['content'] for reply in single_turn_data if reply['role'] == 'user'][0]
#             self.answer = [reply['content'] for reply in single_turn_data if reply['role'] == 'assistant'][0]
#             if 'single_turn_data' in llm_kwargs:
#                 single_turn_data = llm_kwargs.pop('single_turn_data')
#             else:
#                 raise ValueError("Expected 'single_turn_data' in llm_kwargs for non-20q task.")

#         if is_api_model_auto(llm_kwargs['model']):
#             self.tokenizer = None
#         else:
#             self.tokenizer = AutoTokenizer.from_pretrained(llm_kwargs['model'])

#     def __call__(self, messages: List[dict]):
#         if len(messages) and messages[0]['role'] == 'system':
#             messages = messages[1:]
        
#         prompt = self.prompt_handler(question=self.question,
#                                      answer=self.answer,
#                                      chat_history=chat_template(messages)
#                                      )
#         if self.tokenizer is not None:
#             chat = [{'content': prompt, 'role': 'system'}]
#             meta_info = get_meta_info_from_model_name(self.llm_kwargs['model'])
#             prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
#             prompt = prompt + meta_info['response_template'] + "\n\n**user**: "
        
#         # TODO: double check that prompt makes sense
#         cnt = 0
#         while True:
#             cnt += 1
#             response = get_llm_output(prompt, **self.llm_kwargs)

#             if isinstance(response, dict):
#                 try:
#                     keys = response.keys()
#                     current_answer = response.pop('current_answer')
#                     thought = response.pop('thought')
#                     response = response['response']
#                     with open('logs/user_simulator.txt', 'a+') as f:
#                         f.write(f'\n\n[UserSimulator] `current_answer`={current_answer} | `thought`={thought}\n\n')
#                     break
#                 except Exception as e:
#                     print(f'[UserSimulator] {e}')
#             else:
#                 break
#             if cnt > 5:
#                 import pdb; pdb.set_trace()
#         return response.strip()


# # Main Conversation Processing Logic
# def process_conversation(i, dataset, args, assistant_collabllm, assistant_vanilla):
#     qa = dataset['chat'][i]
#     question, answer = qa[-2]['content'], qa[-1]['content']

#     if answer.strip().startswith('{'):
#         answer = extract_json(answer)['answer']
#     print('*************** answer ****************\n', answer)
    
#     conv = []
#     exit_flag = False

#     task = datasets_info[args.dataset]['task']
#     user_kwargs = user_generation_kwargs.copy()
#     if task == '20q':
#         user_kwargs['target_object'] = answer
#     else:
#         user_kwargs['single_turn_data'] = qa

#     user = UserSimulator(
#         task_name=task,
#         **user_kwargs
#     )

#     # Lists to store the results for each turn
#     convs = []
#     pos_responses, neg_responses = [], []

#     for _ in tqdm(range(args.max_new_turns), desc=f"Processing conversation {i}"):
#         user_response = user(conv)
        
#         # Check for termination condition
#         if '[[TERMINATE CHAT]]' in user_response: 
#             exit_flag = True 
#             user_response = user_response.replace('[[TERMINATE CHAT]]', '')
        
#         conv.append({'role': 'user', 'content': user_response})
#         print(f"[Turn {len(conv)}] **User**: {user_response}")
#         if exit_flag:
#             break
        
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future_collabllm = executor.submit(assistant_collabllm, conv, question=question, answer=answer)
#             future_vanilla = executor.submit(assistant_vanilla, conv)
            
#             responses = [future_collabllm.result(), future_vanilla.result()]
        
#         rewards, reward_logs = get_multiturn_rewards(
#             task_name=datasets_info[args.dataset]['task'],
#             single_turn_ds=[qa for _ in range(len(responses))],
#             chat_histories=[conv for _ in range(len(responses))],
#             responses=responses,
#             max_workers=args.max_workers,
#             num_samples=args.num_samples,
#             llm_rw_weight=args.llm_rw_weight,
#             window_size=args.window_size,
#             task_weight=args.task_weight,
#             cost_weight=args.cost_weight,
#             user_generation_kwargs=user_generation_kwargs,
#             assistant_generation_kwargs=assistant_generation_kwargs,
#             reward_generation_kwargs=reward_generation_kwargs,
#             local_model=None, local_tokenizer=None
#         )
        
#         # Select positive and negative responses
#         pos_response = responses[np.argmax(rewards)]
#         neg_response = responses[np.argmin(rewards)]

#         # Append the positive and negative responses, along with the conversation
#         convs.append(copy.deepcopy(conv))
#         pos_responses.append(pos_response)
#         neg_responses.append(neg_response)

#         # Write the conversation and responses to a JSON file
#         write_to_json(convs, pos_responses, neg_responses, i)

#         # Log or print additional details (optional)
#         print(f"[Turn {len(conv)}] Chosen: {pos_response}\n")
#         print(f"[Turn {len(conv)}] Rejected: {neg_response}\n\n")

#     return i, convs, pos_responses, neg_responses


# # Main entry point
# if __name__ == "__main__":
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--target_object', type=str, required=True, help="The object for the 20Q task")
#     args = parser.parse_args()

#     # Assuming `some_data` is already loaded or passed for non-20Q tasks
#     some_data = []  # This would come from your data or pre-processing

#     # Load dataset and models, as needed
#     dataset = load_single_turn_dataset(args.dataset, add_system_prompt=False)

#     assistant_collabllm = LLMAssistant(method='collabllm_cot', **assistant_generation_kwargs)
#     vanilla_generation_kwargs = copy.copy(assistant_generation_kwargs)
#     vanilla_generation_kwargs['json_object'] = False
#     assistant_vanilla = LLMAssistant(method='none', **vanilla_generation_kwargs)

#     for i in tqdm(range(len(dataset['train']['chat']))):
#         i, convs, pos_responses, neg_responses = process_conversation(i, dataset['train'], args, assistant_collabllm, assistant_vanilla)




#  removed random extra code?


from typing import List
from collabllm.utils.api import get_llm_output
from collabllm.prompts import USER_SIMULATOR_PRONPTS
from collabllm.utils.template import chat_template
from collabllm.models import is_api_model_auto
from transformers import AutoTokenizer
from rich import print
import json

class UserSimulator(object):
    def __init__(self, task_name, single_turn_data, **llm_kwargs):
        """
        Initialize the UserSimulator model.
        """
        super().__init__()
        self.task_name = task_name
        self.prompt_handler = USER_SIMULATOR_PRONPTS[task_name]
        self.llm_kwargs = llm_kwargs

        if task_name == '20q':  # aditi addition
            self.object = self.llm_kwargs.get('target_object')
            if self.object is None:
                raise ValueError("Expected 'target_object' in llm_kwargs for 20q task.")
            self.question = self.object
            self.answer = self.object
        else: 
            self.question = [reply['content'] for reply in single_turn_data if reply['role'] == 'user'][0]
            self.answer = [reply['content'] for reply in single_turn_data if reply['role'] == 'assistant'][0]
            if 'single_turn_data' in llm_kwargs:
                single_turn_data = llm_kwargs.pop('single_turn_data')
            else:
                raise ValueError("Expected 'single_turn_data' in llm_kwargs for non-20q task.")
       
        if is_api_model_auto(llm_kwargs['model']):
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_kwargs['model'])

    def __call__(self, messages: List[dict]):
        if len(messages) and messages[0]['role'] == 'system':
            messages = messages[1:]
        
        prompt = self.prompt_handler(question=self.question,
                                     answer=self.answer,
                                     chat_history=chat_template(messages)
                                     )
        
        if self.tokenizer is not None:
            chat = [{'content': prompt, 'role': 'system'}]
            prompt = self.tokenizer.encode(chat[0]['content'], return_tensors='pt')  # Direct encoding without meta-info
        
        cnt = 0
        while True:
            cnt += 1
            response = get_llm_output(prompt, **self.llm_kwargs)

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
        return response.strip()

# Write conversation data to a JSON file
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


def process_conversation(i, dataset, args, assistant_collabllm, assistant_vanilla):
    qa = dataset['chat'][i]
    question, answer = qa[-2]['content'], qa[-1]['content']

    print('*************** answer ****************\n', answer)
    
    conv = []
    exit_flag = False

    user_kwargs = {'target_object': answer} if args.task == '20q' else {'single_turn_data': qa}

    user = UserSimulator(task_name=args.task, **user_kwargs)

    convs = []
    pos_responses, neg_responses = [], []

    for _ in range(args.max_new_turns):
        user_response = user(conv)
        
        if '[[TERMINATE CHAT]]' in user_response: 
            exit_flag = True 
            user_response = user_response.replace('[[TERMINATE CHAT]]', '')
        
        conv.append({'role': 'user', 'content': user_response})
        print(f"[Turn {len(conv)}] **User**: {user_response}")
        if exit_flag:
            break
        
        responses = [assistant_collabllm(conv, question=question, answer=answer), assistant_vanilla(conv)]
        
        # Rewards logic and response selection, simplified
        pos_response = responses[0]  # You can add reward logic here if needed
        neg_response = responses[1]

        convs.append(copy.deepcopy(conv))
        pos_responses.append(pos_response)
        neg_responses.append(neg_response)

        write_to_json(convs, pos_responses, neg_responses, i)

        print(f"[Turn {len(conv)}] Chosen: {pos_response}\n")
        print(f"[Turn {len(conv)}] Rejected: {neg_response}\n\n")

    return i, convs, pos_responses, neg_responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_object', type=str, required=True, help="The object for the 20Q task")
    args = parser.parse_args()

    dataset = {'train': {'chat': []}}  # This should be replaced with the actual dataset loading logic

    assistant_collabllm = lambda conv, question, answer: "Assistant response (collabllm)"
    assistant_vanilla = lambda conv: "Assistant response (vanilla)"

    for i in range(len(dataset['train']['chat'])):
        i, convs, pos_responses, neg_responses = process_conversation(i, dataset['train'], args, assistant_collabllm, assistant_vanilla)
