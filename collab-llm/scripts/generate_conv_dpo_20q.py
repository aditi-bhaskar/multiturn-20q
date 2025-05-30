import argparse
import copy
import concurrent.futures
from datasets import Dataset, DatasetDict, load_dataset

import copy
import random
import logging
import numpy as np
np.set_printoptions(precision=3)

from rich import print
from tqdm import tqdm

import datetime

import re
import sys
sys.path.append('.')

#  aditi addition
import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
os.makedirs("logs", exist_ok=True)

from collabllm.core.multithread import get_multiturn_rewards
from collabllm.datasets import load_single_turn_dataset, datasets_info, split_train_dev_datasets
from collabllm.utils.extract_json_reliable import extract_json

from collabllm.modules import LLMAssistant, UserSimulator

import json


PUSH_REPO = "aditijb/collabllm-20q-filtered-reward"

def parse_args():
    parser = argparse.ArgumentParser()
    def list_of_strings(arg):
      return arg.split(',')

    parser.add_argument('--dataset', type=str, default='math-hard', 
                    help='available datasets under collabllm.datasets.datasets_info')
    
    parser.add_argument('--num_samples', type=int, default=3)

    parser.add_argument('--n_eval_per_dataset', type=int, default=500)
    parser.add_argument('--max_num_conv', type=int, default=1000)

    parser.add_argument('--max_new_turns', type=int, default=10)
    parser.add_argument('--start_obj_num', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=2)

    parser.add_argument('--llm_rw_weight', type=float, default=1)
    parser.add_argument('--task_weight', type=float, default=10)
    parser.add_argument('--cost_weight', type=float, default=1e-3)
    
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_new_tokens', type=int, default=768) # 1024
    parser.add_argument('--temperature', type=float, default=0.8)

    parser.add_argument('--user_model_name', type=str, default='gpt-4o')
    parser.add_argument('--assistant_model_name', type=str, default='gpt-4o')
    parser.add_argument('--reward_model', type=str, default='gpt-4o')
    parser.add_argument('--hf_org', type=str, default='org_name')

    parser.add_argument('--log_step', type=int, default=3)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    
    # parser.add_argument('--max_workers', type=int, default=30)
    parser.add_argument('--max_workers', type=int, default=1)

    # aditi's additions
    parser.add_argument('--target_object', type=str, help="Target object for the 20q task")
    parser.add_argument("--task_name", type=str, default="20q", help="Specify the task (default: 20q)")

    return parser.parse_args()


args = parse_args()
logging.getLogger('tensorflow').setLevel(logging.ERROR)
print('RESUME STATUS: ', args.resume)
assistant_generation_kwargs = {
   "model": args.assistant_model_name,
   "top_p": args.top_p,
   "temperature": args.temperature,
   "max_new_tokens": args.max_new_tokens,
   "json_object": True
}

reward_generation_kwargs = {
   "model": args.reward_model,
   "top_p": 0.9,
   "temperature": 0.2,
   "max_new_tokens": 4096
}

user_generation_kwargs = {
    "model": args.user_model_name,
    "top_p": 1.0,
    "temperature": 1.0,
    "max_new_tokens": 4096,
    "json_object": True
}



def process_conversation(i, dataset, args, assistant_collabllm, assistant_vanilla):

    qa = dataset['chat'][i]  
    question, answer = qa[-2]['content'], qa[-1]['content'] # this is where our previous answer/question was coming from


    if answer.strip().startswith('{'):
        answer = extract_json(answer)['answer']
    print('\n*************** answer ****************\n', answer)
    print('\n*************** question ****************\n', question)  # is technically never used anywhere.


    conv = []
    exit_flag = False

    # ADDED CODE HERE
    task = args.task_name  # aditi modif
    user_kwargs = user_generation_kwargs.copy()

    if task == '20q':
        #  aditi edit to choose a random object
        user_kwargs['target_object'] = answer
        # print(f"\nDEBUG user_kwargs['target_object']={user_kwargs['target_object']}\n") 

        # user_kwargs['target_object'] = answer
        user_kwargs.pop('single_turn_data', None)  # Ensure it's not accidentally passed
    else:
        print("IS NOT IN TASK 20q. EXITING")
        return -1
        # user_kwargs['single_turn_data'] = qa

    user = UserSimulator(
        # is_20q=True,  # added by aditi for 20q task
        task_name=task,
        **user_kwargs
    )

    # Initialize user_response to prevent UnboundLocalError
    user_response = ""

    # Lists to store the results for each turn
    convs = []
    pos_responses, neg_responses = [], []
    chosen_evals, rejected_evals = [], []

    for _ in tqdm(range(args.max_new_turns), desc=f"Processing conversation {i}"):

        # Now, the variable 'user_response' is initialized.
        if '[[TERMINATE CHAT]]' in user_response: 
            exit_flag = True 
            user_response = user_response.replace('[[TERMINATE CHAT]]', '')
        conv.append({'role': 'user', 'content': user_response})
        print(f"[Turn {len(conv)}] **User**: {user_response}")
        if exit_flag:
            break

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_collabllm = executor.submit(assistant_collabllm, conv, question=question, answer=answer)
            future_vanilla = executor.submit(assistant_vanilla, conv)
            
            responses = [future_collabllm.result(), future_vanilla.result()]
        
        user_generation_kwargs['target_object'] = answer  # aditi edit to tell the llm judge about the target object
        rewards, reward_logs = get_multiturn_rewards(
            task_name=args.task_name, # datasets_info[args.dataset]['task'],  # aditi edit
            single_turn_ds=[qa for _ in range(len(responses))],
            chat_histories=[conv for _ in range(len(responses))],
            responses=responses,
            max_workers=args.max_workers,
            num_samples=args.num_samples,
            llm_rw_weight=args.llm_rw_weight,
            window_size=args.window_size,
            task_weight=args.task_weight,
            cost_weight=args.cost_weight,
            user_generation_kwargs=user_generation_kwargs,
            assistant_generation_kwargs=assistant_generation_kwargs,
            reward_generation_kwargs=reward_generation_kwargs,
            target_objects=[answer for _ in range(len(responses))],  # aditi addition -- want a list of the same target object repeated
            local_model=None, local_tokenizer=None,
        )
        if np.argmax(rewards) == np.argmin(rewards):
            reward_stds = [log['reward_std'] for log in reward_logs]
            neg_response = responses[np.argmax(reward_stds)]
            pos_response = responses[np.argmin(reward_stds)]
        else:
            pos_response = responses[np.argmax(rewards)]
            neg_response = responses[np.argmin(rewards)]

        chosen_eval = reward_logs[np.argmax(rewards)]
        rejected_eval = reward_logs[np.argmin(rewards)]

        convs.append(copy.deepcopy(conv))
        pos_responses.append(pos_response)
        neg_responses.append(neg_response)
        chosen_evals.append(chosen_eval)
        rejected_evals.append(rejected_eval)

        conv.append({'role': 'assistant', 'content': pos_response})

        print(f"[Turn {len(conv)}] Rewards {rewards}")
        for key, result in zip(['Chosen', 'Rejected'], [chosen_eval, rejected_eval]):
            print(f"{key}: task_metric_avg={result['task_metric_avg']} | " \
                  f"llm_rw_avg={result['llm_rw_avg']} | token_cost_avg={result['token_cost_avg']}")
        print(f"[Turn {len(conv)}] Chosen: {pos_response}\n")
        print(f"[Turn {len(conv)}] Rejected: {neg_response}\n\n")

        scores, user_responses = [], []
        non_terminated_scores, non_terminated_user_response = [], []

        if 'forward_chat' in reward_logs[0]['rs']:
            for key, item in reward_logs[np.argmax(rewards)]['rs'].items():
                forward_chat = item['forward_chat']
                score = item['average_score']
                scores.append(score)
                user_responses.append(forward_chat)
                if not '[[TERMINATE CHAT]]' in forward_chat[0]['content']:
                    non_terminated_user_response.append(forward_chat[0]['content'])
                    non_terminated_scores.append(score)

            if len(non_terminated_scores) > 0:
                user_response = non_terminated_user_response[np.argmin(non_terminated_scores)]
            else:
                user_response = user_responses[0][0]['content']
        else:
            user_response = user(conv)

    return i, convs, pos_responses, neg_responses, chosen_evals, rejected_evals



# to print all info to my output file
ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
def strip_ansi(text):
    return ansi_escape.sub('', text)

class Tee:
    def __init__(self, logfile):
        self.logfile = logfile

    def write(self, data):
        clean_data = strip_ansi(data)
        self.logfile.write(clean_data)
        self.logfile.flush()

    def flush(self):
        self.logfile.flush()
        
# class Tee:
#     def __init__(self, original, logfile):
#         self.original = original
#         self.logfile = logfile

#     def write(self, data):
#         # Write to terminal as is
#         self.original.write(data)
#         self.original.flush()
#         # Strip ANSI codes before writing to logfile
#         clean_data = strip_ansi(data)
#         self.logfile.write(clean_data)
#         self.logfile.flush()

#     def flush(self):
#         self.original.flush()
#         self.logfile.flush()
        
def main():

    print("\n\nSTARTING MAIN!\n\n")
    print(f"\n\npushing results to {PUSH_REPO}!\n\n")

    

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_path = f"/Users/aditi/Documents/multiturn-20q/collab-llm/logs/full_run_terminal_st_{timestamp}.txt"
    partial_save_path = f"/Users/aditi/Documents/multiturn-20q/collab-llm/logs/partial_convs_{timestamp}.json"

    logfile = open(log_path, "w", encoding='utf-8')
    # sys.stdout = Tee(sys.stdout, logfile)   # only for use in debug
    # sys.stderr = Tee(sys.stderr, logfile)

    sys.stdout = Tee(logfile)
    sys.stderr = Tee(logfile)

    args = parse_args()
    args.dataset = load_dataset(
        "json", 
        data_files={"train": "/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data/eval_single_turn.json"},
        cache_dir="/tmp/aditi", keep_in_memory=True
    )
    dataset = args.dataset["train"]

    # Load previously saved data
    if os.path.exists(partial_save_path):
        with open(partial_save_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        if 'prompt_item' not in saved_data:
            saved_data['prompt_item'] = []
    else:
        saved_data = {
            'idx': [],
            'prompt': [],
            'chosen': [],
            'rejected': [],
            'chosen_eval': [],
            'rejected_eval': [],
            'metadata': [],
            'prompt_item': []
        }

    unique_idx = set(saved_data['idx'])

    random.seed(0)
    idx_all = list(range(len(dataset)))
    random.shuffle(idx_all)
    start = args.start_obj_num
    end = args.start_obj_num + args.max_num_conv
    idx_all = idx_all[start:end]
    idx_todo = [idx for idx in idx_all if idx not in unique_idx]

    if args.task_name != '20q':
        print("NOT RUNNING 20Q TASK; exiting\n\n")
        return -1

    print(f"\n\n\n[INFO] Task name set to: {args.task_name}\n\n\n")
    print(f"\n\n\n[INFO] RUNNING NUMS: start={start}, end={end}\n\n\n")

    assistant_collabllm = LLMAssistant(method='proact_cot_20q', **assistant_generation_kwargs)
    vanilla_generation_kwargs = copy.copy(assistant_generation_kwargs)
    vanilla_generation_kwargs['json_object'] = False
    assistant_vanilla = LLMAssistant(method='none', **vanilla_generation_kwargs)

    for i in tqdm(idx_todo):
        i, convs, pos_responses, neg_responses, chosen_evals, rejected_evals = process_conversation(
            i, dataset, args, assistant_collabllm, assistant_vanilla
        )

        target_object = dataset[i]["chat"][-1]["content"]

        # Remove old entries for this conversation
        indices_to_keep = [j for j, idx in enumerate(saved_data['idx']) if idx != i]
        for key in saved_data:
            saved_data[key] = [saved_data[key][j] for j in indices_to_keep]

        # Append new entries
        saved_data['idx'].extend([i] * len(convs))
        saved_data['prompt'].extend(convs)
        saved_data['chosen'].extend(pos_responses)
        saved_data['rejected'].extend(neg_responses)
        saved_data['chosen_eval'].extend(chosen_evals)
        saved_data['rejected_eval'].extend(rejected_evals)
        saved_data['metadata'].extend([{'user': args.user_model_name, 'assistant': args.assistant_model_name}] * len(convs))
        saved_data['prompt_item'].extend([target_object] * len(convs))

        # Save to disk after every iteration
        with open(partial_save_path, 'w', encoding='utf-8') as f:
            json.dump(saved_data, f, indent=2)

    # Final full backup
    local_save_path = f"/Users/aditi/Documents/multiturn-20q/collab-llm/logs/args_dataset_generated_convs_{timestamp}.json"
    with open(local_save_path, 'w', encoding='utf-8') as f:
        json.dump(saved_data, f, indent=2)

    print(f"Saved conversation data locally to {os.path.abspath(local_save_path)}")

    dataset_converted = Dataset.from_dict(saved_data)
    dataset_dict_for_hf = DatasetDict({"train": dataset_converted})
    dataset_dict_for_hf.push_to_hub(repo_id=PUSH_REPO, private=True)


if __name__ == '__main__':
    main()