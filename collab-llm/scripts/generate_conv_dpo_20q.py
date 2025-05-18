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

from datasets import load_dataset

# todo
#  it should iterate thru all the entries in the dataset -- maybe that's why it doesnt
# make it single turn



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

    # aditi edit/hack to use a different object
    # all_objects = [ds[-1]['content'] for ds in dataset['chat']]
    # answer = random.choice(all_objects)
    # question = "I would like to play a game of 20 questions where I think of an object and you try to guess it using y/n questions."
 

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
            local_model=None, local_tokenizer=None
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
import sys

import re
ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
def strip_ansi(text):
    return ansi_escape.sub('', text)

class Tee:
    def __init__(self, original, logfile):
        self.original = original
        self.logfile = logfile

    def write(self, data):
        # Write to terminal as is
        self.original.write(data)
        self.original.flush()
        # Strip ANSI codes before writing to logfile
        clean_data = strip_ansi(data)
        self.logfile.write(clean_data)
        self.logfile.flush()

    def flush(self):
        self.original.flush()
        self.logfile.flush()


######################## LOAD DATASETS ########################
def main():

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # log_path = f"/Users/aditi/Documents/multiturn-20q/collab-llm/logs/full_run_terminal_{timestamp}.txt"
    log_path = f"/Users/aditi/Documents/multiturn-20q/collab-llm/logs/full_run_terminal_st_{timestamp}.txt"  # for single turn dataset input stuff
    logfile = open(log_path, "w", encoding='utf-8')
    # Replace stdout and stderr with Tee to duplicate output to terminal + logfile
    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = Tee(sys.stderr, logfile)



    args = parse_args()
    # aditi edit to use the local dataset
    args.dataset = load_dataset("json", data_files={
        # OBSOLETE: note that i lost the train file with git conflicts, so i am using the eval file here (much smaller)
        # OBSOLETE: "train": "/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data/eval_processed.json"
        # "train": "/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data/train_processed.json"
        "train": "/Users/aditi/Documents/multiturn-20q/collab-llm/lmrl_gym_20q_data/train_single_turn.json"

    })

    dataset =  args.dataset   #  aditi edit # load_single_turn_dataset(args.dataset, add_system_prompt=False)
    # removed by aditi to get it to run!!
    # if args.resume:
    #     ds = load_dataset(f'{args.hf_org}/collabllm-{args.dataset}', trust_remote_code=True)

    dataset_dict = {}
    for split in ['train']:
        unique_idx = set()
        chosen_list, rejected_list = [], []
        chosen_eval_list, rejected_eval_list = [], []
        idx_list, prompt_list, metadata_list = [], [], []

        # if args.resume:
        #     try:
        #         idx_list, prompt_list, metadata_list = ds[split]['idx'], ds[split]['prompt'], ds[split]['metadata']
        #         chosen_list, rejected_list = ds[split]['chosen'], ds[split]['rejected']
        #         chosen_eval_list, rejected_eval_list = ds[split]['chosen_eval'], ds[split]['rejected_eval']
        #         unique_idx = set(idx_list)
        #     except KeyError:
        #         pass
        
        random.seed(0)
        # idx_all = [i for i in range(min(args.n_eval_per_dataset, len(dataset[split]['chat'])))]  # aditi edit to use n_eval_per_dataset
        #  in dpo/sft we can split into eval/training dataset 

        idx_all = [i for i in range(len(dataset[split]['chat']))]
        random.shuffle(idx_all)
        idx_all = idx_all[:args.max_num_conv]
        idx_todo = [idx for idx in idx_all if idx not in unique_idx]

        # method = 'collabllm_gt_cot' if datasets_info[args.dataset]['task'] == 'question-answering' else 'collabllm_cot'
        # added by aditi; changed the default bc  collabllm_cot doesnt exist
        # method = 'proact_cot_20q' if datasets_info[args.dataset]['task'] == '20q' else 'proact_cot' 

        if args.task_name == '20q':
            method = 'proact_cot_20q' 
        else:
            print("NOT RUNNING 20Q TASK; exiting\n\n")
            return -1

        print(f"\n\n\n[INFO] Task name set to: {args.task_name}\n\n\n")

        assistant_collabllm = LLMAssistant(method=method, **assistant_generation_kwargs)
        vanilla_generation_kwargs = copy.copy(assistant_generation_kwargs)
        vanilla_generation_kwargs['json_object'] = False
        assistant_vanilla = LLMAssistant(method='none', **vanilla_generation_kwargs)
        
        #  with ifx todo, 

        for i in tqdm(idx_todo):
                i, convs, pos_responses, neg_responses, chosen_evals, rejected_evals = process_conversation(i, dataset[split], args, assistant_collabllm, assistant_vanilla)
        # with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        #     futures = {
        #         executor.submit(process_conversation, i, dataset[split], args, assistant_collabllm, assistant_vanilla): i for i in idx_todo
        #     }

            # for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Global tasks'):
            #     i, convs, pos_responses, neg_responses, chosen_evals, rejected_evals = future.result()
        
                idx_list.extend([i] * len(convs))  # Extend with i repeated for each conversation turn
                metadata_list.extend([
                    {'user': args.user_model_name, 
                     'assistant': args.assistant_model_name}] * len(convs))
                prompt_list.extend(convs)
                chosen_list.extend(pos_responses)
                rejected_list.extend(neg_responses)
                chosen_eval_list.extend(chosen_evals)
                rejected_eval_list.extend(rejected_evals)

                if np.unique(idx_list).shape[0] % args.log_step == 0:
                    dataset_dict[split] = Dataset.from_dict({
                        'idx': idx_list,
                        'prompt': prompt_list,
                        'chosen': chosen_list,
                        'rejected': rejected_list,
                        'chosen_eval': chosen_eval_list,
                        'rejected_eval': rejected_eval_list,
                        'metadata': metadata_list
                    })
                    print(f"Dataset Dict Split={split}: {dataset_dict[split]}\n")

            #  only load the one-turn dataset: q=nothing, a=object
            #  list of objects essentially

                    # aditi: removed bc we dont have hf
                    # DatasetDict(dataset_dict).push_to_hub(repo_id=f'{args.hf_org}/collabllm-{args.dataset}', private=True)



        dataset_dict[split] = Dataset.from_dict({
            'idx': idx_list,
            'prompt': prompt_list,
            'chosen': chosen_list,
            'rejected': rejected_list,
            'chosen_eval': chosen_eval_list,
            'rejected_eval': rejected_eval_list,
            'metadata': metadata_list
        })
        print(f"Dataset Dict Split={split}: {dataset_dict[split]}\n")


        # dataset[split][idx]    <-- TODO check what the object is before its processed

        # TODO save this dataset to json!

        # aditi: removed bc we dont have hf
        # TODO create a hf for myself & push to hf  (try it with 2 conversations first)
        # DatasetDict(dataset_dict).push_to_hub(repo_id=f'{args.hf_org}/collabllm-{args.dataset}', private=True)
        #  can log to see the format (can make it public and view dataset in a table)  - set private = false
        # https://huggingface.co/datasets/snap-stanford/pubmed_pipeline-preference_scorer-combined



if __name__ == '__main__':
    main()
