import json
import os
import random
import os.path as osp
import argparse
from tqdm import tqdm
import logging
import datetime

import sys
sys.path.append('.')
from collabllm.datasets import split_train_dev_datasets, load_single_turn_dataset, load_dpo_dataset
from collabllm.evaluator import ChatEvaluator
from collabllm.utils.aggregate import average_nested_dicts
from collabllm.models.generation import run_one_chat_session
from collabllm.models.load import load_model_and_tokenizer
from collabllm.models import is_base_model_auto
from collabllm.prompts import SYSTEM_PROMPT
from collabllm.datasets import datasets_info
from collabllm.models import is_api_model_auto

from datasets import load_dataset

import torch 

# device = "cpu"

def load_hf_dataset(dataset_name, split, n_eval):
    ds = load_dataset(dataset_name, split=split)
    # Optionally limit to n_eval samples
    if n_eval and n_eval < len(ds):
        import random
        random.seed(42)
        ds = ds.shuffle(seed=42).select(range(n_eval))
    return ds


def parse_args():
    parser = argparse.ArgumentParser()
    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument('--dataset', type=str, default='local20q')

    parser.add_argument('--max_new_turns', type=int, default=6)
    parser.add_argument('--n_eval', type=int, default=200)
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'])

    parser.add_argument('--output_dir', type=str, default="./outputs")
    parser.add_argument('--prompt_method', type=str, default="none", choices=['none', 'proact'])

    parser.add_argument('--user_model_name', type=str, default='gpt-4o-mini')
    parser.add_argument('--judge_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--assistant_model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct")

    # generation kwargs
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_new_tokens', type=int, default=2048)

    parser.add_argument('--push_to_blob', action='store_true', help='push to blob')
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--add_sys_prompt', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    return parser.parse_args()


def main():
   args = parse_args()
   logging.disable(logging.CRITICAL)
   print(args)

   # aditi - may 27 2025
   if (args.assistant_model_name == "meta-llama/Llama-3.2-1B-Instruct"):  # base model uses cpu to eval
      device = "cpu" 
   else:
      device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")    # the finetuned model uses device = mps, but i guess we have to check for cpu too?!

   # Hardcoded overrides:
   args.dataset = "local20q"
   args.user_model_name = "gpt-4o-mini"
   args.judge_model = "gpt-4o-mini"

   ######################## CONFIG PATH ########################
   dataset_str = args.dataset.split('/')[-1]
   date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
   is_base_model = is_base_model_auto(args.assistant_model_name)
   model_name = args.assistant_model_name.split("/")[-1]
   if model_name.startswith('checkpoint'):
      model_name = args.assistant_model_name.split("/")[-2] + '_' + model_name
   model_name = model_name + f'_prompt={args.prompt_method}' if is_base_model else model_name
   model_name = model_name + f'_temp={args.temperature}'
   model_name = model_name + f'_{date_str}'

   if not is_base_model and not args.add_sys_prompt:
      print('WARNING !!!!!!! The assistant model may be a finetuned model and add_sys_prompt is False.\n\n\n')
      # return

   output_dir = osp.join(args.output_dir, dataset_str, args.split, model_name)
   save_path = osp.join(output_dir, f'log.json')
   os.makedirs(output_dir, exist_ok=True)

   with open(osp.join(output_dir, 'args.json'), 'w') as f:
      json.dump(vars(args), f, indent=4)

   results = {}
   if osp.exists(save_path):
      with open(save_path, 'r') as f:
         results = json.load(f)
      results = {int(k): v for k, v in results.items()}

   ######################## LOAD DATASET ########################
   split = 'train' if args.split == 'dev' else args.split

   if args.dataset.startswith('local20q'):
      # all my code should be entering this case!!
      print("\n  ***  ADITI is loading the collabllm 20q single turn dataset! ***  \n")
      split_name = 'train' if args.split == 'dev' else args.split
      single_turn_ds = load_single_turn_dataset(args.dataset, add_system_prompt=args.add_sys_prompt)[split_name]
      print(f"\n\n\n\nargs.dataset = {args.dataset}\n\n\n\n")

      eval_indices = list(range(len(single_turn_ds)))
      eval_indices = eval_indices[:args.n_eval]  # aditi edit: crop to only take so many indices!!
   else:
      print("NOT USING CORRECT 20Q DATASET!\n")
      return

   ######################## LOAD MODEL ########################

   print("\n\n BEFORE LOADING MODEL\n")

   # Removed device_map or load_in_4bit for simplicity since device is fixed to cpu
   model, tokenizer = load_model_and_tokenizer(args.assistant_model_name,
                                             max_new_tokens=args.max_new_tokens,
                                             )

   print("\n\n BEFORE MOVING MODEL TO DEVICE\n")

   model = model.to(device)

   print("\n\n TESTING MODEL\n")
   input_text = "Hello, how are you?"
   inputs = tokenizer(input_text, return_tensors="pt")
   inputs = {k: v.to(device) for k, v in inputs.items()}
   outputs = model.generate(**inputs, max_new_tokens=50)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   print("\n\n END TESTING MODEL\n")

   print(f"\n\n[DEBUG] Running eval with max_new_turns={args.max_new_turns}, n_eval={args.n_eval}\n\n")

   model.eval()  # aditi modification
   evaluator = ChatEvaluator(task_name=datasets_info[args.dataset]['task'], judge_model=args.judge_model)
   complete_metrics = evaluator.metrics
   assistant_generation_kwargs = {
      "model": args.assistant_model_name,
      "top_p": args.top_p,
      "temperature": args.temperature,
      "max_new_tokens": args.max_new_tokens,
      "json_object": 'cot' in args.prompt_method
   }
   user_generation_kwargs = {
      "model": args.user_model_name,
      "top_p": 1.0,
      "temperature": 1.0,
      "max_new_tokens": 4096,
      "json_object": True
   }
   if is_api_model_auto(args.assistant_model_name):
      assistant_generation_kwargs['no_repeat_ngram_size'] = 10
      user_generation_kwargs['no_repeat_ngram_size'] = 10

   ######################## START EVALUATION ########################

   for i in range(5):
      print(f"\n\n\n DEBUGGG Example {i} target object: {single_turn_ds[i]['target_object']}")

   for i in tqdm(range(len(eval_indices))):
      idx = eval_indices[i]

      single_turn_data = single_turn_ds[idx]['chat'][-2:]

      chat_history = [single_turn_ds[idx]['chat'][0]] if args.add_sys_prompt else []

      user_generation_kwargs['target_object'] = single_turn_ds[idx]['target_object']

      if idx in results and results[idx]['chat'] is not None:
         chat = results[idx]['chat']
      else:
         chat = run_one_chat_session(
               task_name=datasets_info[args.dataset]['task'],
               single_turn_data=single_turn_data,
               chat_history=chat_history,
               prompt_method=args.prompt_method,
               assistant_generation_kwargs=assistant_generation_kwargs,
               user_generation_kwargs=user_generation_kwargs,
               local_model=model, local_tokenizer=tokenizer,
               max_new_turns=args.max_new_turns,
               is_api_model='auto',
               verbose=True,
         )

      if not idx in results:
         results[idx] = {'chat': chat, 'qa': single_turn_data}

      remaining_metrics = set(complete_metrics) - set(results[idx].keys())
      if len(remaining_metrics) == 0:
         continue

      evaluator.metrics = remaining_metrics

      results[idx].update(evaluator.evaluate(single_turn_data, chat, target_object=single_turn_ds[idx]['target_object']))

      ######################## LOGGING ########################
      if i % args.log_step == 0 or i == len(eval_indices) - 1:
         with open(save_path, 'w') as f:
               json.dump(results, f, indent=4)

         agg_results = average_nested_dicts(results)
         agg_results['n_eval'] = i + 1
         with open(osp.join(output_dir, 'eval.json'), 'w') as f:
               json.dump(agg_results, f, indent=4)
         print(agg_results)


if __name__ == "__main__":
   main()


