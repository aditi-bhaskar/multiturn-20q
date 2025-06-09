# 20 Questions: Multi-turn RLHF for Sparse Rewards

Aditi Bhaskar (aditijb [at] cs.stanford.edu)

### overview

This project implements multiturn RLHF (with DPO training) to solve the 20 questions task from LMRL Gym (Abdulhai et al). Multiturn rewards are inspired by Collab LLM (Wu et al).

Broad research question: Does a proxy reward help a base model (Llama 3.2 1B Instruct) become better at conversing during a task with a sparse reward signal? 

Project goal: Does a proxy metric of information gain or interactivity improve a model's performance on play the 20 questions game?

### datasets + models

See interactive dataset here: [https://huggingface.co/datasets/aditijb/collabllm-20q-interactive](https://huggingface.co/datasets/aditijb/collabllm-20q-interactive)  

See interactive model here: [https://huggingface.co/aditijb/Llama-3.2-1B-Instruct-20q-interactive](https://huggingface.co/aditijb/Llama-3.2-1B-Instruct-20q-interactive)

See infogain dataset here: [https://huggingface.co/datasets/aditijb/collabllm-20q-infogain](https://huggingface.co/datasets/aditijb/collabllm-20q-infogain)  

See infogain model here: [https://huggingface.co/aditijb/Llama-3.2-1B-Instruct-20q-infogain](https://huggingface.co/aditijb/Llama-3.2-1B-Instruct-20q-infogain)

### Mentor: 
Shirley Wu (PhD Candidate in Stanford CS)

## notes on running the project
 
### Setup

python3 -m venv 20q  
source 20q/bin/activate  
pip install -r requirements.txt  
(Note, some other things might need to be installed afterwards)

---

### Important Notes

The following can be run from inside the `collab-llm` folder. Do not run these scripts unless you have set up the remote hf dataset/model to be your own dataset/model! It will not push anywhere unless you have permission. You will also need to export your `export OPENAI_API_KEY=sk-proj-XXXXXX` and your hugging face cli `export HF_TOKEN=hf_XXXX` + `huggingface-cli login`.

---

## Relevant scripts

### Generate data

`./scripts/generate_conv_dpo_20q.sh`


### Push to hf

`python3 scripts/20q_combine_jsons_and_push_to_hf.py`


### Run train

`./scripts/dpo_train_offline_20q.sh`


### Run evals

`./scripts/eval_multiturn_20q.sh aditijb/collabllm-20q q`

`python3 20q_plot_evals2.py`  
(make sure to change the file path inside the `.py` script to match the correct new output)

---

## My major contributions 
note, many files have been changed to make everything work!

### Prompts are at

`collab-llm/collabllm/prompts/llm_judge/20q.txt`  
`collab-llm/collabllm/prompts/llm_assistant/proact_cot_20q.txt`  
`collab-llm/collabllm/prompts/user_simulator_cot/20q.txt`

### Rewards can be found in

`collab-llm/collabllm/metrics/__init__.py`

### Data preprocessing

`collab-llm/collabllm/datasets/twentyq.py`
`collab-llm/collabllm/datasets/twentyq_mt.py`
`collab-llm/lmrl_gym_20q_data/*` (What object examples am I running training/eval on? I get them from the processed LMRL Gym data for the task)

### Large sections of implementation

`collab-llm/scripts/generate_conv_dpo_20q.py`  
`collab-llm/scripts/dpo_train_offline_20q.py`  
`collab-llm/scripts/eval_multiturn_20q.py`
`collab-llm/scripts/20q_*` (many plotting scripts)

### Data (final results) can be found in
note, many results files have been deleted since they were redundant or obsolete.

`collab-llm/outputs/eval/20q/local20q/test/4-PAPER_ACTUAL_MERGED_FINAL`