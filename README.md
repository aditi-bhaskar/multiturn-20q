# spr2025 cs224r custom project

### overview

20questions with multiturn rlhf. Based on the multiturn reward system implemented in Collab LLM (Wu et al) and the 20q task dataset from LMRL Gym (Abdulhai et al). 

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

## My major contributions (note, many files have been changed to make everything work!)

### Prompts are at

`collab-llm/collabllm/prompts/llm_judge/20q.txt`  
`collab-llm/collabllm/prompts/llm_assistant/proact_cot_20q.txt`  
`collab-llm/collabllm/prompts/user_simulator_cot/20q.txt`


### Large sections of implementation (besides plotting scripts that begin with “20q”) are found at

`collab-llm/scripts/generate_conv_dpo_20q.py`  
`collab-llm/scripts/dpo_train_offline_20q.py`  
`collab-llm/scripts/eval_multiturn_20q.py`


### Rewards can be found in

`collab-llm/collabllm/metrics/__init__.py`
