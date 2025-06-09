# spr2025 cs224r custom project

20questions with multiturn rlhf. Based on the multiturn reward system implemented in Collab LLM (Wu et al) and the 20q task dataset from LMRL Gym (Abdulhai et al). 

Research question: Does multiturn rlhf help a base model (Llama 3.2 1B Instruct) become better at playing 20 questions? 

See dataset here [https://huggingface.co/datasets/aditijb/collabllm-20q](https://huggingface.co/datasets/aditijb/collabllm-20q)

See model here [https://huggingface.co/datasets/aditijb/collabllm-20q](https://huggingface.co/datasets/aditijb/collabllm-20q)

Mentor: Shirley Wu (PhD Candidate in Stanford CS)

# spr2025 cs224r custom project

### Setup

python3 -m venv 20q  
source 20q/bin/activate  
pip install -r requirements.txt  
(Note, some other things might need to be installed afterwards)

---

### Important Notes

The following can be run from inside the `collab-llm` folder. Do not run these scripts unless you have set up the remote hf dataset/model to be your own dataset/model! Otherwise it will overwrite my stuff. You will also need to export your `export OPENAI_API_KEY=sk-proj-XXXXXX` and your hugging face cli `export HF_TOKEN=hf_XXXX` + `huggingface-cli login`.

---

### Generate data (see `generate_conv_dpo_20q.sh` / `.py`)

`./scripts/generate_conv_dpo_20q.sh`

---

### Push to hf

`python3 scripts/20q_combine_jsons_and_push_to_hf.py`

---

### Run train

`./scripts/dpo_train_offline_20q.sh`

---

### Run evals

`./scripts/eval_multiturn_20q.sh aditijb/collabllm-20q q`

`python3 20q_plot_evals2.py`  
(make sure to change the file path inside the `.py` script to match the correct new output)

---

### Prompts are at

`collab-llm/collabllm/prompts/llm_judge/20q.txt`  
`collab-llm/collabllm/prompts/llm_assistant/proact_cot_20q.txt`  
`collab-llm/collabllm/prompts/user_simulator_cot/20q.txt`

---

### Large sections of implementation (besides plotting scripts that begin with “20q”) are found at

`collab-llm/scripts/generate_conv_dpo_20q_2mod.py`  
`collab-llm/scripts/dpo_train_offline_20q.py`  
`collab-llm/scripts/eval_multiturn_20q.py`

---

### Rewards can be found in

`collab-llm/collabllm/metrics/__init__.py`
