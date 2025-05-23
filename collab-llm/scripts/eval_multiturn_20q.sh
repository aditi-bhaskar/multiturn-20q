#!/bin/bash

# Usage: ./scripts/eval_multiturn.sh  <dataset> <mode>
# Usage: ./scripts/eval_multiturn.sh bigcodebench dpo_offline
# dpo_online dpo_offline
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset> <mode>"
    exit 1
fi

# Random seed and port setup 
RANDOM_SEED=0  # use $$ instead
PORT=$((56480 + RANDOM_SEED % 10))


######################################
#  aditi tweak these

PROMPT_METHOD="vanilla_llama_3.2_1b"
# PROMPT_METHOD="sftdpo_llama_3.2_1b"

# MAX_NEW_TURNS=20
# N_EVAL=15
MAX_NEW_TURNS=1
N_EVAL=1

######################################

# Fixed configuration for 20Q
# DATASET="aditijb/collabllm-20q"
DATASET="local20q"
# MAX_NEW_TURNS=6
# N_EVAL=180
# MAX_TOKENS=2048
MAX_TOKENS=256  # dont need it to yap during 20q game
OUTPUT_DIR="./outputs/eval/20q"
ADD_SYS_PROMPT_FLAG=""
SPLIT="test"  # automatically uses train (dev) split instead of test split for the evals

# fix user model to gpt-4o for eval
USER_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
ASSISTANT_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct  # vanilla model -- smaller model; should be able to download?

# /name/project/collabllm/outputs/Meta-Llama-3-8B-Instruct_step-1500  #  trained version, after 1500 training steps


# Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_multiturn_20q.py \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --judge_model $JUDGE_MODEL \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --user_model_name $USER_MODEL \
    --prompt_method $PROMPT_METHOD \
    --temperature 0.6 \
    --max_new_turns $MAX_NEW_TURNS \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    $ADD_SYS_PROMPT_FLAG \



# TODO make a test or eval split of my data

# call the script:
# on base llama 8b
# ./scripts/eval_multiturn_20q.sh aditijb/collabllm-20q