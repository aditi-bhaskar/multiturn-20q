#!/bin/bash

# Usage: ./scripts/eval_multiturn.sh  <dataset> <mode>
# Usage: ./scripts/eval_multiturn.sh bigcodebench dpo_offline
# dpo_online dpo_offline
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset> <mode>"
    exit 1
fi

# Parameters
# DATASET_NAME=$1
# MODE=$2

# Load configurations
# source scripts/config.sh
# set_dataset_config $DATASET_NAME
# set_assistant_model $DATASET_NAME $MODE

# Decide whether to add --add_sys_prompt
# ADD_SYS_PROMPT_FLAG=""
# if [ "$ADD_SYSTEM_PROMPT" == "True" ]; then
#     ADD_SYS_PROMPT_FLAG="--add_sys_prompt"
# fi

# Random seed and port setup
# RANDOM_SEED=$$
# PORT=$((56480 + RANDOM_SEED % 10))
RANDOM_SEED=0

# Fixed configuration for 20Q
DATASET="aditijb/collabllm-20q"
PROMPT_METHOD="none"
MAX_NEW_TURNS=6
N_EVAL=180
MAX_TOKENS=2048
OUTPUT_DIR="./outputs/eval/20q"
ADD_SYS_PROMPT_FLAG=""

# fix user model to gpt-4o for eval
USER_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o-mini
ASSISTANT_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct  # vanilla model
# /name/project/collabllm/outputs/Meta-Llama-3-8B-Instruct_step-1500  # my trained version, after 1500 training steps

# Output directory
# OUTDIR=$OUTPUT_DIR/eval/$OUTPUT_DIR_SUFFIX

# Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_multiturn_20q.py \
    --dataset $DATASET \
    --output_dir $OUTDIR \
    --split test \
    --judge_model $JUDGE_MODEL \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --user_model_name $USER_MODEL \
    --prompt_method $PROMPT_METHOD \
    --temperature $TEMP \
    --max_new_turns $MAX_NEW_TURNS \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    $ADD_SYS_PROMPT_FLAG \



# call the script:
# on base llama 8b
# ./scripts/eval_multiturn_20q.sh aditijb/collabllm-20q