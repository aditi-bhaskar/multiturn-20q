#!/bin/bash

# Usage: ./scripts/eval_multiturn.sh


# Random seed and port setup 
RANDOM_SEED=$$  # use $$ instead
# PORT=$((56480 + RANDOM_SEED % 10))
PORT=$((56480 + RANDOM_SEED % 10))


######################################
#  aditi tweak these

PROMPT_METHOD="none"

MAX_NEW_TURNS=20

# for debug:
# MAX_NEW_TURNS=1
# N_EVAL=1

MAX_TOKENS=256  # dont need it to yap during 20q game
OUTPUT_DIR="./outputs/eval/20q"
ADD_SYS_PROMPT_FLAG=""
SPLIT="test"  # automatically uses train (dev) split instead of test split for the evals

######################################

# can tweak these!

######################################
# MODEL USED:
######################################
# ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q  # dpo finetuned for 1 epoch (test)
# ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q-2v  # dpo finetuned for multiple epochs
# ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q-3v  # dpo finetuned for multiple epochs with new dataset
# ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q-reward  # dpo finetuned for multiple epochs with filtered dataset



# ASSISTANT_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct  # vanilla model -- smaller model; should be able to download?
# ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q  # dpo finetuned for 1 epoch (test)
# ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q-reward # dpo finetuned, new dataset, new reward, filtered



######################################
# TEMPERATURE USED
######################################




ASSISTANT_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct  # vanilla model -- smaller model; should be able to download?




N_EVAL=25
START=125
TEMPERATURE=0.7


# Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_multiturn_20q.py \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --prompt_method $PROMPT_METHOD \
    --temperature $TEMPERATURE \
    --max_new_turns $MAX_NEW_TURNS \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    --start $START \
    $ADD_SYS_PROMPT_FLAG \




ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q  # dpo finetuned for 1 epoch (test)


N_EVAL=25
START=125
TEMPERATURE=0.7


# Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_multiturn_20q.py \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --prompt_method $PROMPT_METHOD \
    --temperature $TEMPERATURE \
    --max_new_turns $MAX_NEW_TURNS \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    --start $START \
    $ADD_SYS_PROMPT_FLAG \



ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q-reward # dpo finetuned, new dataset, new reward, filtered

N_EVAL=25
START=125
TEMPERATURE=0.7


# Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_multiturn_20q.py \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --prompt_method $PROMPT_METHOD \
    --temperature $TEMPERATURE \
    --max_new_turns $MAX_NEW_TURNS \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    --start $START \
    $ADD_SYS_PROMPT_FLAG \





ASSISTANT_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct  # vanilla model -- smaller model; should be able to download?




N_EVAL=25
START=150
TEMPERATURE=0.7


# Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_multiturn_20q.py \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --prompt_method $PROMPT_METHOD \
    --temperature $TEMPERATURE \
    --max_new_turns $MAX_NEW_TURNS \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    --start $START \
    $ADD_SYS_PROMPT_FLAG \




ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q  # dpo finetuned for 1 epoch (test)


N_EVAL=25
START=150
TEMPERATURE=0.7


# Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_multiturn_20q.py \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --prompt_method $PROMPT_METHOD \
    --temperature $TEMPERATURE \
    --max_new_turns $MAX_NEW_TURNS \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    --start $START \
    $ADD_SYS_PROMPT_FLAG \



ASSISTANT_MODEL_NAME=aditijb/Llama-3.2-1B-Instruct-20q-reward # dpo finetuned, new dataset, new reward, filtered

N_EVAL=25
START=150
TEMPERATURE=0.7


# Run evaluation
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=$PORT \
    --nnodes=1 --nproc_per_node=1 \
    scripts/eval_multiturn_20q.py \
    --output_dir $OUTPUT_DIR \
    --split $SPLIT \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --prompt_method $PROMPT_METHOD \
    --temperature $TEMPERATURE \
    --max_new_turns $MAX_NEW_TURNS \
    --n_eval $N_EVAL \
    --max_new_tokens $MAX_TOKENS \
    --top_p 0.9 \
    --start $START \
    $ADD_SYS_PROMPT_FLAG \





# # ./scripts/eval_multiturn_20q.sh

