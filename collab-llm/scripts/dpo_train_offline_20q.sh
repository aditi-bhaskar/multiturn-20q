#!/bin/bash

# Source shared parameters
# Usage: ./scripts/eval_multiturn.sh  <dataset>
# Usage: ./scripts/eval_multiturn.sh  aditijb/collabllm-20q
# source scripts/config.sh



fp16=False
bf16=False

#  to make sure i can handle mps operations using my cpu if mps is not available
export PYTORCH_ENABLE_MPS_FALLBACK=1


export USE_SUB=false
export USE_SNAP=true
export USE_GCR=false

# # Ensure a dataset name is provided as input
# if [ -z "$1" ]; then
#     echo "Usage: $0 <dataset>"
#     exit 1
# fi

# Set dataset-specific parameters, start with trained sft model
# set_dataset_config "$1"
# set_assistant_model "$1" "sft"

RANDOM_SEED=0  # use $$ instead
PORT=$((56430 + RANDOM_SEED % 10))

OUTPUT_DIR="./train/20q"

MAX_TOKENS=256
MIN_GAP=0.1  # aditi : idk what this is so i made smth up

NUM_TRAIN_EPOCHS=8  #1
N_EVAL_PER_DATASET=30

# DATASET=aditijb/collabllm-20q

# USER_MODEL=gpt-4o-mini
# JUDGE_MODEL=gpt-4o-mini
ASSISTANT_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct  # vanilla model -- smaller model; should be able to download?

DATASET=aditijb/collabllm-20q  # hardcode the dataset we use for training

# WANDB__SERVICE_WAIT=300 
# CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 torchrun --master_port=$PORT --nnodes=1 --nproc_per_node=7 \
# torchrun --master_port=$PORT --nnodes=1 --nproc_per_node=7 \


BATCHSIZE=4  # originally 2, but larger to make it run faster


# CUDA_VISIBLE_DEVICES=4 \
#     torchrun --master_port=$PORT \
#     --nnodes=1 --nproc_per_node=1 \

    python \
    scripts/dpo_train_offline_20q.py \
    --datasets $DATASET \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size $BATCHSIZE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --assistant_model_name $ASSISTANT_MODEL_NAME \
    --save_total_limit 10 \
    --eval_steps 1 \
    --learning_rate 5e-6 \
    --max_new_tokens $MAX_TOKENS \
    --output_dir $OUTPUT_DIR/dpo_train_offline \
    --n_eval_per_dataset $N_EVAL_PER_DATASET \
    --minimum_gap $MIN_GAP \
    --push_to_hub 

    # --user_model_name $USER_MODEL \
    # --judge_model $JUDGE_MODEL \





# TRAINING NEXT STEPS:

#  TODO: figure out why the mps is giving me soooo many issues. 
# i should be able to just remove all mentions of mps and cuda in the dpo trainer and it should work?!

# /Users/aditi/Documents/multiturn-20q/20q/lib/python3.12/site-packages/trl/trainer/dpo_trainer.py



# Amazon EC2 has nvidia t4 gpu
# learn how to monitor gpu usage

# qwen 0.5B model
