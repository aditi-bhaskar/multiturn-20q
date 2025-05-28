#!/bin/bash

# Source shared parameters
# Usage: ./scripts/dpo_train_offline_20q.sh  <dataset>
# Usage: ./scripts/dpo_train_offline_20q.sh  aditijb/collabllm-20q
# source scripts/config.sh

fp16=False
bf16=False

#  to make sure i can handle mps operations using my cpu if mps is not available
export PYTORCH_ENABLE_MPS_FALLBACK=1

export USE_SUB=false
export USE_SNAP=true
export USE_GCR=false

RANDOM_SEED=0  # use $$ instead
PORT=$((56430 + RANDOM_SEED % 10))

OUTPUT_DIR="./train/20q"

MAX_TOKENS=256
MIN_GAP=0.1  # aditi : idk what this is so i made smth up

N_EVAL_PER_DATASET=10  # reduced from previously 30


ASSISTANT_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct  # vanilla model; small to use

DATASET=aditijb/collabllm-20q-2v  # hardcode the dataset we use for training


# editables!!
NUM_TRAIN_EPOCHS=8  # 1
BATCHSIZE=4  # originally 2, but larger to make it run faster

# first attempt (may 25): num epochs 1, batchsize 4
# second attempt (may 27): num epochs 8, batchsize 4

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




# TRAINING NEXT STEPS:

#  TODO: figure out why the mps is giving me soooo many issues. 
# i should be able to just remove all mentions of mps and cuda in the dpo trainer and it should work?!

# /Users/aditi/Documents/multiturn-20q/20q/lib/python3.12/site-packages/trl/trainer/dpo_trainer.py



# Amazon EC2 has nvidia t4 gpu
# learn how to monitor gpu usage

# qwen 0.5B model





