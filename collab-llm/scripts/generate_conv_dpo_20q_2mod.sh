# ./scripts/generate_conv_dpo_20q_2mod.sh   

# Environment Variables (Adjust to match dataset & model)
export USE_SUB=false
export USE_SNAP=true
export USE_GCR=false


# Dataset configurations
export DATASET=NONE                # Change this for different datasets (e.g., math-hard, humaneval)
# export TEMP=0.5                      # Temperature for sampling (adjust for creativity)
export TEMP=0.7                     # Temperature for sampling (adjust for creativity)
export TEMP_2=0.3

# ^ this is length of game
export MAX_TOKENS=256               # Maximum tokens per generation -- aditi edit
export COST_WEIGHT=5e-4              # Cost weight in reward function (adjust based on preference)
export LLM_RW_WEIGHT=1               # Weight for RLHF reward function
export USER_MODEL=gpt-4o-mini        # Choose the user model (e.g., gpt-4o-mini, gpt-4o)
export N_EVAL=100                   # aditi edit for debugging


# debug time:
# export MAX_NUM_CONV=1           # number of objects/games
# export MAX_NEW_TURNS=2          # Maximum number of new conversation turns per task
# export MAX_NUM_WORKERS=1

# test time:
export MAX_NUM_CONV=75           # number of objects/games
export START_OBJ_NUM=0           # number of objects/games we start from (useful for running multiple times)
export MAX_NEW_TURNS=20           # Maximum number of new conversation turns per task. 20 because we're playing 20q
export MAX_NUM_WORKERS=4

#  completed ranges: 0  -- temp = 0.7
#  running range: 0-74
#  todo range: 75+

export USER_MODEL=gpt-4o-mini
export ASSISTANT_MODEL=gpt-4o-mini
export ASSISTANT_MODEL_2=meta-llama/Llama-3.2-1B-Instruct
export REWARD_MODEL=gpt-4o-mini


CUDA_VISIBLE_DEVICES=0 python scripts/generate_conv_dpo_20q_2mod.py \
    --dataset $DATASET \
    --max_workers $MAX_NUM_WORKERS \
    --num_samples 3 \
    --user_model_name $USER_MODEL \
    --assistant_model_name $ASSISTANT_MODEL \
    --assistant_model_name_2 $ASSISTANT_MODEL_2 \
    --reward_model $REWARD_MODEL \
    --max_new_tokens $MAX_TOKENS \
    --max_new_turns $MAX_NEW_TURNS \
    --start_obj_num $START_OBJ_NUM \
    --window_size 2 \
    --temperature $TEMP \
    --temperature_2 $TEMP_2 \
    --top_p 0.9 \
    --task_weight 1 \
    --llm_rw_weight $LLM_RW_WEIGHT \
    --cost_weight $COST_WEIGHT \
    --n_eval_per_dataset $N_EVAL \
    --max_num_conv $MAX_NUM_CONV \
    --task_name 20q \
    --resume

