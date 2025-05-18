# ./scripts/generate_conv_dpo_20q.sh   


# Environment Variables (Adjust to match dataset & model)
export USE_SUB=false
export USE_SNAP=true
export USE_GCR=false

# TODO ASK SHIRLEY -- WHAT DO ALL OF THESE MEAN?

# Dataset configurations
export DATASET=NONE                # Change this for different datasets (e.g., math-hard, humaneval)
export TEMP=0.5                      # Temperature for sampling (adjust for creativity)
export MAX_NEW_TURNS=20               # Maximum number of new conversation turns per task
# ^ this is length of game
# export MAX_NEW_TURNS=2               # aditi edit for debugging
# export MAX_NEW_TURNS=8               # from original code
# export MAX_TOKENS=1536               # from original code
export MAX_TOKENS=256               # Maximum tokens per generation -- aditi edit
export COST_WEIGHT=5e-4              # Cost weight in reward function (adjust based on preference)
export LLM_RW_WEIGHT=1               # Weight for RLHF reward function
export USER_MODEL=gpt-4o-mini        # Choose the user model (e.g., gpt-4o-mini, gpt-4o)
# export N_EVAL=600                    # Number of evaluations per dataset split
export N_EVAL=100                   # aditi edit for debugging
# export N_EVAL=2                   # aditi edit for debugging
# ^ this is num of games played


# You can also adjust the assistant model and reward model based on your preference
# export ASSISTANT_MODEL=gpt-4o
# export REWARD_MODEL=claude-3-5-sonnet-20240620

export USER_MODEL=gpt-3.5-turbo
export ASSISTANT_MODEL=gpt-3.5-turbo
export REWARD_MODEL=gpt-3.5-turbo


# import os
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a list of possible target objects for the 20Q task
export TARGET_OBJECTS=("apple" "cat" "car" "dog" "airplane")

# Pick a random object from the list to pass as the target object
export TARGET_OBJECT=${TARGET_OBJECTS[$RANDOM % ${#TARGET_OBJECTS[@]}]}

CUDA_VISIBLE_DEVICES=0 python scripts/generate_conv_dpo_20q.py \
    --dataset $DATASET \
    --max_workers 1 \
    --num_samples 3 \
    --user_model_name $USER_MODEL \
    --assistant_model_name $ASSISTANT_MODEL \
    --reward_model $REWARD_MODEL \
    --max_new_tokens $MAX_TOKENS \
    --max_new_turns $MAX_NEW_TURNS \
    --window_size 2 \
    --temperature $TEMP \
    --top_p 0.9 \
    --task_weight 1 \
    --llm_rw_weight $LLM_RW_WEIGHT \
    --cost_weight $COST_WEIGHT \
    --n_eval_per_dataset $N_EVAL \
    --max_num_conv 1 \
    --task_name 20q \
    # --target_object "apple" \
    --resume

    # --target_object "$TARGET_OBJECT"  # Pass the selected target object
    # --max_num_conv 500 \
    # --max_workers 10 \

