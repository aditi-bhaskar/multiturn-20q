import os
import torch
from peft import PeftModel, PeftConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import sys

def load_model_and_tokenizer(model_name, max_new_tokens=2048, is_eval=True):
    is_macos = (torch.backends.mps.is_available() and torch.backends.mps.is_built()) or (
        not torch.cuda.is_available() and torch.backends.mps.is_available()
    )

    print("\nDEBUG: IN LOAD MODEL + TOKENIZER\n\n")
    sys.modules["bitsandbytes"] = None

    use_bnb = False  # Force disable bnb for Mac
    device_map={"": "cpu"}


    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    try:
        # Try loading as a PEFT adapter
        config = PeftConfig.from_pretrained(model_name)
        base_model_name = config.base_model_name_or_path

        print("\nin case 1:")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
        )

        model = PeftModel.from_pretrained(base_model, model_name, is_trainable=not is_eval)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    except Exception:
        print("\nin case 2:")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    print("\nepilogue for model loading:")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if is_eval else "right"
    model.eval()

    return model, tokenizer

