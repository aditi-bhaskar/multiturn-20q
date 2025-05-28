import os
import torch
from peft import PeftModel, PeftConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_name, max_new_tokens=2048, is_eval=True):
    is_macos = (torch.backends.mps.is_available() and torch.backends.mps.is_built()) or (
        not torch.cuda.is_available() and torch.backends.mps.is_available()
    )

    use_bnb = not is_macos
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    try:
        # Try loading as a PEFT adapter
        config = PeftConfig.from_pretrained(model_name)
        base_model_name = config.base_model_name_or_path

        if use_bnb:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                # device_map="auto",
                trust_remote_code=True,
                device_map={"": "cpu"},
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                # device_map="auto",
                trust_remote_code=True,
                device_map={"": "cpu"},
            )

        model = PeftModel.from_pretrained(base_model, model_name, is_trainable=not is_eval)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    except Exception:
        # Fall back to base model
        if use_bnb:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if is_eval else "right"
    model.eval()

    return model, tokenizer

