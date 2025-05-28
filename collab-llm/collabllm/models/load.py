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




# import os
# import torch
# from peft import PeftModel, PeftConfig, get_peft_model
# from trl import AutoModelForCausalLMWithValueHead
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import torch


# def is_unsloth_model_auto(model_name):
#     return 'unsloth' in model_name

# def is_base_model_auto(model_name):
#     return 'meta-llama' in model_name or 'mistralai' in model_name or 'unsloth' in model_name 
# # or 'aditijb' in model_name  # aditi edit so my model is treated as base model



# def load_model_and_tokenizer(model_name, max_new_tokens, 
#                              peft_config=None, model_class=AutoModelForCausalLM, 
#                              is_eval=True, device=None,
#                              load_in_4bit_aditi=False):
#     if device is None:
#         local_rank = os.getenv("LOCAL_RANK")
#         device = "cuda" + str(local_rank) if torch.cuda.is_available() else "cpu"

#     # Detect macOS environment and disable bitsandbytes
#     is_macos = (torch.backends.mps.is_available() and torch.backends.mps.is_built()) or (not torch.cuda.is_available() and torch.backends.mps.is_available())
#     bnb_config = None
#     load_in_4bit = False
#     load_in_8bit = False

#     if not is_macos:
#         load_in_4bit = True
#         load_in_8bit = False # aditi edit may 27 2025
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=False,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )

#     #  only two cases matter to me
#     if is_base_model_auto(model_name):
#         model = model_class.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#             use_cache=False,
#             quantization_config=bnb_config if not is_macos else None,
#             torch_dtype=torch.float16,
#             # load_in_4bit=load_in_4bit if not is_macos else False,
#             # load_in_8bit=load_in_8bit
#         )
#         if peft_config is not None:
#             model = get_peft_model(model, peft_config)
#         model = model.to(device)

#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#     else:
#         # For PEFT models: load base model without bnb, then load adapter
#         config = PeftConfig.from_pretrained(model_name)
#         base_model = model_class.from_pretrained(
#             config.base_model_name_or_path,
#             trust_remote_code=True,
#             torch_dtype=torch.float16,
#             load_in_4bit=False,  # no bnb for base model load
#             load_in_8bit=False,
#             device_map=None
#         )
#         model = PeftModel.from_pretrained(base_model, model_name, is_trainable=not is_eval)
#         model = model.to(device)
#         tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

#     # Set padding and pad token
#     tokenizer.padding_side = 'left' if is_eval else 'right'
#     tokenizer.pad_token = tokenizer.eos_token

#     return model.eval(), tokenizer





# def load_model_and_tokenizer(model_name, 
#                              max_new_tokens, 
#                              peft_config=None,
#                              model_class=AutoModelForCausalLM,
#                              is_eval=True,
#                              device=None,
#                              load_in_4bit_aditi=False): # aditi addition: turn off the 4 bit quantization thing -- only works on linux
#     if device is None:
#         local_rank = os.getenv("LOCAL_RANK")
#         device = "cuda" + str(local_rank) if torch.cuda.is_available() else "cpu"   # aditi edit to fix cpu vs gpu usage
#         # device = "cuda:" + str(local_rank)
    

#     #  aditi edit to AVOID THE BNB!!!
#     is_macos = (torch.backends.mps.is_available() and torch.backends.mps.is_built()) or (torch.cuda.is_available() == False and torch.backends.mps.is_available())
#     if is_macos:
#         bnb_config = None
#         load_in_4bit_aditi = False
#     else:
#         # your bitsandbytes config here
#         load_in_4bit_aditi = True


#     if is_unsloth_model_auto(model_name):
#         from unsloth import FastLanguageModel
#         model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name=model_name,
#             max_seq_length=max_new_tokens,
#             dtype=None,
#             use_cache=False,
#             load_in_4bit=load_in_4bit_aditi # aditi addition
#             )
#         if is_eval:
#             FastLanguageModel.for_inference(model)
#         model = model.to(device)
#     else:
#         # bnb_config = BitsAndBytesConfig(
#         #     load_in_4bit=load_in_4bit_aditi,  # aditi addition
#         #     bnb_4bit_quant_type="nf4",
#         #     bnb_4bit_use_double_quant=False,
#         #     bnb_4bit_compute_dtype=torch.bfloat16
#         #     )

#         #  aditi edit: I force it to NOT use the weird 4 bit float whatever stuff
#         bnb_config = None
#         if load_in_4bit_aditi and not is_macos:  # edited aditi
#             bnb_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=False,
#                 bnb_4bit_compute_dtype=torch.bfloat32   # changed from bfloat16
#             )
#             print("\nusing bnb 4bit bstuff\n")
        
#         if is_base_model_auto(model_name):
#             print(f'[load_model_and_tokenizer] Loading {model_class}: {model_name}')
#             if model_class == AutoModelForCausalLM:
#                 print("\n fail case 1 \n")

#                 model = model_class.from_pretrained(
#                     model_name,
#                     trust_remote_code=True,
#                     use_cache=False,
#                     # device_map={'': device},
#                     quantization_config=bnb_config,
#                     torch_dtype=torch.float16,  # added by aditi to avoid the bits and bytes stuff may 26
#                     # is_trainable=not eval
#                     # load_in_4bit=load_in_4bit_aditi # aditi addition
#                 )
#                 if peft_config is not None:
#                     model = get_peft_model(model, peft_config)
#                 model = model.to(device)
#             elif model_class == AutoModelForCausalLMWithValueHead:
#                 print("\n fail case 2 \n")

#                 assert peft_config is not None
#                 model = model_class.from_pretrained(
#                         model_name,
#                         trust_remote_code=True,
#                         # device_map={'': device},
#                         peft_config=peft_config,
#                         quantization_config=bnb_config,
#                         is_trainable=not is_eval,
#                         torch_dtype=torch.float16,  # added by aditi to avoid the bits and bytes stuff may 26

#                         # load_in_4bit=load_in_4bit_aditi # aditi addition
#                     )
#                 model = model.to(device)

#             tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         else:
#             print(f'[load_model_and_tokenizer] Load peft model from local {model_name}')
#             config = PeftConfig.from_pretrained(model_name)
#             if model_class == AutoModelForCausalLM:
#                 print("\n fail case 3 \n")

#                 base_model = model_class.from_pretrained(config.base_model_name_or_path, 
#                                                         #  device_map={'': device},
#                                                         #  quantization_config=bnb_config,
#                                                         #  load_in_4bit=load_in_4bit_aditi # aditi addition
#                                                         # torch_dtype=torch.float16,  # added by aditi to avoid the bits and bytes stuff may 26

#                                                          )
#                 model = PeftModel.from_pretrained(base_model, model_name, is_trainable=not is_eval)
#                 model = model.to(device)


#             if model_class == AutoModelForCausalLMWithValueHead:
#                 print("\n fail case 4 \n")

#                 model = model_class.from_pretrained(model_name,
#                                                     # device_map={'': device},
#                                                     # local_files_only=True, # need to disable for azure
#                                                     quantization_config=bnb_config,
#                                                     is_trainable=not is_eval,
#                                                     # load_in_4bit=load_in_4bit_aditi # aditi addition
#                                                     torch_dtype=torch.float16,  # added by aditi to avoid the bits and bytes stuff may 26

#                                                     )
#                 model = model.to(device)

#             tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#     if is_eval:
#         tokenizer.padding_side = 'left'
#     else:
#         tokenizer.padding_side = 'right'
#     tokenizer.pad_token = tokenizer.eos_token
#     print(f'[load_model_and_tokenizer] Set default padding side to {tokenizer.padding_side}')
    




#     def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())
#     n_trainable_params, n_params = count_parameters(model)

#     print(f'[load_model_and_tokenizer] Number of trainable parameters: {n_trainable_params}'
#             f' / {n_params} ({n_trainable_params / n_params * 100.:.2f}%)')
#     return model.eval(), tokenizer
