import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from transformers import GPT2Model, GPT2Config


from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModelForCausalLM
)

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


# lora_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=LORA_R, 
#             lora_alpha=LORA_ALPHA, 
#             lora_dropout=LORA_DROPOUT, 
#             target_modules=["c_attn", "c_proj", "c_fc"],
#             fan_in_fan_out=True
#         )


# gpt2_config = GPT2Config.from_pretrained('./models/gpt2')
# model = GPT2Model.from_pretrained('./models/gpt2', config=gpt2_config)

# # params = sum(p.numel() for p in model.parameters())
# # params_r = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # print(params, params_r)

# model = get_peft_model(model, lora_config)


# B, L, dmodel = 2, 10, 768 
# token_embeddings = torch.randn(B, L, dmodel)

# print(model.h)

# outputs = token_embeddings
# for block in model.h:
#     outputs = block(outputs)[0]

# last_block_output = outputs

# print(last_block_output.shape)

edges = torch.randn((2, 10))
edges.repeat(1, 3)
print(edges.shape)