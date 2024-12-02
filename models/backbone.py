import torch
import logging
import torch.nn as nn
import pandas as pd

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model,GPT2LMHeadModel

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModelForCausalLM
)

from config import data_filename
from config.args_config import args


class BIGCity(nn.Module):
    def __init__(self):
        logging.info("Start initializing the BIGCity backbone")
        super(BIGCity, self).__init__()
        
        self.LORA_R = 8
        self.LORA_ALPHA = 32
        self.LORA_DROPOUT = 0.1
        
        self.gpt2_config = None
        self.lora_config = None
        self.gpt2 = None
        self.build_model()
        
        logging.info("Finish initializing the BIGCity backbone")
        
    def forward(self, x):
        print(x.shape)
        hidden =  self.gpt2(inputs_embeds=x)
        print(type(hidden))
    
    def build_model(self):
        self.gpt2_config = GPT2Config.from_pretrained('./models/gpt2')
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.LORA_R, 
            lora_alpha=self.LORA_ALPHA, 
            lora_dropout=self.LORA_DROPOUT, 
            target_modules=["c_attn","c_fc"]
        )
        
        self.gpt2 = GPT2LMHeadModel(config=self.gpt2_config)
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            # print(i, name)
            param.requires_grad = False
        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)
        
        lora_params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        print(lora_params)
        