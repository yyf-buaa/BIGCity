import torch
import logging
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from config.args_config import args
from .layers import MLP

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
        self.mlp_c, self.mlp_t, self.mlp_r = None, None, None
        self.build_model()
        
        logging.info("Finish initializing the BIGCity backbone")
        
    def forward(self, x):   
        for block in self.gpt2.h:
            x = block(x)[0]
            
        clas_out = self.mlp_c(x)
        t_out = self.mlp_t(x)
        reg_out = self.mlp_r(x)
        return clas_out, t_out, reg_out
    
    def build_model(self):
        self.gpt2_config = GPT2Config.from_pretrained('./models/gpt2')
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.LORA_R, 
            lora_alpha=self.LORA_ALPHA, 
            lora_dropout=self.LORA_DROPOUT, 
            target_modules=["c_attn", "c_proj", "c_fc"]
        )
        
        logging.info(f"GPT2 config: \n{self.gpt2_config}")
        logging.info(f"LoRA config: \n{self.lora_config}")
        
        self.gpt2 = GPT2Model(self.gpt2_config)
        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)
        
        logging.info(f"GPT2+LoRA model structure: \n{self.gpt2}")
        
        lora_params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        
        logging.info(f"Total number of LoRA learnable parameters: {lora_params}")
        
        d_model, N = args.d_model, 5269
        self.mlp_c = nn.Linear(d_model, N)
        self.mlp_t = nn.Linear(d_model, 1)
        self.mlp_r = nn.Linear(d_model, 1)
        
        logging.info(f"Downstream tasks mlp: \n"
                     f"Classification: {self.mlp_c} \n"
                     f"Time prediction: {self.mlp_t} \n"
                     f"Regression: {self.mlp_r}")
        