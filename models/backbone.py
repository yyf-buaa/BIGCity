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
        
        self.road_cnt = 5269
        self.d_time_feature = 6
        
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
        B, L = x.shape[0], args.seq_len
        
        # add position embedding
        position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).repeat(B, 1)
        position_embedding = self.gpt2.wpe.weight[position_ids]
        x = x + position_embedding
        
        # transformer block
        for block in self.gpt2.h:
            x = block(x)[0]
        
        # mlp for downstream tasks
        clas_out = self.mlp_c(x)
        time_out = self.mlp_t(x)
        reg_out = self.mlp_r(x)
        
        return clas_out, time_out, reg_out
    
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
        
        self.gpt2 = GPT2Model.from_pretrained('./models/gpt2', config=self.gpt2_config)
        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)
        
        logging.info(f"GPT2+LoRA model structure: \n{self.gpt2}")
        
        lora_params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        
        logging.info(f"Total number of LoRA learnable parameters: {lora_params}")
        
        Dm, Dtf, N = args.d_model, self.d_time_feature, self.road_cnt
        self.mlp_c = nn.Linear(Dm, N)
        self.mlp_t = nn.Linear(Dm, Dtf)
        self.mlp_r = nn.Linear(Dm, 1)
        
        logging.info(f"Downstream tasks mlp: \n"
                     f"Classification: {self.mlp_c} \n"
                     f"Time prediction: {self.mlp_t} \n"
                     f"Regression: {self.mlp_r} \n")
        