import torch
import logging
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from config.args_config import args
from config.global_vars import device
from data_provider import file_loader
from .layers import MLP


class Backbone(nn.Module):
    def __init__(self):
        logging.info("Start initializing the BIGCity backbone")
        super(Backbone, self).__init__()
        
        self.road_cnt = file_loader.road_cnt
        self.traj_category_cnt = file_loader.traj_category_cnt
        self.d_time_feature = 6
        
        self.LORA_R = 8
        self.LORA_ALPHA = 32
        self.LORA_DROPOUT = 0.1
        
        self.gpt2_config = None
        self.lora_config = None
        self.gpt2 = None
        self.tasks_mlp = nn.ModuleDict({
            "road_clas": None,
            "time_reg": None,
            "state_reg": None,
            "tul_clas": None,
        })
        self.build_model()
        
        logging.info("Finish initializing the BIGCity backbone")
        
    def forward(self, x, activate_heads):   
        B, Lpad = x.shape[0], x.shape[1]
        
        # add position embedding
        position_ids = torch.arange(Lpad, dtype=torch.long).unsqueeze(0).repeat(B, 1)
        position_embedding = self.gpt2.wpe.weight[position_ids]
        x = x + position_embedding
        
        # transformer block
        for block in self.gpt2.h:
            x = block(x)[0]

        # final layer norm
        x = self.gpt2.ln_f(x)
            
        # mlp for downstream tasks
        outputs = {
            name: self.tasks_mlp[name](x) for name in activate_heads # if self.tasks_mlp[name] is not None
        }
        
        return outputs
    
    def build_model(self):
        self.gpt2_config = GPT2Config.from_pretrained('./models/gpt2')
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.LORA_R, 
            lora_alpha=self.LORA_ALPHA, 
            lora_dropout=self.LORA_DROPOUT, 
            target_modules=["c_attn", "c_proj", "c_fc"],
            fan_in_fan_out=True
        )
        
        logging.info(f"GPT2 config: \n{self.gpt2_config}")
        logging.info(f"LoRA config: \n{self.lora_config}")
        
        self.gpt2 = GPT2Model.from_pretrained("./models/gpt2", config=self.gpt2_config)
        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)
        
        logging.info(f"GPT2+LoRA model structure: \n{self.gpt2}")
        
        lora_params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        
        logging.info(f"Total number of LoRA learnable parameters: {lora_params}")
        
        Dm, Dtf, N, Nclas = args.d_model, self.d_time_feature, self.road_cnt, self.traj_category_cnt
        self.tasks_mlp = nn.ModuleDict({
            "road_clas": MLP(Dm, Dm, N),     # road clas
            "time_reg": MLP(Dm, Dm, Dtf),    # time reg
            "state_reg": MLP(Dm, Dm, 1),     # traffic state reg
            "tul_clas": MLP(Dm, Dm, Nclas),  # traj reg
        })
        
        logging.info(f"Downstream tasks mlp: \n"
                     f"Road Classification: {self.tasks_mlp["road_clas"]} \n"
                     f"Time prediction: {self.tasks_mlp["time_reg"]} \n"
                     f"Regression: {self.tasks_mlp["state_reg"]} \n"
                     f"Trajectory classification: {self.tasks_mlp["tul_clas"]} \n")
        