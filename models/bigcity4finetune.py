import torch
import torch.nn as nn

from transformers import GPT2Tokenizer

from config.args_config import args

from models.st_tokenizer import StTokenizer
from models.backbone import Backbone
from models.bigcity import BigCity


NEXT_HOP_PROMPT = "Predict the next hop of the trajectory, generate the road segment id corresponding to each" # [CLS]
TRAJ_CLASSIFY_PROMPT = "Classify the trajectory into different categories, generate the Category id corresponding to each" # [CLS]
TIME_REGRESS_PROMPT = "predict the time of the trajectory, regress the time stamp corresponding to each" # [REG]
TRAFFIC_STATE_REGRESS_PROMPT = "Predict the traffic state of the trajectory, regress the traffic state corresponding to each" # [REG]
TRAJ_RECOVER_PROMPT = "Recover the vacant road segments in the trajectory, generate the road segment id corresponding to each" # [CLS]

            
class BigCity4FineTune(BigCity):
    def __init__(self, device):
        super(BigCity4FineTune, self).__init__(device)
        
        checkpoint = torch.load(f"./checkpoints/{args.city}_pretrain_best.pth", weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        for name, param in self.tokenizer.named_parameters():
            param.requires_grad = False
        self.special_token.requires_grad_(False)
        
        self.prompt_tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2")
        self.NEXT_HOP_PROMPT = self.encode_prompt(NEXT_HOP_PROMPT, 0)
        self.TIME_REGRESS_PROMPT = self.encode_prompt(TIME_REGRESS_PROMPT, 1)
        self.TRAFFIC_STATE_REGRESS_PROMPT = self.encode_prompt(TRAFFIC_STATE_REGRESS_PROMPT, 1)
        self.TRAJ_RECOVER_PROMPT = self.encode_prompt(TRAJ_RECOVER_PROMPT, 0)
        self.TRAJ_CLASSIFY_PROMPT = self.encode_prompt(TRAJ_CLASSIFY_PROMPT, 1)
        
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def encode_prompt(self, prompt, special_token_id):
        encoded_prompt = self.prompt_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_embedding = self.backbone.gpt2.wte(encoded_prompt)
        
        special_token = self.special_token(torch.tensor([special_token_id]).to(self.device)).unsqueeze(1)
        
        prompt_with_special_token = torch.cat([prompt_embedding, special_token], dim=1)
        
        return prompt_with_special_token

        
    def forward(self, task_name, batch_road_id, batch_time_id, batch_time_features, batch_label):
        batch_tokens = self.tokenizer(batch_road_id, batch_time_id, batch_time_features)
        B, L, D = batch_tokens.shape
        Dtf = 6

        if task_name == "next_hop":
            clas_token = self.special_token(torch.tensor([0]).to(self.device)).expand(B, 1, D)
            batch_psm_tokens = torch.cat([self.NEXT_HOP_PROMPT.expand(B, -1, -1), batch_tokens, clas_token], dim=1) # L = 18 + L + 1
            
            next_hop_road_clas_output = self.backbone(batch_psm_tokens, ["road_clas"])["road_clas"] 
            
            predict_road_id = next_hop_road_clas_output[:, -1, :] # (B, Nroad)
            
            next_hop_loss = self.cross_entropy(predict_road_id, batch_label)
            return next_hop_loss
        
        elif task_name == "traj_classify":
            clas_token = self.special_token(torch.tensor([0]).to(self.device)).expand(B, 1, D)
            batch_psm_tokens = torch.cat([self.TRAJ_CLASSIFY_PROMPT.expand(B, -1, -1), batch_tokens, clas_token], dim=1) # L = 16 + L + 1
            
            tul_clas_output = self.backbone(batch_psm_tokens, ["tul_clas"])["tul_clas"]
            
            predict_classify_id = tul_clas_output[:, -1, :] # (B, Nclas)
                    
            traj_classify_loss = self.cross_entropy(predict_classify_id, batch_label)
            return traj_classify_loss
        
        elif task_name == "time_reg":
            reg_token = self.special_token(torch.tensor([1]).to(self.device)).expand(B, L, D)
            batch_psm_tokens = torch.cat([self.TIME_REGRESS_PROMPT.expand(B, -1, -1), batch_tokens, reg_token], dim=1) # L = 16 + L + L
            
            time_reg_output= self.backbone(batch_psm_tokens, ["time_reg"])["time_reg"]
            
            predict_time_features = time_reg_output[:, -L:, :] # (B, L, 6)
            
            time_features_loss = self.mse(predict_time_features, batch_label)
            return time_features_loss
            
        elif task_name == "traffic_state_reg":
            reg_token = self.special_token(torch.tensor([1]).to(self.device)).expand(B, L, D)
            batch_psm_tokens = torch.cat([self.TRAFFIC_STATE_REGRESS_PROMPT.expand(B, -1, -1), batch_tokens, reg_token], dim=1) # L = 17 + L + L

            traffic_state_reg_output = self.backbone(batch_psm_tokens, ["state_reg"])["state_reg"]
            
            predict_traffic_states = traffic_state_reg_output[:, -L:, :] # (B, L, 1)
            
            traffic_state_loss = self.mse(predict_traffic_states.squeeze(2), batch_label) 
            return traffic_state_loss
        
        elif task_name == "traj_recover":
            num_mask_per_seq = int(args.mask_rate * L)
            
            clas_token = self.special_token(torch.tensor([0]).to(self.device)).expand(B, num_mask_per_seq, D)
            batch_psm_tokens = torch.cat([self.TRAJ_RECOVER_PROMPT.expand(B, -1, -1), batch_tokens, clas_token], dim=1) # L = 19 + L + num_mask_per_seq

            traj_recover_road_clas_output = self.backbone(batch_psm_tokens, ["road_clas"])["road_clas"]
            
            predict_road_id = traj_recover_road_clas_output[:, -num_mask_per_seq:, :]
            
            traj_recover_loss = self.cross_entropy(predict_road_id.reshape(-1, predict_road_id.shape[-1]), batch_label.view(-1))
            return traj_recover_loss
            
