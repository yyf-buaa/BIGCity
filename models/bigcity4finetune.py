import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers import GPT2Tokenizer

from config.args_config import args
from config.global_vars import device

from models.st_tokenizer import StTokenizer
from models.backbone import Backbone
from models.bigcity import BigCity


NEXT_HOP_PROMPT = "Predict the next hop of the trajectory, generate the road segment id corresponding to each" # [CLS]
TIME_REGRESS_PROMPT = "predict the time of the trajectory, regress the time stamp corresponding to each" # [REG]
TRAFFIC_STATE_REGRESS_PROMPT = "Predict the traffic state of the trajectory, regress the traffic state corresponding to each" # [REG]
TRAJ_RECOVER_PROMPT = "Recover the vacant road segments in the trajectory, generate the road segment id corresponding to each" # [CLS]
TRAJ_CLASSIFY_PROMPT = "Classify the trajectory into different categories, generate the Category id corresponding to each" # [CLS]
            
class BigCity4FineTune(BigCity):
    def __init__(self):
        super(BigCity4FineTune, self).__init__()
        
        self.task_name = "state_reg"
        
        checkpoint = torch.load("./checkpoints/xa_checkpoint1.pth", weights_only=True)
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
        
    def encode_prompt(self, prompt, special_token_id):
        encoded_prompt = self.prompt_tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_embedding = self.backbone.gpt2.wte(encoded_prompt)
        
        special_token = self.special_token(torch.tensor([special_token_id]).to(device)).unsqueeze(1)
        
        prompt_with_special_token = torch.cat([prompt_embedding, special_token], dim=1)
        
        return prompt_with_special_token

        
    def forward(self, batch_road_id, batch_time_id, batch_time_features, batch_road_flow, mask, num_mask):
        batch_tokens = self.tokenizer(batch_road_id, batch_time_id, batch_time_features)
        B, L, D = batch_tokens.shape
        Dtf = 6
                
        if self.task_name == "next_hop":
            clas_token = self.special_token(torch.tensor([0]).to(device)).expand(B, 1, 1)
            batch_psm_tokens = torch.cat([self.NEXT_PROMPT.expand(B, -1, -1), batch_tokens, clas_token], dim=1)
            clas_output, time_output, reg_output = self.backbone(batch_psm_tokens) 
            clas_indices = torch.arange(-1, 0).to(device)
            predict_road_id = clas_output[:, clas_indices, :]          
            road_id_loss = F.cross_entropy(predict_road_id.reshape(-1, self.backbone.road_cnt), batch_road_id[:, -1])
            return road_id_loss
        
        elif self.task_name == "time_reg":
            reg_token = self.special_token(torch.tensor([1]).to(device)).expand(B, L, D)
            batch_psm_tokens = torch.cat([self.TIME_REGRESS_PROMPT.expand(B, -1, -1), batch_tokens, reg_token], dim=1)
            clas_output, time_output, reg_output = self.backbone(batch_psm_tokens)
            predict_time_features = time_output[:, -L:, :]
            time_features_loss = F.mse_loss(predict_time_features.reshape(-1, Dtf), batch_time_features.view(-1, Dtf))
            return time_features_loss
            
        elif self.task_name == "state_reg":
            reg_token = self.special_token(torch.tensor([1]).to(device)).expand(B, L, D)
            batch_psm_tokens = torch.cat([self.TRAFFIC_STATE_REGRESS_PROMPT.expand(B, -1, -1), batch_tokens, reg_token], dim=1)
            clas_output, time_output, reg_output = self.backbone(batch_psm_tokens)
            predict_states = reg_output[:, -L:, :]
            traffic_state_loss = F.mse_loss(predict_states.reshape(-1), batch_road_flow.view(-1)) 
            return traffic_state_loss
        
        elif self.task_name == "traj_recover":
            clas_token = self.special_token(torch.tensor([0]).to(device)).expand(B, num_mask, D)
            mask_batch_tokens = batch_tokens.masked_fill(mask.unsqueeze(-1).expand(-1, -1, D) == 0, 0)
            batch_psm_tokens = torch.cat([self.TRAJ_RECOVER_PROMPT.expand(B, -1, -1), mask_batch_tokens, clas_token], dim=1)
            clas_output, time_output, reg_output = self.backbone(batch_psm_tokens)
            clas_indices = torch.arange(-num_mask, 0).to(device)
            predict_road_id = clas_output[:, clas_indices, :]
            road_id_loss = F.cross_entropy(predict_road_id.reshape(-1, self.backbone.road_cnt), batch_road_id[mask == 0])
            return road_id_loss
            
        elif self.task_name == "traj_classify":
            clas_token = self.special_token(torch.tensor([0]).to(device)).expand(B, 1, 1)
            batch_psm_tokens = torch.cat([self.TRAJ_CLASSIFY_PROMPT.expand(B, -1, -1), batch_tokens, clas_token], dim=1)
            clas_output, time_output, reg_output = self.backbone(batch_psm_tokens) 
            clas_indices = torch.arange(-1, 0).to(device)
            predict_classify_id = clas_output[:, clas_indices, :]          
            classify_id_loss = F.cross_entropy(predict_classify_id.reshape(-1, self.backbone.road_cnt), batch_road_id)
            return classify_id_loss

        # return super(BigCity4FineTune, self).forward(batch_road_id, batch_time_id, batch_time_features, mask, num_mask)
