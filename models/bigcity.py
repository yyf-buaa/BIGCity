import torch
import torch.nn as nn

from config.args_config import args
from models.st_tokenizer import StTokenizer
from models.backbone import Backbone


class BigCity(nn.Module):
    def __init__(self, device):
        super(BigCity, self).__init__()
        
        self.device = device
        
        self.tokenizer = StTokenizer(device)
        
        # 0: clas; 1: reg
        self.special_token = nn.Embedding(num_embeddings=2, embedding_dim=args.d_model)
                
        self.backbone = Backbone(device)
        
 
    def forward(self, batch_road_id, batch_time_id, batch_time_features, mask, num_mask):
        # get tokens
        batch_tokens = self.tokenizer(batch_road_id, batch_time_id, batch_time_features)
        
        B, L, D = batch_tokens.shape
        
        # mask
        mask_batch_tokens = batch_tokens.masked_fill(mask.unsqueeze(-1).expand(-1, -1, D) == 0, 0)
        # batch_tokens[mask == 0, :] = 0 # Which is better?
        
        # add special tokens: (clas, reg)
        clas_token = self.special_token(torch.tensor([0]).to(self.device)).expand(B, num_mask, D)  # (B, num_mask, D)
        reg_token = self.special_token(torch.tensor([1]).to(self.device)).expand(B, num_mask, D)   # (B, num_mask, D)
        special_tokens = torch.stack([clas_token, reg_token], dim=2).view(B, -1, D)  # (B, 2*num_mask, D)
        batch_psm_tokens = torch.cat([mask_batch_tokens, special_tokens], dim=1) # (B, L + 2*num_mask, D)
        
        # backbone
        output = self.backbone(batch_psm_tokens, ["road_clas", "time_reg", "state_reg"])
        clas_output, time_output, reg_output = output["road_clas"], output["time_reg"], output["state_reg"]
        
        # get output special tokens
        clas_indices = torch.arange(-2 * num_mask, 0, 2).to(self.device)
        reg_indices = torch.arange(-2 * num_mask + 1, 1, 2).to(self.device)
        
        return clas_output[:, clas_indices, :], time_output[:, reg_indices, :], reg_output[:, reg_indices, :]
        
