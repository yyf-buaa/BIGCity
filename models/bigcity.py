import torch
import torch.nn as nn

from config.args_config import args
from models.st_tokenizer import StTokenizer
from models.backbone import Backbone


class BigCity(nn.Module):
    def __init__(self):
        super(BigCity, self).__init__()
        
        self.tokenizer = StTokenizer()
        self.clas_token = nn.Parameter(torch.full((args.d_model,), 1, dtype=torch.float32))
        self.reg_token = nn.Parameter(torch.full((args.d_model,), 2, dtype=torch.float32))
        
        self.backbone = Backbone()
        
 
    def forward(self, batch_road_id, batch_time_id, batch_time_features, mask, num_mask):
        # get tokens
        batch_tokens = self.tokenizer(batch_road_id, batch_time_id, batch_time_features)
        
        B, L, D = batch_tokens.shape
        
        # mask
        mask_batch_tokens = batch_tokens.masked_fill(mask.unsqueeze(-1).expand(-1, -1, D) == 0, 0)
        
        # add special tokens: (clas, reg)
        clas_token = self.clas_token.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        reg_token = self.reg_token.unsqueeze(0).unsqueeze(0)    # (1, 1, D)
        special_tokens = torch.cat([clas_token, reg_token] * num_mask, dim=1) # (1, 2*num_mask, D)
        special_tokens = special_tokens.repeat(B, 1, 1) # (B, 2*num_mask, D)
        batch_psm_tokens = torch.cat([mask_batch_tokens, special_tokens], dim=1) # (B, L + 2*num_mask, D)
        
        # backbone
        clas_output, time_output, reg_output = self.backbone(batch_psm_tokens)
        
        # get output special tokens
        clas_indices = torch.arange(-2 * num_mask, 0, 2)
        reg_indices = torch.arange(-2 * num_mask + 1, 1, 2) 
        
        return clas_output[:, clas_indices, :], time_output[:, reg_indices, :], reg_output[:, reg_indices, :]
        
