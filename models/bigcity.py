import torch
import torch.nn as nn

from config.args_config import args
from models.st_tokenizer import StTokenizer
from models.backbone import Backbone
from config.global_vars import device


class BigCity(nn.Module):
    def __init__(self):
        super(BigCity, self).__init__()
        
        self.tokenizer = StTokenizer().to(device)
        self.clas_token = nn.Parameter(torch.full((args.d_model,), 1, dtype=torch.float32), requires_grad=True).to(device)
        self.reg_token = nn.Parameter(torch.full((args.d_model,), 2, dtype=torch.float32), requires_grad=True).to(device)
        
        self.backbone = Backbone().to(device)
        
 
    def forward(self, batch_road_id, batch_time_id, batch_time_features, mask, num_mask):
        # get tokens
        batch_tokens = self.tokenizer(batch_road_id, batch_time_id, batch_time_features)
        
        B, L, D = batch_tokens.shape
        
        # mask
        mask_batch_tokens = batch_tokens.masked_fill(mask.unsqueeze(-1).expand(-1, -1, D) == 0, 0)
        
        # add special tokens: (clas, reg)
        clas_token = self.clas_token.expand(B, num_mask, D)  # (B, num_mask, D)
        reg_token = self.reg_token.expand(B, num_mask, D)   # (B, num_mask, D)
        special_tokens = torch.stack([clas_token, reg_token], dim=2).view(B, -1, D)  # (B, 2*num_mask, D)
        batch_psm_tokens = torch.cat([mask_batch_tokens, special_tokens], dim=1) # (B, L + 2*num_mask, D)

        # backbone
        clas_output, time_output, reg_output = self.backbone(batch_psm_tokens)
        
        # get output special tokens
        clas_indices = torch.arange(-2 * num_mask, 0, 2).to(device)
        reg_indices = torch.arange(-2 * num_mask + 1, 1, 2).to(device)
        
        return clas_output[:, clas_indices, :], time_output[:, reg_indices, :], reg_output[:, reg_indices, :]
        
