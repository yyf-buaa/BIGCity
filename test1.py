import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from transformers import GPT2Model, GPT2Config

def padding_mask(B, L):
    mask = torch.ones(B, L)
    num_mask = int(0.5 * L)
    for i in range(B):
        indices_to_mask = torch.randperm(L, dtype=torch.long)[:num_mask]
        mask[i][indices_to_mask] = 0
    return mask, num_mask


def psm_input(batch_road_id, batch_time_index, mask, num_mask):
        B, T = batch_road_id.shape
        batch_masked_road_id = batch_road_id.masked_fill(mask == 0, 888)
        special_token = torch.tensor([11, 12, 13])
        special_token = torch.tile(special_token, (B, num_mask))
        special_time = torch.zeros(B, 3 * num_mask)
        return torch.cat([batch_masked_road_id, special_token], dim=1), torch.cat([batch_time_index, special_time],
                                                                                  dim=1)

B, L, d = 3, 6, 4
mask, mask_num = padding_mask(B, L)
print(mask)
# mask = mask.unsqueeze(-1).expand(-1, -1, d)
# print(mask)
# batch_tokens = torch.randn((B, L, d))

# print(batch_tokens)
# print(batch_tokens.masked_fill(mask == 0, 0))
# # zzz
# print(batch_tokens)
batch_road_id = torch.randint(0, 10, (B, L))
batch_time_id = torch.randint(0, 10, (B, L))

print(batch_road_id)
x = batch_road_id[mask == 0]
print(x)

# a, b = psm_input(batch_road_id, batch_time_id, mask, mask_num)
# print(a)
# print(b)
# print(a.shape, b.shape)
# predict_road_id = torch.tensor([[[0.8, 0.3, 0.5], [0.6, 0.1, 0.3]], [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]])
# real_road_id = torch.tensor([[0, 2], [1, 1]])

# predict_road_id_flat = predict_road_id.view(-1, 3)  # Flatten to [batch_size * seq_len, num_classes]
# real_road_id_flat = real_road_id.view(-1)  # Flatten to [batch_size * seq_len]

# # Apply Cross-Entropy Loss
# road_id_loss = F.cross_entropy(predict_road_id_flat, real_road_id_flat)
# print(road_id_loss)
a = torch.randn(2, 3, 1)
b = torch.randn(2, 3)
print(F.mse_loss(a, b))
