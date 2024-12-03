import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from transformers import GPT2Model, GPT2Config


# 配置和模型初始化
gpt2_config = GPT2Config.from_pretrained('./models/gpt2')
gpt2 = GPT2Model.from_pretrained('./models/gpt2', config=gpt2_config)

# 假设我们有一个 batch 和 token_embeddings
B, L, dmodel = 2, 10, 768  # 示例: batch size = 2, sequence length = 10, embedding size = 768
token_embeddings = torch.randn(B, L, dmodel)  # (B, L, dmodel)

# 获取位置嵌入
position_embeddings = gpt2.wpe.weight  # (max_position_embeddings, d_model)

# 生成位置索引
position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).repeat(B, 1)
print(position_ids)

# # 获取对应位置的嵌入
# position_embeds = position_embeddings[position_ids]  # (B, L, d_model)
# print(position_embeds.shape)

# # 将位置嵌入与token嵌入相加
# input_embeddings = token_embeddings + position_embeds  # (B, L, d_model)

# # 显示结果
# print(input_embeddings.shape)  # 应该是 (B, L, d_model)