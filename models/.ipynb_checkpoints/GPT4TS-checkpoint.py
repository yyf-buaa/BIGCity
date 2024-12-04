from models.st_tokenizer import StTokenizer
from models.backbone import BIGCity
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import GPT2Tokenizer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from layers.Embed import DataEmbedding, DataEmbedding_wo_time,TrajDataEmbedding
import logging
import os
import copy

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tokenizer = StTokenizer()
        self.city = BIGCity()
 
    def forward(self, input1, input2, input3):  # 根据 StTokenizer 需要的参数添加更多输入
        # 假设 StTokenizer 需要两个输入参数 input1 和 input2
        # 这里可能需要一些预处理步骤来合并或转换输入
        tokenizer_output = self.tokenizer(input1, input2, input3)  # 调用 StTokenizer 的 forward 方法
        # 如果 BIGCity 只需要 tokenizer 的输出作为输入
        output1, output2, output3 = self.city(tokenizer_output)
        # 根据需要返回输出
        return output1, output2, output3
        
