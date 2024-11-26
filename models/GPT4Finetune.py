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
from models import GPT4TS
import logging
import os
import copy


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.road_num = configs.road_num
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.city = configs.city
        pretrain_configs = copy.deepcopy(configs)
        pretrain_configs.task_name = 'Pretrain'
        self.pretrain_model = GPT4TS.Model(pretrain_configs).float()
        self.checkpoint_pth = os.path.join('./checkpoints/' + configs.checkpoint_name, 'checkpoint.pth')
        self.inst_tokenizer = GPT2Tokenizer.from_pretrained('../gpt2')
        self.gpt2_checkpoint_pth =  os.path.join('./checkpoints/' + configs.gpt2_checkpoint_name, 'checkpoint.pth')
        saved_state_dict = torch.load(self.checkpoint_pth,map_location = 'cpu')
        new_state_dict = {}
        for key, value in saved_state_dict.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            new_state_dict[key] = value
        self.pretrain_model.load_state_dict(new_state_dict, strict=False)
        gpt2_state_dict = torch.load(self.gpt2_checkpoint_pth,map_location='cpu')
        new_state_dict = {}
        for key, value in gpt2_state_dict.items():
            if key.startswith('module.gpt2.'):
                key = key[len('module.gpt2.'):]
            new_state_dict[key] = value
        # import ipdb
        # ipdb.set_trace()
        self.pretrain_model.gpt2.load_state_dict(new_state_dict, strict=False)
        logging.info('Finish loading pretrain model')
        logging.info(f"seq_len={self.seq_len}\n")
        self.d_ff = configs.d_ff
        for p in self.pretrain_model.parameters():
            p.requires_grad = False #冻住预训练的base-model
        for p in self.pretrain_model.time_ffn.parameters():
            p.requires_grad = True
        for p in self.pretrain_model.flow_ffn.parameters():
            p.requires_grad = True
        for p in self.pretrain_model.classify_layer.parameters():
            p.requires_grad = True 
        self.gpt2 = self.pretrain_model.gpt2
        self.enc_embedding = self.pretrain_model.enc_embedding
        for i, (name, param) in enumerate(self.pretrain_model.named_parameters()):
             logging.info( str(name) + '\t' + str(param.shape) + '\t' +
                str(param.device) + '\t' + str(param.requires_grad)) 


    def forward(self, x_id, x_mark_enc,inst_id):
        outputs_road, output_time, output_flow = self.task_tuning(x_id, x_mark_enc,inst_id)
        return outputs_road, output_time, output_flow

    def task_tuning(self, x_id, x_mark_enc,inst_id):
        x_enc = self.token_embedding(x_id)
        B, L, M = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        inst_out = self.gpt2.get_input_embeddings(inst_id)
        enc_out = torch.cat((inst_out,enc_out),dim=1)
        hidden = self.gpt2(inputs_embeds=enc_out).hidden_states[-1]
        hidden_cls = self.get_match_hidden(hidden, x_id, self.road_num + 1)
        output_roads = self.classify_layer(hidden_cls)
        hidden_tem = self.get_match_hidden(hidden, x_id, self.road_num + 2)
        output_time = self.time_ffn(hidden_tem)
        output_time = output_time.squeeze(dim=-1)
        hidden_flow = self.get_match_hidden(hidden, x_id, self.road_num + 3)
        output_flow = self.flow_ffn(hidden_flow)
        output_flow = output_flow.squeeze(dim=-1)
        return output_roads, output_time, output_flow
    
    def get_match_hidden(self, hidden, x_id, token_id):
        B, T, d = hidden.shape
        indices = torch.nonzero(x_id == token_id, as_tuple=True)
        extracted_vectors = hidden[indices[0], indices[1]]
        extracted_vectors = extracted_vectors.reshape(B, -1, d)
        return extracted_vectors

    def pretrain(self, x_id, x_mark_enc):
        x_enc = self.token_embedding(x_id)

        B, L, M = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        hidden = self.gpt2(inputs_embeds=enc_out).hidden_states[-1]
        hidden_cls = self.get_match_hidden(hidden, x_id, self.road_num + 1)
        output_roads = self.classify_layer(hidden_cls)
        hidden_tem = self.get_match_hidden(hidden, x_id, self.road_num + 2)
        output_time = self.time_ffn(hidden_tem)
        output_time = output_time.squeeze(dim=-1)
        hidden_flow = self.get_match_hidden(hidden, x_id, self.road_num + 3)
        output_flow = self.flow_ffn(hidden_flow)
        output_flow = output_flow.squeeze(dim=-1)
        return output_roads, output_time, output_flow

    def token_embedding(self, x_id):
        road_embedding = self.st_tokenizer()
        B, L = x_id.shape
        x_enc = road_embedding[x_id].to(x_id.device)
        return x_enc


    
