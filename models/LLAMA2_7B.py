from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch import optim
import pandas as pd
from transformers import GPT2ForSequenceClassification
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BertTokenizer, BertModel
from einops import rearrange
from layers.Embed import DataEmbedding, DataEmbedding_wo_time,TrajDataEmbedding
from layers.global_attn import GlobalAttnLayer
import ipdb

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
# CLS_TOKEN = torch.ones(1,128)
# TEM_TOKEN = torch.ones(1,128)
# TEM_TOKEN *= 2
# FLOW_TOKEN = torch.ones(1,128)
# FLOW_TOKEN *= 3
# MASK_TOKEN = torch.zeros(1,128)
class ST_Tokenizer(nn.Module):
    #读取两个embedding矩阵，然后一个一维卷积+global attn，输出的是一个(N+4)*d的矩阵
     def __init__(self,city):
        super(ST_Tokenizer, self).__init__()
        self.static_embedding = torch.from_numpy(np.load('../dataset/{}/road_embedding/road_embedding_HHGCLV3_{}_128.npy'.format(city,city))).float()#N*d
        self.dynamic_embedding = torch.from_numpy(np.load('../dataset/{}/road_dyna_embedding.npy'.format(city))).float() #N*T*d
        N,T,d,d1 = self.dynamic_embedding.shape
        self.conv = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=T)
        self.global_attn = GlobalAttnLayer(2*d,d,8)
     def forward(self):
        self.device = self.conv.weight.device
        self.static_embedding = self.static_embedding.to(self.device)
        self.dynamic_embedding = self.dynamic_embedding.to(self.device)
        dynamic_embedding = self.dynamic_embedding.permute(0, 2, 1).to(self.device)
        # import ipdb
        # ipdb.set_trace()
        dynamic_embedding = self.conv(dynamic_embedding)
        dynamic_embedding = dynamic_embedding.squeeze(-1)
        road_embedding = torch.cat((self.static_embedding,dynamic_embedding),dim=-1)
        road_embedding = self.global_attn(road_embedding,road_embedding)
        rows = torch.arange(4).unsqueeze(1)  
        special_token = rows.repeat(1, 128)  
        special_token = special_token.to(self.device)
        road_embedding = torch.cat((road_embedding,special_token),dim=0)
        return road_embedding

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
        self.dropout = configs.dropout
        logging.info(f"seq_len={self.seq_len}\n")
        self.d_ff = configs.d_ff
        self.st_tokenizer = ST_Tokenizer(self.city)
        self.enc_embedding =TrajDataEmbedding(c_in=128,d_model=configs.d_model,embed_type=configs.embed)
        model_dir = '/root/dataDisk/Bigscity_flow/Llama-2-7b-hf/'
        self.llama = LlamaForCausalLM.from_pretrained(model_dir, output_hidden_states=True)
        logging.info(self.llama)
        
        for i, (name, param) in enumerate(self.llama.named_parameters()):
            param.requires_grad = False
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=LORA_R, 
            lora_alpha=LORA_ALPHA, 
            lora_dropout=LORA_DROPOUT, 
            target_modules=[
                            #"c_attn",
                            #"C_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj"
                            ]
        )
        self.llama = get_peft_model(self.llama, lora_config)
        if self.task_name == 'imputation':
            self.ln_proj = nn.LayerNorm(configs.d_model)
            # self.out_layer = nn.Linear(
            #     configs.d_model, 
            #     configs.c_out, 
            #     bias=True)
            self.out_layer = nn.Linear(configs.d_model, configs.road_num)
        if self.task_name == "ETA":
            # self.act = F.gelu
            # self.ln_proj = nn.LayerNorm(configs.d_model)
            self.out_layer = nn.Linear(configs.d_model, 1)
        if self.task_name == 'anomaly_detection':
            self.ln_proj = nn.LayerNorm(configs.d_ff)
            self.out_layer = nn.Linear(
                configs.d_ff, 
                configs.c_out, 
                bias=True)
        if self.task_name == 'classification':
            # self.act = F.gelu
            # self.dropout = nn.Dropout(0.1)
            #self.ln_proj = nn.LayerNorm(configs.d_model)
            self.out_layer = nn.Linear(configs.d_model, 2)
            #self.softmax = nn.LogSoftmax(dim=-1)
        
        if self.task_name == 'Pretrain':
            self.classify_layer = nn.Linear(configs.d_model, configs.road_num)
            self.time_ffn = nn.Linear(configs.d_model, 1)
            self.flow_ffn = nn.Linear(configs.d_model, 1)
    
    

    def forward(self, x_id, x_mark_enc):
        if self.task_name == 'Pretrain':
            outputs_road, output_time,output_flow = self.pretrain(x_id,x_mark_enc)
            return outputs_road, output_time,output_flow
        return None

    def eta(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape
        enc_out=self.enc_embedding(x_enc, x_mark_enc) # [B,T,C]
        dec_out = self.llama(inputs_embeds=enc_out).last_hidden_state
        # dec_out = self.ln_proj(dec_out)
        outputs1 = dec_out[:,1:,:]
        outputs = self.out_layer(outputs1)#[B,T-1,1]
        outputs = outputs.squeeze(dim=-1)
        return outputs

    def get_match_hidden(self,hidden,x_id,token_id):
        B,T,d = hidden.shape
        indices = torch.nonzero(x_id == token_id, as_tuple=True)
        extracted_vectors = hidden[indices[0], indices[1]]
        extracted_vectors = extracted_vectors.reshape(B,-1,d)
        return extracted_vectors
        
    def pretrain(self, x_id,x_mark_enc):
        x_enc = self.token_embedding(x_id)
        
        B, L, M = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        hidden =  self.llama(inputs_embeds=enc_out).hidden_states[-1]
        hidden_cls = self.get_match_hidden(hidden,x_id,self.road_num+1)
        output_roads = self.classify_layer(hidden_cls)  
        hidden_tem = self.get_match_hidden(hidden,x_id,self.road_num+2)     
        output_time = self.time_ffn(hidden_tem)
        output_time = output_time.squeeze(dim=-1)
        hidden_flow = self.get_match_hidden(hidden,x_id,self.road_num+3)  
        output_flow = self.flow_ffn(hidden_flow)
        output_flow = output_flow.squeeze(dim=-1)
        return output_roads,output_time,output_flow

    def token_embedding(self,x_id):
        road_embedding = self.st_tokenizer()
        B,L = x_id.shape
        x_enc = road_embedding[x_id].to(x_id.device)
        return x_enc




    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        
        dec_out=self.ln_proj(dec_out)
        # # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, L, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(
        #               1, L, 1))
        dec_out = self.out_layer(dec_out)
        #logging.info("dec_out shape is="+str(dec_out.shape))
        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        # enc_out = rearrange(enc_out, 'b l m -> b m l')
        # enc_out = self.padding_patch_layer(enc_out)
        # enc_out = enc_out.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # enc_out = self.predict_linear(enc_out)
        # enc_out = rearrange(enc_out, 'b m n p -> b n (m p)')

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        # dec_out = dec_out.reshape(B, -1)
        
        # dec_out = self.ln(dec_out)
        dec_out = self.out_layer(dec_out)
        # logging.info(dec_out.shape)
        # dec_out = dec_out.reshape(B, self.pred_len + self.seq_len, -1)
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        
        return dec_out

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape
        
        # Normalization from Non-stationary Transformer

        seg_num = 25
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        enc_out = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))
        
        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        
        outputs = outputs[:, :, :self.d_ff]
        # outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer

        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * \
                  (stdev[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = dec_out + \
                  (means[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')

        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc,x_len):
        # logging.info(x_enc.shape)
        B, L, M = x_enc.shape
        # import pdb
        # pdb.set_trace()
        enc_out=self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        #dec_out = self.ln_proj(dec_out)
        #outputs = dec_out[:,-1,:]
        d = dec_out.shape[-1]
        outputs = torch.gather(dec_out,1,x_len.view(B,1,1).expand(B,1,d)-1)
        outputs = outputs.squeeze(dim = -2)
        # import pdb
        # pdb.set_trace()

        outputs = self.out_layer(outputs)  
        #outputs = self.softmax(outputs)
        #logging.info(outputs.shape)
        # import pdb
        # pdb.set_trace()
        return outputs
        
    
