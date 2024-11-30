from typing import Optional
import numpy as np
import pandas as pd
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import GPT2ForSequenceClassification
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model,GPT2LMHeadModel
from transformers import GPT2Config
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from layers.Embed import DataEmbedding, DataEmbedding_wo_time,TrajDataEmbedding
from layers.global_attn import GlobalAttnLayer
import ipdb
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftModelForCausalLM
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


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=heads, concat=False)

    def forward(self, x, edge_index, edge_weights):
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_weights))
        x = self.conv2(x, edge_index, edge_attr=edge_weights)
        return x


class ST_Tokenizer(nn.Module):
    def __init__(self, city):
        self.city = city
        super(ST_Tokenizer, self).__init__()
        # self.static_embedding = torch.from_numpy(np.load('/home/wangwenrui/dataset/{}/road_embedding/road_embedding_HHGCLV3_{}_128.npy'.format(city,city))).float() #N*d
        # static embedding
        self._load_geo()
        self._load_rel()
        max_size = torch.max(self.static_embedding, dim=0).values
        print(self.static_embedding[:10, :])
        print("ssssssssssssssss", self.static_embedding.shape)
        self.static_embedding_layers = nn.ModuleList(
            [nn.Embedding(num_embeddings=size+1, embedding_dim=128) 
             for size in max_size]
        )
        print(self.static_embedding_layers)
        print("##############", self.static_embedding.shape, self.static_embedding.shape[1]*128)
        self.spatial_encoder = nn.Sequential(
            MLP(input_size=self.static_embedding.shape[1]*128, hidden_size=128, output_size=128),
            GAT(in_channels=128, out_channels=128, heads=2),
            MLP(input_size=128, hidden_size=128, output_size=128)
        )
        # dynamic embedding
        self.dynamic_embedding = torch.from_numpy(np.load('/home/wangwenrui/dataset/{}/road_dyna_embedding.npy'.format(city))).float() # N*T*d
        print("ddddddddddddddddd", self.dynamic_embedding.shape)
        N, T, d = self.dynamic_embedding.shape
        kernel_size = 1
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=kernel_size, padding=padding)
        self.global_attn = GlobalAttnLayer(2*d, d, 8)

    def forward(self):
        self.device = self.conv.weight.device
        self.static_embedding = self.static_embedding.to(self.device) # N * d
        self.edge_index = torch.from_numpy(self.edge_index).to(self.device)
        self.edge_weight = torch.from_numpy(self.edge_weight).to(self.device)
        self.static_embedding = torch.cat([self.static_embedding_layers[i](self.static_embedding[:, i]) for i in range(self.static_embedding.size(1))], dim=1)
        print("@@@@@", self.static_embedding.shape)
        self.static_embedding = self.spatial_encoder[0](self.static_embedding)
        self.static_embedding = self.spatial_encoder[1](self.static_embedding, self.edge_index, self.edge_weight)
        self.static_embedding = self.spatial_encoder[2](self.static_embedding)

        dynamic_embedding = self.dynamic_embedding.permute(0, 2, 1).to(self.device)  # N*d*T
        # import ipdb
        # ipdb.set_trace()
        # 显存太大，分批次进行
        # batch_size = 16
        # N = dynamic_embedding.size(0)
        # num_batches = (N + batch_size - 1) // batch_size  
        # conv_results = []
        # for i in range(num_batches):
        #     start_idx = i * batch_size
        #     end_idx = min((i + 1) * batch_size, N)
        #     batch_embedding = dynamic_embedding[start_idx:end_idx].to(self.device)
        #     batch_output = self.conv(batch_embedding)
        #     batch_embedding.to('cpu')
        #     conv_results.append(batch_output)
        # logging.info(f"length of conv_results: {len(conv_results)}")
        # logging.info(f"each one of conv_results: {conv_results[0].shape}")
        # dynamic_embedding = torch.cat(conv_results, dim=0)
        print(dynamic_embedding.shape)
        dynamic_embedding = self.conv(dynamic_embedding)
        print(dynamic_embedding.shape)
        self.static_embedding = self.static_embedding.unsqueeze(2).repeat(1, 1, dynamic_embedding.shape[-1])
        print("**********", self.static_embedding.shape, self.dynamic_embedding.shape)
        print("**********", self.static_embedding.permute(0, 2, 1).shape, self.dynamic_embedding.permute(0, 2, 1).shape)
        road_embedding = torch.cat((self.static_embedding.permute(0, 2, 1), dynamic_embedding.permute(0, 2, 1)), dim=-1)
        print("********", road_embedding.shape)
        # road_embedding = self.global_attn(road_embedding,road_embedding) # N * T * d
        N, T, d = road_embedding.shape
        special_token = torch.zeros(4, T, d).to(self.device)
        for i in range(4):
            special_token[i] = i
        road_embedding = torch.cat((road_embedding, special_token),dim=0)
        return road_embedding # N+4 * T * d
    
    def _load_geo(self):
        """ read the static features of roads """
        self.feature_file = '../dataset/{}/roadmap_{}/road_features_{}.csv'.format(self.city, self.city, self.city)
        feature_file = pd.read_csv(self.feature_file)
        self.static_embedding = torch.tensor(feature_file.to_numpy(), dtype=torch.long)  # N * d
        self.road_num = len(feature_file)

    def _load_rel(self):
        """ read the adjacent relation between roads """
        self.rel_file = '../dataset/{}/roadmap_{}/roadmap_{}.rel'.format(self.city, self.city, self.city)
        relfile = pd.read_csv(self.rel_file)
        weight_col = None
        for col in relfile.columns:
            if 'weight' in col:
                weight_col = col
        assert weight_col is not None

        relfile = relfile[['origin_id', 'destination_id', weight_col]]

        self.edge_index = []
        self.edge_weight = []
        for row in relfile.values:
            self.edge_index.append([row[0], row[1]])
            self.edge_weight.append(row[-1])
        
        self.edge_index = np.array(self.edge_index, dtype='int64').T
        self.edge_weight = np.array(self.edge_weight, dtype='float32')
        return relfile


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        configs.road_num = 5269
        configs.seq_len = 64
        self.is_ln = configs.ln
        self.road_num = configs.road_num
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.city = configs.city
        self.dropout = configs.dropout
        logging.info(f"seq_len={self.seq_len}\n")
        self.d_ff = configs.d_ff
        self.st_tokenizer = ST_Tokenizer(self.city, )
        self.enc_embedding =TrajDataEmbedding(c_in=128,d_model=configs.d_model,embed_type=configs.embed)
        custom_config = GPT2Config.from_pretrained('./models/gpt2')
        custom_config.num_hidden_layers = configs.gpt_layers
        self.gpt2 = GPT2LMHeadModel(config=custom_config)
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            param.requires_grad = False
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=LORA_R, 
            lora_alpha=LORA_ALPHA, 
            lora_dropout=LORA_DROPOUT, 
            target_modules=[
                            "c_attn",
                            # "C_proj",
                            "c_fc"
                            ]
        )
        self.gpt2 = get_peft_model(self.gpt2, lora_config)
        lora_params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        print(f"LoRA 参数总量: {lora_params}")
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
        
        if self.task_name == 'pretrain':
            self.classify_layer = nn.Linear(configs.d_model, configs.road_num)
            self.time_ffn = nn.Linear(configs.d_model, 1)
            self.flow_ffn = nn.Linear(configs.d_model, 1)
    
    

    def forward(self, x_id, x_time, x_mark_enc):
        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        #     dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        #     return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        # if self.task_name == 'imputation':
        #     dec_out = self.imputation(
        #         x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'anomaly_detection':
        #     dec_out = self.anomaly_detection(x_enc)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'classification':
        #     dec_out = self.classification(x_enc, x_mark_enc,x_len)
        #     return dec_out  # [B, N]
        # if self.task_name == "ETA":
        #     dec_out = self.eta(x_enc, x_mark_enc)
        #     return dec_out
        if self.task_name == 'pretrain':
            outputs_road, output_time, output_flow = self.pretrain(x_id, x_time, x_mark_enc)
            return outputs_road, output_time, output_flow
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

    def pretrain(self, x_id, x_time, x_mark_enc):
        # x_id, x_time: B * T
        x_enc = self.token_embedding(x_id, x_time)
        B, L, M = x_enc.shape
        logging.info(x_enc.shape)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        hidden =  self.gpt2(inputs_embeds=enc_out).hidden_states[-1]
        hidden_cls = self.get_match_hidden(hidden,x_id,self.road_num+1)
        output_roads = self.classify_layer(hidden_cls)  
        hidden_tem = self.get_match_hidden(hidden,x_id,self.road_num+2)     
        output_time = self.time_ffn(hidden_tem)
        output_time = output_time.squeeze(dim=-1)
        hidden_flow = self.get_match_hidden(hidden,x_id,self.road_num+3)  
        output_flow = self.flow_ffn(hidden_flow)
        output_flow = output_flow.squeeze(dim=-1)
        return output_roads, output_time, output_flow

    def token_embedding(self, x_id, x_time):
        road_embedding = self.st_tokenizer()  # N * T * d
        B, L = x_id.shape
        x_enc = road_embedding[x_id, x_time].to(x_id.device)
        return x_enc # B * L * d

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
        
