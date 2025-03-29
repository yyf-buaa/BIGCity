import torch
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import global_vars
from config.args_config import args
from .layers import MLP, GAT, CrossAttention
from data_provider.file_loader import file_loader

class StTokenizer(nn.Module):
    def __init__(self, device):
        logging.info("Start initializing the ST tokenizer.")
        super(StTokenizer, self).__init__()
        
        self.device = device
        
        self.slide_window_size = 6
        self.d_vec = 128
        self.d_time_features = 6
        self.start_time = global_vars.start_time
        self.end_time = global_vars.end_time
        self.interval = global_vars.interval
        
        self.edge_cnt = None
        self.edges = None
        self.edge_weight = None
        self.load_relation()
        
        self.road_cnt = None
        self.static_features = None
        self.load_static_features()
        
        self.time_slots_cnt = None
        self.dynamic_features = None
        self.load_dynamic_features()
        
        self.static_origin_embedding = None
        self.static_embedding_layers = None
        self.dynamic_embedding_layers = None
        self.dynamic_embedding = None
        self.time_embedding_layers = None
        self.cross_attention = None
        self.final_mlp = None
        self.build_tokenizer()
        
        logging.info("Finish initializing the ST tokenizer.")

    def forward(self, batch_road_id, batch_time_id, batch_time_features):
        B, N, M, L = batch_road_id.shape[0], self.road_cnt, self.edge_cnt, args.seq_len
        
        # embedding of the static original discrete data
        se = torch.cat([self.static_origin_embedding[i](self.static_features[:, i]) for i in range(self.static_features.size(1))], dim=1)
        # static layer 0: MLP
        se = self.static_embedding_layers[0](se)
        
        # static layer 1: GAT
        se = self.static_embedding_layers[1](se, self.edges, self.edge_weight)
        
        # static layer 2: MLP
        static_embedding = self.static_embedding_layers[2](se)
        
        # get static tokens
        static_embedding_result = static_embedding[batch_road_id]
        
        if not args.pre_dyna:
            # dynamic layer 0: MLP
            de = self.dynamic_features[:, batch_time_id[:, 0]]
            de = self.dynamic_embedding_layers[0](de)

            # dynamic layer 1: GAT(batch)
            edges = self.edges.repeat(1, B) + (torch.arange(B) * N).repeat_interleave(M).to(self.device)
            edge_weight = self.edge_weight.repeat(B)
            de = de.permute(1, 0, 2).reshape(-1, de.shape[-1])
            de = self.dynamic_embedding_layers[1](de, edges, edge_weight)
            de = de.view(B, N + 1, -1).permute(1, 0, 2)
            
            # dynamic layer 2: MLP
            dynamic_embedding = self.dynamic_embedding_layers[2](de)
            
            # get dynamic tokens
            dynamic_embedding_result = dynamic_embedding[batch_road_id, torch.arange(B).unsqueeze(1).expand(B, L)]
        else:
            # get dynamic tokens from pre-trained embedding
            dynamic_embedding_result = self.dynamic_embedding[batch_road_id, torch.arange(B).unsqueeze(1).expand(B, L)]
            
        # concat stactic/dynamic tokens
        road_embedding_result = torch.cat((static_embedding_result, dynamic_embedding_result), dim=-1)
        
        # cross attention
        road_embedding_result = self.cross_attention(road_embedding_result)
        
        # time layer 1: Linear
        time_embedding_result = self.time_embedding_layers[0](batch_time_features)

        # concat time features
        embedding_result = torch.cat((road_embedding_result, time_embedding_result), dim=-1)
        
        # final MLP: token dimension is converted to d_model, ready for gpt2
        embedding_result = self.final_mlp(embedding_result)
        
        return embedding_result # (B, L, d_model)
    
    def load_relation(self):
        self.edge_cnt = file_loader.get_edge_cnt()
        self.edges = file_loader.get_edges().to(self.device)
        self.edge_weight = file_loader.get_edge_weight().to(self.device)
        
    def load_static_features(self):
        self.road_cnt = file_loader.get_road_cnt()
        static_features = file_loader.get_static_features().to(self.device)
        
        # fill in 1 empty road segment id
        row_zeros = torch.zeros(1, static_features.shape[1], dtype=static_features.dtype).to(self.device)
        self.static_features = torch.cat((static_features, row_zeros), dim=0)
        
    def load_dynamic_features(self):
        self.time_slots_cnt = file_loader.get_time_slots_cnt()
        dynamic_features = file_loader.get_dynamic_features().to(self.device)
            
        N, T, S = self.road_cnt, self.time_slots_cnt, self.slide_window_size

        padded_dynamic_features = torch.cat((torch.zeros(N, S).to(self.device), dynamic_features), dim=1)
        dynamic_features= torch.stack([padded_dynamic_features[:, j - S:j] for j in range(S, T + S)], dim=1)
        
        # fill in 1 empty road segment id and 1 empty time slot id
        row_zeros = torch.zeros(1, dynamic_features.shape[1], S, dtype=dynamic_features.dtype).to(self.device)
        dynamic_features = torch.cat((dynamic_features, row_zeros), dim=0)
        col_zeros = torch.zeros(dynamic_features.shape[0], 1, S, dtype=dynamic_features.dtype).to(self.device)
        self.dynamic_features = torch.cat((dynamic_features, col_zeros), dim=1)
        
    def build_tokenizer(self):
        logging.info("Start building static ST tokenizer.")
        
        Demb, Dtf, Dmodel = self.d_vec, self.d_time_features, args.d_model
        
        max_size = torch.max(self.static_features, dim=0).values
        self.static_origin_embedding = nn.ModuleList([
            nn.Embedding(num_embeddings=size+1, embedding_dim=Demb) for size in max_size
        ])
        self.static_embedding_layers = nn.Sequential(
            MLP(self.static_features.shape[1]*Demb, Demb, Demb),
            GAT(in_channels=Demb, out_channels=Demb, heads=2),
            MLP(Demb, Demb, Demb)
        )

        logging.info("Finish building static ST tokenizer. \n"
                    f"static_origin_embedding: \n{self.static_origin_embedding} \n"
                    f"static_embedding_layers: \n{self.static_embedding_layers} \n")
        
        
        logging.info("Start building dynamic ST tokenizer.")
        
        S = self.slide_window_size   
        
        self.dynamic_embedding_layers = nn.Sequential(
            MLP(S, Demb, Demb),
            GAT(in_channels=Demb, out_channels=Demb, heads=2),
            MLP(Demb, Demb, Demb),
        )
        if args.pre_dyna:
            self.dynamic_embedding = torch.from_numpy(np.load(global_vars.road_dynamic_embedding_file))
            self.dynamic_embedding.requires_grad = False
            logging.info(f"Pre-trained dyna embeddings are used. Shape: {self.dynamic_embedding.shape}")
        
        logging.info("Finish building dynamic ST tokenizer. \n"
                      f"dynamic_embedding_layers: \n{self.dynamic_embedding_layers} \n")
        
        self.cross_attention = CrossAttention(2 * Demb)
        
        self.time_embedding_layers = nn.Sequential(
            nn.Linear(Dtf, Demb)
        )
        
        self.final_mlp = MLP(3 * Demb, 3 * Demb, Dmodel)
