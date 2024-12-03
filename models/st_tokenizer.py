import torch
import logging
import torch.nn as nn
import pandas as pd

from config import data_filename
from config.args_config import args

from .layers import MLP, GAT, CrossAttention

from tqdm import tqdm


class StTokenizer(nn.Module):
    def __init__(self):
        logging.info("Start initializing the ST tokenizer.")
        super(StTokenizer, self).__init__()
        
        self.slide_window_size = 6
        self.d_vec = 128
        self.d_time_features = 6
        self.start_time = pd.to_datetime("2018-10-01T00:00:00Z")
        self.end_time = pd.to_datetime("2018-11-30T23:30:00Z")
        self.interval = 1800
        
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
        self.time_embedding_layers = None
        self.cross_attention = None
        self.final_mlp = None
        self.build_tokenizer()
        
        logging.info("Finish initializing the ST tokenizer.")

    def forward(self, road_id_batch, time_id_batch, time_feature_batch):
             
        B, N, M, L = road_id_batch.shape[0], self.road_cnt, self.edge_cnt, args.seq_len
        
        # embedding of the static original discrete data
        se = torch.cat([self.static_origin_embedding[i](self.static_features[:, i]) for i in range(self.static_features.size(1))], dim=1)
        # static layer 0: MLP
        se = self.static_embedding_layers[0](se)
        
        # static layer 1: GAT
        se = self.static_embedding_layers[1](se, self.edges, self.edge_weight)
        
        # static layer 2: MLPS
        static_embedding = self.static_embedding_layers[2](se)
        
        # dynamic layer 0: MLP
        de = self.dynamic_features[:, time_id_batch[:, 0]]
        de = self.dynamic_embedding_layers[0](de)

        # dynamic layer 1: GAT(batch)
        edges = self.edges.repeat(1, B) + (torch.arange(B) * N).repeat_interleave(M) 
        edge_weight = self.edge_weight.repeat(B)
        de = de.permute(1, 0, 2).reshape(-1, de.shape[-1])
        de = self.dynamic_embedding_layers[1](de, edges, edge_weight)
        de = de.view(B, N, -1).permute(1, 0, 2)
        
        # dynamic layer 2: MLP
        dynamic_embedding = self.dynamic_embedding_layers[2](de)
        
        # get tokens from embedding matrix
        static_embedding_result = static_embedding[road_id_batch]
        dynamic_embedding_result = dynamic_embedding[road_id_batch, torch.arange(B).unsqueeze(1).expand(B, L)]
        
        # concat stactic/dynamic tokens
        road_embedding_result = torch.cat((static_embedding_result, dynamic_embedding_result), dim=-1)
        
        # cross attention
        road_embedding_result = self.cross_attention(road_embedding_result)
        
        # time layer 1: Linear
        time_embedding_result = self.time_embedding_layers[0](time_feature_batch)

        # concat time features
        embedding_result = torch.cat((road_embedding_result, time_embedding_result), dim=-1)
        
        # final MLP: token dimension is converted to d_model, ready for gpt2
        embedding_result = self.final_mlp(embedding_result)
        
        return embedding_result # (B, L, d_model)
    
    def load_relation(self):
        logging.info("Start reading adjacency file.")
        
        rel_data = pd.read_csv(data_filename.road_relation_file)
        
        self.edge_cnt = len(rel_data)
        
        self.edges = torch.tensor(rel_data[['origin_id', 'destination_id']].to_numpy(dtype='int64'), dtype=torch.int64).T
        self.edge_weight = torch.tensor(rel_data['geographical_weight'].to_numpy(dtype='float32'), dtype=torch.float32)
        
        logging.info("Finish reading adjacency file. \n"
                    f"The number of edges in the graph: {len(rel_data)}. \n")
    
    def load_static_features(self):
        logging.info("Start reading static features file.")
        
        static_data = pd.read_csv(data_filename.road_static_file)
        
        self.road_cnt = len(static_data)
        
        self.static_features = torch.tensor(static_data.to_numpy(), dtype=torch.long)
        
        logging.info("Finish reading static features file. \n"
                    f"The number of vertices in the graph: {self.road_cnt}, \n"
                    f"Shape of static features: {self.static_features.shape} \n")
        
    def load_dynamic_features(self):
        logging.info("Start reading dynamic features file.")
        
        dynamic_data = pd.read_csv(data_filename.road_dynamic_file)
        
        self.time_slots_cnt = int((self.end_time - self.start_time).total_seconds() // self.interval) + 1
        
        self.dynamic_features = torch.zeros((self.road_cnt, self.time_slots_cnt), dtype=torch.float32)  
        for row in tqdm(dynamic_data.values[0: 2928*10], desc='Processing rows', total=dynamic_data.shape[0]):
            time_id = row[0] % self.time_slots_cnt
            road_id = row[3]
            self.dynamic_features[road_id, time_id] = row[4]
            
        N, T, S = self.road_cnt, self.time_slots_cnt, self.slide_window_size      

        padded_dynamic_features = torch.cat((torch.zeros(N, S), self.dynamic_features), dim=1)
        self.dynamic_features= torch.stack([padded_dynamic_features[:, j - S:j] for j in range(S, T + S)], dim=1)
        
        logging.info("Finish reading dynamic features file. \n"
                    f"Shape of dynamic features: {self.dynamic_features.shape} \n")
        
    def build_tokenizer(self):
        logging.info("Start building static ST tokenizer.")
        
        Demb, Dtf, Dmodel = self.d_vec, self.d_time_features, args.d_model
        
        max_size = torch.max(self.static_features, dim=0).values
        self.static_origin_embedding = nn.ModuleList([nn.Embedding(num_embeddings=size+1, embedding_dim=Demb) for size in max_size])
        self.static_embedding_layers = nn.Sequential(
            MLP(input_size=self.static_features.shape[1]*Demb, hidden_size=Demb, output_size=Demb),
            GAT(in_channels=Demb, out_channels=Demb, heads=2),
            MLP(input_size=Demb, hidden_size=Demb, output_size=Demb)
        )

        logging.info("Finish building static ST tokenizer. \n"
                    f"static_origin_embedding: \n{self.static_origin_embedding} \n"
                    f"static_embedding_layers: \n{self.static_embedding_layers} \n")
        
        
        logging.info("Start building dynamic ST tokenizer.")
        
        S = self.slide_window_size   
        
        self.dynamic_embedding_layers = nn.Sequential(
            MLP(input_size=S, hidden_size=Demb, output_size=Demb),
            GAT(in_channels=Demb, out_channels=Demb, heads=2),
            MLP(input_size=Demb, hidden_size=Demb, output_size=Demb),
        )
        
        logging.info("Finish building dynamic ST tokenizer. \n"
                      f"dynamic_embedding_layers: \n{self.dynamic_embedding_layers} \n")
        
        self.cross_attention = CrossAttention(2 * Demb)
        
        self.time_embedding_layers = nn.Sequential(
            nn.Linear(Dtf, Demb)
        )
        
        self.final_mlp = MLP(3 * Demb, 3 * Demb, Dmodel)
