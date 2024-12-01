import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import pandas as pd

from config import data_filename

from tqdm import tqdm


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



class StTokenizer(nn.Module):
    def __init__(self):
        logging.info("Start initializing the ST tokenizer.")
        super().__init__()
        
        self.slide_window_size = 6
        self.d_vec = 128
        self.start_time = pd.to_datetime("2018-10-01T00:00:00Z")
        self.end_time = pd.to_datetime("2018-11-30T23:30:00Z")
        self.interval = 1800
        
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
        self.static_embedding = None
        self.dynamic_embedding_layers = None
        self.dynamic_embedding = None
        self.build_tokenizer()
        
        logging.info("Finish initializing the ST tokenizer.")

    def forward(self, x_id, x_time):          
        e = torch.cat([self.static_origin_embedding[i](self.static_features[:, i]) for i in range(self.static_features.size(1))], dim=1)
        e = self.static_embedding_layers[0](e)
        e = self.static_embedding_layers[1](e, self.edges, self.edge_weight)
        static_embedding = self.static_embedding_layers[2](e)
        
        dynamic_embedding = self.dynamic_features[x_id, x_time]
        print(self.dynamic_features[3767, 10])
        print(dynamic_embedding.shape)
        print(dynamic_embedding)
        
        
        # road_embedding[x_id, x_time]
        
        
        
    
    def load_relation(self):
        logging.info("Start reading adjacency file.")
        
        rel_data = pd.read_csv(data_filename.road_relation_file)
        
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
        for row in tqdm(dynamic_data.values, desc='Processing rows', total=dynamic_data.shape[0]):
            time_id = row[0] % self.time_slots_cnt
            road_id = row[3]
            self.dynamic_features[road_id, time_id] = row[4]
            
        N, T, S = self.road_cnt, self.time_slots_cnt, self.slide_window_size      

        padded_dynamic_features = torch.cat((torch.zeros(N, T), self.dynamic_features), dim=1)
        self.dynamic_features= torch.stack([padded_dynamic_features[:, j - S:j] for j in range(S, T + S)], dim=1)
        
        print(self.dynamic_features)
        
        logging.info("Finish reading dynamic features file. \n"
                    f"Shape of dynamic features: {self.dynamic_features.shape} \n")
        
    def build_tokenizer(self):
        logging.info("Start building static ST tokenizer.")
        
        D = self.d_vec
        
        max_size = torch.max(self.static_features, dim=0).values
        self.static_origin_embedding = nn.ModuleList([nn.Embedding(num_embeddings=size+1, embedding_dim=D) for size in max_size])
        self.static_embedding_layers = nn.Sequential(
            MLP(input_size=self.static_features.shape[1]*D, hidden_size=D, output_size=D),
            GAT(in_channels=D, out_channels=D, heads=2),
            MLP(input_size=D, hidden_size=D, output_size=D)
        )

        logging.info("Finish building static ST tokenizer. \n"
                    f"static_origin_embedding: \n{self.static_origin_embedding} \n"
                    f"static_embedding_layers: \n{self.static_embedding_layers} \n")
        
        
        logging.info("Start building dynamic ST tokenizer.")
        
        S = self.slide_window_size   
        
        self.dynamic_embedding_layers = nn.Sequential (
            MLP(input_size=S, hidden_size=D, output_size=D),
            GAT(in_channels=D, out_channels=D, heads=2),
            # TODO CONV1D
        )
        
        logging.info("Finish building dynamic ST tokenizer. \n"
                      f"dynamic_embedding_layers: \n{self.dynamic_embedding_layers} \n")
        