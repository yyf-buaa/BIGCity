import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

class ST_Tokenizer(nn.Module):
    def __init__(self, city):
        self.city = city
        super(ST_Tokenizer, self).__init__()
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
