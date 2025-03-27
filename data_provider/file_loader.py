import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import logging
import os

from config import global_vars
from config import logging_config
from config.args_config import args


class FileLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileLoader, cls).__new__(cls)
            cls._instance.edges = None
            cls._instance.edge_weight = None
            cls._instance.edge_cnt = None

            cls._instance.static_features = None
            cls._instance.dynamic_features = None
            cls._instance.road_cnt = None
            cls._instance.time_slots_cnt = None

            cls._instance.traj_data = None
            cls._instance.traj_cnt = None
            cls._instance.traj_category_cnt = None
        return cls._instance

    def load_road_relation_file(self):
        relation_cache = global_vars.road_relation_tensor_file
        if os.path.exists(relation_cache):
            logging.info(f"Loading cached edges and weights from {relation_cache}")
            cache = torch.load(relation_cache, weights_only=True)
            self.edges, self.edge_weight, self.edge_cnt = cache['edges'], cache['edge_weight'], cache['edge_cnt']
        else:
            logging.info("Reading adjacency file from csv.")
            rel_data = pd.read_csv(global_vars.road_relation_file)

            self.edge_cnt = len(rel_data)
            self.edges = torch.tensor(rel_data[['origin_id', 'destination_id']].to_numpy(dtype='int64'), dtype=torch.int64).T
            self.edge_weight = torch.tensor(rel_data['geographical_weight'].to_numpy(dtype='float32'), dtype=torch.float32)

            torch.save({"edges": self.edges, "edge_weight": self.edge_weight, "edge_cnt": self.edge_cnt}, relation_cache)
            logging.info(f"Saved adjacency tensor cache to {relation_cache}")
            
        logging.info(f"Edges loaded. Shape: {self.edges.shape}, Edge count: {self.edge_cnt}")

    def load_road_features_file(self):
        static_cache = global_vars.road_static_tensor_file
        if os.path.exists(static_cache):
            logging.info(f"Loading cached static features from {static_cache}")
            cache = torch.load(static_cache, weights_only=True)
            self.static_features, self.road_cnt = cache['static_features'], cache['road_cnt']
        else:
            logging.info("Reading road static features file from csv.")
            static_data = pd.read_csv(global_vars.road_static_file)
            
            self.road_cnt = len(static_data)
            self.static_features = torch.tensor(static_data.to_numpy(), dtype=torch.int64)
            
            torch.save({"static_features": self.static_features, "road_cnt": self.road_cnt}, static_cache)
            logging.info(f"Saved static features tensor cache to {static_cache}")
        logging.info(f"Static features loaded. Shape: {self.static_features.shape}")

        dynamic_cache = global_vars.road_dynamic_tensor_file
        if os.path.exists(dynamic_cache):
            logging.info(f"Loading cached dynamic features from cache {dynamic_cache}")
            cache = torch.load(dynamic_cache, weights_only=True)
            self.dynamic_features, self.time_slots_cnt = cache["dynamic_features"], cache["time_slots_cnt"]
        else:
            logging.info("Reading road dynamic features file from csv.")
            dynamic_data = pd.read_csv(global_vars.road_dynamic_file)[['dyna_id', 'entity_id', 'traffic_speed']]
            dynamic_data_tensor = torch.tensor(dynamic_data.values, dtype=torch.float32)
            
            self.time_slots_cnt = int((global_vars.end_time - global_vars.start_time).total_seconds() // global_vars.interval) + 1
            
            self.dynamic_features = torch.zeros((self.road_cnt, self.time_slots_cnt), dtype=torch.float32)
            
            # Data format required!!!
            time_ids = (dynamic_data_tensor[:, 0] % self.time_slots_cnt).long()
            road_ids = dynamic_data_tensor[:, 1].long()
            values = dynamic_data_tensor[:, 2]
            self.dynamic_features[road_ids, time_ids] = values
            # for row in tqdm(dynamic_data.values, desc='Processing dynamic rows', total=dynamic_data.shape[0]):
            #     time_id = row[0] % self.time_slots_cnt
            #     road_id = row[1]
            #     self.dynamic_features[road_id, time_id] = row[2]

            missing_ids = set(range(self.road_cnt)) - set(dynamic_data['entity_id'])
            for id in missing_ids:
                prev_line, next_line = id - 1, max(self.road_cnt - 1, id + 1)
                self.dynamic_features[id] = (self.dynamic_features[prev_line] + self.dynamic_features[next_line]) / 2

            row_sum = torch.sum(self.dynamic_features, dim=1)
            non_zero_count = torch.sum(self.dynamic_features != 0, dim=1)
            row_mean = row_sum / non_zero_count
            row_mean[non_zero_count == 0] = 0
            mask = (self.dynamic_features == 0)
            self.dynamic_features[mask] = row_mean.unsqueeze(1).expand_as(self.dynamic_features)[mask]
            
            min_value = torch.min(self.dynamic_features)
            max_value = torch.max(self.dynamic_features)
            self.dynamic_features = (self.dynamic_features - min_value) / (max_value - min_value)
            
            torch.save({"dynamic_features": self.dynamic_features, "time_slots_cnt": self.time_slots_cnt}, dynamic_cache)
            logging.info(f"Saved dynamic features tensor cache to {dynamic_cache}")
        logging.info(f"Dynamic features loaded. Shape: {self.dynamic_features.shape}")

    def load_traj_dataset_file(self):
        logging.info("Start reading trajectory data file.")
        traj_data = pd.read_csv(global_vars.cur_traj_file, delimiter=';')
        traj_data_full = pd.read_csv(global_vars.traj_file, delimiter=';')

        traj_data = traj_data.sample(frac=args.sample_rate)
        traj_data.reset_index(drop=True, inplace=True)

        self.traj_data = traj_data
        self.traj_cnt = len(traj_data)
        self.traj_category_cnt = len(set(traj_data_full["usr_id"]))

        logging.info(f"Trajectory data loaded. Count: {self.traj_cnt}, Categories: {self.traj_category_cnt}")

    def load_all(self):
        if self.edges is None or self.edge_weight is None:
            self.load_road_relation_file()
        if self.static_features is None or self.dynamic_features is None:
            self.load_road_features_file()
        if self.traj_data is None:
            self.load_traj_dataset_file()

    # Getter methods with checks:
    def get_edges(self):
        if self.edges is None:
            raise RuntimeError("edges not loaded. Please call load_all() first.")
        return self.edges

    def get_edge_weight(self):
        if self.edge_weight is None:
            raise RuntimeError("edge_weight not loaded.")
        return self.edge_weight

    def get_static_features(self):
        if self.static_features is None:
            raise RuntimeError("static_features not loaded.")
        return self.static_features

    def get_dynamic_features(self):
        if self.dynamic_features is None:
            raise RuntimeError("dynamic_features not loaded.")
        return self.dynamic_features

    def get_traj_data(self):
        if self.traj_data is None:
            raise RuntimeError("traj_data not loaded.")
        return self.traj_data
    
    def get_road_cnt(self):
        return self.road_cnt
    
    def get_time_slots_cnt(self):
        return self.time_slots_cnt
    
    def get_traj_cnt(self):
        return self.traj_cnt
    
    def get_traj_category_cnt(self):
        return self.traj_category_cnt

    def get_meta_info(self):
        return {
            "road_cnt": self.road_cnt,
            "time_slots_cnt": self.time_slots_cnt,
            "traj_cnt": self.traj_cnt,
            "traj_category_cnt": self.traj_category_cnt
        }

file_loader = FileLoader()
