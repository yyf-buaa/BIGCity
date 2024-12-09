import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import logging
import os

from config import global_vars
from config import logging_config


def load_road_relation_file():
    logging.info("Start reading adjacency file.")
    
    rel_data = pd.read_csv(global_vars.road_relation_file)    
    
    edge_cnt = len(rel_data)
    
    edges = torch.tensor(rel_data[['origin_id', 'destination_id']].to_numpy(dtype='int64'), dtype=torch.int64).T
    edge_weight = torch.tensor(rel_data['geographical_weight'].to_numpy(dtype='float32'), dtype=torch.float32)  
    
    logging.info("Finish reading adjacency file. \n"
                    f"The number of edges in the graph: {len(rel_data)}. \n")
    return edges, edge_weight, edge_cnt

def load_road_features_file():
    logging.info("Start reading static features file.")
    
    static_data = pd.read_csv(global_vars.road_static_file)    
    
    road_cnt = len(static_data)
    
    static_features = torch.tensor(static_data.to_numpy(), dtype=torch.int64)
    
    logging.info("Finish reading static features file. \n"
                    f"The number of vertices in the graph: {road_cnt}, \n"
                    f"Shape of static features: {static_features.shape} \n")
    
    logging.info("Start reading dynamic features file.")
    
    time_slots_cnt = int((global_vars.end_time - global_vars.start_time).total_seconds() // global_vars.interval) + 1
    
    if os.path.exists(global_vars.road_dynamic_tensor_file):
        logging.info("Load existing tensor file.")
        dynamic_features = torch.load(global_vars.road_dynamic_tensor_file, weights_only=True)
        
        # all_zero_rows = torch.all(dynamic_features == 0, dim=1)
        # zero_row_indices = torch.nonzero(all_zero_rows, as_tuple=False).squeeze()
        # print(zero_row_indices, zero_row_indices.shape)
    else:
        logging.info("Tensor file not found, start loading raw data")
        dynamic_data = pd.read_csv(global_vars.road_dynamic_file)
        
        dynamic_features = torch.zeros((road_cnt, time_slots_cnt), dtype=torch.float32)  
        for row in tqdm(dynamic_data.values, desc='Processing rows', total=dynamic_data.shape[0]):
            time_id = row[0] % time_slots_cnt
            road_id = row[3]
            dynamic_features[road_id, time_id] = row[4]
            
        missing_ids = set(range(road_cnt)) - set(dynamic_data['entity_id'])
        for id in missing_ids:
            prev_line, next_line = id - 1, max(road_cnt - 1, id + 1)
            dynamic_features[id] = (dynamic_features[prev_line] + dynamic_features[next_line]) / 2
        
        row_sum = torch.sum(dynamic_features, dim=1)
        non_zero_count = torch.sum(dynamic_features != 0, dim=1)
        row_mean = row_sum / non_zero_count
        row_mean[non_zero_count == 0] = 0
        mask = (dynamic_features == 0)
        dynamic_features[mask] = row_mean.unsqueeze(1).expand_as(dynamic_features)[mask]
 
        torch.save(dynamic_features, global_vars.road_dynamic_tensor_file)
        logging.info(f"Save the file to {global_vars.road_dynamic_tensor_file}")
            
    logging.info("Finish reading dynamic features file. \n"
                    f"Shape of dynamic features: {dynamic_features.shape} \n")
        
    return static_features, dynamic_features, road_cnt, time_slots_cnt


edges, edge_weight, edge_cnt = load_road_relation_file()
static_features, dynamic_features, road_cnt, time_slots_cnt = load_road_features_file()

    
        
        