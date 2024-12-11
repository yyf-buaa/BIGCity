from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import logging
from ast import literal_eval

from config import global_vars
from config.args_config import args
from . import file_loader
from utils.timefeatures import time_features

class DatasetTraj(Dataset):
    def __init__(self):
        logging.info("Start loading trajectory dataset.")
        
        super().__init__()
        
        self.dataset_len = None
        self.read_traj_data()
        
        logging.info("Finish loading trajectory dataset.")
    
    def read_traj_data(self):
        logging.info("Start reading trajectory data file.")
        
        traj_data = pd.read_csv(global_vars.traj_file_short, delimiter=';')
        traj_data = traj_data.sample(frac = args.sample_rate)
        traj_data.reset_index(drop=True, inplace=True)
        
        self.dataset_len = len(traj_data)
        
        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("Finish reading road id.")
        
        self.traj_time_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(global_vars.start_time.timestamp())) // global_vars.interval).to(torch.int32)
        logging.info("Finish reading time id.")
        
        # self.traj_time_offset = self.traj_time_lists - self.traj_time_lists[:, 0:1]

        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("Finish reading time features.")
        
        self.traj_road_flow = file_loader.dynamic_features[self.traj_road_id_lists, self.traj_time_id_lists]
        # self.traj_road_flow = self.traj_road_flow - self.traj_road_flow[:, 0:1]
        logging.info("Finish reading road flow.")
        
        logging.info(
                f"Finish reading trajectory data file. \n"
                f"Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_lists: {self.traj_time_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of traj_road_flow: {self.traj_road_flow.shape} \n")
        
    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
                self.traj_time_id_lists[index],  
                self.traj_time_features_lists[index],
                self.traj_road_flow[index])
            
    def __len__(self):
        return self.dataset_len

        