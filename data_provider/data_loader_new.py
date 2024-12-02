from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import logging
from ast import literal_eval

from config import data_filename
from utils.timefeatures import time_features

class DatasetTraj(Dataset):
    def __init__(self):
        logging.info("Start loading trajectory dataset.")
        
        super().__init__()
        self.seq_len = 64
        self.read_traj_data()
        
        logging.info("Finish loading trajectory dataset.")
    
    def read_traj_data(self):
        logging.info("Start reading trajectory data file.")
        
        traj_data = pd.read_csv(data_filename.traj_file_short, delimiter=';').sample(frac=1)
        traj_data.reset_index(drop=True, inplace=True)
        
        self.dataset_len = len(traj_data)
    
        self.traj_road_id_lists = torch.tensor([ x[:self.seq_len] if len(x) > self.seq_len else x + [x[-1]] * (self.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int32)
        
        self.traj_time_lists = torch.tensor([ x[:self.seq_len] if len(x) > self.seq_len else x + [x[-1]] * (self.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.long)
        

        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(pd.to_datetime("2018-10-01T00:00:00Z").timestamp())) // 1800).to(torch.int32)
        # print(self.traj_time_id_lists)
        
        self.traj_time_offset = self.traj_time_lists - self.traj_time_lists[:, 0:1]
        # print(self.traj_time_offset)
        
        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'),freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, self.seq_len, data_stamp.shape[-1]))
        
        logging.info(
                f"Finish reading trajectory data file. \n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_lists: {self.traj_time_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape} \n")
        
    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
                self.traj_time_id_lists[index],  
                self.traj_time_features_lists[index],
                self.traj_time_offset[index])
            
    def __len__(self):
        return self.dataset_len

        