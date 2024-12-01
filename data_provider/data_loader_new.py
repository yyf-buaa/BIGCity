from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from ast import literal_eval

import config.logging_config
from config import args_config
from config import data_filename
from utils.timefeatures import time_features

class DatasetTraj(Dataset):
    def __init__(self):
        logging.info("Start loading trajectory dataset.")
        
        super().__init__()
        self.seq_len = 64
        self.read_traj_data()
    
    def read_traj_data(self):
        logging.info("Start reading trajectory data file.")
        
        traj_data = pd.read_csv(data_filename.traj_file_short, delimiter=';').sample(frac = 0.8)
        traj_data.reset_index(drop=True, inplace=True)
        
        self.dataset_len = len(traj_data)
    
        self.traj_road_id_lists = np.array([ x[:self.seq_len] if len(x) > self.seq_len else x + [x[-1]] * (self.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))])
        self.traj_time_lists = np.array([ x[:self.seq_len] if len(x) > self.seq_len else x + [x[-1]] * (self.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))])
        
        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'),freq='s').transpose(1, 0)
        self.traj_time_features_lists = np.reshape(data_stamp, (self.dataset_len, self.seq_len, data_stamp.shape[-1]))
        
        logging.info(
                f"Finish reading trajectory data file. \n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_lists: {self.traj_time_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape} \n")
        
    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
               self.traj_time_lists[index],  
               self.traj_time_features_lists[index])
            
    def __len__(self):
        return self.dataset_len

        