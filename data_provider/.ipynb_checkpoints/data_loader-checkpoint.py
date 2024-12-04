from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import logging
from ast import literal_eval
import os
from config import data_filename
from config.args_config import args
from utils.timefeatures import time_features

class Dataset_Traj(Dataset):
    def __init__(self):
        logging.info("Start loading trajectory dataset.")
        
        super().__init__()
        self.seq_len = args.seq_len
        self.read_traj_data()
        
        logging.info("Finish loading trajectory dataset.")
    
    def read_flow(self):
        if os.path.exists(os.path.join(args.root_path, args.city, 'traj_road_flow.npy'.format(args.city))):
            self.traj_road_flow = np.load(os.path.join(args.root_path, args.city, 'traj_road_flow.npy'.format(args.city)))
        else:
            self.traj_road_flow = np.zeros(shape=[len(traj_raw), args.seq_len])
            road_flow = pd.read_csv(os.path.join(args.root_path, args.city, '{}.dyna'.format(args.city)))
            road_flow.set_index('dyna_id', inplace=True)
            for i in tqdm(range(len(traj_raw))):
                for j in range(args.seq_len):
                    time = traj_times[i][j]
                    starttime = 1538352000
                    road_geo_id = traj_road_ids[i][j]
                    time_index = (time - starttime)//1800
                    if "{}_{}".format(road_geo_id,time_index) in road_flow.index:
                        self.traj_road_flow[i][j] = road_flow.loc["{}_{}".format(road_geo_id, time_index)]['traffic_speed']
            np.save(os.path.join(args.root_path, args.city, 'traj_road_flow.npy'), args.traj_road_flow)
    
    def read_traj_data(self):
        logging.info("Start reading trajectory data file.")
        
        traj_data = pd.read_csv(os.path.join(args.root_path, args.city, 'traj_{}_11.csv'.format(args.city)), delimiter=';')
        traj_data = traj_data.sample(frac = args.sample_rate)
        traj_data.reset_index(drop=True, inplace=True)
        
        self.dataset_len = len(traj_data)
        print('road_finish')
        self.traj_road_id_lists = torch.tensor([ x[:self.seq_len] if len(x) > self.seq_len else x + [x[-1]] * (self.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int32)
        print('time_finish')
        self.traj_time_lists = torch.tensor([ x[:self.seq_len] if len(x) > self.seq_len else x + [x[-1]] * (self.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.long)
        print('time_index_finish')
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(pd.to_datetime("2018-10-01T00:00:00Z").timestamp())) // 1800).to(torch.int32)
        print(self.traj_time_id_lists)
        
        self.traj_time_offset = self.traj_time_lists - self.traj_time_lists[:, 0:1]
        print(self.traj_time_offset)
        print('time_offset_finish')
        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'),freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, self.seq_len, data_stamp.shape[-1]))
        print('time_features_finish')
        self.read_flow()
        
        logging.info(
                f"Finish reading trajectory data file. \n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_lists: {self.traj_time_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of traj_road_flow: {self.traj_road_flow.shape} \n")
        
    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
                self.traj_time_id_lists[index],  
                self.traj_time_features_lists[index],
                self.traj_time_offset[index],
                self.traj_road_flow[index])
            
    def __len__(self):
        return self.dataset_len

        