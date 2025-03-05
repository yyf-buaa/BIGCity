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
from data_provider import file_loader

class DatasetTraj(Dataset):
    def __init__(self):
        logging.info("Start loading trajectory dataset.")
        
        super().__init__()
        
        self.dataset_len = None
        self.read_traj_data()
        
        logging.info("Finish loading trajectory dataset.")
    
    def read_traj_data(self):
        
        traj_data = file_loader.traj_data
        
        self.dataset_len = file_loader.traj_data_cnt
        
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
                f"Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of traj_road_flow: {self.traj_road_flow.shape} \n")
        
    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
                self.traj_time_id_lists[index],  
                self.traj_time_features_lists[index],
                self.traj_road_flow[index])
            
    def __len__(self):
        return self.dataset_len


class DatasetNextHop(Dataset):
    def __init__(self):
        logging.info("Start loading next hop dataset.")
        
        super().__init__()
        
        self.dataset_len = None
        self.read_traj_data()
        
        logging.info("Finish loading next hop dataset.")
    
    def read_traj_data(self):

        traj_data = file_loader.traj_data
        
        self.dataset_len = file_loader.traj_data_cnt
        
        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len - 1] + x[args.seq_len - 2] if len(x) > args.seq_len else x[:-1] + [x[-2]] * (args.seq_len - len(x) + 1) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("Finish reading road id.")
        
        self.traj_time_lists = torch.tensor([ x[:args.seq_len - 1] + x[args.seq_len - 2] if len(x) > args.seq_len else x[:-1] + [x[-2]] * (args.seq_len - len(x) + 1) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(global_vars.start_time.timestamp())) // global_vars.interval).to(torch.int32)
        logging.info("Finish reading time id.")

        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("Finish reading time features.")
        
        self.nexthop_labels = torch.tensor([ x[args.seq_len - 1]if len(x) > args.seq_len else x[-1]  
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("Finish reading next hop label.")
        
        logging.info(
                f"Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of nexthop_labels: {self.nexthop_labels.shape}, \n")
        
    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
                self.traj_time_id_lists[index],  
                self.traj_time_features_lists[index],
                self.nexthop_labels[index])
            
    def __len__(self):
        return self.dataset_len


class DatasetTrajClassify(Dataset):
    def __init__(self):
        logging.info("Start loading trajectory classify dataset.")
        
        super().__init__()
        
        self.dataset_len = None
        self.read_traj_data()
        
        logging.info("Finish loading trajectory classify dataset.")
    
    def read_traj_data(self):

        traj_data = file_loader.traj_data
        
        self.dataset_len = file_loader.traj_data_cnt
        
        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("Finish reading road id.")
        
        self.traj_time_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(global_vars.start_time.timestamp())) // global_vars.interval).to(torch.int32)
        logging.info("Finish reading time id.")

        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("Finish reading time features.")
        
        self.classify_labels = torch.tensor(traj_data["usr_id"], dtype=torch.int64)
        logging.info("Finish reading classify label.")
        
        logging.info(
                f"Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of classify_labels: {self.classify_labels.shape}, \n")
        
    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
                self.traj_time_id_lists[index],  
                self.traj_time_features_lists[index],
                self.classify_labels[index])
            
    def __len__(self):
        return self.dataset_len


class DatasetTimeReg(Dataset):
    def __init__(self):
        logging.info("Start loading time regression dataset.")
        
        super().__init__()
        
        self.dataset_len = None
        self.read_traj_data()
        
        logging.info("Finish loading time regression dataset.")
    
    def read_traj_data(self):

        traj_data = file_loader.traj_data
        
        self.dataset_len = file_loader.traj_data_cnt
        
        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("Finish reading road id.")
        
        self.traj_time_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(global_vars.start_time.timestamp())) // global_vars.interval).to(torch.int32)
        logging.info("Finish reading time id.")
        
        self.traj_time_features_lists = torch.zeros((self.dataset_len, args.seq_len, 6), dtype=torch.float32)
        logging.info("Finish reading time features.")

        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.time_regression_labels = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("Finish reading time regression labels.")
        
        logging.info(
                f"Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n",
                f"Shape of time_regression_labels: {self.time_regression_labels.shape}, \n")
        
    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
                self.traj_time_id_lists[index],  
                self.traj_time_features_lists[index],
                self.time_regression_labels[index])
            
    def __len__(self):
        return self.dataset_len


class DatasetTrafficStateReg(Dataset):
    
    def __init__(self):
        logging.info("Start loading traffic state regression dataset.")
        
        super().__init__()
        
        self.dataset_len = None
        self.read_traj_data()
        
        logging.info("Finish loading traffic state regression dataset.")
    
    def read_traj_data(self):

        traj_data = file_loader.traj_data
        
        self.dataset_len = file_loader.traj_data_cnt
        
        self.traj_road_id_lists, self.traj_time_id_lists = self.generate_sequences(
            self.dataset_len, args.seq_len, file_loader.time_slots_cnt, args.seq_len * 1.5, file_loader.road_cnt, 0.5)
        logging.info("Finish reading road id.")
        logging.info("Finish reading time id.")
        
        self.traj_time_lists = torch.tensor(global_vars.start_time.timestamp()).to(torch.int64) + self.traj_time_id_lists * global_vars.interval
        
        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists  = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("Finish reading time features.")
        
        self.traffic_state_regression_labels = file_loader.dynamic_features[self.traj_road_id_lists, self.traj_time_id_lists]
        logging.info("Finish reading time regression labels.")
        
        logging.info(
                f"Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of traffic_state_regression_labels: {self.traffic_state_regression_labels.shape}, \n")
        
    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
                self.traj_time_id_lists[index],  
                self.traj_time_features_lists[index],
                self.traffic_state_regression_labels[index])
            
    def __len__(self):
        return self.dataset_len

    
    
    def generate_sequences(self, K, L, T, A, N, P):
        """
        Generate K sequences with probability P of strictly increasing sequences,
        and 1-P of non-strictly increasing sequences.
        """
        sequences = []
        labels = []

        for _ in range(K):
            while True:
                x = np.random.randint(0, T)
                ub = min(T - 1, x + A)
                available = ub - x + 1

                if available < L:
                    continue

                if A < L or np.random.rand() >= P:
                    seq = np.sort(np.random.choice(np.arange(x, ub + 1), size=L, replace=True))
                else:
                    seq = np.sort(np.random.choice(np.arange(x, ub + 1), size=L, replace=False))

                sequences.append(seq)
                label = np.random.randint(0, N)
                labels.append(np.full(L, label))
                break

        sequences = torch.tensor(np.array(sequences), dtype=torch.int64)
        labels = torch.tensor(np.array(labels), dtype=torch.int64)

        return labels, sequences

class DatasetTrajRecover(Dataset):
    pass


