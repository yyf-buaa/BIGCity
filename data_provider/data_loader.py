from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
import logging
from ast import literal_eval

from config import global_vars
from config import random_seed
from config.args_config import args
from . import file_loader
from utils.timefeatures import time_features
from data_provider.file_loader import file_loader


class DatasetTraj(Dataset):
    def __init__(self):
        logging.info("Start loading trajectory dataset.")
        
        super().__init__()
        
        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None
        self.traj_time_id_lists = None
        self.traj_time_features_lists = None
        self.traj_road_flow = None
        
        if os.path.exists(global_vars.cached_traj_dataset):
            self.load_cached_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()
        
        logging.info(
                f"(DatasetTraj) Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of traj_road_flow: {self.traj_road_flow.shape} \n")
        
        logging.info("Finish loading trajectory dataset.")
    
    def read_traj_data(self):
        
        traj_data = file_loader.get_traj_data()
        
        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("(DatasetTraj) Finish reading road id.")
        
        self.traj_time_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(global_vars.start_time.timestamp())) // global_vars.interval).to(torch.int32)
        logging.info("(DatasetTraj) Finish reading time id.")
        
        # self.traj_time_offset = self.traj_time_lists - self.traj_time_lists[:, 0:1]

        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("(DatasetTraj) Finish reading time features.")
        
        self.traj_road_flow = file_loader.dynamic_features[self.traj_road_id_lists, self.traj_time_id_lists]
        # self.traj_road_flow = self.traj_road_flow - self.traj_road_flow[:, 0:1]
        logging.info("(DatasetTraj) Finish reading road flow.")
    
    def cache_traj_data(self):
        logging.info(f"Caching trajectory dataset to {global_vars.cached_traj_dataset}")
        torch.save({
            'traj_road_id_lists': self.traj_road_id_lists,
            'traj_time_id_lists': self.traj_time_id_lists,
            'traj_time_features_lists': self.traj_time_features_lists,
            'traj_road_flow': self.traj_road_flow
        }, global_vars.cached_traj_dataset)
        
    def load_cached_traj_data(self):
        logging.info(f"Loading cached trajectory dataset from {global_vars.cached_traj_dataset}")
        cached_data = torch.load(global_vars.cached_traj_dataset, weights_only=True)
        self.traj_road_id_lists = cached_data['traj_road_id_lists']
        self.traj_time_id_lists = cached_data['traj_time_id_lists']
        self.traj_time_features_lists = cached_data['traj_time_features_lists']
        self.traj_road_flow = cached_data['traj_road_flow']
        
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
        
        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None
        self.traj_time_id_lists = None
        self.traj_time_features_lists = None
        self.nexthop_labels = None
        
        if os.path.exists(global_vars.cached_next_hop_dataset):
            self.load_cached_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()
        
        logging.info(
                f"(DatasetNextHop) Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of nexthop_labels: {self.nexthop_labels.shape}, \n")
        
        logging.info("Finish loading next hop dataset.")
    
    def read_traj_data(self):
        traj_data = file_loader.get_traj_data()
        
        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len - 1] + x[args.seq_len - 2] if len(x) > args.seq_len else x[:-1] + [x[-2]] * (args.seq_len - len(x) + 1) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("(DatasetNextHop) Finish reading road id.")
        
        self.traj_time_lists = torch.tensor([ x[:args.seq_len - 1] + x[args.seq_len - 2] if len(x) > args.seq_len else x[:-1] + [x[-2]] * (args.seq_len - len(x) + 1) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(global_vars.start_time.timestamp())) // global_vars.interval).to(torch.int32)
        logging.info("(DatasetNextHop) Finish reading time id.")

        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("(DatasetNextHop) Finish reading time features.")
        
        self.nexthop_labels = torch.tensor([ x[args.seq_len - 1]if len(x) > args.seq_len else x[-1]  
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("(DatasetNextHop) Finish reading next hop label.")
        
    def cache_traj_data(self):
        logging.info(f"Caching next hop dataset to {global_vars.cached_next_hop_dataset}")
        torch.save({
            'traj_road_id_lists': self.traj_road_id_lists,
            'traj_time_id_lists': self.traj_time_id_lists,
            'traj_time_features_lists': self.traj_time_features_lists,
            'nexthop_labels': self.nexthop_labels
        }, global_vars.cached_next_hop_dataset)
        
    def load_cached_traj_data(self):
        logging.info(f"Loading cached next hop dataset from {global_vars.cached_next_hop_dataset}")
        cached_data = torch.load(global_vars.cached_next_hop_dataset, weights_only=True)
        self.traj_road_id_lists = cached_data['traj_road_id_lists']
        self.traj_time_id_lists = cached_data['traj_time_id_lists']
        self.traj_time_features_lists = cached_data['traj_time_features_lists']
        self.nexthop_labels = cached_data['nexthop_labels']
        
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
        
        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None
        self.traj_time_id_lists = None
        self.traj_time_features_lists = None
        self.classify_labels = None
        
        if os.path.exists(global_vars.cached_traj_classify_dataset):
            self.load_cached_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()
        
        logging.info(
                f"(DatasetTrajClassify) Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of classify_labels: {self.classify_labels.shape}, \n")
        
        logging.info("Finish loading trajectory classify dataset.")
    
    def read_traj_data(self):

        traj_data = file_loader.get_traj_data()
        
        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("(DatasetTrajClassify) Finish reading road id.")
        
        self.traj_time_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(global_vars.start_time.timestamp())) // global_vars.interval).to(torch.int32)
        logging.info("(DatasetTrajClassify) Finish reading time id.")

        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("(DatasetTrajClassify) Finish reading time features.")
        
        self.classify_labels = torch.tensor(traj_data["usr_id"], dtype=torch.int64)
        logging.info("(DatasetTrajClassify) Finish reading classify label.")
        
    def cache_traj_data(self):
        logging.info(f"Caching trajectory classify dataset to {global_vars.cached_traj_classify_dataset}")
        torch.save({
            'traj_road_id_lists': self.traj_road_id_lists,
            'traj_time_id_lists': self.traj_time_id_lists,
            'traj_time_features_lists': self.traj_time_features_lists,
            'classify_labels': self.classify_labels
        }, global_vars.cached_traj_classify_dataset)
        
    def load_cached_traj_data(self):
        logging.info(f"Loading cached trajectory classify dataset from {global_vars.cached_traj_classify_dataset}")
        cached_data = torch.load(global_vars.cached_traj_classify_dataset, weights_only=True)
        self.traj_road_id_lists = cached_data['traj_road_id_lists']
        self.traj_time_id_lists = cached_data['traj_time_id_lists']
        self.traj_time_features_lists = cached_data['traj_time_features_lists']
        self.classify_labels = cached_data['classify_labels']
        
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
        
        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None
        self.traj_time_id_lists = None
        self.traj_time_features_lists = None
        self.time_regression_labels = None
        
        if os.path.exists(global_vars.cached_time_reg_dataset):
            self.load_cached_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()
        
        logging.info(
                f"(DatasetTimeReg) Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of time_regression_labels: {self.time_regression_labels.shape}, \n")
        
        logging.info("Finish loading time regression dataset.")
    
    def read_traj_data(self):

        traj_data = file_loader.get_traj_data()
        
        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        logging.info("(DatasetTimeReg) Finish reading road id.")
        
        self.traj_time_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(global_vars.start_time.timestamp())) // global_vars.interval).to(torch.int32)
        logging.info("(DatasetTimeReg) Finish reading time id.")
        
        self.traj_time_features_lists = torch.zeros((self.dataset_len, args.seq_len, 6), dtype=torch.float32)
        logging.info("(DatasetTimeReg) Finish reading time features (masked).")

        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.time_regression_labels = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("(DatasetTimeReg) Finish reading time regression labels.")
        
    def cache_traj_data(self):
        logging.info(f"Caching time regression dataset to {global_vars.cached_time_reg_dataset}")
        torch.save({
            'traj_road_id_lists': self.traj_road_id_lists,
            'traj_time_id_lists': self.traj_time_id_lists,
            'traj_time_features_lists': self.traj_time_features_lists,
            'time_regression_labels': self.time_regression_labels
        }, global_vars.cached_time_reg_dataset)
        
    def load_cached_traj_data(self):
        logging.info(f"Loading cached time regression dataset from {global_vars.cached_time_reg_dataset}")
        cached_data = torch.load(global_vars.cached_time_reg_dataset, weights_only=True)
        self.traj_road_id_lists = cached_data['traj_road_id_lists']
        self.traj_time_id_lists = cached_data['traj_time_id_lists']
        self.traj_time_features_lists = cached_data['traj_time_features_lists']
        self.time_regression_labels = cached_data['time_regression_labels']
        
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
        
        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None
        self.traj_time_id_lists = None
        self.traj_time_features_lists = None
        self.traffic_state_regression_labels = None
        
        if os.path.exists(global_vars.cached_traffic_state_reg_dataset):
             self.load_cached_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()
           
        logging.info(
                f"(DatasetTrafficStateReg) Number of trajectories: {self.dataset_len}\n"
                f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
                f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
                f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
                f"Shape of traffic_state_regression_labels: {self.traffic_state_regression_labels.shape}, \n")
        
        logging.info("Finish loading traffic state regression dataset.")
    
    def read_traj_data(self):

        traj_data = file_loader.get_traj_data()
        
        road_cnt = file_loader.get_road_cnt()
        time_slots_cnt = file_loader.get_time_slots_cnt()
        
        self.traj_road_id_lists, self.traj_time_id_lists = self.generate_sequences(
            self.dataset_len, args.seq_len, time_slots_cnt, args.seq_len * 1.5, road_cnt, 0.5)
        logging.info("(DatasetTrafficStateReg) Finish reading road id.")
        logging.info("(DatasetTrafficStateReg) Finish reading time id (unmasked).")
        
        self.traj_time_lists = torch.tensor(global_vars.start_time.timestamp()).to(torch.int64) + self.traj_time_id_lists * global_vars.interval
        
        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists  = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        logging.info("(DatasetTrafficStateReg) Finish reading time features.")
        
        self.traffic_state_regression_labels = file_loader.dynamic_features[self.traj_road_id_lists, self.traj_time_id_lists]
        logging.info("(DatasetTrafficStateReg) Finish reading time regression labels.")
        
        self.traj_time_id_lists.fill_(time_slots_cnt)
        logging.info("(DatasetTrafficStateReg) Finish reading time id (masked).")
     
    def cache_traj_data(self):
        logging.info(f"Caching traffic state regression dataset to {global_vars.cached_traffic_state_reg_dataset}")
        torch.save({
            'traj_road_id_lists': self.traj_road_id_lists,
            'traj_time_id_lists': self.traj_time_id_lists,
            'traj_time_features_lists': self.traj_time_features_lists,
            'traffic_state_regression_labels': self.traffic_state_regression_labels
        }, global_vars.cached_traffic_state_reg_dataset)
    
    def load_cached_traj_data(self):
        logging.info(f"Loading cached traffic state regression dataset from {global_vars.cached_traffic_state_reg_dataset}")
        cached_data = torch.load(global_vars.cached_traffic_state_reg_dataset, weights_only=True)
        self.traj_road_id_lists = cached_data['traj_road_id_lists']
        self.traj_time_id_lists = cached_data['traj_time_id_lists']
        self.traj_time_features_lists = cached_data['traj_time_features_lists']
        self.traffic_state_regression_labels = cached_data['traffic_state_regression_labels']
        
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
    def __init__(self):
        logging.info("Start loading trajectory recover dataset.")
        
        super().__init__()
        
        self.dataset_len = file_loader.get_traj_cnt()
        self.traj_road_id_lists = None
        self.traj_time_id_lists = None
        self.traj_time_features_lists = None
        self.traj_recover_labels = None
        
        if os.path.exists(global_vars.cached_traj_recover_dataset):
            self.load_cached_traj_data()
        else:
            self.read_traj_data()
            self.cache_traj_data()
            
        logging.info(
            f"(DatasetTrajRecover) Number of trajectories: {self.dataset_len}\n"
            f"Shape of traj_road_id_lists: {self.traj_road_id_lists.shape}, \n"
            f"Shape of traj_time_id_lists: {self.traj_time_id_lists.shape}, \n"
            f"Shape of traj_time_features_lists: {self.traj_time_features_lists.shape}, \n"
            f"Shape of traj_recover_labels: {self.traj_recover_labels.shape}, \n")
        
        logging.info("Finish loading trajectory recover dataset.")
    
    def read_traj_data(self):
        
        traj_data = file_loader.get_traj_data()
        
        road_cnt = file_loader.get_road_cnt()
        time_slots_cnt = file_loader.get_time_slots_cnt()
        
        mask, num_mask = self.padding_mask(self.dataset_len, args.seq_len)
        
        self.traj_road_id_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["path"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_recover_labels = self.traj_road_id_lists[mask == 0].reshape(self.dataset_len, -1)
        logging.info("(DatasetTrajRecover) Finish reading recover label.")
                
        self.traj_road_id_lists[mask == 0] = road_cnt
        logging.info("(DatasetTrajRecover) Finish reading road id (masked).")
        
        self.traj_time_lists = torch.tensor([ x[:args.seq_len] if len(x) > args.seq_len else x + [x[-1]] * (args.seq_len - len(x)) 
                            for x in traj_data["tlist"].apply(lambda x: literal_eval(x))], dtype=torch.int64)
        self.traj_time_id_lists = ((self.traj_time_lists - torch.tensor(global_vars.start_time.timestamp())) // global_vars.interval).to(torch.int32)
        self.traj_time_id_lists[mask == 0] = time_slots_cnt
        logging.info("(DatasetTrajRecover) Finish reading time id (masked).")
        
        data_stamp = time_features(pd.to_datetime(self.traj_time_lists.flatten(), unit='s'), freq='s').transpose(1, 0)
        data_stamp = torch.from_numpy(data_stamp)
        self.traj_time_features_lists = torch.reshape(data_stamp, (self.dataset_len, args.seq_len, data_stamp.shape[-1]))
        self.traj_time_features_lists[mask == 0, :] = 0
        logging.info("(DatasetTrajRecover) Finish reading time features (masked).")
    
    def cache_traj_data(self):
        logging.info(f"Caching trajectory recover dataset to {global_vars.cached_traj_recover_dataset}")
        torch.save({
            'traj_road_id_lists': self.traj_road_id_lists,
            'traj_time_id_lists': self.traj_time_id_lists,
            'traj_time_features_lists': self.traj_time_features_lists,
            'traj_recover_labels': self.traj_recover_labels
        }, global_vars.cached_traj_recover_dataset)
    
    def load_cached_traj_data(self):
        logging.info(f"Loading cached trajectory recover dataset from {global_vars.cached_traj_recover_dataset}")
        cached_data = torch.load(global_vars.cached_traj_recover_dataset, weights_only=True)
        self.traj_road_id_lists = cached_data['traj_road_id_lists']
        self.traj_time_id_lists = cached_data['traj_time_id_lists']
        self.traj_time_features_lists = cached_data['traj_time_features_lists']
        self.traj_recover_labels = cached_data['traj_recover_labels']

    def __getitem__(self, index):
        return (self.traj_road_id_lists[index],
                self.traj_time_id_lists[index],  
                self.traj_time_features_lists[index],
                self.traj_recover_labels[index])
            
    def __len__(self):
        return self.dataset_len
    
    def padding_mask(self, B, L):
        mask = torch.ones(B, L)
        num_mask = int(args.mask_rate * L)
        for i in range(B):
            indices_to_mask = torch.randperm(L, dtype=torch.long)[:num_mask]
            mask[i][indices_to_mask] = 0
        return mask, num_mask

