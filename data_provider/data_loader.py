import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
import datetime
import math
import warnings
import logging
from tqdm import tqdm
warnings.filterwarnings('ignore')
from transformers import GPT2Tokenizer



def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content_list = [line.strip() for line in file.readlines()]
    filtered_content_list = [line for line in content_list if line]
    return filtered_content_list


class Dataset_Inst(Dataset):
    def __init__(self,root_path, flag='train', city='xa', size=None,embedding_model='HHGCLV3',is_pretrain = False,sample_rate = 1):
        self.tokenizer = GPT2Tokenizer.from_pretrained("../gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.root_path = root_path
        self.is_pretrain = is_pretrain
        self.city = city
        self.flag = flag
        self.embedding_model = embedding_model
        self.len = 0
        self.__read_data__()

    def __read_data__(self):
        inst_list = read_txt(os.path.join(self.root_path, 'inst_dataset', self.flag,'Inst.txt'))
        input_ids_list = [self.tokenizer.encode_plus(inst,padding = 'max_length',max_length = 400, add_special_tokens=True, truncation=True,return_tensors="pt")['input_ids'] for inst in inst_list]
        self.input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask_list = [self.tokenizer.encode_plus(inst,padding = 'max_length',max_length = 400, add_special_tokens=True, truncation=True,return_tensors="pt")['attention_mask'] for inst in inst_list]
        self.attention_mask = torch.cat(attention_mask_list, dim=0)

        traj_raw = pd.read_csv(os.path.join(self.root_path, self.city, 'traj_{}_11_task_tuning.csv'.format(self.city)),
                               delimiter=';')
        traj_road_ids = []
        traj_times = []
        self.target_road = np.load(os.path.join(self.root_path, self.city, 'traj_{}_11_task_tuning_target_road.csv'.format(self.city)))
        self.target_time = np.load(
            os.path.join(self.root_path, self.city, 'traj_{}_11_task_tuning_target_time.csv'.format(self.city)))
        self.target_flow = np.load(
            os.path.join(self.root_path, self.city, 'traj_{}_11_task_tuning_target_flow.csv'.format(self.city)))
        self.traj_len = np.zeros(len(traj_raw))
        for i in range(len(traj_raw)):
            path = traj_raw.loc[i, 'path']
            path = path[1:len(path) - 1].split(',')
            path = [int(s) for s in path]
            if len(path) >= self.seq_len:
                path = path[:self.seq_len]
                self.traj_len[i] = self.seq_len
            else:
                self.traj_len[i] = len(path)
                pad_len = self.seq_len - len(path)
                path = path + ([path[-1]] * pad_len)
            traj_road_ids.append(path)

            tlist = traj_raw.loc[i, 'tlist']
            tlist = tlist[1:len(tlist) - 1].split(',')
            tlist = [int(t) for t in tlist]
            if len(tlist) >= self.seq_len:
                tlist = tlist[:self.seq_len]
            else:
                pad_len = self.seq_len - len(tlist)
                tlist = tlist + ([tlist[-1]] * pad_len)
            traj_times.append(tlist)
        traj_road_ids = np.array(traj_road_ids)
        logging.info('traj_road_ids shape is' + str(traj_road_ids.shape))
        self.traj_road_ids = traj_road_ids
        traj_times = np.array(traj_times)
        logging.info('traj_times shape is' + str(traj_times.shape))
        if os.path.exists(os.path.join(self.root_path, self.city, 'traj_road_flow.npy'.format(self.city))):
            self.traj_road_flow = np.load(
                os.path.join(self.root_path, self.city, 'traj_road_flow.npy'.format(self.city)))
        else:
            self.traj_road_flow = np.zeros(shape=[len(traj_raw), self.seq_len])
            road_flow = pd.read_csv('/root/dataDisk/Bigscity_flow/dataset/xa/xa.dyna')
            road_flow.set_index('time_id', inplace=True)
            for i in tqdm(range(len(traj_raw))):
                for j in range(self.seq_len):
                    time = traj_times[i][j]
                    starttime = 1538352000
                    road_geo_id = traj_road_ids[i][j]
                    time_index = (time - starttime) // 1800
                    if "{}_{}".format(road_geo_id, time_index) in road_flow.index:
                        self.traj_road_flow[i][j] = road_flow.loc["{}_{}".format(road_geo_id, time_index)][
                            'traffic_speed']
            np.save('/root/dataDisk/Bigscity_flow/dataset/xa/traj_road_flow.npy', self.traj_road_flow)

        datetime_array = pd.to_datetime(traj_times.flatten(), unit='s')
        datetime_index = pd.DatetimeIndex(datetime_array)
        data_stamp = time_features(datetime_index, freq='s')
        data_stamp = data_stamp.transpose(1, 0)
        self.traj_time_feature = np.reshape(data_stamp, (len(traj_raw), self.seq_len, data_stamp.shape[-1]))
        logging.info("finish time embedding:shape=" + str(self.traj_time_feature.shape))

    def __getitem__(self,index):
        return self.input_ids[index],self.attention_mask[index], self.traj_road_ids[index], self.traj_time_feature[index], self.road_times[index],self.traj_len[index],self.target_road[index],self.target_time[index],self.target_flow[index]
            
    def __len__(self):
        return self.len


class Dataset_Traj(Dataset): 
    def __init__(self,root_path, flag='train', city='xa', size=None, embedding_model='HHGCLV3', is_pretrain = False, sample_rate = 1):
        if city == 'bj':
            self.seq_len = 128
        else:
            self.seq_len = 64
        self.root_path = root_path
        self.is_pretrain = is_pretrain
        self.city = city
        self.flag = flag
        self.embedding_model = embedding_model
        self.sample_rate = sample_rate
        self.__read_data__()

    def __read_data__(self):
        if self.is_pretrain:
            traj_raw = pd.read_csv(os.path.join(self.root_path, self.city, 'traj_{}_11.csv'.format(self.city)), delimiter=';')
            traj_raw = traj_raw.sample(frac = self.sample_rate)
            traj_raw.reset_index(drop=True, inplace=True)

        else:
            traj_raw = pd.read_csv(os.path.join(self.root_path, self.city, 'traj_{}_11_{}.csv'.format(self.city,self.flag)), delimiter=';')
        # 用于存储轨迹所经过的道路ID
        traj_road_ids = []  
        # 用于存储轨迹经过每个道路的时间戳
        traj_times = []
        traj_time_indexs = []
        # 用于存储轨迹持续时间的列表，单位分钟
        duration_times = []  
        # 用于存储每条道路上轨迹出现的持续时间
        road_times = []
        self.traj_len  = np.zeros(len(traj_raw))
        for i in range(len(traj_raw)):
            path = traj_raw.loc[i, 'path']
            path = path[1:len(path) - 1].split(',')
            path = [int(s) for s in path]
            if len(path) >= self.seq_len:
                path = path[:self.seq_len]
                self.traj_len[i] = self.seq_len
            else:
                self.traj_len[i] = len(path)
                pad_len = self.seq_len - len(path)
                path = path + ([path[-1]] * pad_len)
            traj_road_ids.append(path)

            tlist = traj_raw.loc[i, 'tlist']
            tlist = tlist[1:len(tlist) - 1].split(',')
            tlist = [int(t) for t in tlist]
            if len(tlist) >= self.seq_len:
                tlist = tlist[:self.seq_len]
            else:
                pad_len = self.seq_len - len(tlist)
                tlist = tlist + ([tlist[-1]] * pad_len)
            
            duration_times.append((tlist[-1]-tlist[0]) / 60)
            road_time = [(tlist[i+1]-tlist[i]) for i in range(len(tlist)-1)]
            road_time = [0] + road_time
            road_times.append(road_time)
            traj_times.append(tlist)
            traj_time_index = [(t - 1538352000) // 1800 for t in tlist]
            traj_time_indexs.append(traj_time_index)

        self.duration_times = np.array(duration_times)
        self.road_times = np.array(road_times)
        traj_road_ids = np.array(traj_road_ids)
        logging.info('traj_road_ids shape is'+str(traj_road_ids.shape))
        self.traj_road_ids = traj_road_ids
        traj_times = np.array(traj_times)
        self.traj_time_indexs = np.array(traj_time_indexs)
        logging.info('traj_times shape is'+str(traj_times.shape))
        print(traj_times[0: 5])
        if os.path.exists(os.path.join(self.root_path, self.city, 'traj_road_flow.npy'.format(self.city))):
            self.traj_road_flow = np.load(os.path.join(self.root_path, self.city, 'traj_road_flow.npy'.format(self.city)))
        else:
            self.traj_road_flow = np.zeros(shape=[len(traj_raw), self.seq_len])
            road_flow = pd.read_csv(os.path.join(self.root_path, self.city, '{}.dyna'.format(self.city)))
            road_flow.set_index('dyna_id', inplace=True)
            for i in tqdm(range(len(traj_raw))):
                for j in range(self.seq_len):
                    time = traj_times[i][j]
                    starttime = 1538352000
                    road_geo_id = traj_road_ids[i][j]
                    time_index = (time - starttime)//1800
                    if "{}_{}".format(road_geo_id,time_index) in road_flow.index:
                        self.traj_road_flow[i][j] = road_flow.loc["{}_{}".format(road_geo_id, time_index)]['traffic_speed']
            np.save(os.path.join(self.root_path, self.city, 'traj_road_flow.npy'), self.traj_road_flow)
        
        datetime_array = pd.to_datetime(traj_times.flatten(), unit='s')
        datetime_index = pd.DatetimeIndex(datetime_array)
        data_stamp = time_features(datetime_index,freq='s')
        data_stamp = data_stamp.transpose(1, 0)
        self.traj_time_feature = np.reshape(data_stamp, (len(traj_raw), self.seq_len, 
                                            data_stamp.shape[-1]))
        logging.info("finish time embedding:shape=" + str(self.traj_time_feature.shape))


    def __getitem__(self, index):
        return (self.traj_road_ids[index],
               self.traj_time_feature[index],  
               self.road_times[index],
               self.traj_len[index],
               self.duration_times[index],
               self.traj_road_flow[index],
               self.traj_time_indexs[index])
            
    def __len__(self):
        return len(self.traj_road_ids)
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)