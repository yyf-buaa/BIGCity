import os
import torch
import logging
import traceback
import numpy as np
import pandas as pd
from ast import literal_eval

from datetime import datetime

import config.logging_config
import config.random_seed
from config.args_config import args
from config import data_filename
# from exp.exp_pretrain import Exp_Pretrain

from utils.timefeatures import time_features
from data_provider.data_loader_new import DatasetTraj
from torch.utils.data import DataLoader

from models.st_tokenizer import StTokenizer
from models.backbone import BIGCity

from tqdm import tqdm

def main():

    # print(data_filename.road_static_file)
    # print(data_filename.road_dynamic_file)
    # print(data_filename.road_relation_file)
    # print(data_filename.traj_file)
    # print(data_filename.traj_file_short)
    
    dataset = DatasetTraj()
    st = StTokenizer()
    big_city = BIGCity()
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    for i in data_loader:
        x = st(i[0], i[1], i[2])
        x = big_city(x)
        # break
        # loss backward.....
    
    
    
    
    
    
    
    # N, T, S = 2, 13, 6
    
    # dynamic_features = torch.randn((N, T))
    # print(dynamic_features)
    
    # padded_dynamic_features = torch.cat((torch.zeros(N, S), dynamic_features), dim=1)
    # dynamic_features= torch.stack([padded_dynamic_features[:, j - S:j] for j in range(S, T + S)], dim=1)

    # print(dynamic_features)


    # args = args_config.parser.parse_args()
    # print(args)
    
    # if args.use_gpu and args.use_multi_gpu:
    #     args.dvices = args.devices.replace(' ', '')
    #     device_ids = args.devices.split(',')
    #     args.device_ids = [int(id_) for id_ in device_ids]
    #     args.gpu = args.device_ids[0]

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())
