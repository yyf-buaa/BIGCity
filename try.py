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

from models.GPT4TS_new import StTokenizer

from tqdm import tqdm

def main():

    print(torch.cuda.is_available())
    
    print(args)
    
    dataset = DatasetTraj()
    print(args.batch_size, args.num_workers)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    print(data_filename.road_static_file)
    print(data_filename.road_dynamic_file)
    print(data_filename.road_relation_file)
    print(data_filename.traj_file)
    print(data_filename.traj_file_short)
    
    st = StTokenizer()
    st(torch.tensor([3767, 5267]), torch.tensor([2000, 2000]))



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
