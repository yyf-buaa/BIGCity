import os
import torch
import logging
import traceback
import numpy as np

from datetime import datetime

import config.logging_config
import config.random_seed
from config import args_config
from exp.exp_pretrain import Exp_Pretrain
# from exp.exp_task_tuning import Exp_Task_Tuning


def main():

    print(torch.cuda.is_available())
    args = args_config.parser.parse_args()
    print(args)
    
    road_embedding = np.load(os.path.join(args.root_path, args.city,
        'road_embedding', 'road_embedding_{}_{}_128.npy'.format(args.embedding_model, args.city)))
    print(os.path.join(args.root_path, args.city,
        'road_embedding', 'road_embedding_{}_{}_128.npy'.format(args.embedding_model, args.city)))
    args.road_num = len(road_embedding)
    print(road_embedding.shape)
    
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    exp = Exp_Pretrain(args)
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    setting = '{}_{}_{}_{}_{}_{}_gpt{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_epoch{}_{} '.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.city,
                args.embedding_model,
                args.gpt_layers,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, 
                0,
                args.train_epochs,
                current_time_str)
    exp.train(setting)
    # logging.info(args)
    
    
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())
