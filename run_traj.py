import argparse
import os
import torch, logging
import random
import numpy as np
from datetime import datetime
import pandas as pd
import traceback

from exp.exp_pretrain_llama import Exp_Pretrain_Llama
from exp.exp_pretrain import Exp_Pretrain
from exp.exp_task_tuning import Exp_Task_Tuning
import utils.logging_config

logging.getLogger().setLevel(logging.INFO)
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Trajectory LM')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='imputation',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')

# data loader
parser.add_argument('--data', type=str, required=True, default='Traj', help='dataset type')
parser.add_argument('--root_path', type=str, default='../dataset/', help='root path of the data file')
parser.add_argument('--city', type=str, default='xa', help='city of the dataset')
parser.add_argument('--embedding_model',type=str,default='HHGCLV3',help='road_embedding_model')

parser.add_argument('--freq', type=str, default='s',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# inputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=5)
# pre-train multi loss alpha
parser.add_argument('--loss_alpha', type=float, default=0.3)
parser.add_argument('--loss_beta', type=float, default=0.3)
parser.add_argument('--loss_gamma', type=float, default=0.4)
parser.add_argument('--checkpoint_name', type=str, default=None)
parser.add_argument('--gpt2_checkpoint_name', type=str, default=None)
parser.add_argument('--sample_rate', type=float, default=1)


def main():
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    road_embedding = np.load(os.path.join(args.root_path, args.city,
        'road_embedding', 'road_embedding_{}_{}_128.npy'.format(args.embedding_model, args.city)))
    args.road_num = len(road_embedding)
    # print('road_num={}'.format(args.road_num))
    logging.info('road_num={}'.format(args.road_num))

    if args.city == 'bj':
        args.seq_len = 128
    else:
        args.seq_len = 64
        
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # print('Args in experiment:')
    # print(args)
    logging.info('Args in experiment:')
    logging.info(args)

    if args.task_name == 'pretrain':
        Exp = Exp_Pretrain_Llama
    elif args.task_name == 'task_tuning':
        Exp = Exp_Task_Tuning
    elif args.task_name == 'RL_tuning':
        Exp = Exp_RL
    else:
        Exp = None

    if args.is_training:
        for ii in range(args.itr):
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
                ii,
                args.train_epochs,
                current_time_str)
            exp = Exp(args)  # set experiments
            # print('\n>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            logging.info('\n>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            # print('\n>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            logging.info('\n>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        setting = '{}_{}_{}_{}_{}_{}_gpt{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_epoch{}_{}'.   format(
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
            args.des, ii,args.train_epochs,current_time_str)
        exp = Exp(args)  # set experiments
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        logging.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


try:
	main()
except Exception as e:
    logging.error("\n" + traceback.format_exc())
