import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='BigCity Training Config')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='imputation',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=False, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='Traj', help='dataset type')
    parser.add_argument('--dataset_path', type=str, default='../dataset/', help='root path of the data file') # use
    parser.add_argument('--city', type=str, default='xa', help='city of the dataset') # use
    parser.add_argument('--embedding_model',type=str,default='HHGCLV3',help='road_embedding_model')
    parser.add_argument('--seq_len',type=int,default=128,help='trak sequence len') # use

    parser.add_argument('--freq', type=str, default='s',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--log_path', type=str, default='./log/', help='location of log file') # use
    

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio') # use

    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--d_model', type=int, default=768, help='dimension of model') # use
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
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers') # use
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs') # use
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data') # use
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience') # use
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate') # use
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', default=False, action='store_true', help='use gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--device', type=str, default='-1', help='device ids of multile gpus, -1 means cpu')

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
    parser.add_argument('--loss_alpha', type=float, default=0.4) # use
    parser.add_argument('--loss_beta', type=float, default=30) # use
    parser.add_argument('--loss_gamma', type=float, default=60) # use
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--gpt2_checkpoint_name', type=str, default=None)
    parser.add_argument('--sample_rate', type=float, default=1)

    parser.add_argument('--pre_dyna', default=False, action='store_true', help='Pre-trained dynamic embeddings are used if set')
    parser.add_argument('--develop', default=False, action='store_true', help='If set to true, the short dataset is loaded')
    parser.add_argument('--wandb_mode', type=str, default="offline", help='Wandb mode: offline/online')
    parser.add_argument('--no_traffic_state', default=False, action='store_true', help='no traffic state (bj or other dataset)')
    
    return parser

args = get_args_parser().parse_args()