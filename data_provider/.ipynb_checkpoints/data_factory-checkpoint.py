from data_provider.data_loader import Dataset_Traj
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

import logging

data_dict = {
    'Traj': Dataset_Traj
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq
        
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        logging.info(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        data_set = Data()
        logging.info('road='+str(args.road_num))
        batch_size = args.batch_size
        if args.task_name == 'Sim_L1':
            if flag == 'train' or flag == 'val':
                batch_size = args.batch_size
                shuffle_flag = True
            else:
                batch_size = 5000
                shuffle_flag = False
        logging.info(flag)
        logging.info(len(data_set))
        if args.task_name == 'Sim':
            shuffle_flag = False
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers,
        )
        return data_set, data_loader
