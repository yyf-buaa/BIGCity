from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,get_learning_rate,log_gradients
from utils.metrics import metric
import torch, logging
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import os
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
from utils.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter
tb = SummaryWriter()
warnings.filterwarnings('ignore')

def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
        .float())


class Exp_Pretrain(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pretrain, self).__init__(args)
        self.road_num = 5269
        self.task_name = args.task_name
        self.city = args.city
        self.bidir_adj_mx = False
        self._load_geo()
        self._load_rel()
        self.adj_mx = torch.from_numpy(self.adj_mx)
        if args.city == 'bj':
            self.seq_len = 128
        else:
            self.seq_len = 64
        self.alpha = args.loss_alpha
        self.beta = args.loss_beta
        # print("!@#!@#!@#!@#!")
        # raise Exception("stop in pretrain init")

    def construct_mask_tensor(self, x_id):
        B, T = x_id.shape
        N, N = self.adj_mx.shape
        result_tensor = self.adj_mx[x_id[:,:-2]]
        mask_tensor = torch.where(result_tensor == 1, torch.tensor(0.0), torch.tensor(-1000.0))
        # import pdb
        # pdb.set_trace()
        return mask_tensor

    def _load_geo(self):
        self.geo_file = '../dataset/{}/roadmap_{}/roadmap_{}.geo'.format(self.city,self.city,self.city)
        geofile = pd.read_csv(self.geo_file)
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, geo_id in enumerate(self.geo_ids):
            self.geo_to_ind[geo_id] = index
            self.ind_to_geo[index] = geo_id
        logging.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))
        return geofile
    
       
    def _load_rel(self):
        self.rel_file = '../dataset/{}/roadmap_{}/roadmap_{}.rel'.format(self.city,self.city,self.city)
        relfile = pd.read_csv('../dataset/{}/roadmap_{}/roadmap_{}.rel'.format(self.city,self.city,self.city))
        weight_col = None
        print(relfile.columns)
        for col in relfile.columns:
            if 'weight' in col:
                weight_col = col
        assert weight_col is not None
        logging.info("weight_col {}".format(weight_col))

        relfile = relfile[['origin_id', 'destination_id', weight_col]]

        self.adj_mx = np.zeros((self.road_num, self.road_num), dtype=np.int32)
        self.edge_index = []
        self.edge_weight = []
        for row in relfile.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
            self.edge_index.append([self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]])
            self.edge_weight.append(row[-1])
            if self.bidir_adj_mx:
                self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
                self.edge_index.append([self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]])
                self.edge_weight.append(row[-1])
        self.edge_index = np.array(self.edge_index).T
        self.num_edges = self.edge_index.shape[1]
        self.edge_weight = np.array(self.edge_weight, dtype='float32')
        logging.info("Loaded file " + self.rel_file + ', adj_mx.shape=' + str(self.adj_mx.shape) + ', edges=' + str(self.adj_mx.sum()))
        logging.info("edge_index.shape= " + str(self.edge_index.shape) + ", edge_weight shape= "
                          + str(self.edge_weight.shape) + ', edges=' + str(self.edge_index.shape[1]))
        self.adj_mx = self.adj_mx + np.eye(self.road_num, self.road_num)
        return relfile
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        para_num = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logging.info('Traj_LM parameter numbers: {}'.format(para_num))   
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight)
        return

    # def _select_criterion(self):
    #     criterion = nn.MSELoss(reduction='none')
    #     return criterion
    
    def _build_lr_scheduler(self):
        self.lr_scheduler = CosineLRScheduler(
                    self.model_optim, self.args.train_epochs, lr_min=0, decay_rate=0.1,
                    warmup_t=40, warmup_lr_init=self.args.learning_rate/20, t_in_epochs=True
                    )
        return

    def cal_loss_time(self,outputs_time,targets_time,batch_traj_len):
        #outputs_time:[B,T-2],targets_time[B,T-2]
        mask = padding_mask(lengths=(batch_traj_len-1), max_len=self.seq_len)
        weights = mask.type_as(mask)[:,1:-1].to(self.device)
        criterion = nn.MSELoss(reduction='none')
        loss = criterion(outputs_time, targets_time) * weights
        loss_time = torch.mean(loss)
        # import pdb
        # pdb.set_trace()
        return loss_time

    def cal_loss_road(self, outputs_road, tragets_road, batch_traj_len):
        mask = padding_mask(lengths=(batch_traj_len), max_len=self.seq_len)
        weights = mask.type_as(mask)[:,1:-1].to(self.device).reshape(-1)
        outputs_road_old = outputs_road
        outputs_road = outputs_road.reshape(-1, outputs_road.shape[-1])
        tragets_road = tragets_road.reshape(-1)
        criterion = nn.NLLLoss(reduction='none')
        loss = criterion(outputs_road, tragets_road) * weights
        loss_road = torch.mean(loss)
        # import pdb
        # pdb.set_trace()
        return loss_road

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self._select_optimizer()
        self._build_lr_scheduler()
        
        for i, data in tqdm(enumerate(train_loader)):
            print(len(data))
            break

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_traj_id, batch_x, batch_x_mark, batch_road_time, batch_traj_len, 
            batch_duration_time, __temp) in tqdm(enumerate(train_loader)):
                iter_count += 1
                self.model_optim.zero_grad()
                batch_traj_id = batch_traj_id.to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_road_time = batch_road_time.float().to(self.device)
                #batch_duration_time = batch_duration_time.float().to(self.device)
                # batch_mask_tensor = self.construct_mask_tensor(batch_traj_id).to(self.device)
                batch_mask_tensor = None
                outputs_road, outputs_time = self.model(batch_x, batch_x_mark, None)
                loss_time = self.cal_loss_time(outputs_time, batch_road_time[:,1:], batch_traj_len)
                loss_road = self.cal_loss_road(outputs_road, batch_traj_id[:, 1:-1], batch_traj_len)
                # logging.info(loss_time.item())
                # logging.info(loss_road.item())
                loss = loss_road*self.alpha + loss_time*self.beta
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    logging.info("\titers: {0}, epoch: {1} |loss: {2:.7f}".format(i + 1, epoch + 1,loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logging.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
                loss.backward()
                self.model_optim.step()


            logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            logging.info("Epoch: {0}, Steps: {1} |Pre Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            tb.add_scalar('Pretrain Loss',train_loss,epoch)
            lr = get_learning_rate(self.model_optim)
            tb.add_scalar('Learning_rate',lr,epoch)
            early_stopping(train_loss, self.model, path)
            self.lr_scheduler.step(epoch + 1)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test_in_train(self, test_loader):
        pass


    def test(self, setting, test=0):
        return
