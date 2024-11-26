from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,get_learning_rate,log_gradients
from utils.metrics import metric
import torch,logging
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import logging
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
from layers.Embed import DataEmbedding, DataEmbedding_wo_time,TrajDataEmbedding
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
        self.road_num = args.road_num
        self.mask_token_id = self.road_num
        self.cls_token_id = self.road_num + 1
        self.tem_token_id = self.road_num + 2
        self.flow_token_id = self.road_num + 3
        self.task_name = args.task_name
        self.city = args.city
        self.bidir_adj_mx = False
        # self._load_geo()
        # self._load_rel()
        # self.adj_mx = torch.from_numpy(self.adj_mx)
        if args.city == 'bj':
            self.seq_len = 128
        else:
            self.seq_len = 64
        self.alpha = args.loss_alpha
        self.beta = args.loss_beta
        self.gamma = args.loss_gamma

    def construct_mask_tensor(self,x_id):
        B, T = x_id.shape
        N, N = self.adj_mx.shape
        result_tensor = self.adj_mx[x_id[:,:-2]]
        mask_tensor = torch.where(result_tensor == 1, torch.tensor(0.0), torch.tensor(-1000.0))
        return mask_tensor

    def _load_geo(self):
        self.geo_file = '../dataset/{}/roadmap_{}/roadmap_{}.geo'.format(self.city,self.city,self.city)
        geofile = pd.read_csv('../dataset/{}/roadmap_{}/roadmap_{}.geo'.format(self.city,self.city,self.city))
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
        logging.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape) + ', edges=' + str(self.adj_mx.sum()))
        logging.info("edge_index shape= " + str(self.edge_index.shape) + ", edge_weight shape= "
                          + str(self.edge_weight.shape) + ', edges=' + str(self.edge_index.shape[1]))
        self.adj_mx = self.adj_mx + np.eye(self.road_num, self.road_num)
        return relfile
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        para_num = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logging.info('Traj_LM parameter numbers: {}'.format(para_num))
        logging.info(str(model))
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
                    warmup_t=40, warmup_lr_init=self.args.learning_rate/20, t_in_epochs=True)
        return

    def psm_input(self, batch_x, batch_time, num_mask):
        B, T = batch_x.shape
        special_token = torch.tensor([self.cls_token_id, self.tem_token_id, self.flow_token_id])
        special_token = torch.tile(special_token, (B, num_mask))
        special_time = torch.zeros(B, 3 * num_mask)
        return torch.cat([batch_x, special_token], dim=1), torch.cat([batch_time, special_time], dim=1)

    def psm_input_time(self, batch_x_mark, mask_road, num_mask):
        B, T, d = batch_x_mark.shape
        mask_road = mask_road.unsqueeze(-1)
        mask = mask_road.expand(B, T, d)
        batch_x_mark = batch_x_mark.masked_fill(mask == 0, 0)
        special_token = torch.zeros(size=[B,3*num_mask,d])
        return torch.cat([batch_x_mark,special_token], dim=1)

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
        loss_time_func = nn.MSELoss()
        loss_road_func = nn.CrossEntropyLoss()
        loss_flow_func = nn.MSELoss()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_traj_id, batch_x_mark, batch_road_time, batch_traj_len, batch_duration_time, batch_road_flow, batch_traj_time_index) in tqdm(enumerate(train_loader)):
                iter_count += 1
                self.model_optim.zero_grad()
                B, T = batch_traj_id.shape
                d_t = batch_x_mark.shape[2]
                mask = torch.rand((T))
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask[0] = 1
                mask[-1] = 1
                num_mask = torch.sum(mask == 0).item()
                mask = mask.unsqueeze(0)  # (1, T)
                mask_road = mask.expand(B, T)  # (B, T)
                x_id = batch_traj_id.masked_fill(mask == 0, self.mask_token_id)
                batch_x_id, batch_traj_time_index = self.psm_input(x_id, batch_traj_time_index, num_mask)
                batch_x_mark = self.psm_input_time(batch_x_mark, mask_road, num_mask)
                batch_x_id = batch_x_id.to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_road_time = batch_road_time.float().to(self.device)
                batch_road_flow = batch_road_flow.float().to(self.device)
                batch_traj_id = batch_traj_id.to(self.device)
                batch_traj_time_index = batch_traj_time_index.to(self.device)
                outputs_road, outputs_time, outputs_flow = self.model(batch_x_id, batch_traj_time_index, batch_x_mark)
                
                loss_time = loss_time_func( torch.flatten(outputs_time,start_dim=0,end_dim=-1),batch_road_time[mask_road == 0])
                loss_road = loss_road_func(torch.flatten(outputs_road,start_dim=0,end_dim=-2), batch_traj_id[mask_road == 0])
                loss_flow = loss_flow_func(torch.flatten(outputs_flow,start_dim=0,end_dim=-1), batch_road_flow[mask_road == 0])
                # logging.info(loss_time.item())
                # logging.info(loss_road.item())
                loss = loss_road*self.alpha + loss_time*self.beta + loss_flow*self.gamma
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    # logging.info("\titers: {0}, epoch: {1} |loss: {2:.7f}".format(i + 1, epoch + 1,loss.item()))
                    logging.info("\titers: {0}, epoch: {1} |loss: {2:.7f}".format(i + 1, epoch + 1,loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # logging.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
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
