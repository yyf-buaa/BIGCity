from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, get_learning_rate, log_gradients
from utils.metrics import metric
import torch, logging
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
from layers.Embed import DataEmbedding, DataEmbedding_wo_time, TrajDataEmbedding

tb = SummaryWriter()
warnings.filterwarnings('ignore')


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

        if args.city == 'bj':
            self.seq_len = 128
        else:
            self.seq_len = 64
        self.alpha = args.loss_alpha
        self.beta = args.loss_beta
        self.gamma = args.loss_gamma
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                      weight_decay=self.args.weight)
        self.lr_scheduler = CosineLRScheduler(
            self.model_optim, self.args.train_epochs, lr_min=0, decay_rate=0.1,
            warmup_t=40, warmup_lr_init=self.args.learning_rate / 20, t_in_epochs=True)

        self.loss_time_func = nn.MSELoss(reduction='mean')
        self.loss_road_func = nn.CrossEntropyLoss(reduction='mean')
        self.loss_flow_func = nn.MSELoss(reduction='mean')

    def padding_mask(self, B, T):
        mask = torch.ones(B, T)
        num_mask = int(self.args.mask_rate * T)
        for i in range(B):
            indices_to_mask = torch.randperm(T - 2, dtype=torch.long)[:num_mask]
            mask[i][indices_to_mask + 1] = 0
        return mask, num_mask

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=list(map(int, self.args.devices.split(','))))
        para_num = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logging.info('Traj_LM parameter numbers: {}'.format(para_num))
        logging.info(str(model))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def psm_input(self, batch_road_id, batch_time_index, mask, num_mask):
        B, T = batch_road_id.shape
        batch_masked_road_id = batch_road_id.masked_fill(mask == 0, self.mask_token_id)
        special_token = torch.tensor([self.cls_token_id, self.tem_token_id, self.flow_token_id])
        special_token = torch.tile(special_token, (B, num_mask))
        special_time = torch.zeros(B, 3 * num_mask)
        return torch.cat([batch_masked_road_id, special_token], dim=1), torch.cat([batch_time_index, special_time],
                                                                                  dim=1)

    def psm_input_time(self, batch_time_features, mask, num_mask):
        B, T, d = batch_time_features.shape
        mask = mask.unsqueeze(-1)
        mask = mask.expand(B, T, d)
        batch_masked_time_features = batch_time_features.masked_fill(mask == 0, self.mask_token_id)
        special_token = torch.zeros(size=[B, 3 * num_mask, d])
        return torch.cat([batch_masked_time_features, special_token], dim=1)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        train_steps = len(train_loader)
        time_now = time.time()
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (
            batch_road_id, batch_time_features, batch_road_time, batch_traj_len, batch_duration_time, batch_road_flow,
            batch_traj_time_index) in tqdm(enumerate(train_loader)):
                iter_count += 1
                self.model_optim.zero_grad()
                print('road time:', batch_road_time.shape)
                B, T = batch_road_id.shape
                d_t = batch_time_features.shape[2]

                mask, num_mask = self.padding_mask(B, T)
                batch_masked_road_id, batch_traj_time_index = self.psm_input(batch_road_id, batch_traj_time_index, mask,
                                                                             num_mask)  # mask and psm
                batch_masked_time_features = self.psm_input_time(batch_time_features, mask, num_mask)  # mask and psm

                batch_masked_road_id = batch_masked_road_id.to(self.device)
                batch_masked_time_features = batch_masked_time_features.float().to(self.device)
                batch_road_time = batch_road_time.float().to(self.device)
                batch_road_flow = batch_road_flow.float().to(self.device)
                batch_road_id = batch_road_id.to(self.device)
                batch_traj_time_index = batch_traj_time_index.to(self.device)

                outputs_road, outputs_time, outputs_flow = self.model(batch_masked_road_id, batch_traj_time_index,
                                                                      batch_masked_time_features)
                print('output:', outputs_road, outputs_time, outputs_flow)

                loss_time = self.loss_time_func(torch.flatten(outputs_time, start_dim=0, end_dim=-1),
                                                batch_road_time[mask == 0])
                loss_road = self.loss_road_func(torch.flatten(outputs_road, start_dim=0, end_dim=-2),
                                                batch_road_id[mask == 0])
                loss_flow = self.loss_flow_func(torch.flatten(outputs_flow, start_dim=0, end_dim=-1),
                                                batch_road_flow[mask == 0])
                logging.info(loss_road.item())
                logging.info(loss_time.item())
                logging.info(loss_flow.item())
                loss = loss_road * self.alpha + loss_time * self.beta + loss_flow * self.gamma
                train_loss.append(loss.item())

                if (i + 1) % 2 == 0:
                    logging.info("\titers: {0}, epoch: {1} |loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    logging.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

                # batch end
                iter_count = 0
                time_now = time.time()
                loss.backward()
                self.model_optim.step()

            # epoch end
            logging.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            logging.info("Epoch: {0}, Steps: {1} |Pre Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            tb.add_scalar('Pretrain Loss', train_loss, epoch)
            lr = get_learning_rate(self.model_optim)
            tb.add_scalar('Learning_rate', lr, epoch)
            self.early_stopping(train_loss, self.model, self.path)
            self.lr_scheduler.step(epoch + 1)

        # train end
        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test_in_train(self, test_loader):
        pass

    def test(self, setting, test=0):
        return
