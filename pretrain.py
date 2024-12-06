import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import traceback
import numpy as np
import pandas as pd
from ast import literal_eval

from datetime import datetime

import config.logging_config
import config.random_seed
from config.args_config import args
from config.global_vars import device

from data_provider.data_loader import DatasetTraj
from data_provider import file_loader
from torch.utils.data import DataLoader

from models.st_tokenizer import StTokenizer
from models.backbone import Backbone
from models.bigcity import BigCity

from utils.tools import EarlyStopping
from utils.scheduler import CosineLRScheduler

from tqdm import tqdm

def padding_mask(B, L):
    mask = torch.ones(B, L)
    num_mask = int(args.mask_rate * L)
    for i in range(B):
        indices_to_mask = torch.randperm(L, dtype=torch.long)[:num_mask]
        mask[i][indices_to_mask] = 0
    return mask.to(device), num_mask

def main():
    dataset = DatasetTraj()
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    bigcity = BigCity()
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    optimizer = torch.optim.Adam(bigcity.parameters(), lr=args.learning_rate,
                                    weight_decay=args.weight)
    lr_scheduler = CosineLRScheduler(
        optimizer, args.train_epochs, lr_min=0, decay_rate=0.1,
        warmup_t=40, warmup_lr_init=args.learning_rate / 20, t_in_epochs=True)

    
    for epoch in range(1, args.train_epochs + 1):
        print(f"epoch: {epoch}")
        train_loss = []
        for batch in tqdm(data_loader, desc="batch", total=len(data_loader)):
            
            batch_road_id, batch_time_id, batch_time_features, batch_road_flow = batch
            
            # to gpu
            batch_road_id = batch_road_id.to(device)
            batch_time_id = batch_time_id.to(device)
            batch_time_features = batch_time_features.to(device)
            batch_road_flow = batch_road_flow.to(device)

            B, L, N, Dtf = batch_road_id.shape[0], batch_road_id.shape[1], file_loader.road_cnt, 6
            
            # get mask
            mask, num_mask = padding_mask(B, L)
            
            # forward
            predict_road_id, predict_time_features, predict_road_flow = bigcity(
                batch_road_id, batch_time_id, batch_time_features, mask, num_mask
            )
            
            # get masked real value
            real_road_id = batch_road_id[mask == 0]
            real_time_features = batch_time_features[mask == 0]
            real_road_flow = batch_road_flow[mask == 0]

            # loss
            road_id_loss = F.cross_entropy(predict_road_id.view(-1, N), real_road_id)
            time_features_loss = F.mse_loss(predict_time_features.view(-1, Dtf), real_time_features)
            road_flow_loss = F.mse_loss(predict_road_flow.view(-1), real_road_flow)
            
            loss = road_id_loss * args.loss_alpha + time_features_loss * args.loss_beta + road_flow_loss * args.loss_gamma
            train_loss.append(loss.item())
            
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss_ave = np.average(train_loss)
        print(train_loss_ave)
        early_stopping(train_loss_ave, bigcity, args.checkpoints) # TODO
        lr_scheduler.step(epoch)
    
    
if __name__ == "__main__":
    print(device)
    try:
        main()
        pass
    except Exception as e:
        logging.error("\n" + traceback.format_exc())
