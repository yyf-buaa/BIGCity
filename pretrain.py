import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import traceback
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import config.logging_config
import config.random_seed
from config.args_config import args
from config.global_vars import device

from data_provider.data_loader import DatasetTraj
from data_provider import file_loader

from models.st_tokenizer import StTokenizer
from models.backbone import Backbone
from models.bigcity import BigCity

from utils.tools import EarlyStopping
from utils.scheduler import CosineLRScheduler


dataset = DatasetTraj()
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
bigcity = BigCity()

early_stopping = EarlyStopping(patience=args.patience, verbose=True)
optimizer = torch.optim.Adam(bigcity.parameters(), lr=args.learning_rate, weight_decay=args.weight)
lr_scheduler = CosineLRScheduler(
    optimizer, args.train_epochs, lr_min=0, decay_rate=0.1,
    warmup_t=40, warmup_lr_init=args.learning_rate / 20, t_in_epochs=True)

# Initialize loss records at the beginning of the training loop
road_id_losses, time_features_losses, road_flow_losses, total_losses = [], [], [], []

def padding_mask(B, L):
    mask = torch.ones(B, L)
    num_mask = int(args.mask_rate * L)
    for i in range(B):
        indices_to_mask = torch.randperm(L, dtype=torch.long)[:num_mask]
        mask[i][indices_to_mask] = 0
    return mask.to(device), num_mask

def train():
    for epoch in range(1, args.train_epochs + 1):
        print(f"Epoch: {epoch}")
        train_loss = []
        for batch in tqdm(data_loader, desc="batch", total=len(data_loader)):
            batch_road_id, batch_time_id, batch_time_features, batch_road_flow = batch
            
            # Move data to GPU
            batch_road_id = batch_road_id.to(device)
            batch_time_id = batch_time_id.to(device)
            batch_time_features = batch_time_features.to(device)
            batch_road_flow = batch_road_flow.to(device)

            B, L, N, Dtf = batch_road_id.shape[0], batch_road_id.shape[1], file_loader.road_cnt, 6
            
            # Get mask
            mask, num_mask = padding_mask(B, L)
            
            # Forward pass
            predict_road_id, predict_time_features, predict_road_flow = bigcity(
                batch_road_id, batch_time_id, batch_time_features, mask, num_mask
            )
            
            # Get masked real values
            real_road_id = batch_road_id[mask == 0]
            real_time_features = batch_time_features[mask == 0]
            real_road_flow = batch_road_flow[mask == 0]

            # Calculate individual losses
            road_id_loss = F.cross_entropy(predict_road_id.view(-1, N), real_road_id)
            time_features_loss = F.mse_loss(predict_time_features.view(-1, Dtf), real_time_features)
            road_flow_loss = F.mse_loss(predict_road_flow.view(-1), real_road_flow)
            
            # Calculate total loss with scaling factors
            loss = road_id_loss * args.loss_alpha + time_features_loss * args.loss_beta + road_flow_loss * args.loss_gamma
            train_loss.append(loss.item())
            
            # Record the losses for each component
            road_id_losses.append(road_id_loss.item())
            time_features_losses.append(time_features_loss.item())
            road_flow_losses.append(road_flow_loss.item())
            total_losses.append(loss.item())
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Calculate average training loss for this epoch
        train_loss_ave = np.average(train_loss)
        print(f"Epoch {epoch}, Average Loss: {train_loss_ave}")

        # Early stopping and scheduler step
        early_stopping(train_loss_ave, bigcity, args.checkpoints)
        lr_scheduler.step(epoch)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': bigcity.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss }, os.path.join(args.checkpoints, f'checkpoint{epoch}.pth'))

def main():
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Generating loss plots...")

    finally:
        # After training or interruption, plot the losses
        plt.figure(figsize=(10, 6))

        plt.plot(total_losses, label='Total Loss', color='blue', linestyle='-', linewidth=2)
        plt.plot(road_id_losses, label='Road ID Loss', color='red', linestyle='--', linewidth=1.5)
        plt.plot(time_features_losses, label='Time Features Loss', color='green', linestyle=':', linewidth=1.5)
        plt.plot(road_flow_losses, label='Road Flow Loss', color='purple', linestyle='-.', linewidth=1.5)

        plt.xlabel('Iterations (batches)', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.title('Losses during Training', fontsize=14)
        plt.legend()
        plt.grid(True)

        plot_filename = "./image/training_losses.png"
        plt.savefig(plot_filename)
        print(f"Loss plot saved to {plot_filename}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())