import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

import config.logging_config
import config.random_seed
from config.args_config import args
from config.global_vars import device

from data_provider.data_loader import DatasetTraj
from data_provider import file_loader

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
mse = nn.MSELoss()
cross_entropy = nn.CrossEntropyLoss()

# Initialize loss records at the beginning of the training loop
road_id_losses, time_features_losses, road_flow_losses, total_losses = [], [], [], []


def save_loss_image():
    # After training or interruption, plot the losses
    # Plot individual loss graphs
    loss_names = ["Total Loss", "Road ID Loss", "Time Features Loss", "Road Flow Loss"]
    loss_data = [total_losses, road_id_losses, time_features_losses, road_flow_losses]
    loss_colors = ["blue", "red", "green", "purple"]
    loss_styles = ["-", "--", ":", "-."]

    for i, (name, data, color, style) in enumerate(zip(loss_names, loss_data, loss_colors, loss_styles)):
        plt.figure(figsize=(12, 8))
        plt.plot(data, label=name, color=color, linestyle=style, linewidth=1.5)
        plt.xlabel("Iterations (batches)", fontsize=12)
        plt.ylabel("Loss Value", fontsize=12)
        plt.title(f"{name} during Training", fontsize=14)
        plt.legend()
        plt.grid(True)
        
        # Save each plot
        plot_filename = f"./image/{name.replace(' ', '_').lower()}.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to avoid overlap or memory issues

def padding_mask(B, L):
    mask = torch.ones(B, L)
    num_mask = int(args.mask_rate * L)
    for i in range(B):
        indices_to_mask = torch.randperm(L, dtype=torch.long)[:num_mask]
        mask[i][indices_to_mask] = 0
    return mask.to(device), num_mask

def train():
    for epoch in range(1, args.train_epochs + 1):
        logging.info(f"Epoch: {epoch}")
        epoch_loss, epoch_road_id_loss, epoch_time_loss, epoch_flow_loss = [], [], [], []
        for batchidx, batch in enumerate(tqdm(data_loader, desc="batch", total=len(data_loader))):
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
            road_id_loss = cross_entropy(predict_road_id.view(-1, N), real_road_id)
            time_features_loss = mse(predict_time_features.view(-1, Dtf), real_time_features)
            road_flow_loss = mse(predict_road_flow.view(-1), real_road_flow)
            
            # Calculate total loss with scaling factors
            loss = road_id_loss * args.loss_alpha + time_features_loss * args.loss_beta + road_flow_loss * args.loss_gamma
            epoch_loss.append(loss.item())
            epoch_road_id_loss.append(road_id_loss.item())
            epoch_time_loss.append(time_features_loss.item())
            epoch_flow_loss.append(road_flow_loss.item())
            
            # Record the losses for each component
            total_losses.append(loss.item())
            road_id_losses.append(road_id_loss.item())
            time_features_losses.append(time_features_loss.item())
            road_flow_losses.append(road_flow_loss.item())
            
            # Log losses to wandb
            wandb.log({
                "Batch": batchidx,
                "Batch Total Loss": loss.item(),
                "Batch Road ID Loss": road_id_loss.item(),
                "Batch Time Features Loss": time_features_loss.item(),
                "Batch Road Flow Loss": road_flow_loss.item(),
            })
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
        # Calculate average training loss for this epoch
        epoch_loss_ave, epoch_road_id_loss_ave, epoch_time_loss_ave, epoch_flow_loss_ave = np.average(
            epoch_loss), np.average(epoch_road_id_loss), np.average(epoch_time_loss), np.average(epoch_flow_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch}, Average Loss: {epoch_loss_ave}")
        logging.info(f"Learning Rate: {current_lr}")
        wandb.log({
            "Epoch Average Total Loss": epoch_loss_ave,
            "Epoch Average Road ID Loss": epoch_road_id_loss_ave,
            "Epoch Average Time Features Loss": epoch_time_loss_ave,
            "Epoch Average Road Flow Loss": epoch_flow_loss_ave,
            "Learning Rate": current_lr
        })
        
        # Save checkpoint & loss plot
        save_loss_image()
        torch.save({
            'epoch': epoch,
            'model_state_dict': bigcity.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss }, os.path.join(args.checkpoints, f'{args.city}_checkpoint{epoch}.pth'))

        # Early stopping and scheduler step
        early_stopping(epoch_loss_ave, bigcity, args.checkpoints)
        lr_scheduler.step(epoch)
    

def main():
    wandb.init(mode="offline", project="bigcity", config=args, name="pretrain")

    try:
        train()
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user. Generating loss plots...")
    finally:
        save_loss_image()
        
        logging.info(f"Loss plot saved to ./image/")
        logging.info(f"total_losses: {total_losses}")
        logging.info(f"road_id_losses: {road_id_losses}")
        logging.info(f"time_features_losses: {time_features_losses}")
        logging.info(f"road_flow_losses: {road_flow_losses}")
        
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())