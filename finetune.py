import os
import torch
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

from models.bigcity4finetune import BigCity4FineTune

from utils.tools import EarlyStopping
from utils.scheduler import CosineLRScheduler


dataset = DatasetTraj()
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
bigcity = BigCity4FineTune()

early_stopping = EarlyStopping(patience=args.patience, verbose=True)
optimizer = torch.optim.Adam(bigcity.parameters(), lr=args.learning_rate, weight_decay=args.weight)
lr_scheduler = CosineLRScheduler(
    optimizer, args.train_epochs, lr_min=0, decay_rate=0.1,
    warmup_t=40, warmup_lr_init=args.learning_rate / 20, t_in_epochs=True)

# Initialize loss records at the beginning of the training loop
total_losses = []


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
        epoch_loss = []
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
            loss = bigcity(
                batch_road_id, batch_time_id, batch_time_features, batch_road_flow, mask, num_mask
            )
            
            epoch_loss.append(loss.item())
            total_losses.append(loss.item())
            
            # Log losses to wandb
            wandb.log({
                "Batch": batchidx,
                "Batch Total Loss": loss.item(),
            })
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
        # Calculate average training loss for this epoch
        epoch_loss_ave = np.average(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch}, Average Loss: {epoch_loss_ave}")
        logging.info(f"Learning Rate: {current_lr}")
        wandb.log({
            "Epoch Average Total Loss": epoch_loss_ave,
            "Learning Rate": current_lr
        })
        
        # Save checkpoint
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
        logging.info(f"Loss plot saved to ./image/")
        logging.info(f"total_losses: {total_losses}")
        
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())