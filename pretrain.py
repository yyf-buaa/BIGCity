import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
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
from utils.masking import padding_mask
from utils.plot_losses import save_loss_image, save_losses_to_csv

losses = {
    "total": [],
    "road_id": [],
    "time_features": [],
    "road_flow": []
}


def train():
    dataset = DatasetTraj()
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    bigcity = BigCity()
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(bigcity.parameters(), lr=args.learning_rate, weight_decay=args.weight)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.train_epochs * 0.1), 
        num_training_steps=args.train_epochs
    )
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    data_loader_len = len(data_loader)
    
    
    for epoch in range(1, args.train_epochs + 1):
        
        progress_bar = tqdm(enumerate(data_loader), 
                            total=len(data_loader), 
                            desc=f"Epoch {epoch}/{args.train_epochs}", 
                            unit="batch"
        )
        
        for batchidx, batch in progress_bar:
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
            
            # Record the losses for each component
            losses["total"].append(loss.item())
            losses["road_id"].append(road_id_loss.item())
            losses["time_features"].append(time_features_loss.item())
            losses["road_flow"].append(road_flow_loss.item())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.2f}"}) 
            
            # Log losses to wandb
            wandb.log({
                "batch": batchidx,
                "batch_total_loss": loss.item(),
                "batch_road_id_loss": road_id_loss.item(),
                "batch_time_features_loss": time_features_loss.item(),
                "batch_road_flow_loss": road_flow_loss.item(),
            })
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate average training loss for this epoch
        epoch_loss_ave = np.mean(losses["total"][-data_loader_len:])
        epoch_road_id_loss_ave = np.mean(losses["road_id"][-data_loader_len:])
        epoch_time_features_loss_ave = np.mean(losses["time_features"][-data_loader_len:])
        epoch_road_flow_loss_ave = np.mean(losses["road_flow"][-data_loader_len:])
        
        # Log average losses to wandb
        wandb.log({
            "epoch_average_total_loss": epoch_loss_ave,
            "epoch_average_road_id_loss": epoch_road_id_loss_ave,
            "epoch_average_time_features_loss": epoch_time_features_loss_ave,
            "epoch_average_road_flow_loss": epoch_road_flow_loss_ave,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # Save checkpoint & loss plot
        torch.save({
            'epoch': epoch,
            'model_state_dict': bigcity.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss }, os.path.join(args.checkpoints, f'{args.city}_pretrain_checkpoint{epoch}.pth'))

        # Early stopping and lr scheduler step
        early_stopping(epoch_loss_ave, bigcity, args.checkpoints)
        scheduler.step()

def main():
    project_name = "bigcity-dev" if args.develop else "bigcity"
    
    wandb.init(mode="online", project=project_name, config=args, name="pretrain")

    try:
        train()
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user.")
    finally:
        logging.info(f"Saving losses to {config.logging_config.cur_log_dir}.")
        save_loss_image(losses)
        save_losses_to_csv(losses)
        
        logging.info(f"Finishing training.")
        
    wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())