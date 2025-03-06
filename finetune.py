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

from data_provider import file_loader
from data_provider.data_loader import DatasetNextHop, DatasetTrajClassify, DatasetTimeReg, DatasetTrafficStateReg, DatasetTrajRecover

from models.bigcity4finetune import BigCity4FineTune

from utils.tools import EarlyStopping
from utils.scheduler import CosineLRScheduler
from utils.round_iterator import RoundRobinIterator


datasets = {
    "next_hop": DatasetNextHop(),
    "traj_classify": DatasetTrajClassify(),
    "time_reg": DatasetTimeReg(),
    "traffic_state_reg": DatasetTrafficStateReg(),
    "traj_recover": DatasetTrajRecover()
}

total_losses = {
    "next_hop": [],
    "traj_classify": [],
    "time_reg": [],
    "traffic_state_reg": [],
    "traj_recover": []
}

dataloaders = {
    name: DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for name, dataset in datasets.items()
}
bigcity = BigCity4FineTune()

early_stopping = EarlyStopping(patience=args.patience, verbose=True)
optimizer = torch.optim.Adam(bigcity.parameters(), lr=args.learning_rate, weight_decay=args.weight)
lr_scheduler = CosineLRScheduler(
    optimizer, args.train_epochs, lr_min=0, decay_rate=0.1,
    warmup_t=40, warmup_lr_init=args.learning_rate / 20, t_in_epochs=True)


def train():
    for epoch in range(1, args.train_epochs + 1):
        iterator = RoundRobinIterator(dataloaders)
        
        epoch_losses = {
            "next_hop": [],
            "traj_classify": [],
            "time_reg": [],
            "traffic_state_reg": [],
            "traj_recover": []
        }
        
        progress_bar = tqdm(
            enumerate(iterator), 
            total=len(iterator), 
            desc=f"Epoch {epoch}/{args.train_epochs}", 
            unit="batch"
        )
        
        for batch_idx, (task_name, (batch_road_id, batch_time_id, batch_time_features, batch_label)) in progress_bar:
            progress_bar.set_description(f"Epoch {epoch}/{args.train_epochs} - Task: {task_name: <18}")
            
            # Move data to GPU
            batch_road_id = batch_road_id.to(device)
            batch_time_id = batch_time_id.to(device)
            batch_time_features = batch_time_features.to(device)    
            batch_label = batch_label.to(device)
            
            # Forward pass (loss has been calculated)
            loss = bigcity(
                task_name, batch_road_id, batch_time_id, batch_time_features, batch_label
            )
            
            # Record loss
            epoch_losses[task_name].append(loss.item())
            total_losses[task_name].append(loss.item())
            
            # Log losses to wandb
            wandb.log({f"{task_name}_batch_loss": loss.item()})
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate average training loss for this epoch
        average_losses = {f"{task_name}_epoch_average_loss": np.average(task_losses) 
                          for task_name, task_losses in epoch_losses.items()}
        print(average_losses)
        wandb.log(average_losses)
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': bigcity.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss }
            , os.path.join(args.checkpoints, f'{args.city}_finetune_checkpoint{epoch}.pth'))
        
        # Step the learning rate scheduler
        lr_scheduler.step(epoch)

def main():
    wandb.init(mode="offline", project="bigcity", config=args, name="pretrain")

    try:
        train()
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user.")
    finally:
        logging.info(f"total_losses: {total_losses}")
        
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())