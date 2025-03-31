import os
import logging
import traceback
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from config.logging_config import init_logger, make_log_dir
from config.args_config import args

from data_provider.file_loader import file_loader
from data_provider.data_loader import DatasetNextHop, DatasetTrajClassify, DatasetTimeReg, DatasetTrafficStateReg, DatasetTrajRecover

from models.bigcity4finetune import BigCity4FineTune

from utils.tools import EarlyStopping
from utils.round_iterator import RoundRobinIterator
from utils.plot_losses import save_loss_image, save_losses_to_csv


losses = {
    "next_hop": [],
    "traj_classify": [],
    "time_reg": [],
    "traffic_state_reg": [],
    "traj_recover": []
}


def train(device):
    file_loader.load_all()
    
    datasets = {
        "next_hop": DatasetNextHop(),
        "traj_classify": DatasetTrajClassify(),
        "time_reg": DatasetTimeReg(),
        "traffic_state_reg": DatasetTrafficStateReg(),
        "traj_recover": DatasetTrajRecover(),
    }
    
    dataloaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for name, dataset in datasets.items()
    }
    
    bigcity = BigCity4FineTune(device).to(device)
    
    optimizer = torch.optim.Adam(bigcity.parameters(), lr=args.learning_rate, weight_decay=args.weight)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.train_epochs * 0.1), 
        num_training_steps=args.train_epochs
    )
    
    early_stopping = EarlyStopping("finetune", patience=args.patience, verbose=True)
    
    data_loader_len = file_loader.get_traj_cnt()
    for epoch in range(1, args.train_epochs + 1):
        iterator = RoundRobinIterator(dataloaders)
        
        progress_bar = tqdm(
            enumerate(iterator), 
            total=len(iterator), 
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
            losses[task_name].append(loss.item())
            
            # Log losses to wandb
            wandb.log({
                "batch": batch_idx,
                f"{task_name}_batch_loss": loss.item()
            })
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate average training loss for this epoch
        average_losses = {f"{task_name}_epoch_average_loss": np.mean(task_losses[-data_loader_len:]) 
                          for task_name, task_losses in losses.items()}
        average_losses["learning_rate"] = optimizer.param_groups[0]['lr']
        logging.info(average_losses)
        wandb.log(average_losses)
        
        # Step the learning rate scheduler
        early_stopping.save_checkpoint(bigcity, optimizer, epoch, f"{epoch}")
        scheduler.step(epoch)

def main():
    project_name = "bigcity-dev" if args.develop else "bigcity"
    wandb.init(mode=args.wandb_mode, project=project_name, config=args, name="finetune")
    
    log_dir = make_log_dir(args.log_path)
    init_logger(log_dir)
    
    device = torch.device("cpu" if args.device == "-1" and torch.cuda.is_available() else f"cuda:{args.device}")
    logging.info(f"Using device: {device}")
    
    try:
        train(device)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    finally:
        logging.info(f"Saving losses to {log_dir}.")
        save_loss_image(losses, log_dir)
        save_losses_to_csv(losses, log_dir)
        
        logging.info(f"Finishing training.")
        
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())