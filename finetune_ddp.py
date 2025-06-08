import os
import logging
import traceback
import numpy as np
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

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

def setup_ddp(rank, world_size, device_ids):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(device_ids[rank])

def cleanup_ddp():
    dist.destroy_process_group()

def train(rank, world_size, device_ids, log_dir):

    setup_ddp(rank, world_size, device_ids)
    
    if rank == 0:
        project_name = "bigcity-dev" if args.develop else "bigcity"
        wandb.init(mode=args.wandb_mode, project=project_name, config=args, name=f"finetune-{args.city}")
    
    init_logger(log_dir)
    
    device = torch.device(f"cuda:{device_ids[rank]}")
    
    file_loader.load_all(rank)

    datasets = {
        "next_hop": DatasetNextHop(),
        "traj_classify": DatasetTrajClassify(),
        "time_reg": DatasetTimeReg(),
        "traffic_state_reg": DatasetTrafficStateReg(),
        "traj_recover": DatasetTrajRecover(),
    }
    
    dataloaders = {}
    for name, dataset in datasets.items():
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)
        dataloaders[name] = data_loader
    
    
    bigcity = BigCity4FineTune(device).to(device)
    bigcity = nn.parallel.DistributedDataParallel(bigcity, device_ids=[device_ids[rank]], find_unused_parameters=True)
    
    optimizer = torch.optim.Adam(bigcity.parameters(), lr=args.learning_rate, weight_decay=args.weight)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.train_epochs * 0.1), 
        num_training_steps=args.train_epochs
    )
    
    early_stopping = EarlyStopping("finetune", patience=args.patience, verbose=True)
    
    data_loader_len = file_loader.get_traj_cnt()
    try:
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
                loss, _ = bigcity(
                    task_name, batch_road_id, batch_time_id, batch_time_features, batch_label
                )
                
                # Record the losses for each component
                losses[task_name].append(loss.item())
                
                # Log losses to wandb
                if rank == 0:
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

            if rank == 0:
                logging.info(average_losses)
                wandb.log(average_losses)
                
                early_stopping.save_checkpoint(bigcity, optimizer, epoch, f"{epoch}")
                
            scheduler.step()
    
    finally:
        if rank == 0:
            logging.info(f"Saving losses to {log_dir}.")
            save_loss_image(losses, log_dir)
            save_losses_to_csv(losses, log_dir)
            
            wandb.finish()
        
        cleanup_ddp()

def main():
    log_dir = make_log_dir(args.log_path, args.checkpoint_path)
    init_logger(log_dir)
    
    try:
        device_ids = [int(x) for x in args.device.split(',')]
        world_size = len(device_ids)
        if args.device == '-1':
            logging.error("CPU training is not supported.")
            return
        elif world_size == 1:
            logging.info(f"Single GPU training is not supported, use pretrain.py instead.")
            return
        else:
            logging.info(f"Using devices: {device_ids}")
            mp.spawn(train, args=(world_size, device_ids, log_dir), nprocs=world_size, join=True)
    
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    
    finally:
        logging.info(f"Finishing training.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())