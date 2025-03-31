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
from data_provider.data_loader import DatasetTraj

from models.bigcity import BigCity

from utils.tools import EarlyStopping
from utils.masking import padding_mask
from utils.plot_losses import save_loss_image, save_losses_to_csv

losses = {
    "total": [],
    "road_id": [],
    "time_features": [],
    "road_flow": []
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
        wandb.init(mode=args.wandb_mode, project=project_name, config=args, name="pretrain")
    
    init_logger(log_dir)
    
    device = torch.device(f"cuda:{device_ids[rank]}")
    
    file_loader.load_all(rank)

    dataset = DatasetTraj()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)
    
    bigcity = BigCity(device).to(device)
    bigcity = nn.parallel.DistributedDataParallel(bigcity, device_ids=[device_ids[rank]], find_unused_parameters=True)
    
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(bigcity.parameters(), lr=args.learning_rate, weight_decay=args.weight)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.train_epochs * 0.1), 
        num_training_steps=args.train_epochs
    )
    
    early_stopping = EarlyStopping("pretrain", patience=args.patience, verbose=True)
    
    
    data_loader_len = len(data_loader)
    try:
        for epoch in range(1, args.train_epochs + 1):
            
            progress_bar = tqdm(enumerate(data_loader), 
                                total=len(data_loader), 
                                desc=f"Rank {rank} - Epoch {epoch}/{args.train_epochs}", 
                                unit="batch",
                                disable=(rank != 0),
            )
            
            for batchidx, batch in progress_bar:
                batch_road_id, batch_time_id, batch_time_features, batch_road_flow = batch
                
                # Move data to GPU
                batch_road_id = batch_road_id.to(device)
                batch_time_id = batch_time_id.to(device)
                batch_time_features = batch_time_features.to(device)
                batch_road_flow = batch_road_flow.to(device)
                

                B, L, N, Dtf = batch_road_id.shape[0], batch_road_id.shape[1], file_loader.get_road_cnt(), 6
                
                # Get mask
                mask, num_mask = padding_mask(B, L)
                mask = mask.to(device)
                
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
                            
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                progress_bar.set_postfix({"loss": f"{loss.item():.2f}"}) 
                
                # Log losses to wandb
                if rank == 0:
                    wandb.log({
                        "batch": batchidx,
                        "batch_total_loss": loss.item(),
                        "batch_road_id_loss": road_id_loss.item(),
                        "batch_time_features_loss": time_features_loss.item(),
                        "batch_road_flow_loss": road_flow_loss.item(),
                    })                        

            # Calculate average training loss for this epoch
            epoch_loss_ave = np.mean(losses["total"][-data_loader_len:])
            epoch_road_id_loss_ave = np.mean(losses["road_id"][-data_loader_len:])
            epoch_time_features_loss_ave = np.mean(losses["time_features"][-data_loader_len:])
            epoch_road_flow_loss_ave = np.mean(losses["road_flow"][-data_loader_len:])
            
            if rank == 0:
                # Log average losses to wandb
                wandb.log({
                    "epoch_average_total_loss": epoch_loss_ave,
                    "epoch_average_road_id_loss": epoch_road_id_loss_ave,
                    "epoch_average_time_features_loss": epoch_time_features_loss_ave,
                    "epoch_average_road_flow_loss": epoch_road_flow_loss_ave,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                early_stopping(epoch_loss_ave, bigcity, optimizer, epoch)
                
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