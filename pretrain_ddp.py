import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
import logging
import traceback
import numpy as np
import wandb
from tqdm import tqdm

import config.logging_config
import config.random_seed
from config.args_config import args
from config.global_vars import device as global_device
from data_provider.data_loader import DatasetTraj
from data_provider import file_loader
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


def parse_device(device_str):
    if device_str.strip() == "-1":
        return "cpu", None
    devices = device_str.split(",")
    if len(devices) == 1:
        return "single", int(devices[0])
    else:
        return "ddp", [int(d) for d in devices]


def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    current_device = torch.device(f"cuda:{args.local_rank}")
    return current_device


def train(current_device, device_type):
    dataset = DatasetTraj()

    if device_type == "ddp":
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    bigcity = BigCity().to(current_device)

    if device_type == "ddp":
        bigcity = DDP(bigcity, device_ids=[args.local_rank], output_device=args.local_rank)

    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(bigcity.parameters(), lr=args.learning_rate, weight_decay=args.weight)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.train_epochs * 0.1),
        num_training_steps=args.train_epochs
    )

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(1, args.train_epochs + 1):

        if device_type == "ddp":
            data_loader.sampler.set_epoch(epoch)

        progress_bar = tqdm(enumerate(data_loader),
                            total=len(data_loader),
                            desc=f"Epoch {epoch}/{args.train_epochs}",
                            unit="batch",
                            disable=(device_type == "ddp" and args.local_rank != 0)
                            )

        for batchidx, batch in progress_bar:
            batch_road_id, batch_time_id, batch_time_features, batch_road_flow = batch

            batch_road_id = batch_road_id.to(current_device)
            batch_time_id = batch_time_id.to(current_device)
            batch_time_features = batch_time_features.to(current_device)
            batch_road_flow = batch_road_flow.to(current_device)

            B, L, N, Dtf = batch_road_id.shape[0], batch_road_id.shape[1], file_loader.road_cnt, 6
            mask, num_mask = padding_mask(B, L)

            predict_road_id, predict_time_features, predict_road_flow = bigcity(
                batch_road_id, batch_time_id, batch_time_features, mask, num_mask
            )

            real_road_id = batch_road_id[mask == 0]
            real_time_features = batch_time_features[mask == 0]
            real_road_flow = batch_road_flow[mask == 0]

            road_id_loss = cross_entropy(predict_road_id.view(-1, N), real_road_id)
            time_features_loss = mse(predict_time_features.view(-1, Dtf), real_time_features)
            road_flow_loss = mse(predict_road_flow.view(-1), real_road_flow)

            loss = road_id_loss * args.loss_alpha + time_features_loss * args.loss_beta + road_flow_loss * args.loss_gamma

            losses["total"].append(loss.item())
            losses["road_id"].append(road_id_loss.item())
            losses["time_features"].append(time_features_loss.item())
            losses["road_flow"].append(road_flow_loss.item())

            if args.local_rank == 0 or device_type != "ddp":
                progress_bar.set_postfix({"loss": f"{loss.item():.2f}"})

                wandb.log({
                    "batch": batchidx,
                    "batch_total_loss": loss.item(),
                    "batch_road_id_loss": road_id_loss.item(),
                    "batch_time_features_loss": time_features_loss.item(),
                    "batch_road_flow_loss": road_flow_loss.item(),
                })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.local_rank == 0 or device_type != "ddp":
            data_loader_len = len(data_loader)
            epoch_loss_ave = np.mean(losses["total"][-data_loader_len:])
            epoch_road_id_loss_ave = np.mean(losses["road_id"][-data_loader_len:])
            epoch_time_features_loss_ave = np.mean(losses["time_features"][-data_loader_len:])
            epoch_road_flow_loss_ave = np.mean(losses["road_flow"][-data_loader_len:])

            wandb.log({
                "epoch_average_total_loss": epoch_loss_ave,
                "epoch_average_road_id_loss": epoch_road_id_loss_ave,
                "epoch_average_time_features_loss": epoch_time_features_loss_ave,
                "epoch_average_road_flow_loss": epoch_road_flow_loss_ave,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            torch.save({
                'epoch': epoch,
                'model_state_dict': bigcity.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, os.path.join(args.checkpoints, f'{args.city}_pretrain_checkpoint{epoch}.pth'))

            early_stopping(epoch_loss_ave, bigcity, args.checkpoints)

        scheduler.step()


def main():
    device_type, device_ids = parse_device(args.device)
    print(f"device type: {device_type} | args: {device_ids}")

    if device_type == "cpu":
        current_device = torch.device("cpu")
    elif device_type == "single":
        current_device = torch.device(f"cuda:{device_ids}")
    else:  # ddp
        current_device = setup_ddp()
    
    raise Exception("Stop here")
    

    if args.local_rank == 0 or device_type != "ddp":
        project_name = "bigcity-dev" if args.develop else "bigcity"
        wandb.init(mode="offline", project=project_name, config=args, name="pretrain")

    try:
        train(current_device, device_type)
    except KeyboardInterrupt:
        logging.info("\nTraining interrupted by user.")
    finally:
        if args.local_rank == 0 or device_type != "ddp":
            logging.info(f"Saving losses to {config.logging_config.cur_log_dir}.")
            save_loss_image(losses)
            save_losses_to_csv(losses)
            wandb.finish()
        logging.info(f"Training finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())
