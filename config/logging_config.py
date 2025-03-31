import logging
import os
from datetime import datetime
import torch.distributed as dist

from config.args_config import args

def make_log_dir(log_dir=args.log_path, checkpoints_dir=args.checkpoint_path):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
            
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cur_log_dir = os.path.join(log_dir, timestamp)
    
    return cur_log_dir

def init_logger(cur_log_dir: str):

    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    
    if rank == 0:
        os.makedirs(cur_log_dir, exist_ok=True)

        log_filename = os.path.join(cur_log_dir, f"{os.path.basename(cur_log_dir)}.log")

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            f"%(asctime)s - rank{rank} - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            f"%(asctime)s - rank{rank} - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        logging.getLogger().setLevel(logging.WARNING)