import logging
import os
from .args_config import args
from datetime import datetime

log_dir = "./log"
checkpoints_dir = args.checkpoints
plot_dir = "./plot"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
    
# if not os.path.exists(plot_dir):
#     os.makedirs(plot_dir)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_dir, f"{timestamp}.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
