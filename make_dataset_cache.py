from config.args_config import args
from config import global_vars
from config.logging_config import init_logger, make_log_dir

from data_provider.file_loader import file_loader
from data_provider.data_loader import DatasetTraj, DatasetNextHop, DatasetTrajClassify, DatasetTimeReg, DatasetTrafficStateReg, DatasetTrajRecover

import os
import logging
from multiprocessing import Pool

def construct_dataset(dataset_class):
    dataset_instance = dataset_class()
    print(f"Constructed {dataset_class.__name__}")

def main():
    
    log_dir = make_log_dir()
    init_logger(log_dir)
    
    files = [
        ("Road Relation File", global_vars.road_relation_file),
        ("Road Relation Tensor File", global_vars.road_relation_tensor_file),
        ("Road Static File", global_vars.road_static_file),
        ("Road Static Tensor File", global_vars.road_static_tensor_file),
        ("Road Dynamic File", global_vars.road_dynamic_file),
        ("Road Dynamic Tensor File", global_vars.road_dynamic_tensor_file),
        ("Dataset Meta File", global_vars.dataset_meta_file),
        ("Trajectory File", global_vars.traj_file),
        ("Trajectory Short File", global_vars.traj_file_short),
    ]

    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    for name, file_path in files:
        if os.path.exists(file_path):
            logging.info(f"{name} {GREEN}exists{RESET}: {file_path}")
        else:
            logging.info(f"{name} {RED}does not exist{RESET}: {file_path}")
    
    file_loader.load_all()
    
    dataset_classes = [
        DatasetTraj,
        DatasetNextHop,
        DatasetTrajClassify,
        DatasetTimeReg,
        DatasetTrafficStateReg,
        DatasetTrajRecover,
    ]
    
    print(len(dataset_classes))

    with Pool(processes=len(dataset_classes)) as pool:
        pool.map(construct_dataset, dataset_classes)

if __name__ == "__main__":
    main()