from config.args_config import args
from config.logging_config import init_logger

from data_provider.file_loader import file_loader
from data_provider.data_loader import DatasetTraj, DatasetNextHop, DatasetTrajClassify, DatasetTimeReg, DatasetTrafficStateReg, DatasetTrajRecover

import multiprocessing
from multiprocessing import Pool

def construct_dataset(dataset_class):
    dataset_instance = dataset_class()
    print(f"Constructed {dataset_class.__name__}")

def main():
    
    init_logger()
    
    file_loader.load_all()
    
    dataset_classes = [
        DatasetTraj,
        DatasetNextHop,
        DatasetTrajClassify,
        DatasetTimeReg,
        DatasetTrafficStateReg,
        DatasetTrajRecover
    ]
    
    print(len(dataset_classes))

    with Pool(processes=len(dataset_classes)) as pool:
        pool.map(construct_dataset, dataset_classes)

if __name__ == "__main__":
    main()