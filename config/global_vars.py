import os
import torch
import pandas as pd
from config.args_config import args

device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")

city_root_path = os.path.join(args.root_path, args.city)

road_relation_file = os.path.join(city_root_path, f"roadmap_{args.city}", f"roadmap_{args.city}.rel")
road_relation_tensor_file = os.path.join(city_root_path, f'cached_{args.city}_relation.pth')

road_static_file = os.path.join(city_root_path, f"roadmap_{args.city}", f"road_features_{args.city}.csv")
road_static_tensor_file = os.path.join(city_root_path, f'cached_{args.city}_static.pth')

road_dynamic_file = os.path.join(city_root_path, f'{args.city}.dyna')
road_dynamic_tensor_file = os.path.join(city_root_path, f'cached_{args.city}_dynamic.pth')

traj_file = os.path.join(city_root_path, f'traj_{args.city}_11.csv')
traj_file_short = os.path.join(city_root_path, f'traj_{args.city}_11_short.csv')
cur_traj_file = traj_file_short if args.develop else traj_file

def generate_cached_file_name(dataset_name, is_short=False):
    suffix = "_short" if is_short else ""
    return os.path.join(city_root_path, f"cached_{dataset_name}_dataset{suffix}.pth")

datasets = ["traj", "next_hop", "traj_classify", "time_reg", "traffic_state_reg", "traj_recover"]
cached_files = {dataset: generate_cached_file_name(dataset, args.develop) for dataset in datasets}

cached_traj_dataset = cached_files["traj"]
cached_next_hop_dataset = cached_files["next_hop"]
cached_traj_classify_dataset = cached_files["traj_classify"]
cached_time_reg_dataset = cached_files["time_reg"]
cached_traffic_state_reg_dataset = cached_files["traffic_state_reg"]
cached_traj_recover_dataset = cached_files["traj_recover"]

road_dynamic_embedding_file = os.path.join(city_root_path, "road_dyna_embedding.npy")

start_time = pd.to_datetime("2018-10-01T00:00:00Z")
end_time = pd.to_datetime("2018-11-30T23:30:00Z")
interval = 1800
