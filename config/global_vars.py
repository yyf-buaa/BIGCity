import os
import torch
import pandas as pd
from config.args_config import args

device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
road_relation_file = os.path.join(args.root_path, args.city, f"roadmap_{args.city}", f"roadmap_{args.city}.rel")

road_static_file = os.path.join(args.root_path, args.city, f"roadmap_{args.city}", f"road_features_{args.city}.csv")
road_static_tensor_file = os.path.join(args.root_path, args.city, f"roadmap_{args.city}", f"road_features_{args.city}.pth")

road_dynamic_file = os.path.join(args.root_path, args.city, f'{args.city}.dyna')
road_dynamic_tensor_file = os.path.join(args.root_path, args.city, f'{args.city}.pth')

traj_file = os.path.join(args.root_path, args.city, f'traj_{args.city}_11.csv')
traj_file_short = os.path.join(args.root_path, args.city, f'traj_{args.city}_11_short.csv')

road_dynamic_embedding_file = os.path.join(args.root_path, args.city, "road_dyna_embedding.npy")

start_time = pd.to_datetime("2018-10-01T00:00:00Z")
end_time = pd.to_datetime("2018-11-30T23:30:00Z")
interval = 1800
