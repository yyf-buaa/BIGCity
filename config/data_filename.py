import os
from config.args_config import args

road_features_file = os.path.join(args.root_path, args.city, f"roadmap_{args.city}/roadmap_{args.city}.rel")

traj_file = os.path.join(args.root_path, args.city, f'traj_{args.city}_11.csv')
traj_file_short = os.path.join(args.root_path, args.city, f'traj_{args.city}_11_short.csv')