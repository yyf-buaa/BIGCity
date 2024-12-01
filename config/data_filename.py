import os
from config.args_config import args

road_relation_file = os.path.join(args.root_path, args.city, f"roadmap_{args.city}", f"roadmap_{args.city}.rel")

road_static_file = os.path.join(args.root_path, args.city, f"roadmap_{args.city}", f"road_features_{args.city}.csv")

road_dynamic_file = os.path.join(args.root_path, args.city, f'{args.city}.dyna')

traj_file = os.path.join(args.root_path, args.city, f'traj_{args.city}_11.csv')
traj_file_short = os.path.join(args.root_path, args.city, f'traj_{args.city}_11_short.csv')

