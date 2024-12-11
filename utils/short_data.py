from ..config import args_config
import os
import pandas as pd

if __name__ == "__main__":
    args = args_config.args
    
    traj_file_name = os.path.join(args.root_path, args.city, f'traj_{args.city}_11.csv')
    
    traj_data = pd.read_csv(traj_file_name, delimiter=';')
    
    print(traj_data.head())
    
    traj_raw_short = traj_data.head(100)
    
    traj_short_file_name = os.path.join(args.root_path, args.city, f'traj_{args.city}_11_short.csv')
    
    traj_raw_short.to_csv(traj_short_file_name, index=False, sep=';')