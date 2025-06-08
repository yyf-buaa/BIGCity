import logging
import traceback
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import label_binarize

from config.logging_config import init_logger, make_log_dir
from config.args_config import args

from data_provider.file_loader import file_loader
from data_provider.data_loader import DatasetNextHop, DatasetTrajClassify, DatasetTimeReg, DatasetTrafficStateReg, DatasetTrajRecover

from models.bigcity4finetune import BigCity4FineTune


def eval_next_hop(preds, labels):
    print(preds.shape, labels.shape)
    pred_proba = F.softmax(preds, dim=1)
    pred_class = torch.argmax(pred_proba, dim=1)
    
    acc = accuracy_score(labels, pred_class)
    
    mrr_at_5 = 0
    n = len(labels)
    for i in range(n):
        top_5_indices = torch.topk(pred_proba[i], 5).indices
        if labels[i] in top_5_indices:
            rank = (top_5_indices == labels[i]).nonzero(as_tuple=True)[0].item() + 1
            mrr_at_5 += 1 / rank
    mrr_at_5 /= n
    
    ndcg_at_5 = 0
    for i in range(n):
        top_5_indices = torch.topk(pred_proba[i], 5).indices
        top_5_probs = pred_proba[i][top_5_indices]
        dcg = 0
        for j, prob in enumerate(top_5_probs):
            if top_5_indices[j] == labels[i]:
                dcg += (2**1 - 1) / (j + 1)
        ideal_dcg = 0
        for j in range(5):
            ideal_dcg += (2**1 - 1) / (j + 1)
        ndcg = dcg / ideal_dcg if ideal_dcg != 0 else 0
        ndcg_at_5 += ndcg
    ndcg_at_5 /= n
    
    logging.info(f"NEXT_HOP ACC: {acc:.4f}, MRR@5: {mrr_at_5:.4f}, NDCG@5: {ndcg_at_5:.4f}")
    
    return acc, mrr_at_5, ndcg_at_5

def eval_traj_classify(preds, labels):
    pred_proba = F.softmax(preds, dim=1)
    pred_class = torch.argmax(pred_proba, dim=1)
    label_binarized = label_binarize(labels, classes=list(range(preds.shape[1])))
    
    acc = accuracy_score(labels, pred_class)
    
    f1 = f1_score(labels, pred_class, average='macro')
    
    auc = roc_auc_score(label_binarized, pred_proba, multi_class='ovr')
    
    logging.info(f"TRAJ_CLASSIFY ACC: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    return acc, f1, auc


def eval_time_reg(preds, labels):
    preds = preds.reshape(preds.shape[0], -1)
    labels = labels.reshape(labels.shape[0], -1)
    
    mae = mean_absolute_error(labels, preds)
    
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    
    mape = mean_absolute_percentage_error(labels, preds)
    
    logging.info(f"TIME_REG MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
    
    return mae, rmse, mape
    
def eval_traffic_state_reg(preds, labels):
    preds = preds.reshape(preds.shape[0], -1)
    labels = labels.reshape(labels.shape[0], -1)
    
    mae = mean_absolute_error(labels, preds)
    
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    
    mape = mean_absolute_percentage_error(labels, preds)
    
    logging.info(f"TRAFFIC_STATE_REG MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}")
    
    return mae, rmse, mape
    

def eval_traj_recover(preds, labels):
    preds = preds.reshape(-1, preds.shape[-1])
    labels = labels.reshape(-1)
    pred_proba = F.softmax(preds, dim=1)
    pred_class = torch.argmax(pred_proba, dim=1)
    
    acc = accuracy_score(labels, pred_class)
    
    f1 = f1_score(labels, pred_class, average='macro')
    
    logging.info(f"TRAJ_RECOVER ACC: {acc:.4f}, F1: {f1:.4f}")
    
    return acc, f1

def evaluate(device):
    file_loader.load_all()
    
    datasets = {
        "next_hop": DatasetNextHop(),
        "traj_classify": DatasetTrajClassify(),
        "time_reg": DatasetTimeReg(),
        "traffic_state_reg": DatasetTrafficStateReg(),
        "traj_recover": DatasetTrajRecover(),
    }
    
    test_datasets = {
        name: random_split(dataset, [len(dataset) - int(0.1 * len(dataset)), int(0.1 * len(dataset))])[1]
        for name, dataset in datasets.items()
    }
    
    dataloaders = {
        name: DataLoader(dataset, batch_size=1, shuffle=True)
        for name, dataset in test_datasets.items()
    }
    
    eval_funcs = {
        "next_hop": eval_next_hop,
        "traj_classify": eval_traj_classify,
        "time_reg": eval_time_reg,
        "traffic_state_reg": eval_traffic_state_reg,
        "traj_recover": eval_traj_recover,
    }
    
    bigcity = BigCity4FineTune(device, f"./checkpoints/{args.city}_finetune_1.pth").to(device)
    bigcity.eval()
    
    for task_name, dataloader in dataloaders.items():
        logging.info(f"Evaluating task: {task_name}")
        
        progress_bar = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            unit="batch"
        )
        
        all_preds, all_labels = [], []
        
        for batch_idx, (batch_road_id, batch_time_id, batch_time_features, batch_label) in progress_bar:
            progress_bar.set_description(f"Task: {task_name: <18}")
            
            batch_road_id = batch_road_id.to(device)
            batch_time_id = batch_time_id.to(device)
            batch_time_features = batch_time_features.to(device)    
            batch_label = batch_label.to(device)
            
            with torch.no_grad():
                _, batch_pred = bigcity(
                    task_name, batch_road_id, batch_time_id, batch_time_features, batch_label
                )
                all_preds.append(batch_pred.cpu())
                all_labels.append(batch_label.cpu())
        
        preds_tensor = torch.cat(all_preds, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        eval_funcs[task_name](preds_tensor, labels_tensor)


def main():
    project_name = "bigcity-dev" if args.develop else "bigcity"
    wandb.init(mode=args.wandb_mode, project=project_name, config=args, name=f"evaluate-{args.city}")
    
    log_dir = make_log_dir(args.log_path)
    init_logger(log_dir)
    
    device = torch.device("cpu" if args.device == "-1" and torch.cuda.is_available() else f"cuda:{args.device}")
    logging.info(f"Using device: {device}")
    
    try:
        evaluate(device)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    finally:
        logging.info(f"Saving losses to {log_dir}.")
        
        logging.info(f"Finishing training.")
        
    wandb.finish()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())