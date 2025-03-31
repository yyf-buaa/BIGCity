import numpy as np
import torch, logging
import matplotlib.pyplot as plt
import os
from config.args_config import args
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    # if args.lradj == 'type7':
    #     lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}
    # if args.lradj == 'type6':
    #     lr_adjust = {epoch: args.learning_rate * (0.6 ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    # if epoch in lr_adjust.keys():
    lr = None
    for param_group in optimizer.param_groups:
        # import pdb
        # pdb.set_trace()
        lr = param_group['lr']*0.9
        param_group['lr'] = lr
    logging.info('Updating learning rate to {}'.format(lr))
    
def log_gradients(model, writer, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, global_step=step)

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        return lr


class EarlyStopping:
    def __init__(self, task_name, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.task_name = task_name

    def __call__(self, val_loss, model, optimizer, epoch):
        if epoch % 5 == 0:
            self.save_checkpoint(model, optimizer, epoch, f'{epoch}')
        if epoch == args.train_epochs:
            self.save_checkpoint(model, optimizer, epoch, 'final')
        
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, optimizer, epoch, "best", val_loss)
            self.adj_lr = False
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}, loss: {val_loss:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, optimizer, epoch, "best", val_loss)
            self.counter = 0
            self.early_stop = False
            
    def save_checkpoint(self, model, optimizer, epoch, name, val_loss=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.checkpoint_path, f'{args.city}_{self.task_name}_{name}.pth'))
        if val_loss is not None:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
