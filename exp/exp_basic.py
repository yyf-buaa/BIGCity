import os
import torch
import logging
from models import bigcity
from models import GPT4Finetune
from models import LLAMA2_7B
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'GPT4TS': bigcity,
            'GPT4Finetune': GPT4Finetune,
            'LLAMA':LLAMA2_7B
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu)) 
            logging.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            logging.info('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
