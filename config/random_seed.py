import torch
import random
import numpy as np

fix_seed = 2021

torch.manual_seed(fix_seed)

random.seed(fix_seed)

np.random.seed(fix_seed)

torch.cuda.manual_seed(fix_seed)