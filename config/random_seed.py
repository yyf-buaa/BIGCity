import torch
import random
import numpy as np

fix_seed = 2021

torch.manual_seed(fix_seed)

random.seed(fix_seed)

np.random.seed(fix_seed)

torch.cuda.manual_seed(fix_seed)

def set_random_seeds(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False