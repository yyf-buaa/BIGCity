import torch,logging
from config.global_vars import device
from config.args_config import args

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
    
def padding_mask(B, L):
    mask = torch.ones(B, L)
    num_mask = int(args.mask_rate * L)
    for i in range(B):
        indices_to_mask = torch.randperm(L, dtype=torch.long)[:num_mask]
        mask[i][indices_to_mask] = 0
    return mask.to(device), num_mask
