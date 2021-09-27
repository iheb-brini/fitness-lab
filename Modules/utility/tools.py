import torch

def try_gpu(i=0): 
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus(): 
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    nb_gpu = torch.cuda.device_count()
    if nb_gpu > 0:
        return [torch.device(f'cuda:{i}') for i in range(nb_gpu)]
    return [torch.device('cpu'), ]
