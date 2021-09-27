from typing import List
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


def summary(layers: torch.nn.Module, in_size: List) -> None:
    r"""Pytorch model summary

    Args:
        layers (torch.nn.Module)
        in_size (List)
    ---------
    Examples 
    --------- ::

        net = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(28*28, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 10)
                    )
        summary(net, (1, 28, 28))
    """
    if len(in_size) == 3:
        in_size = (1, *in_size)
    X = torch.randn(size=in_size, dtype=torch.float32)
    print('input shape: \t', X.shape)

    for layer in layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape: \t', X.shape)