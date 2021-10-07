import torch

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn((n_samples, z_dim), device=device)
