import torch
import torch.nn.functional as F

def rbf_kernel(x, y, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between x and y.
    x: (N, D)
    y: (M, D)
    Returns: (N, M)
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1) # (N, 1, D)
    y = y.unsqueeze(0) # (1, M, D)

    # ||x - y||^2
    dist = torch.sum((x - y) ** 2, dim=-1) # (N, M)
    return torch.exp(-dist / (2.0 * sigma * sigma))

def mmd_loss(x, y, sigma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of samples x and y using RBF kernel.
    x: (N, D)
    y: (M, D)
    """
    xx = rbf_kernel(x, x, sigma)
    yy = rbf_kernel(y, y, sigma)
    xy = rbf_kernel(x, y, sigma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

def huber_loss(pred, target, delta=1.0):
    """
    Computes Huber loss between predictions and targets.
    """
    return F.huber_loss(pred, target, delta=delta)
