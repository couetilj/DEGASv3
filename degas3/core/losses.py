"""Loss functions: MMD, DEGAS rank loss, and the Cox negative partial log-likelihood."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Multi-bandwidth Gaussian kernel matrix over the stacked source and target."""
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    l2 = ((total0 - total1) ** 2).sum(2)
    bandwidth = fix_sigma if fix_sigma else torch.sum(l2.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidths = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    return sum(torch.exp(-l2 / b) for b in bandwidths) / len(bandwidths)


def MMD_loss(source, target, kernel_mul=2.0, kernel_num=20, fix_sigma=None):
    """Maximum mean discrepancy between two embedding batches of equal size."""
    n = int(source.size()[0])
    k = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    return torch.mean(k[:n, :n]) + torch.mean(k[n:, n:]) - torch.mean(k[:n, n:]) - torch.mean(k[n:, :n])


def Rmatrix(surv):
    """Risk-set indicator: out[i, j] = 1 when time_j >= time_i."""
    t = surv.cpu().detach().numpy().reshape(-1)
    return torch.tensor((t[None, :] >= t[:, None]).astype(int))


def rank_loss(hazard, time, censor, device):
    """Original DEGAS Cox-style loss (kept for ``loss_type='rank_loss'``)."""
    risk = Rmatrix(time).to(device)
    return -torch.mean(
        (hazard - torch.log(torch.sum(torch.exp(hazard).expand(time.shape[0], time.shape[0]) * risk, axis=1)))
        * censor
    )


class NegativeLogLikelihood(nn.Module):
    """Cox negative partial log-likelihood (after DeepSurv.pytorch)."""

    def __init__(self, device: torch.device | str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

    def forward(self, risk_pred, y, e, device=None):
        mask = torch.ones(y.shape[0], y.shape[0]).to(self.device)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        return -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)
