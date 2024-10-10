import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, x, y):
        inputs = x['mask']
        targets = y['mask']
        
        diff = torch.abs(inputs - targets)

        if self.beta > 0:
            loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)

        else:
            loss = diff

        return torch.mean(loss)