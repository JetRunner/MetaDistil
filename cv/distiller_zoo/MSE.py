from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


class MSEWithTemperature(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(MSEWithTemperature, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = y_s / self.T
        p_t = y_t / self.T
        loss = F.mse_loss(p_s, p_t)
        return loss
