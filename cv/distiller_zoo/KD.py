from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class CustomDistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(CustomDistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = (p_t * (p_t.log() - p_s)).sum(dim=1).mean(dim=0) * (self.T ** 2)
        return loss


if __name__ == '__main__':
    kl_1 = DistillKL(3)
    kl_2 = CustomDistillKL(3)
    student = torch.tensor([[3., 7., 1.], [4., 8., 5.], [0.25, 0.76, 0.2], [0.42, 0.99, 0.8]])
    teacher = torch.tensor([[4., 6., 5.], [2., 7., 6.], [0.28, 0.94, 0.1], [0.49, 0.75, 0.0]])
    print(kl_1(student, teacher), kl_2(student, teacher))
    assert kl_1(student, teacher) - kl_2(student, teacher) < 0.000001
