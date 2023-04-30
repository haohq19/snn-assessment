import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spike_func import *

class TETLoss(nn.Module):
    def __init__(self):
        super(TETLoss, self).__init__()

    def forward(self, pred, gt):
        n_step = pred.shape[2]
        loss = torch.zeros(0)
        for step in n_step:
            loss += nn.CrossEntropyLoss(pred[..., step], gt)
        return 1/n_step * loss