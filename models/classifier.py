import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spike_func import *


class RateCodingClassifier(nn.Module):
    def __init__(self, n_dim, n_class, device):
        super(RateCodingClassifier, self).__init__()
        self.n_class = n_class
        self.device = device
        self.classifier = nn.Linear(n_dim, n_class)
        self.spike_func = SpikeFunc.apply

    def forward(self, input):
        shape = input.shape
        n_step = input.shape[2]
        h0_mem = h0_spike = h0_sumspike = torch.zeros(shape[0], self.n_class, device=self.device)

        for step in range(n_step):
            x = input[..., step]
            h0_mem, h0_spike = self.mem_update(self.classifier, x, h0_mem, h0_spike)
            h0_sumspike += h0_spike

        output = h0_sumspike / n_step
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike


class TemporalCodingClassifier(nn.Module):
    def __init__(self, n_dim, n_class, device):
        super(TemporalCodingClassifier, self).__init__()
        self.n_class = n_class
        self.device = device
        self.classifier = nn.Linear(n_dim, n_class)
        self.spike_func = SpikeFunc.apply

    def forward(self, input):
        shape = input.shape
        n_step = input.shape[2]
        h0_mem = h0_spike = torch.zeros(shape[0], self.n_class, device=self.device)
        output = torch.zeros(shape[0], self.n_class, n_step, device=self.device)

        for step in range(n_step):
            x = input[..., step]
            h0_mem, h0_spike = self.mem_update(self.classifier, x, h0_mem, h0_spike)
            output[..., step] = h0_spike
        for step in range(n_step):
            output[..., step] = h0_spike
        return h0_spike

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike


class LinearClassifier(nn.Module):

    def __init__(self, n_dim, n_class, device):
        super(LinearClassifier, self).__init__()
        self.n_class = n_class
        self.device = device
        self.classifier = nn.Linear(n_dim, n_class)

    def forward(self, input):
        shape = input.shape
        n_step = input.shape[2]
        output = torch.zeros(shape[0], self.n_class, device=self.device)

        for step in range(n_step):
            output += self.classifier(input[..., step])

        return output/n_step



