import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spike_func import *


class Encoder(nn.Module):

    def __init__(self, device):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)

        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = input.shape[-1]
        c0_mem = c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        p0_mem = p0_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        c1_mem = c1_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        p1_mem = p1_spike = torch.zeros(shape[0], 64, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        c2_mem = c2_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        p2_mem = p2_spike = torch.zeros(shape[0], 128, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        c3_mem = c3_spike = torch.zeros(shape[0], 256, int(shape[2] / 16), int(shape[3] / 16), device=self.device)
        p3_mem = p3_spike = torch.zeros(shape[0], 256, int(shape[2] / 32), int(shape[3] / 32), device=self.device)
        output = torch.zeros(shape[0], 256,  int(shape[2] / 32), int(shape[3] / 32), n_step, device=self.device)
        for step in range(n_step):
            x = input[..., step]
            c0_mem, c0_spike = self.mem_update(self.conv0, x, c0_mem, c0_spike)
            p0_mem, p0_spike = self.mem_update_pool(F.avg_pool2d, c0_spike, p0_mem, p0_spike, _pool=2)
            p0_spike = F.dropout(p0_spike, p=0.5)
            c1_mem, c1_spike = self.mem_update(self.conv1, p0_spike, c1_mem, c1_spike)
            p1_mem, p1_spike = self.mem_update_pool(F.avg_pool2d, c1_spike, p1_mem, p1_spike, _pool=2)
            p1_spike = F.dropout(p1_spike, p=0.5)
            c2_mem, c2_spike = self.mem_update(self.conv2, p1_spike, c2_mem, c2_spike)
            p2_mem, p2_spike = self.mem_update_pool(F.avg_pool2d, c2_spike, p2_mem, p2_spike, _pool=2)
            p2_spike = F.dropout(p2_spike, p=0.5)
            c3_mem, c3_spike = self.mem_update(self.conv3, p2_spike, c3_mem, c3_spike)
            p3_mem, p3_spike = self.mem_update_pool(F.avg_pool2d, c3_spike, p3_mem, p3_spike, _pool=2)
            p3_spike = F.dropout(p3_spike, p=0.5)
            output[..., step] = p3_spike
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, _pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, _pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike
    
class Decoder(nn.Module):

    def __init__(self, device):
        super(Decoder, self).__init__()
        self.convTrans0 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0)  #  4 * 4
        self.convTrans1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0)  # 8 * 8
        self.convTrans2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=0)  # 16 * 16
        self.convTrans3 = nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=17, stride=1, padding=0)  # 32 * 32
        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = input.shape[-1]
        c0_mem = c0_spike = torch.zeros(shape[0], 128, shape[2] * 4, shape[3] * 4, device=self.device)
        c1_mem = c1_spike = torch.zeros(shape[0], 64, shape[2] * 8, shape[3] * 8, device=self.device)
        c2_mem = c2_spike = torch.zeros(shape[0], 64, shape[2] * 16, shape[3] * 16, device=self.device)
        output = torch.zeros(shape[0], 2, shape[2] * 32, shape[3] * 32, n_step, device=self.device)
        for step in range(n_step):
            x = input[..., step]
            c0_mem, c0_spike = self.mem_update(self.convTrans0, x, c0_mem, c0_spike)
            c0_spike = F.dropout(c0_spike, p=0.5)
            c1_mem, c1_spike = self.mem_update(self.convTrans1, c0_spike, c1_mem, c1_spike)
            c1_spike = F.dropout(c1_spike, p=0.5)
            c2_mem, c2_spike = self.mem_update(self.convTrans2, c1_spike, c2_mem, c2_spike)
            output[..., step] = self.convTrans3(c2_spike)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike


class AutoEncoder0(nn.Module):
    
    def __init__(self, device):
        super(AutoEncoder0, self).__init__()
        self.encoder = Encoder(device=device)
        self.decoder = Decoder(device=device)
        self.device = device
    
    def forward(self, input):
        f = self.encoder(input)
        output = self.decoder(f)
        loss = torch.mean((input - output) * (input - output))
        return loss
    