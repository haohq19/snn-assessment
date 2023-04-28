import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spike_func import *
    
class SCNN2(nn.Module):

    def __init__(self, device):
        super(SCNN2, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        # torch.cuda.empty_cache()
        shape = input.shape
        n_step = input.shape[4]
        c0_mem = c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        c1_mem = c1_spike = torch.zeros(shape[0], 128, shape[2], shape[3], device=self.device)
        p1_mem = p1_spike = torch.zeros(shape[0], 128, int(shape[2]/2), int(shape[3]/2), device=self.device)
        c2_mem = c2_spike = torch.zeros(shape[0], 256, int(shape[2]/2), int(shape[3]/2), device=self.device)
        p2_mem = p2_spike = torch.zeros(shape[0], 256, int(shape[2]/4), int(shape[3]/4), device=self.device)
        output = torch.zeros(shape[0], 256 * int(shape[2]/4) * int(shape[3]/4), n_step, device=self.device)
        for step in range(n_step):
            x = input[..., step]
            c0_mem, c0_spike = self.mem_update(self.conv0, x, c0_mem, c0_spike)
            c0_spike = F.dropout(c0_spike, p=0.5)
            c1_mem, c1_spike = self.mem_update(self.conv1, c0_spike, c1_mem, c1_spike)
            p1_mem, p1_spike = self.mem_update_pool(F.avg_pool2d, c1_spike, p1_mem, p1_spike, _pool=2)
            p1_spike = F.dropout(p1_spike, p=0.5)
            c2_mem, c2_spike = self.mem_update(self.conv2, p1_spike, c2_mem, c2_spike)
            p2_mem, p2_spike = self.mem_update_pool(F.avg_pool2d, c2_spike, p2_mem, p2_spike, _pool=2)
            p2_spike = F.dropout(p2_spike, p=0.5)
            output[..., step] = p2_spike.view(shape[0], -1)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, _pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, _pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike
    
    
class SCNN3(nn.Module):

    def __init__(self, device):
        super(SCNN3, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)

        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        # torch.cuda.empty_cache()
        shape = input.shape
        n_step = input.shape[4]
        c0_mem = c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        p0_mem = p0_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        c1_mem = c1_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        p1_mem = p1_spike = torch.zeros(shape[0], 64, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        c2_mem = c2_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        p2_mem = p2_spike = torch.zeros(shape[0], 128, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        c3_mem = c3_spike = torch.zeros(shape[0], 256, int(shape[2] / 16), int(shape[3] / 16), device=self.device)
        p3_mem = p3_spike = torch.zeros(shape[0], 256, int(shape[2] / 32), int(shape[3] / 32), device=self.device)
        output = torch.zeros(shape[0], 256 * int(shape[2] / 32) * int(shape[3] / 32), n_step, device=self.device)
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
            output[..., step] = p3_spike.view(shape[0], -1)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, _pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, _pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike


class SCNN0(nn.Module):

    def __init__(self, device):
        super(SCNN0, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = input.shape[4]
        c0_mem = c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        p0_mem = p0_spike = torch.zeros(shape[0], 64, int(shape[2]/2), int(shape[3]/2), device=self.device)
        output = torch.zeros(shape[0], 64 * int(shape[2]/2) * int(shape[3]/2), n_step, device=self.device)
        for step in range(n_step):
            x = input[..., step]
            c0_mem, c0_spike = self.mem_update(self.conv0, x, c0_mem, c0_spike)
            p0_mem, p0_spike = self.mem_update_pool(F.avg_pool2d, c0_spike, p0_mem, p0_spike, _pool=2)
            p0_spike = F.dropout(p0_spike, p=0.5)
            output[..., step] = p0_spike.view(shape[0], -1)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, _pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, _pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike


class SCNN1(nn.Module):

    def __init__(self, device):
        super(SCNN1, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = input.shape[4]
        c0_mem = c0_spike = torch.zeros(shape[0], 8, shape[2], shape[3], device=self.device)
        p0_mem = p0_spike = torch.zeros(shape[0], 8, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        c1_mem = c1_spike = torch.zeros(shape[0], 16, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        p1_mem = p1_spike = torch.zeros(shape[0], 16, int(shape[2] / 16), int(shape[3] / 16), device=self.device)
        output = torch.zeros(shape[0], 16 * int(shape[2] / 16) * int(shape[3] / 16), n_step, device=self.device)
        for step in range(n_step):
            x = input[..., step]
            c0_mem, c0_spike = self.mem_update(self.conv0, x, c0_mem, c0_spike)
            p0_mem, p0_spike = self.mem_update_pool(F.avg_pool2d, c0_spike, p0_mem, p0_spike, _pool=4)
            p0_spike = F.dropout(p0_spike, p=0.5)
            c1_mem, c1_spike = self.mem_update(self.conv1, p0_spike, c1_mem, c1_spike)
            p1_mem, p1_spike = self.mem_update_pool(F.avg_pool2d, c1_spike, p1_mem, p1_spike, _pool=4)
            p1_spike = F.dropout(p1_spike, p=0.5)
            output[..., step] = p1_spike.view(shape[0], -1)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, _pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, _pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike


class SCNN4(nn.Module):

    def __init__(self, device):
        super(SCNN4, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = input.shape[4]
        c0_mem = c0_spike = torch.zeros(shape[0], 32, shape[2], shape[3], device=self.device)
        p0_mem = p0_spike = torch.zeros(shape[0], 32, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        c1_mem = c1_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        p1_mem = p1_spike = torch.zeros(shape[0], 64, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        c2_mem = c2_spike = torch.zeros(shape[0], 64, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        p2_mem = p2_spike = torch.zeros(shape[0], 64, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        c3_mem = c3_spike = torch.zeros(shape[0], 128, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        p3_mem = p3_spike = torch.zeros(shape[0], 128, int(shape[2] / 16), int(shape[3] / 16), device=self.device)
        c4_mem = c4_spike = torch.zeros(shape[0], 128, int(shape[2] / 16), int(shape[3] / 16), device=self.device)
        p4_mem = p4_spike = torch.zeros(shape[0], 128, int(shape[2] / 32), int(shape[3] / 32), device=self.device)
        output = torch.zeros(shape[0], 128 * int(shape[2] / 32) * int(shape[3] / 32), n_step, device=self.device)
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
            c4_mem, c4_spike = self.mem_update(self.conv4, p3_spike, c4_mem, c4_spike)
            p4_mem, p4_spike = self.mem_update_pool(F.avg_pool2d, c4_spike, p4_mem, p4_spike, _pool=2)
            p4_spike = F.dropout(p4_spike, p=0.5)
            output[..., step] = p4_spike.view(shape[0], -1)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, _pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, _pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike