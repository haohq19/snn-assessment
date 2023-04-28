import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spike_func import *


class SResnet18(nn.Module):

    def __init__(self, device):
        super(SResnet18, self).__init__()
        # conv0
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        # block0
        # self.block0 = SResBlock(_in_channels=64, _out_channels=64, _kernel_size=3, _padding=1, device=device)
        self.b0_conv0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b0_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # block1
        # self.block1 = SResBlock(_in_channels=64, _out_channels=64, _kernel_size=3, _padding=1, device=device)
        self.b1_conv0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b1_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # conv1
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # conv2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # block2
        # self.block2 = SResBlock(_in_channels=128, _out_channels=128, _kernel_size=3, _padding=1, device=device)
        self.b2_conv0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b2_conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # conv3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # conv4
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # block3
        # self.block3 = SResBlock(_in_channels=256, _out_channels=256, _kernel_size=3, _padding=1, device=device)
        self.b3_conv0 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.b3_conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # conv5
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # conv6
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # block4
        # self.block4 = SResBlock(_in_channels=512, _out_channels=512, _kernel_size=3, _padding=1, device=device)
        self.b4_conv0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.b4_conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = shape[4]
        # conv0 [64, 32, 32]
        c0_mem = c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        # block0 [64, 32, 32]
        b0_c0_mem = b0_c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        b0_c1_mem = b0_c1_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        # block1 [64, 32, 32]
        b1_c0_mem = b1_c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        b1_c1_mem = b1_c1_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        # conv1 [128, 32, 32]
        c1_mem = c1_spike = torch.zeros(shape[0], 128, shape[2], shape[3], device=self.device)
        # conv2 [128, 32, 32]
        c2_mem = c2_spike = torch.zeros(shape[0], 128, shape[2], shape[3], device=self.device)
        # pool0 [128, 16, 16]
        p0_mem = p0_spike = torch.zeros(shape[0], 128, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        # block2 [128, 16, 16]
        b2_c0_mem = b2_c0_spike = torch.zeros(shape[0], 128, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        b2_c1_mem = b2_c1_spike = torch.zeros(shape[0], 128, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        # conv3 [256, 16, 16]
        c3_mem = c3_spike = torch.zeros(shape[0], 256, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        # conv4 [256, 16, 16]
        c4_mem = c4_spike = torch.zeros(shape[0], 256, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        # pool 1 [256, 8, 8]
        p1_mem = p1_spike = torch.zeros(shape[0], 256, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # block3
        b3_c0_mem = b3_c0_spike = torch.zeros(shape[0], 256, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        b3_c1_mem = b3_c1_spike = torch.zeros(shape[0], 256, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # conv5 [512, 8, 8]
        c5_mem = c5_spike = torch.zeros(shape[0], 512, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # conv6 [512, 8, 8]
        c6_mem = c6_spike = torch.zeros(shape[0], 512, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # pool2 [512, 4, 4]
        p2_mem = p2_spike = torch.zeros(shape[0], 512, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        # block4
        b4_c0_mem = b4_c0_spike = torch.zeros(shape[0], 512, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        b4_c1_mem = b4_c1_spike = torch.zeros(shape[0], 512, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        # output
        output = torch.zeros(shape[0], 512 * int(shape[2] / 8) * int(shape[3] / 8), n_step, device=self.device)
        for step in range(n_step):
            x = input[..., step]
            # conv0
            c0_mem, c0_spike = self.mem_update(self.conv0, x, c0_mem, c0_spike)
            # block0
            b0_c0_mem, b0_c0_spike = self.mem_update(self.b0_conv0, c0_spike, b0_c0_mem, b0_c0_spike)
            b0_c1_mem, b0_c1_spike = self.mem_update(self.b0_conv1, b0_c0_spike, b0_c1_mem, b0_c1_spike)
            # block1
            b1_c0_mem, b1_c0_spike = self.mem_update(self.b1_conv0, b0_c1_spike + c0_spike, b1_c0_mem, b1_c0_spike)
            b1_c1_mem, b1_c1_spike = self.mem_update(self.b1_conv1, b1_c0_spike, b1_c1_mem, b1_c1_spike)
            # conv1
            c1_mem, c1_spike = self.mem_update(self.conv1, b1_c1_spike + b0_c1_spike, c1_mem, c1_spike)
            # conv2
            c2_mem, c2_spike = self.mem_update(self.conv2, c1_spike, c2_mem, c2_spike)
            # pool0
            p0_mem, p0_spike = self.mem_update_pool(F.avg_pool2d, c2_spike, p0_mem, p0_spike, pool=2)
            # block2
            b2_c0_mem, b2_c0_spike = self.mem_update(self.b2_conv0, p0_spike, b2_c0_mem, b2_c0_spike)
            b2_c1_mem, b2_c1_spike = self.mem_update(self.b2_conv1, b2_c0_spike, b2_c1_mem, b2_c1_spike)
            # conv3
            c3_mem, c3_spike = self.mem_update(self.conv3, b2_c1_spike + p0_spike, c3_mem, c3_spike)
            # conv4
            c4_mem, c4_spike = self.mem_update(self.conv4, c3_spike, c4_mem, c4_spike)
            # pool1
            p1_mem, p1_spike = self.mem_update_pool(F.avg_pool2d, c4_spike, p1_mem, p1_spike, pool=2)
            # block3
            b3_c0_mem, b3_c0_spike = self.mem_update(self.b3_conv0, p1_spike, b3_c0_mem, b3_c0_spike)
            b3_c1_mem, b3_c1_spike = self.mem_update(self.b3_conv1, b3_c0_spike, b3_c1_mem, b3_c1_spike)
            # conv5
            c5_mem, c5_spike = self.mem_update(self.conv5, b3_c1_spike + p1_spike, c5_mem, c5_spike)
            # conv6
            c6_mem, c6_spike = self.mem_update(self.conv6, c5_spike, c6_mem, c6_spike)
            # pool2
            p2_mem, p2_spike = self.mem_update_pool(F.avg_pool2d, c6_spike, p2_mem, p2_spike, pool=2)
            # block4
            b4_c0_mem, b4_c0_spike = self.mem_update(self.b4_conv0, p2_spike, b4_c0_mem, b4_c0_spike)
            b4_c1_mem, b4_c1_spike = self.mem_update(self.b4_conv1, b4_c0_spike, b4_c1_mem, b4_c1_spike)
            # output
            output[..., step] = (b4_c1_spike + p2_spike).view(shape[0], -1)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        # torch.cuda.empty_cache()
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, pool=2):
        # torch.cuda.empty_cache()
        _mem = _mem * decay * (1 - _spike) + _func(_x, pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike



class SResnet5(nn.Module):

    def __init__(self, device):
        super(SResnet5, self).__init__()
        # conv0
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        # conv1
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # block0
        self.b0_conv0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b0_conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # conv2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = shape[4]
        # conv0 [64, 32, 32]
        c0_mem = c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        # pool0 [128, 16, 16]
        p0_mem = p0_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        # conv1 [128, 16, 16]
        c1_mem = c1_spike = torch.zeros(shape[0], 128, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        # pool1
        p1_mem = p1_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # block0 [64, 32, 32]
        b0_c0_mem = b0_c0_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        b0_c1_mem = b0_c1_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # conv2 [128, 32, 32]
        c2_mem = c2_spike = torch.zeros(shape[0], 256, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # pool2
        p2_mem = p2_spike = torch.zeros(shape[0], 256, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        # output
        output = torch.zeros(shape[0], 256 * int(shape[2] / 8) * int(shape[3] / 8), n_step, device=self.device)
        for step in range(n_step):
            x = input[..., step]
            # conv0
            c0_mem, c0_spike = self.mem_update(self.conv0, x, c0_mem, c0_spike)
            # pool0
            p0_mem, p0_spike = self.mem_update_pool(F.avg_pool2d, c0_spike, p0_mem, p0_spike, pool=2)
            # conv1
            c1_mem, c1_spike = self.mem_update(self.conv1, p0_spike, c1_mem, c1_spike)
            # pool1
            p1_mem, p1_spike = self.mem_update_pool(F.avg_pool2d, c1_spike, p1_mem, p1_spike, pool=2)
            # block0
            b0_c0_mem, b0_c0_spike = self.mem_update(self.b0_conv0, p1_spike, b0_c0_mem, b0_c0_spike)
            b0_c1_mem, b0_c1_spike = self.mem_update(self.b0_conv1, b0_c0_spike, b0_c1_mem, b0_c1_spike)
            # conv2
            c2_mem, c2_spike = self.mem_update(self.conv2, b0_c1_spike + p1_spike, c2_mem, c2_spike)
            # pool2
            p2_mem, p2_spike = self.mem_update_pool(F.avg_pool2d, c2_spike, p2_mem, p2_spike, pool=2)
            output[..., step] = p2_spike.view(shape[0], -1)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike


class SResnet4(nn.Module):

    def __init__(self, device):
        super(SResnet4, self).__init__()
        # conv0
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=5, stride=1, padding=2)
        # block0
        self.b0_conv0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b0_conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # conv1
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = shape[4]
        # conv0 [128, 32, 32]
        c0_mem = c0_spike = torch.zeros(shape[0], 128, shape[2], shape[3], device=self.device)
        # pool0 [128, 8, 8]
        p0_mem = p0_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # block0 [128, 8, 8]
        b0_c0_mem = b0_c0_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        b0_c1_mem = b0_c1_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # pool1 [128, 4, 4]
        p1_mem = p1_spike = torch.zeros(shape[0], 128, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        # conv1 [256, 2, 2]
        c1_mem = c1_spike = torch.zeros(shape[0], 256, int(shape[2] / 16), int(shape[3] / 16), device=self.device)
        # pool2 [256, 1, 1]
        p2_mem = p2_spike = torch.zeros(shape[0], 256, int(shape[2] / 32), int(shape[3] / 32), device=self.device)
        # output [256]
        output = torch.zeros(shape[0], 256 * int(shape[2] / 32) * int(shape[3] / 32), n_step, device=self.device)
        for step in range(n_step):
            x = input[..., step]
            # conv0
            c0_mem, c0_spike = self.mem_update(self.conv0, x, c0_mem, c0_spike)
            # pool0
            p0_mem, p0_spike = self.mem_update_pool(F.avg_pool2d, c0_spike, p0_mem, p0_spike, pool=4)
            # block0
            b0_c0_mem, b0_c0_spike = self.mem_update(self.b0_conv0, p0_spike, b0_c0_mem, b0_c0_spike)
            b0_c1_mem, b0_c1_spike = self.mem_update(self.b0_conv1, b0_c0_spike, b0_c1_mem, b0_c1_spike)
            # pool1
            p1_mem, p1_spike = self.mem_update_pool(F.avg_pool2d, b0_c1_spike + p0_spike, p1_mem, p1_spike, pool=2)
            # conv1
            c1_mem, c1_spike = self.mem_update(self.conv1, p1_spike, c1_mem, c1_spike)
            # pool2
            p2_mem, p2_spike = self.mem_update_pool(F.avg_pool2d, c1_spike, p2_mem, p2_spike, pool=2)
            output[..., step] = p2_spike.view(shape[0], -1)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike


class SResnet6(nn.Module):

    def __init__(self, device):
        super(SResnet6, self).__init__()
        # conv0
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        # conv1
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # block0
        self.b0_conv0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b0_conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # conv2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = shape[4]
        # conv0 [64, 32, 32]
        c0_mem = c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        # pool0 [128, 16, 16]
        p0_mem = p0_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        # conv1 [128, 16, 16]
        c1_mem = c1_spike = torch.zeros(shape[0], 128, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        # pool1
        p1_mem = p1_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # block0 [64, 8, 8]
        b0_c0_mem = b0_c0_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        b0_c1_mem = b0_c1_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # conv2 [128, 8, 8]
        c2_mem = c2_spike = torch.zeros(shape[0], 128, int(shape[2] / 4), int(shape[3] / 4), device=self.device)
        # pool2
        p2_mem = p2_spike = torch.zeros(shape[0], 128, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        # conv3 [128, 4, 4]
        c3_mem = c3_spike = torch.zeros(shape[0], 128, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        # pool2
        p3_mem = p3_spike = torch.zeros(shape[0], 128, int(shape[2] / 32), int(shape[3] / 32), device=self.device)
        # output
        output = torch.zeros(shape[0], 128 * int(shape[2] / 32) * int(shape[3] / 32), n_step, device=self.device)
        for step in range(n_step):
            x = input[..., step]
            # conv0
            c0_mem, c0_spike = self.mem_update(self.conv0, x, c0_mem, c0_spike)
            # pool0
            p0_mem, p0_spike = self.mem_update_pool(F.avg_pool2d, c0_spike, p0_mem, p0_spike, pool=2)
            # conv1
            c1_mem, c1_spike = self.mem_update(self.conv1, p0_spike, c1_mem, c1_spike)
            # pool1
            p1_mem, p1_spike = self.mem_update_pool(F.avg_pool2d, c1_spike, p1_mem, p1_spike, pool=2)
            # block0
            b0_c0_mem, b0_c0_spike = self.mem_update(self.b0_conv0, p1_spike, b0_c0_mem, b0_c0_spike)
            b0_c1_mem, b0_c1_spike = self.mem_update(self.b0_conv1, b0_c0_spike, b0_c1_mem, b0_c1_spike)
            # conv2
            c2_mem, c2_spike = self.mem_update(self.conv2, b0_c1_spike + p1_spike, c2_mem, c2_spike)
            # pool2
            p2_mem, p2_spike = self.mem_update_pool(F.avg_pool2d, c2_spike, p2_mem, p2_spike, pool=2)
            # conv3
            c3_mem, c3_spike = self.mem_update(self.conv2, p2_spike, c3_mem, c3_spike)
            # pool3
            p3_mem, p3_spike = self.mem_update_pool(F.avg_pool2d, c3_spike, p3_mem, p3_spike, pool=4)
            output[..., step] = p3_spike.view(shape[0], -1)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike


class SResnet7(nn.Module):

    def __init__(self, device):
        super(SResnet7, self).__init__()
        # conv0
        self.conv0 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        # block0
        self.b0_conv0 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.b0_conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # conv1
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # block1
        self.b1_conv0 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.b1_conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # conv2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.spike_func = SpikeFunc.apply
        self.device = device

    def forward(self, input):
        shape = input.shape
        n_step = shape[4]

        # conv0 [64, 32, 32]
        c0_mem = c0_spike = torch.zeros(shape[0], 64, shape[2], shape[3], device=self.device)
        # pool0 [64, 16, 16]
        p0_mem = p0_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)

        # block0 [64, 16, 16]
        b0_c0_mem = b0_c0_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        b0_c1_mem = b0_c1_spike = torch.zeros(shape[0], 64, int(shape[2] / 2), int(shape[3] / 2), device=self.device)

        # conv1 [128, 16, 16]
        c1_mem = c1_spike = torch.zeros(shape[0], 128, int(shape[2] / 2), int(shape[3] / 2), device=self.device)
        # pool1 [128, 4, 4]
        p1_mem = p1_spike = torch.zeros(shape[0], 128, int(shape[2] / 8), int(shape[3] / 8), device=self.device)

        # block1 [128, 4, 4]
        b1_c0_mem = b1_c0_spike = torch.zeros(shape[0], 128, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        b1_c1_mem = b1_c1_spike = torch.zeros(shape[0], 128, int(shape[2] / 8), int(shape[3] / 8), device=self.device)

        # conv2 [64, 4, 4]
        c2_mem = c2_spike = torch.zeros(shape[0], 64, int(shape[2] / 8), int(shape[3] / 8), device=self.device)
        # pool2 [64, 1, 1]
        p2_mem = p2_spike = torch.zeros(shape[0], 64, int(shape[2] / 32), int(shape[3] / 32), device=self.device)

        # output
        output = torch.zeros(shape[0], 64 * int(shape[2] / 32) * int(shape[3] / 32), n_step, device=self.device)
        for step in range(n_step):

            x = input[..., step]
            # conv0
            c0_mem, c0_spike = self.mem_update(self.conv0, x, c0_mem, c0_spike)
            # pool0
            p0_mem, p0_spike = self.mem_update_pool(F.avg_pool2d, c0_spike, p0_mem, p0_spike, pool=2)

            # block0
            b0_c0_mem, b0_c0_spike = self.mem_update(self.b0_conv0, p0_spike, b0_c0_mem, b0_c0_spike)
            b0_c1_mem, b0_c1_spike = self.mem_update(self.b0_conv1, b0_c0_spike, b0_c1_mem, b0_c1_spike)

            # conv1
            c1_mem, c1_spike = self.mem_update(self.conv1, p0_spike + b0_c1_spike, c1_mem, c1_spike)
            # pool1
            p1_mem, p1_spike = self.mem_update_pool(F.avg_pool2d, c1_spike, p1_mem, p1_spike, pool=4)

            # block1
            b1_c0_mem, b1_c0_spike = self.mem_update(self.b1_conv0, p1_spike, b1_c0_mem, b1_c0_spike)
            b1_c1_mem, b1_c1_spike = self.mem_update(self.b1_conv1, b1_c0_spike, b1_c1_mem, b1_c1_spike)

            # conv2
            c2_mem, c2_spike = self.mem_update(self.conv2, b1_c0_spike + p1_spike, c2_mem, c2_spike)
            # pool2
            p2_mem, p2_spike = self.mem_update_pool(F.avg_pool2d, c2_spike, p2_mem, p2_spike, pool=4)

            output[..., step] = p2_spike.view(shape[0], -1)

        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

    def mem_update_pool(self, _func, _x, _mem, _spike, pool=2):
        _mem = _mem * decay * (1 - _spike) + _func(_x, pool)
        _spike = self.spike_func(_mem)
        return _mem, _spike