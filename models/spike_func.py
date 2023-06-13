import torch


class SpikeFunc(torch.autograd.Function):
    thresh = 0.3
    lens = 0.5

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(SpikeFunc.thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - SpikeFunc.thresh) < SpikeFunc.lens
        return grad_input * temp.float() / (2 * SpikeFunc.lens)

    @classmethod
    def init(cls, thresh, lens):
        cls.thresh = thresh
        cls.lens = lens
