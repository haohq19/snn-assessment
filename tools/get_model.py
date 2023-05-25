import torch
import torch.nn as nn
from models.model import SpikeModel


def get_backbone(config, device):
    return SpikeModel(config, device)


def get_head(config, device):
    return SpikeModel(config, device)


class Model(nn.Module):

    def __init__(self, config, device):
        super(Model, self).__init__()
        self.device = device
        self.backbone = get_backbone(config['backbone'], self.device)
        self.head = get_head(config['head'], self.device)

    def forward(self, input):
        feature = self.backbone(input)
        return self.head(feature)


def get_model(config, device):
    model = Model(config, device)
    print(config)
    return model
