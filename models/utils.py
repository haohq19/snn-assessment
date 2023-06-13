import torch.nn as nn
from models.model import *


def _make_layer(layer_config):

    if layer_config['type'] == 'conv2d':
        in_channels = layer_config['in_channels'] if 'in_channels' in layer_config else 1
        out_channels = layer_config['out_channels'] if 'out_channels' in layer_config else 1
        kernel_size = layer_config['kernel_size'] if 'kernel_size' in layer_config else 1
        stride = layer_config['stride'] if 'stride' in layer_config else 1
        padding = layer_config['padding'] if 'padding' in layer_config else 0
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    elif layer_config['type'] == 'fc':
        in_features = layer_config['in_features'] if 'in_features' in layer_config else 1
        out_features = layer_config['out_features'] if 'out_features' in layer_config else 1
        return nn.Linear(
            in_features=in_features,
            out_features=out_features
        )
    elif layer_config['type'] == 'avgpool2d':
        kernel_size = layer_config['kernel_size'] if 'kernel_size' in layer_config else 1
        stride = layer_config['stride'] if 'stride' in layer_config else 2
        return nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride
        )
    else:
        print('undefined layer type, check model config')
        exit(1)


def _membrane_shape(layer_config, in_shape):

    if layer_config['type'] == 'conv2d':

        bs, in_channels, in_width, in_height = in_shape

        out_channels = layer_config['out_channels'] if 'out_channels' in layer_config else 1
        kernel_size = layer_config['kernel_size'] if 'kernel_size' in layer_config else 1
        stride = layer_config['stride'] if 'stride' in layer_config else 1
        padding = layer_config['padding'] if 'padding' in layer_config else 0

        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        return [bs, out_channels, out_width, out_height]

    elif layer_config['type'] == 'fc':

        bs, in_features = in_shape

        out_features = layer_config['out_features'] if 'out_features' in layer_config else 1

        return [bs, out_features]

    elif layer_config['type'] == 'avgpool2d':
        bs, in_channels, in_width, in_height = in_shape

        kernel_size = layer_config['kernel_size'] if 'kernel_size' in layer_config else 1
        stride = layer_config['stride'] if 'stride' in layer_config else 2

        out_channels = in_channels
        out_width = ((in_width - kernel_size) // stride) + 1
        out_height = ((in_height - kernel_size) // stride) + 1

        return [bs, out_channels, out_width, out_height]

    else:
        print('undefined layer type, check model config')
        exit(1)


if __name__ == '__main__':
    b = 0
    if b:
        print(1)
    y, z = 1, 1
    print(y)


