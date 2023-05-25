import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spike_func import *
from models.utils import _make_layer, _membrane_shape


class SpikeModel(nn.Module):

    def __init__(self, model_config, device):
        super(SpikeModel, self).__init__()
        self.config = model_config
        self.device = device
        self.dropout_p = self.config['dropout_p']
        self.decay = self.config['decay']
        self._build_model(config=self.config['layers'])
        self.spike_func = SpikeFunc.apply

    def forward(self, input):
        """
        forward of model
        :param input: size (bs, channels, width, height, n_steps)
        :return: output, size (bs, features, n_steps)
        """
        # torch.cuda.empty_cache()
        bs = input.shape[0]
        in_shape = input.shape[0: 4]
        n_steps = input.shape[4]
        shapes = self._init_membrane(self.config, in_shape)
        potentials = []
        spikes = []

        for shape in shapes:
            potentials.append(torch.zeros(*shape, device=self.device))
            spikes.append(torch.zeros(*shape, device=self.device))
        # output spike
        out_features = shapes[-1][1] * shapes[-1][2] * shapes[-1][3]
        out_shape = [bs, out_features, n_steps]
        output = torch.zeros(out_shape, device=self.device)

        for step in range(n_steps):
            spikes[0] = input[..., step]
            for i, layers in enumerate(self.layers):
                # residual connect
                if self.residual_indices[i]:
                    potentials[i + 1], spikes[i + 1] = self.mem_update(
                        i,
                        spikes[i] + spikes[i + self.residual_indices[i]],
                        potentials[i + 1],
                        spikes[i + 1]
                    )
                else:
                    potentials[i + 1], spikes[i + 1] = self.mem_update(
                        i,
                        spikes[i],
                        potentials[i + 1],
                        spikes[i + 1]
                    )
                # dropout
                if self.dropout_indices[i]:
                    spikes[i + 1] = F.dropout(spikes[i + 1], p=self.dropout_p)
            output[..., step] = spikes[-1].view(bs, -1)
        return output

    def _build_model(self, config):
        """
        build the layers of a model
        :param model_config: config to describe the model
        :return: None
        """
        layers = []
        self.residual_indices = []
        self.dropout_indices = []
        for _, layer_config in enumerate(config):
            layers.append(_make_layer(layer_config=layer_config))

            if 'residual' in layer_config:
                self.residual_indices.append(layer_config['residual'])
            else:
                self.residual_indices.append(0)
            if 'dropout' in layer_config:
                self.residual_indices.append(layer_config['dropout'])
            else:
                self.residual_indices.append(0)

        self.layers = nn.ModuleList(layers)

    def _init_membrane(self, model_config, in_shape):
        """
        calculate the shape of the membrane of each layer
        :param model_config: config to describe the model
        :param in_shape: shape of input data, size (bs, channels, width, height)
        :return: list of shapes of each layer, N layers -> N + 1 shapes
        """
        shapes = []
        out_shape = in_shape
        shapes.append(out_shape)
        for _, layer_config in enumerate(model_config):
            out_shape = _membrane_shape(
                layer_config=layer_config,
                in_shape=out_shape
            )
            shapes.append(out_shape)
        return shapes

    def mem_update(self, _i, _x, _potential, _spike):
        """
        update membrane potential and spike
        :param _i: index of layer
        :param _x: input data
        :param _potential: potential of current layer
        :param _spike: spike of current layer
        :return: new potential and spike of current layer
        """
        _potential = _potential * self.decay * (1 - _spike) + self.layers[_i](_x)
        _spike = self.spike_func(_potential)
        return _potential, _spike


