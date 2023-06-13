import yaml
from tools.get_model import *
from tools.get_data import *
config = {
    'type': 'SResnet6',
    'device': 'cuda',
    'model': {
        'backbone': {
            'dropout_p': 0.5,
            'decay': 0.5,
            'layers': [
                {
                    'type': 'conv2d',
                    'in_channels': 2,
                    'out_channels': 64,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                },
                {
                    'type': 'avgpool2d',
                    'kernel_size': 2,
                    'stride': 2
                },
                {
                    'type': 'conv2d',
                    'in_channels': 64,
                    'out_channels': 128,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                },
                {
                    'type': 'avgpool2d',
                    'kernel_size': 2,
                    'stride': 2
                },
                {
                    'type': 'conv2d',
                    'in_channels': 128,
                    'out_channels': 128,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                },
                {
                    'type': 'conv2d',
                    'in_channels': 128,
                    'out_channels': 128,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                },
                {
                    'type': 'conv2d',
                    'in_channels': 128,
                    'out_channels': 128,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1,
                    'residual': -2
                },
                {
                    'type': 'avgpool2d',
                    'kernel_size': 2,
                    'stride': 2
                },
                {
                    'type': 'conv2d',
                    'in_channels': 128,
                    'out_channels': 128,
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                },
                {
                    'type': 'avgpool2d',
                    'kernel_size': 4,
                    'stride': 4
                }
            ]
        },
        'head': {
            'dropout_p': 0.5,
            'decay': 0.5,
            'layers':[
                {
                    'type': 'fc',
                    'in_features': 256,
                    'out_features': 11
                }
            ]
        }
    },
    'dataset': {
        'train': {
            'type': 'dg',
            'group_name': 'train',
            'n_steps': 50,
            'n_classes': 0,
            'ds': 4,
            'dt': 1000,
            'batch_size': 1,
            'num_workers': 0,
            'shuffle': True,
            'drop_last': True
        },
        'fine-tune': {
            'type': 'dg',
            'group_name': 'train',
            'n_steps': 50,
            'n_classes': 0,
            'ds': 4,
            'dt': 1000,
            'batch_size': 1,
            'num_workers': 0,
            'shuffle': True,
            'drop_last': True
        },
        'test': {
            'type': 'dg',
            'group_name': 'test',
            'n_steps': 50,
            'n_classes': 0,
            'ds': 4,
            'dt': 1000,
            'batch_size': 1,
            'num_workers': 0,
            'shuffle': False,
            'drop_last': False
        }
    },
    'run': {
        'type': 'train',
        'n_epochs': 300,
        'load_backbone_weight': False,
        'load_head_weight': False,
        'save_name': '',
        'load_fine_tune_weight': False,
        'fine_tune_dataset': '',
    },
    'optimizer': {
        'learning_rate': 1e-4
    },
    'schedular': {
        'lr_decay_epoch': 100,
        'decay': 0.3
    },
    'criterion': {
        'loss': 'MSELoss'
    },
    'save': {
        'timestamp': '',
        'save_name': '',
        'path': ''
    }
}


class Config:

    def __init__(self):
        self.config = {}

    def update_config(self, config):
        self.config.update(config)

    def load_file(self):
        with open('models/SResnet6.yaml', 'r') as file:
            self.config.update(yaml.load(file, Loader=yaml.Loader))

    def save_file(self):
        with open('../models/SResnet6.yaml', 'w+') as file:
            yaml.dump(self.config, file)

if __name__ == '__main__':
    cfg = Config()
    cfg.update_config(config)
    cfg.save_file()
    model = get_model(config['model'], config['device'])
    data = get_data(config['dataset']['train'])
    print(1)
