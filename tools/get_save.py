import re
import os
import glob
import torch
import torch.nn as nn


def sort_key(file_name):
    n = re.findall(r'_(\d+)\.', file_name)  # 使用正则表达式提取文件名中的n
    if n:
        return int(n[0])
    else:
        return 0


def load_save(model, config):
    """
    load weight from save
    :param model: model to load weights
    :param config: current config
    :return: model with loaded weights and records
    """
    save_name = config['save']
    load_backbone_weight = config['load_backbone_weight']
    load_head_weight = config['load_head_weight']
    load_fine_tune_weight = config['load_fine_tune_weight']
    fine_tune_dataset = config['fine_tune_dataset']

    train_loss_record = []  # 040202 after only
    test_loss_record = []  # 040202 after only
    train_acc_record = []  # 040202 after only
    acc_record = []  # test_acc_record
    epoch = 0  # current epoch
    old_config = {}

    save_path = './saves/' + save_name + '/'
    if load_fine_tune_weight is False:
        saves = glob.glob(os.path.join(save_path, '*.pth'))
        saves = [save for save in saves if 'ft' not in save]
    else:
        saves = glob.glob(os.path.join(save_path, '*ft_{}*.pth'.format(fine_tune_dataset)))
    if len(saves) > 0:
        saves = sorted(saves, key=sort_key, reverse=True)
        latest_save = saves[0]
        print('utils.get_model.get_model: load weights from ' + latest_save)
        state = torch.load(latest_save)
        weight = state['weight']
        model_dict = model.static_dict()
        if load_backbone_weight:
            for k, v in weight.items():
                if 'backbone' in k:
                    model_dict.update({k: v})
        if load_head_weight:
            for k, v in weight.items():
                if 'head' in k:
                    model_dict.update({k: v})
        model.load_state_dict(model_dict)
        train_loss_record = state['train_loss_record']  # 040202 after only
        test_loss_record = state['test_loss_record']  # 040202 after only
        train_acc_record = state['train_acc_record']  # 040202 after only
        acc_record = state['acc_record']
        epoch = state['epoch']
        old_config = state['config']
    else:
        print('utils.get_model.get_model: 0 pt-weights found, initialize weights')
    return model, train_loss_record, test_loss_record, train_acc_record, acc_record, epoch, old_config
