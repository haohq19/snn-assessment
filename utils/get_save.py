import re
from spike_data.dvsgesture_dataset import *
from spike_data.nmnist_dataset import *
from spike_data.caltech_dataset import *
from spike_data.cifar_dataset import *
from models.scnn import *
from models.sresnet import *
from models.classifier import *
from models.spike_func import *



def sort_key(file_name):
    n = re.findall(r'_(\d+)\.', file_name)  # 使用正则表达式提取文件名中的n
    if n:
        return int(n[0])
    else:
        return 0


def get_save(
        weight_name,  # 权重名称
        load_backbone_only=False
):
    train_loss_record = []  # 040202 after only
    test_loss_record = []  # 040202 after only
    train_acc_record = []  # 040202 after only
    acc_record = []  # test_acc_record
    epoch = 0  # current epoch

    weight_save_path = './saves/' + weight_name + '/'
    pt_weights = glob.glob(os.path.join(weight_save_path, '*.pth'))
    ft_weights = glob.glob(os.path.join(weight_save_path, '*ft*.pth'))
    pt_weights = list(set(pt_weights).difference(set(ft_weights)))
    if len(pt_weights) > 0:
        pt_weights = sorted(pt_weights, key=sort_key, reverse=True)
        latest_pt_weight = pt_weights[0]
        print('utils.get_model.get_model: load weights from ' + latest_pt_weight)
        state = torch.load(latest_pt_weight)
        if load_backbone_only is True:
            model_dict = model.state_dict()
            pt_dict = state['model']
            for k, v in pt_dict.items():
                if 'backbone' in k:
                    model_dict.update({k: v})
        train_loss_record = state['train_loss_record']  # 040202 after only
        test_loss_record = state['test_loss_record']  # 040202 after only
        train_acc_record = state['train_acc_record']  # 040202 after only
        acc_record = state['acc_record']
        epoch = state['epoch']
    else:
        print('utils.get_model.get_model: 0 pt-weights found, initialize weights')
    return model_dict, train_loss_record, test_loss_record, train_acc_record, acc_record, epoch
