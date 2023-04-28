import argparse
import numpy as np
from data_loader.dvsgesture_dataset import *
from data_loader.nmnist_dataset import *
from data_loader.caltech_dataset import *
from data_loader.cifar_dataset import *
from models.scnn import *
from models.sresnet import *
from LogME.LogME import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='snn evaluation')
parser.add_argument('-g', help='Number of GPU to use', default=7, type=int)
parser.add_argument('-m', help='Mode, 0: init params, 1: load params', default=1, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.g)


def lr_scheduler(_optimizer, _epoch, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if _epoch % lr_decay_epoch == 0 and _epoch > 0:
        for param_group in _optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.3
    return _optimizer


class Model(nn.Module):

    def __init__(self, device, m_name):
        super(Model, self).__init__()
        self.device = device
        if m_name == 'sres4':
            self.backbone = SResnet4(device=self.device)
        elif m_name == 'sres5':
            self.backbone = SResnet5(device=self.device)
        elif m_name == 'sres18':
            self.backbone = SResnet18(device=self.device)
        elif m_name == 'scnn3':
            self.backbone = SCNN3(device=self.device)
        elif m_name == 'scnn2':
            self.backbone = SCNN2(device=self.device)
        elif m_name == 'scnn1':
            self.backbone = SCNN1(device=self.device)
        elif m_name == 'scnn0':
            self.backbone = SCNN0(device=self.device)

    def forward(self, input):
        output = self.backbone(input)
        return output


def get_data_loader(dataset_name, group_name, n_step, n_class, ds, dt, batch_size, num_workers):
    print(dataset_name + ' ' + group_name)
    if dataset_name == 'dg':
        size = [2, int(128 / ds), int(128 / ds)]
        dataset = DvsGestureDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, drop_last=False)
        return dataloader
    elif dataset_name == 'nm':
        dataset = NMnistDataset(n_step=n_step, n_class=n_class, group_name=group_name)
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, drop_last=False)
        return dataloader
    elif dataset_name == 'ct':
        size = [2, int(256 / ds), int(192 / ds)]
        dataset = CaltechDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, drop_last=False)
        return dataloader
    elif dataset_name == 'cf':
        size = [2, int(128 / ds), int(128 / ds)]
        dataset = CifarDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, drop_last=False)
        return dataloader


def get_model(device, m_name, model_save_path, load_backbone_only=True):
    model = Model(device=device, m_name=m_name)
    mode = args.m
    if mode == 1:
        pt_files = glob.glob(os.path.join(model_save_path, '*.pth'))
        ft_files = glob.glob(os.path.join(model_save_path, '*ft*.pth'))
        checkpoint_files = list(set(pt_files).difference(set(ft_files)))
        if len(checkpoint_files) > 0:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            print('load parameters from ' + latest_checkpoint)
            state = torch.load(latest_checkpoint)
            if load_backbone_only == True:
                model_dict = model.state_dict()
                pt_dict = state['model']
                for k, v in pt_dict.items():
                    if 'backbone' in k:
                        model_dict.update({k: v})
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state['model'])
        else:
            print('0 saves found, initialize parameters')
    else:
        print('initialize parameters')
    return model


def logME_eval(model, test_loader):
    model.to(device)
    model.eval()
    features = []
    labels = []
    for _, (input, label) in enumerate(test_loader):
        batch_size = input.shape[0]
        n_step = input.shape[4]

        # weight
        weight = np.zeros([batch_size, 1, n_step], dtype=float)
        # norm
        image = input.numpy()
        norm = np.zeros([batch_size, 1, n_step], dtype=float)
        for i in range(n_step):
            weight[..., i] = (1 - 0.2 ** (n_step - i)) / (1 - 0.2)
        for i in range(batch_size):
            norm[i] = np.linalg.norm(image[i])
        # label
        label = np.argmax(label.numpy(), axis=1)
        # feature
        feature = model(input.to(device)).detach().cpu().numpy()
        # for i in range(len(feature)):
        #     sample = feature[i]
        #
        #     fig = plt.figure(dpi=300)
        #     ax = fig.add_subplot(111)
        #     im = ax.imshow(sample, cmap='jet', aspect='auto')
        #     plt.title(label[i])
        #     ax.set_aspect(1)
        #     plt.show()
        # softmax
        exp_f = np.exp(feature)
        softmax_f = exp_f / np.sum(np.sum(exp_f, axis=1, keepdims=True), axis=2, keepdims=True)
        feature = softmax_f.sum(axis=2)
        # for i in range(batch_size):
        #     fnorm[i] = np.linalg.norm(feature[i])
        # feature = feature / fnorm
        # feature = feature / norm
        # feature = feature.sum(axis=2) / n_step
        features.append(feature)
        labels.append(label)

    features = np.vstack(features)
    labels = np.hstack(labels)
    logme = LogME()
    return logme.fit(features, labels)


def sign(x, y):
    if x >= y:
        return 1
    if x < y:
        return -1


def relative_coefficient(ls):
    result = 0
    for i in range(len(ls)):
        for j in range(len(ls) - i - 1):
            result += sign(ls[i], ls[i + j + 1])
    return result * 2 / len(ls) / (len(ls) - 1)



if __name__ == '__main__':
    print('== logME-eval ==')
    number = '042509'
    train_ds_name = 'dg'
    test_ds_name = 'dg'
    m_name = 'sres4'

    n_step = 100
    batch_size = 8
    dt = 10 * 1000  # temporal resolution, in us
    ds = 4  # spatial resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 16
    n_class_test = 0
    n_classes = []
    if train_ds_name == 'dg':
        n_classes = [11, 9, 7, 5, 3, 0]
        # n_classes = [11]
    elif train_ds_name == 'nm':
        n_classes = [10, 8, 6, 4, 2, 0]
    elif train_ds_name == 'ct':
        n_classes = [100, 80, 60, 40, 20, 0]
    elif train_ds_name == 'cf':
        n_classes = [10, 8, 6, 4, 2, 0]
        # n_classes = [10]

    if test_ds_name == 'dg':
        n_class_test = 11
    elif test_ds_name == 'nm':
        n_class_test = 10
    elif test_ds_name == 'ct':
        n_class_test = 100
    elif test_ds_name == 'cf':
        n_class_test = 10

    test_loader = get_data_loader(dataset_name=test_ds_name, group_name='test', n_step=n_step, n_class=n_class_test,
                                  ds=ds, dt=dt, batch_size=batch_size, num_workers=num_workers)
    scores = []
    for n_class in n_classes:
        model_name_c = '{}_{}_{}_{}c'.format(number, train_ds_name, m_name, n_class)
        model_save_path = './models_save/' + model_name_c + '/'
        model = get_model(device, m_name, model_save_path)
        score = logME_eval(model, test_loader)
        scores.append(score)
        print(str(n_class) + ' ' + str(score))
    for score in scores:
        print(score)
    print(relative_coefficient(scores))
