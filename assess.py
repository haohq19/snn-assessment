import argparse
import os
import re
import numpy as np
from tools.get_data import *
from tools.get_model import *
from LogME.LogME import *
from APriorLogEvidence.aple import *

parser = argparse.ArgumentParser(description='snn assessment')
parser.add_argument('--gpu', help='Number of GPU to use', default=7, type=int)
parser.add_argument('--ld', help='Mode, 0: init params, 1: load params', default=1, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

def assess(model, test_loader):
    model.to(device)
    model.eval()
    features = []
    labels = []
    for _, (input, label) in enumerate(test_loader):
        # batch_size = input.shape[0]
        # n_step = input.shape[4]
        # weight
        # weight = np.zeros([batch_size, 1, n_step], dtype=float)
        # norm
        # image = input.numpy()
        # norm = np.zeros([batch_size, 1, n_step], dtype=float)
        # for i in range(n_step):
        #     weight[..., i] = (1 - 0.2 ** (n_step - i)) / (1 - 0.2)
        # for i in range(batch_size):
        #     norm[i] = np.linalg.norm(image[i])
        label = np.argmax(label.numpy(), axis=1)
        feature = model(input.to(device)).detach().cpu().numpy()
        # for i in range(len(feature)):
        #     sample = feature[i]
        #     fig = plt.figure(dpi=300)
        #     ax = fig.add_subplot(111)
        #     im = ax.imshow(sample, cmap='jet', aspect='auto')
        #     plt.title(label[i])
        #     ax.set_aspect(1)
        #     plt.show()
        # softmax
        # exp_f = np.exp(feature)
        # softmax_f = exp_f / np.sum(np.sum(exp_f, axis=1, keepdims=True), axis=2, keepdims=True)
        # feature = softmax_f.sum(axis=2)
        # for i in range(batch_size):
        #     fnorm[i] = np.linalg.norm(feature[i])
        # feature = feature / fnorm
        # feature = feature / norm
        feature = feature.sum(axis=2)
        # feature = feature / np.sqrt(np.sum(feature * feature, axis=1, keepdims=True))
        features.append(feature)
        labels.append(label)

    features = np.vstack(features)
    labels = np.hstack(labels)
    logme = LogME()
    # return logme.fit(features, labels)
    return aple(features, labels)

def sign(x, y):
    if x >= y:
        return 1
    if x < y:
        return -1


def kendall_relative_coefficient(x, y=None):
    if y is None:
        y = list(reversed(range(0, len(x))))
    result = 0
    for i in range(len(x)):
        for j in range(len(x) - i - 1):
            result += sign(x[i], x[i + j + 1]) * sign(y[i], y[i + j + 1])
    return result * 2 / len(x) / (len(x) - 1)



if __name__ == '__main__':
    print('assess: assess models')
    number = '040801'
    model_name = 'sres5'
    train_ds_name = 'dg'
    test_ds_name = 'dg'

    n_step = 50
    batch_size = 1

    dt = 20 * 1000  # temporal resolution, in us
    ds = 4  # spatial resolution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0 # 16


    folder_path = './models_save'
    _weight_name = '{}_{}_{}'.format(number, train_ds_name, model_name)

    dir_names = [name for name in os.listdir(folder_path)
                 if os.path.isdir(os.path.join(folder_path, name))]

    weight_names = [name for name in dir_names if name.startswith(_weight_name)]

    n_classes = [int(re.search(f"{_weight_name}_(\d+)c", name).group(1))
               for name in weight_names]
    n_classes = sorted(n_classes, reverse=True)


    test_loader, n_class_total = get_data(dataset_name=test_ds_name,
                                          group_name='test',
                                          n_step=n_step,
                                          n_class=0,
                                          ds=ds,
                                          dt=dt,
                                          batch_size=batch_size,
                                          num_workers=num_workers
                                          )
    scores = []
    for n_class in n_classes:
        weight_name = _weight_name + '_{}c'.format(n_class)
        weight_save_path = './models_save/' + weight_name + '/'
        model, _, _, _, _, _ = get_model(device=device,
                                         n_class_target=n_class_total,
                                         model_name=model_name,
                                         weight_name=weight_name,
                                         load_param=True,
                                         load_backbone_only=True,
                                         use_backbone_only=True
                                         )
        score = assess(model, test_loader)
        scores.append(score)
        print(str(n_class) + ': ' + str(score))
    print()
    for score in scores:
        print(score)
    print()
    print('kendall: ' + str(kendall_relative_coefficient(scores)))
