import os
import time
import torch
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import imageio
import scipy
from models.scnn import *
from data_loader.nmnist_dataset import *
from data_loader.dvsgesture_dataset import *
from logME_test import *

# 训练过程可视化
def visualize_loss(
    number,         # 模型编号
    train_ds_name,  # 模型训练数据集名称
    m_name,         # 模型名称
    n_class,        # 训练集种类数
    n_class_test    # 测试集种类数
    ):
    
    model_name = '{}_{}_{}_{}c'.format(number, train_ds_name, m_name, n_class)
    model_save_path = './models_save/' + model_name + '/'
    train_loss_record = []
    test_loss_record = []
    train_acc_record = []
    acc_record = []
    epoch = 0
    pt_files = glob.glob(os.path.join(model_save_path, '*.pth'))
    ft_files = glob.glob(os.path.join(model_save_path, '*ft*.pth'))
    pt_files = list(set(pt_files).difference(set(ft_files)))
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    if len(pt_files) > 0:
        print('loading parameters from: ' + model_save_path)
        print('{} saves found'.format(len(pt_files)))
        pt_files.sort(key=os.path.getctime)
        pt_files = pt_files[-1:]
        for ckpt in pt_files:
            state = torch.load(ckpt)
            train_loss_record = state['train_loss_record']  # 040202 after only
            test_loss_record = state['test_loss_record']  # 040202 after only
            train_acc_record = state['train_acc_record']  # 040202 after only
            acc_record = state['acc_record']
            epoch = state['epoch']
            epoch_record = range(epoch)
            ax[0].plot(epoch_record, train_loss_record, color='blue', linewidth=1.0, linestyle='-',
                       label='train loss')

            ax[1].plot(epoch_record, train_acc_record, color='blue', linewidth=1.0, linestyle='-',
                       label='train accuracy')

    if len(ft_files) > 0:
        print('loading parameters from: ' + model_save_path)
        print('{} saves found'.format(len(ft_files)))
        ft_files.sort(key=os.path.getctime)
        ft_files = ft_files[-1:]
        for ckpt in ft_files:
            state = torch.load(ckpt)
            test_loss_record = state['test_loss_record']  # 040202 after only
            acc_record = state['acc_record']
            epoch = state['epoch'] + epoch
            epoch_record = range(epoch)
            ax[0].plot(epoch_record, test_loss_record, color='red', linewidth=1.0, linestyle='-',
                       label='test loss')
            ax[1].plot(epoch_record, acc_record, color='red', linewidth=1.0, linestyle='-',
                       label='test accuracy')

    ax[0].set(title="train: %d classes  fine-tune: %d classes" % (n_class, n_class_test),
              xlabel="epoch", ylabel="MSE loss")
    ax[1].set(xlabel="epoch", ylabel="accuracy (%)")
    fig.set(dpi=300)
    ax[0].set_xlim(left=0, right=600)
    ax[1].set_xlim(left=0, right=600)
    ax[0].set_ylim(bottom=0, top=5)
    ax[1].set_ylim(bottom=0, top=100)
    ax[0].legend()
    ax[1].legend()
    ax[0].grid(which='both')
    ax[1].grid(which='both')
    plt.show()


def visualize_data(
        inputs,  # batch tensor  (bs, channel, width, height, n_step)
        labels   # 1-hot label tensor  (bs, n_class, n_step)
):
    batch_size = inputs.shape[0]
    n_step = inputs.shape[4]
    images = inputs.numpy()
    labels = np.argmax(labels.numpy(), axis=1)
    n_column = int((n_step + 9) / 10)
    for i in range(batch_size):
        fig, axs = plt.subplots(n_column, 10, figsize=(20, 2 * (n_column + 1)))
        axs = axs.reshape(-1, 10)  # 将子图区域数组转换为 N 行 10 列的二维数组
        for c in range(n_column):
            for r in range(10):
                if r + 10 * c < n_step:
                    img_p = images[i, 0, ..., r + 10 * c]
                    img_n = images[i, 1, ..., r + 10 * c]
                    axs[c, r].imshow(img_p, vmin=0, cmap='Reds')
                    axs[c, r].imshow(img_n, vmin=0, cmap='Greens', alpha=0.5)
                axs[c, r].axis('off')

        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        fig.suptitle('label: {}'.format(labels[i]), fontsize=50)
        plt.show()


def get_softmax_features(feature, label, index):
    softmax_f_sum_list = []
    batch_size = len(feature)
    label = np.argmax(label.numpy(), axis=1)
    exp_f = np.exp(feature)
    softmax_f = exp_f / np.sum(np.sum(exp_f, axis=1, keepdims=True), axis=2, keepdims=True)
    softmax_f_sum = softmax_f.sum(axis=2)
    for bs in range(batch_size):
        if label[bs] == index:
            softmax_f_sum_list.append(softmax_f_sum[i])
    return np.vstack(softmax_f_sum_list)


def visualize_feature(model, device, test_loader):
    model.to(device)
    model.eval()
    label_3_feat = []
    label_4_feat = []
    for _, (input, label) in enumerate(test_loader):



        batch_size = input.shape[0]
        n_step = input.shape[4]

        # label
        label = np.argmax(label.numpy(), axis=1)
        # feature
        feature = model(input.to(device)).detach().cpu().numpy()
        exp_f = np.exp(feature)
        softmax_f = exp_f / np.sum(np.sum(exp_f, axis=1, keepdims=True), axis=2, keepdims=True)
        softmax_f = softmax_f.sum(axis=2)
        for i in range(batch_size):
            if label[i] == 1:
                label_3_feat.append(softmax_f[i])
            elif label[i] == 2:
                label_4_feat.append(softmax_f[i])
    label_3_feat = np.vstack(label_3_feat)
    label_4_feat = np.vstack(label_4_feat)

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.imshow(label_3_feat, cmap='jet', aspect='auto')
    plt.title('label: 1')
    plt.show()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.imshow(label_4_feat, cmap='jet', aspect='auto')
    plt.title('label: 2')
    plt.show()

    return 0

def visualize_feat(model, device, test_loader):
    model.to(device)
    model.eval()
    for _, (input, label) in enumerate(test_loader):
        #  visualize_data(input, label)

        batch_size = input.shape[0]
        n_step = input.shape[4]
        image = input.view(batch_size, -1, n_step)
        feat_cross_entropy = np.zeros((n_step, n_step))
        img_cross_entropy = np.zeros((n_step, n_step))

        # label
        label = np.argmax(label.numpy(), axis=1)
        # feature
        feature = model(input.to(device)).detach().cpu().numpy()



        for i in range(batch_size):
            if label[i] == 3 or label[i] == 4:
        #
        #         img_sample = image[i]
        #         for t_1 in range(n_step):
        #             for t_2 in range(n_step):
        #                 f1 = np.exp(img_sample[:, t_1]) / np.sum(np.exp(img_sample[:, t_1]))
        #                 f2 = np.exp(img_sample[:, t_2]) / np.sum(np.exp(img_sample[:, t_2]))
        #                 ce = -1/len(img_sample) * np.sum(f1 * np.log(f2) + (1 - f1) * np.log(1 - f2))
        #                 img_cross_entropy[t_1][t_2] = ce
        #
        #         fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        #         cmap = plt.cm.Reds
        #         im = ax.matshow(img_cross_entropy,cmap=cmap)
        #         fig.colorbar(im, ax=ax)
        #         plt.title('label: {}'.format(label[i]))
        #         plt.show()
        #
        #         sample = feature[i]
        #         fig = plt.figure(dpi=300)
        #         ax = fig.add_subplot(111)
        #         ax.imshow(sample, cmap='coolwarm', aspect='auto')
        #         ax.set_aspect(1)
        #         plt.title('label: {}'.format(label[i]))
        #         plt.show()
        #
        #         for t_1 in range(n_step):
        #             for t_2 in range(n_step):
        #                 f1 = np.exp(sample[:, t_1]) / np.sum(np.exp(sample[:, t_1]))
        #                 f2 = np.exp(sample[:, t_2]) / np.sum(np.exp(sample[:, t_2]))
        #                 ce = -1/len(sample) * np.sum(f1 * np.log(f2) + (1 - f1) * np.log(1 - f2))
        #                 feat_cross_entropy[t_1][t_2] = ce
        #         ce_mean = feat_cross_entropy.mean()
        #         ce_std = feat_cross_entropy.std()
        #         print('{}: label: {}, mean: {}, std: {}'.format(i, label[i], ce_mean, ce_std))
        #         fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        #         cmap = plt.cm.Reds
        #         vmin = 0.024
        #         vmax = 0.028
        #         im = ax.matshow(feat_cross_entropy,cmap=cmap, vmin=vmin, vmax=vmax)
        #         fig.colorbar(im, ax=ax)
        #         plt.title('label: {}'.format(label[i]))
        #         plt.show()
        #         break

if __name__ == '__main__':
    number = '041902'
    train_ds_name = 'dg'
    test_ds_name = 'dg'
    m_name = 'scnn3'

    n_step = 5
    batch_size = 40
    dt = 10 * 1000  # temporal resolution, in us
    ds = 4  # spatial resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0
    n_class_test = 0

    if test_ds_name == 'dg':
        n_class_test = 11
    elif test_ds_name == 'nm':
        n_class_test = 10
    elif test_ds_name == 'ct':
        n_class_test = 100
    elif test_ds_name == 'cf':
        n_class_test = 10

    test_loader = get_data_loader(dataset_name=test_ds_name, group_name='train', n_step=n_step, n_class=n_class_test,
                                  ds=ds, dt=dt, batch_size=batch_size, num_workers=num_workers)
    model_name = '{}_{}_{}_{}c'.format(number, train_ds_name, m_name, n_class_test)
    model_save_path = './models_save/' + model_name + '/'
    model = get_model(device, m_name, model_save_path)

    visualize_feature(model, device, test_loader)


