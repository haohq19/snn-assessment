import torch.utils.data as data
import torch
import scipy.io as sio
import h5py
import time
import numpy as np


class NMnistDataset(data.Dataset):
    def __init__(self, group_name, n_class=10, n_step=10):
        super(NMnistDataset, self).__init__()
        self.n_class = n_class
        if group_name == 'test':
            start_time = time.time()
            dataset = sio.loadmat(r'./dataset/NMnist/NMNIST_test_data.mat')
            self.images = dataset['image']
            self.labels = dataset['label']
            mask = np.argmax(self.labels, axis=1) < n_class
            self.labels = self.labels[mask]
            self.images = self.images[mask]
            self.images = self.images.transpose(0, 3, 1, 2, 4)
            self.images = self.images[..., 1:-1, 1:-1, :n_step]
            print('generate n-mnist test set, used %.3fs' % (time.time() - start_time))

        elif group_name == 'train':
            start_time = time.time()
            dataset = h5py.File(r'./dataset/NMnist/NMNIST_train_data.mat')
            self.images = dataset['image'][()]
            self.labels = dataset['label'][()]
            self.labels = self.labels.transpose(1, 0)
            self.images = self.images.transpose(4, 1, 3, 2, 0)
            mask = np.argmax(self.labels, axis=1) < n_class
            self.labels = self.labels[mask]
            self.images = self.images[mask]
            self.images = self.images[..., 1:-1, 1:-1, :n_step]
            print('generate n-mnist train set, used %.3fs' % (time.time() - start_time))

        self.n_sample = len(self.labels)

    def __getitem__(self, index):
        inputs = torch.from_numpy(self.images[index]).float()
        labels = torch.from_numpy(self.labels[index]).float()
        return inputs, labels

    def __len__(self):
        return self.n_sample


if __name__ == '__main__':
    test_dataset = NMnistDataset('nmnist_r', n_class=10, n_step=10)
    train_dataset = NMnistDataset('nmnist_h', n_class=10, n_step=10)

    print('error')

# import torch.utils.data as data
# import torch
# import scipy.io as sio
# import h5py
# import time
#
#
# class NMnistDataset(data.Dataset):
#     def __init__(self, path, method='h', n_class=5, n_step=10):
#         super(NMnistDataset, self).__init__()
#         self.n_class = n_class
#         if method == 'nmnist_r':
#             start = time.time()
#             dataset = sio.loadmat(path)
#             self.images = torch.from_numpy(dataset['image']).float()
#             self.labels = torch.from_numpy(dataset['label']).float()
#             _, label = torch.max(self.labels, axis=1)
#             mask = label < n_class
#             self.labels = self.labels[mask]
#             self.images = self.images[mask]
#             self.images = self.images.permute(0, 3, 1, 2, 4)
#             self.images = self.images[..., 1:-1, 1:-1, :n_step].contiguous()
#             end = time.time()
#             print(end - start)
#
#         elif method == 'nmnist_h':
#             start = time.time()
#             dataset = h5py.File(path)
#             self.images = torch.from_numpy(dataset['image'][()]).float()
#             self.labels = torch.from_numpy(dataset['label'][()]).float()
#             self.labels = self.labels.permute(1, 0)
#             self.images = self.images.permute(4, 1, 3, 2, 0)
#             _, label = torch.max(self.labels, axis=1)
#             mask = label < n_class
#             self.labels = self.labels[mask]
#             self.images = self.images[mask]
#             self.images = self.images[..., 1:-1, 1:-1, :n_step].contiguous()
#             end = time.time()
#             print(end-start)
#
#         self.n_sample = len(self.images)
#
#     def __getitem__(self, index):
#         img, label = self.images[index], self.labels[index]
#         return img, label
#
#     def __len__(self):
#         return self.n_sample
