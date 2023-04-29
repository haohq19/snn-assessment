import struct
import numpy as np
import h5py
import glob
import os
import bisect
import time
import torch.utils.data

class CifarDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            n_step=10,
            n_class=100,
            group_name='train',
            size=None,
            ds=4,
            dt=1000):
        super(CifarDataset, self).__init__()
        if size is None:
            size = [2, 32, 32]
        self.filename = './dataset/cifar_events_' + group_name + '_{}.hdf5'.format(n_class)
        self.n_step = n_step
        self.n_class_total = 10
        if n_class == 0:
            self.n_class = self.n_class_total
        else:
            self.n_class = n_class
        self.group_name = group_name
        self.size = size
        self.ds = ds
        self.dt = dt
        if not os.path.isfile(self.filename):
            create_events_hdf5(self.filename, self.n_class, self.group_name)
            print('create dataset' + self.filename)
        self.f = h5py.File(self.filename, 'r', swmr=True, libver="latest")
        self.n_sample = len(self.f)


    def __getitem__(self, index):
        sub_group = self.f[str(index)]
        label = sub_group['labels'][()]
        time = sub_group['time']
        data = sub_group['data']
        tbegin = time[()][0]
        tend = time[()][-1] - self.n_step * self.dt
        start_time = tbegin
        if tbegin < tend:
            start_time = np.random.randint(tbegin, tend)
        idx_beg = find_first(time, start_time)
        idx_end = find_first(time[idx_beg:], start_time + self.n_step * self.dt) + idx_beg
        data = chunk_evs_pol(time[idx_beg:idx_end], data[idx_beg:idx_end], dt=self.dt, n_step=self.n_step,
                             size=self.size, ds=self.ds)
        return torch.from_numpy(data).float(), torch.from_numpy(one_hot(label[0], 10)).float()

    def __len__(self):
        return self.n_sample


def aedat_to_events(folder_name, group_name):
    aedat_list = glob.glob('./dataset/CIFAR10/' + folder_name + '/*.aedat')
    aedat_list = sorted(aedat_list)
    events = []
    n_event = len(aedat_list)
    n_event_train = int(n_event/10 * 9)
    n_event_test = n_event - n_event_train
    if group_name == 'train':
        aedat_list = aedat_list[:n_event_train]
        n_event = n_event_train
    elif group_name == 'test':
        aedat_list = aedat_list[-n_event_test:]
        n_event = n_event_test
    for index in range(n_event):
        with open(aedat_list[index], 'rb') as f:
            for i in range(5):
                f.readline()
            data = np.fromfile(f, dtype='uint8')
            event_len = int(len(data)/8)
            event_x = 127 - (np.right_shift(data[3::8], 1) & 0x7F)
            event_y = data[2::8] & 0x7F
            event_p = data[3::8] & 0x01
            event_t = np.zeros(event_len, dtype='uint32')
            event_t |= data[4::8]
            event_t <<= 8
            event_t |= data[5::8]
            event_t <<= 8
            event_t |= data[6::8]
            event_t <<= 8
            event_t |= data[7::8]
            event = np.vstack([event_t, event_x, event_y, event_p]).T
            events.append(event)
    return events



def create_events_hdf5(hdf5_filename, n_class, group_name):
    folder_path = './dataset/CIFAR10'
    folder_list = os.listdir(folder_path)
    folder_list = sorted(folder_list)
    n_grp = 0
    start_time = time.time()
    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        key = 0
        for folder_name in folder_list:
            if key == n_class:
                break
            events = aedat_to_events(folder_name, group_name)
            print(len(events))
            if len(events) > 0:
                for i in range(len(events)):
                    subgroup = f.create_group(str(n_grp))
                    dataset_t = subgroup.create_dataset('time', events[i][:, 0].shape, dtype=np.uint32)
                    dataset_d = subgroup.create_dataset('data', events[i][:, 1:].shape, dtype=np.uint8)
                    dataset_l = subgroup.create_dataset('labels', (1,), dtype=np.uint32)
                    dataset_t[...] = events[i][:, 0]
                    dataset_d[...] = events[i][:, 1:]
                    dataset_l[...] = key
                    n_grp += 1
                key += 1
    print('generate CIFAR10 ' + group_name + ' dataset, used %.3fs' % (time.time() - start_time))



def one_hot(mbt, num_classes):
    out = np.zeros(num_classes)
    out[mbt] = 1
    return out


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)


def chunk_evs_pol(times, datas, dt=1000, n_step=60, size=None, ds=4):
    if size is None:
        size = [2, 32, 32]
    t_start = times[0]
    ts = range(t_start, t_start + n_step * dt, dt)
    chunks = np.zeros(size + [len(ts)], dtype='int8')
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(times[idx_end:], t + dt)
        if idx_end > idx_start:
            ee = datas[idx_start:idx_end]
            pol, x, y = ee[:, 2], ee[:, 0] // ds, ee[:, 1] // ds
            np.add.at(chunks, (pol, x, y, i), 1)
        idx_start = idx_end
    return chunks


if __name__ == "__main__":
    cifar_dataset = CifarDataset(
        n_step=10,
        n_class=10,
        group_name='train',
        size=[2, 32, 32],
        ds=4,
        dt=1000
    )
    for i in range(cifar_dataset.__len__()):
        data, label = cifar_dataset.__getitem__(i)
    dataloader = torch.utils.data.DataLoader(cifar_dataset, batch_size=256)
    for i, (inputs, labels) in enumerate(dataloader):
        print(str(i))

