import struct
import numpy as np
import h5py
import glob
import os
import bisect
import time
import torch.utils.data


class CaltechDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            n_step=10,
            n_class=100,
            group_name='train',
            size=None,
            ds=8,
            dt=1000):
        super(CaltechDataset, self).__init__()
        if size is None:
            size = [2, 32, 32]
        self.n_step = n_step
        self.n_class_total = 100
        if n_class == 0:
            self.n_class = self.n_class_total
        else:
            self.n_class = n_class
        self.group_name = group_name
        self.filename = './dataset/caltech_events_' + self.group_name + '_{}.hdf5'.format(self.n_class)
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
        return torch.from_numpy(data).float(), torch.from_numpy(one_hot(label[0], 100)).float()

    def __len__(self):
        return self.n_sample


def bin_to_events(key, folder_name, group_name):
    bin_list = glob.glob('./dataset/NCaltech101/Caltech101/' + folder_name + '/*.bin')
    label_list = glob.glob('./dataset/NCaltech101/Caltech101_annotations/' + folder_name + '/*.bin')
    bin_list = sorted(bin_list)
    label_list = sorted(label_list)
    events = []
    labels = []
    n_event = len(bin_list)
    n_event_train = n_event - 10  # int(n_event/10 * 9)
    n_event_test = 10  # n_event - n_event_train
    if group_name == 'train':
        bin_list = bin_list[:n_event_train]
        label_list = label_list[:n_event_train]
        n_event = n_event_train
    elif group_name == 'test':
        bin_list = bin_list[-n_event_test:]
        label_list = label_list[-n_event_test:]
        n_event = n_event_test
    for i in range(n_event):
        with open(bin_list[i], 'rb') as f:
            # data = f.read()
            # event_x = np.array(list(data[0::5]))
            # event_y = np.array(list(data[1::5]))
            # event_p = np.array([byte >> 7 for byte in data[2::5]])
            # event_t = np.array([(int(b1 & 0x7f) << 16) + (int(b2) << 8) + int(b3)
            #                     for b1, b2, b3 in zip(data[2::5], data[3::5], data[4::5])])
            # event = np.vstack([event_t, event_x, event_y, event_p]).T
            # events.append(event)
            ### chat gpt god in god ###
            data = np.fromfile(f, dtype='uint8')
            event_x = data[0::5]
            event_y = data[1::5]
            event_p = np.right_shift(data[2::5], 7)
            event_t = np.zeros(len(event_x), dtype='uint32')
            event_t |= data[2::5] & 0x7F
            event_t <<= 8
            event_t |= data[3::5]
            event_t <<= 8
            event_t |= data[4::5]
            event = np.vstack([event_t, event_x, event_y, event_p]).T
            events.append(event)
        with open(label_list[i], 'rb') as f:
            data = np.fromfile(f, dtype='int16')
            label = np.zeros(5)
            label[0] = key
            label[1] = data[2]
            label[2] = data[3]
            label[3] = data[6] - data[2]
            label[4] = data[7] - data[3]
            labels.append(label)

    return events, labels



def create_events_hdf5(hdf5_filename, n_class, group_name):
    folder_path = './dataset/NCaltech101/Caltech101'
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
            events, labels = bin_to_events(key, folder_name, group_name)
            # print(len(events))
            if len(events) > 0:
                for i in range(len(events)):
                    subgroup = f.create_group(str(n_grp))
                    dataset_t = subgroup.create_dataset('time', events[i][:, 0].shape, dtype=np.uint32)
                    dataset_d = subgroup.create_dataset('data', events[i][:, 1:].shape, dtype=np.uint8)
                    dataset_l = subgroup.create_dataset('labels', labels[i].shape, dtype=np.uint32)
                    dataset_t[...] = events[i][:, 0]
                    dataset_d[...] = events[i][:, 1:]
                    dataset_l[...] = labels[i]
                    n_grp += 1
                key += 1
    print('generate caltech-101 ' + group_name + ' dataset, used %.3fs' % (time.time() - start_time))

def one_hot(mbt, num_classes):
    out = np.zeros(num_classes)
    out[mbt] = 1
    return out


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)


def chunk_evs_pol(times, datas, size, dt=1000, n_step=10, ds=4):
    t_start = times[0]
    ts = range(t_start, t_start + n_step * dt, dt)
    chunks = np.zeros(size + [len(ts)], dtype='int8')
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(times[idx_end:], t + dt)
        if idx_end > idx_start:
            ee = datas[idx_start:idx_end]
            p, x, y = ee[:, 2], ee[:, 0] // ds, ee[:, 1] // ds
            np.add.at(chunks, (p, x, y, i), 1)
        idx_start = idx_end
    return chunks


if __name__ == "__main__":
    caltech_dataset = CaltechDataset(group_name='train', n_class=76)
    dataloader = torch.utils.data.DataLoader(caltech_dataset, batch_size=256)
    for i, (inputs, labels) in enumerate(dataloader):
         print(str(i))
    #     # print(labels)

