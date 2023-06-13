import struct
import numpy as np
import h5py
import glob
import os
import bisect
from spike_data.utils import *
import torch.utils.data


class DvsGestureDataset(torch.utils.data.Dataset):

    num_instances = 11

    def __init__(
            self,
            n_step=10,
            n_class=0,
            group_name='train',
            size=None,
            ds=4,
            dt=1000):
        """
        Args:
            n_step: time step of the spike neural network
            n_class: classes of the labels the generated dataset to contain, 0 means all
            group_name: train or test
            size: size of the input image, usually 2*32*32
            ds: spatial resolution
            dt: temporal resolution, in us
        """
        super(DvsGestureDataset, self).__init__()
        if size is None:
            size = [2, 32, 32]
        self.n_step = n_step
        if n_class == 0:
            self.n_class = DvsGestureDataset.num_instances
        else:
            self.n_class = n_class
        self.group_name = group_name
        self.size = size
        self.ds = ds
        self.dt = dt

        self.filename = 'spike_data/dataset/dvs_gestures_events_' + self.group_name + '_{}.hdf5'.format(self.n_class)
        if not os.path.isfile(self.filename):
            create_events_hdf5(self.filename, self.n_class, self.group_name)

        self.f = h5py.File(self.filename, 'r', swmr=True, libver="latest")
        self.n_subgroup = len(self.f)
        self.n_sample_ls = []
        self.n_sample = 0
        for sub_grp in range(self.n_subgroup):
            sub_grp_len = len(self.f[str(sub_grp)]['labels'][()])
            self.n_sample_ls.append(sub_grp_len)
            self.n_sample += sub_grp_len
        self.map = np.zeros(self.n_sample, dtype=int)
        self.n_sample_cnt = np.zeros(self.n_subgroup, dtype=int)
        i_begin = 0
        for sub_grp in range(self.n_subgroup):
            sub_grp_len = self.n_sample_ls[sub_grp]
            i_end = i_begin + sub_grp_len
            self.map[i_begin: i_end] = sub_grp
            i_begin += sub_grp_len
        for sub_grp in range(self.n_subgroup - 1):
            sub_grp_len = self.n_sample_ls[sub_grp]
            self.n_sample_cnt[sub_grp + 1] = self.n_sample_cnt[sub_grp] + sub_grp_len
        # print(self.map)
        # print(self.n_sample_cnt)

    def __getitem__(self, index):
        sub_grp_index = self.map[index]
        item_index = index - self.n_sample_cnt[sub_grp_index]
        sub_grp = self.f[str(sub_grp_index)]
        labels = sub_grp['labels'][()]
        label = labels[item_index, :]
        t_begin = label[1]
        t_end = label[2] - 2 * self.n_step * self.dt
        start_time = t_begin
        if (self.group_name == 'train') and (t_begin < t_end):
            start_time = np.random.randint(t_begin, t_end)
        label = label[0] - 1
        times = sub_grp['time']
        datas = sub_grp['data']
        i_begin = bisect.bisect_left(times, start_time)
        i_end = bisect.bisect_left(times[i_begin:], start_time + self.n_step * self.dt) + i_begin

        if i_end <= i_begin:
            return self.__getitem__(index+1)

        frames = events_to_frames(
            times[i_begin: i_end],
            datas[i_begin: i_end],
            dt=self.dt,
            ds=self.ds,
            n_step=self.n_step,
            size=self.size
        )
        return torch.from_numpy(frames).float(), torch.from_numpy(one_hot(label, self.n_class_total)).float()

    def __len__(self):
        return self.n_sample

    @classmethod
    def get_instances_num(cls):
        return DvsGestureDataset.num_instances


def gather_aedat(file_path, start_id, end_id):
    files = []
    for i in range(start_id, end_id):
        search_mask = file_path + '/user' + "{0:02d}".format(i) + '*.aedat'
        file = glob.glob(search_mask)
        files += file
    return files


def aedat_to_events(filename, label_range):
    label_filename = filename[:-6] + '_labels.csv'
    labels = np.loadtxt(label_filename, skiprows=1, delimiter=',',
                        dtype='uint32')  # ((label, start_time, end_time), ... ) 12 labels
    events = []
    with open(filename, 'rb') as f:
        for i in range(5):
            f.readline()
        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head) == 0:
                break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]

            if (eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber * eventsize), 'uint32')
                event_bytes = event_bytes.reshape(-1, 2)

                x = (event_bytes[:, 0] >> 17) & 0x00001FFF
                y = (event_bytes[:, 0] >> 2) & 0x00001FFF
                p = (event_bytes[:, 0] >> 1) & 0x00000001
                t = event_bytes[:, 1]
                events.append([t, x, y, p])

            else:
                f.read(eventnumber * eventsize)
    events = np.column_stack(events)
    events = events.astype('uint32')
    selected_events = np.zeros([4, 0], 'uint32')
    selected_labels = np.zeros([0, 3], 'uint32')
    for label in labels:
        if label[0] - 1 in range(label_range):
            start = np.searchsorted(events[0, :], label[1])
            end = np.searchsorted(events[0, :], label[2])
            selected_events = np.column_stack([selected_events, events[:, start:end]])
            selected_labels = np.row_stack([selected_labels, label])
    return selected_events.T, selected_labels


def compute_start_time(labels, pad):
    l0 = np.arange(len(labels[:, 0]), dtype='int')
    np.random.shuffle(l0)
    label = labels[l0[0], 0]
    tbegin = labels[l0[0], 1]
    tend = labels[l0[0], 2] - pad
    try:
        start_time = np.random.randint(tbegin, tend)
        return start_time, label
    except BaseException:
        return tbegin, label


def create_events_hdf5(hdf5_filename, n_class, group_name):
    files = []
    if group_name == 'train':
        files = gather_aedat(os.path.join('spike_data/dataset/DvsGesture'), 1, 24)
        print("generate dvs-gesture train dataset")
    elif group_name == 'test':
        files = gather_aedat(os.path.join('spike_data/dataset/DvsGesture'), 24, 30)
        print("generate dvs-gesture test dataset")
    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        key = 0
        for file in files:
            events, labels = aedat_to_events(file, n_class)
            if len(events) > 0:
                subgroup = f.create_group(str(key))
                dataset_t = subgroup.create_dataset('time', events[:, 0].shape, dtype=np.uint32)
                dataset_d = subgroup.create_dataset('data', events[:, 1:].shape, dtype=np.uint8)
                dataset_l = subgroup.create_dataset('labels', labels.shape, dtype=np.uint32)
                dataset_t[...] = events[:, 0]
                dataset_d[...] = events[:, 1:]
                dataset_l[...] = labels
                key += 1


if __name__ == "__main__":
    dvs_gesture_dataset = DvsGestureDataset(group_name='train', n_class=12)
    for i in range(dvs_gesture_dataset.__len__()):
        if i > 1170:
            print(str(i))
        data, label = dvs_gesture_dataset.__getitem__(i)
    dataloader = torch.utils.data.DataLoader(dvs_gesture_dataset, batch_size=256)
    for i, (inputs, labels) in enumerate(dataloader):
        print(str(i))
        # print(labels)

