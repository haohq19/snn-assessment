import glob
import os
import torch.utils.data
from spike_data.utils import *


class CaltechDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            n_step=10,
            n_class=0,
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
        self.filename = 'spike_data/dataset/caltech_events_' + self.group_name + '_{}.hdf5'.format(self.n_class)
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
        times = sub_group['time']
        events = sub_group['data']
        t_begin = times[()][0]
        t_end = times[()][-1] - self.n_step * self.dt
        start_time = t_begin
        if (self.group_name == 'train') and (t_begin < t_end):
            start_time = np.random.randint(t_begin, t_end)
        i_begin = bisect.bisect_left(times, start_time)
        i_end = bisect.bisect_left(times[i_begin:], start_time + self.n_step * self.dt) + i_begin
        frames = events_to_frames(
            times[i_begin: i_end],
            events[i_begin: i_end],
            dt=self.dt,
            ds=self.ds,
            n_step=self.n_step,
            size=self.size
        )
        return torch.from_numpy(frames).float(), torch.from_numpy(one_hot(label[0], 100)).float()

    def __len__(self):
        return self.n_sample


def bin_to_events(key, folder_name, group_name):
    bin_list = glob.glob('spike_data/dataset/NCaltech101/Caltech101/' + folder_name + '/*.bin')
    label_list = glob.glob('spike_data/dataset/NCaltech101/Caltech101_annotations/' + folder_name + '/*.bin')
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
    folder_path = 'spike_data/dataset/NCaltech101/Caltech101'
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


if __name__ == "__main__":
    caltech_dataset = CaltechDataset(group_name='train', n_class=76)
    dataloader = torch.utils.data.DataLoader(caltech_dataset, batch_size=256)
    for i, (inputs, labels) in enumerate(dataloader):
         print(str(i))
    #     # print(labels)

