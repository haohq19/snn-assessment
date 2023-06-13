import glob
import os
import torch.utils.data
from spike_data.utils import *


class CifarDataset(torch.utils.data.Dataset):

    num_instances = 10

    def __init__(
            self,
            n_step=10,
            n_class=0,
            group_name='train',
            size=None,
            ds=4,
            dt=1000):
        super(CifarDataset, self).__init__()
        if size is None:
            size = [2, 32, 32]
        self.n_step = n_step
        if n_class == 0:
            self.n_class = CifarDataset.num_instances
        else:
            self.n_class = n_class
        self.group_name = group_name
        self.size = size
        self.ds = ds
        self.dt = dt
        self.filename = 'spike_data/dataset/cifar_events_' + self.group_name + '_{}.hdf5'.format(self.n_class)
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
        if (self.group_name == 'train') and (t_begin < t_end):  # for test set, start time is fixed
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
        return torch.from_numpy(frames).float(), torch.from_numpy(one_hot(label[0], self.n_class_total)).float()

    def __len__(self):
        return self.n_sample

    @classmethod
    def get_instances_num(cls):
        return CifarDataset.num_instances


def aedat_to_events(folder_name, group_name):
    aedat_list = glob.glob('spike_data/dataset/CIFAR10/' + folder_name + '/*.aedat')
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
    folder_path = 'spike_data/dataset/CIFAR10'
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

