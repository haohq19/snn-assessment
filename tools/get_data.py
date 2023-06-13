from spike_data.dvsgesture_dataset import *
from spike_data.nmnist_dataset import *
from spike_data.caltech_dataset import *
from spike_data.cifar_dataset import *

def get_data(config):
    dataset_name = config['type']
    group_name = config['group_name']
    ds = config['ds']
    dt = config['dt']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    n_step = config['n_steps']
    n_class = config['n_classes'],
    shuffle = config['shuffle'],
    drop_last = config['drop_last']
    print('utils.get_data.get_data: ' + dataset_name + ' ' + group_name)

    if dataset_name == 'dg':
        size = [2, int(128 / ds), int(128 / ds)]
        dataset = DvsGestureDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)

        if batch_size == 0:
            batch_size = dataset.__len__()

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers, drop_last=drop_last)
        return dataloader
    elif dataset_name == 'nm':
        dataset = NMnistDataset(n_step=n_step, n_class=n_class, group_name=group_name)

        if batch_size == 0:
            batch_size = dataset.__len__()

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers, drop_last=drop_last)
        return dataloader
    elif dataset_name == 'ct':
        size = [2, int(256 / ds), int(192 / ds)]
        dataset = CaltechDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)

        if batch_size == 0:
            batch_size = dataset.__len__()

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers, drop_last=drop_last)
        return dataloader
    elif dataset_name == 'cf':
        size = [2, int(128 / ds), int(128 / ds)]
        dataset = CifarDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)

        if batch_size == 0:
            batch_size = dataset.__len__()

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                 num_workers=num_workers, drop_last=drop_last)
        return dataloader


def get_instances_num(config):
    dataset_name = config['type']
    if dataset_name == 'dg':
        return DvsGestureDataset.get_instances_num()
    elif dataset_name == 'cf':
        return CifarDataset.get_instances_num()
    elif dataset_name == 'ct':
        return CaltechDataset.get_instances_num()
    elif dataset_name == 'nm':
        return NMnistDataset.get_instances_num()
