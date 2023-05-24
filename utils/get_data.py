from spike_data.dvsgesture_dataset import *
from spike_data.nmnist_dataset import *
from spike_data.caltech_dataset import *
from spike_data.cifar_dataset import *

def get_data(
        dataset_name,  # 数据集名称
        group_name,  # 训练/测试
        n_step,  # 时间步数量（帧数）
        n_class,  # 数据集内种类数
        ds,  # 空间分辨率
        dt,  # 时间分辨率
        batch_size,
        num_workers
):
    """

    Args:
        dataset_name: name of dataset
        group_name: train or test
        n_step: time steps
        n_class: classes included in the sub-dataset
        ds: spatial resolution
        dt: temporal resolution
        batch_size: batch size
        num_workers: num of workers

    Returns: dataloader, n_class_target

    """

    print('utils.get_data.get_data: ' + dataset_name + ' ' + group_name)

    if dataset_name == 'dg':
        size = [2, int(128 / ds), int(128 / ds)]
        dataset = DvsGestureDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)
        n_class_target = dataset.n_class_total
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers, drop_last=False)
        return dataloader, n_class_target
    elif dataset_name == 'nm':
        dataset = NMnistDataset(n_step=n_step, n_class=n_class, group_name=group_name)
        n_class_target = dataset.n_class_total
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers, drop_last=False)
        return dataloader, n_class_target
    elif dataset_name == 'ct':
        size = [2, int(256 / ds), int(192 / ds)]
        dataset = CaltechDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)
        n_class_target = dataset.n_class_total
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers, drop_last=False)
        return dataloader, n_class_target
    elif dataset_name == 'cf':
        size = [2, int(128 / ds), int(128 / ds)]
        dataset = CifarDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)
        n_class_target = dataset.n_class_total
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers, drop_last=False)
        return dataloader, n_class_target
