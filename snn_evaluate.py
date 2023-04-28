import argparse
from data_loader.dvsgesture_dataset import *
from data_loader.nmnist_dataset import *
from data_loader.caltech_dataset import *
from data_loader.cifar_dataset import *
from models.scnn import *
from models.sresnet import *
from LogME.LogME import *

parser = argparse.ArgumentParser(description='snn evaluation')
parser.add_argument('-g', help='Number of GPU to use', default=0, type=int)
parser.add_argument('-t', help='Duration of one time slice (ms)', default=10, type=int)
parser.add_argument('-r', help='Learning rate', default=4, type=int)
parser.add_argument('-c', help='Number of classes', default=0, type=int)
parser.add_argument('-m', help='Mode, 0: init params, 1: load params', default=1, type=int)
parser.add_argument('-f', help='Train, 0: train, 1: evaluate, 2 fine-tune', default=2, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.g)


def lr_scheduler(_optimizer, _epoch, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if _epoch % lr_decay_epoch == 0 and _epoch > 0:
        for param_group in _optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.3
    return _optimizer


class Model(nn.Module):

    def __init__(self, device, n_class, m_name):
        super(Model, self).__init__()
        self.device = device
        if m_name == 'sres4':
            self.backbone = SResnet4(device=self.device)
            self.classifier = nn.Linear(256 * 1 * 1, n_class)
        elif m_name == 'sres5':
            self.backbone = SResnet5(device=self.device)
            self.classifier = nn.Linear(256 * 4 * 4, n_class)
        elif m_name == 'sres6':
            self.backbone = SResnet6(device=self.device)
            self.classifier = nn.Linear(128 * 1 * 1, n_class)
        elif m_name == 'sres7':
            self.backbone = SResnet7(device=self.device)
            self.classifier = nn.Linear(64 * 1 * 1, n_class)
        elif m_name == 'sres18':
            self.backbone = SResnet18(device=self.device)
            self.classifier = nn.Linear(512 * 4 * 4, n_class)
        elif m_name == 'scnn4':
            self.backbone = SCNN4(device=self.device)
            self.classifier = nn.Linear(128 * 1 * 1, n_class)
        elif m_name == 'scnn3':
            self.backbone = SCNN3(device=self.device)
            self.classifier = nn.Linear(256 * 1 * 1, n_class)
        elif m_name == 'scnn2':
            self.backbone = SCNN2(device=self.device)
            self.classifier = nn.Linear(256 * 8 * 8, n_class)
        elif m_name == 'scnn1':
            self.backbone = SCNN1(device=self.device)
            self.classifier = nn.Linear(64 * 1 * 1, n_class)
        elif m_name == 'scnn0':
            self.backbone = SCNN0(device=self.device)
            self.classifier = nn.Linear(64 * 16 * 16, n_class)
        self.n_class = n_class
        self.spike_func = SpikeFunc.apply


    def forward(self, input):
        input = self.backbone(input)
        shape = input.shape
        n_step = input.shape[2]
        h0_mem = h0_spike = h0_sumspike = torch.zeros(shape[0], self.n_class, device=self.device)

        for step in range(n_step):
            x = input[..., step]
            h0_mem, h0_spike = self.mem_update(self.classifier, x, h0_mem, h0_spike)
            h0_sumspike += h0_spike

        output = h0_sumspike / n_step
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike

# 数据集统一调用接口
def get_data_loader(
    dataset_name,   # 数据集名称
    group_name,     # 训练/测试
    n_step,         # 时间步数量（帧数）
    n_class,        # 数据集内种类数
    ds,             # 空间分辨率
    dt,             # 时间分辨率
    batch_size,
    num_workers
    ):
    
    print(dataset_name + ' ' + group_name)
    if dataset_name == 'dg':
        size = [2, int(128 / ds), int(128 / ds)]
        dataset = DvsGestureDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, drop_last=False)
        return dataloader
    elif dataset_name == 'nm':
        dataset = NMnistDataset(n_step=n_step, n_class=n_class, group_name=group_name)
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, drop_last=False)
        return dataloader
    elif dataset_name == 'ct':
        size = [2, int(256 / ds), int(192 / ds)]
        dataset = CaltechDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, drop_last=False)
        return dataloader
    elif dataset_name == 'cf':
        size = [2, int(128 / ds), int(128 / ds)]
        dataset = CifarDataset(n_step=n_step, n_class=n_class, group_name=group_name, size=size, ds=ds, dt=dt)
        if batch_size == 0:
            batch_size = dataset.__len__()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, drop_last=False)
        return dataloader

# 模型调用统一接口
def get_model(
    device,                   # 加载到设备
    dataset,                  # 数据集名称
    m_name,                   # 模型名称
    load_backbone_only=False  # 加载模式
    ):
    
    max_n_class = 0
    if dataset == 'dg':
        max_n_class = 11
    elif dataset == 'nm':
        max_n_class = 10
    elif dataset == 'ct':
        max_n_class = 100
    elif dataset == 'cf':
        max_n_class = 10
    model = Model(device=device, n_class=max_n_class, m_name=m_name)
    train_loss_record = []  # 040202 after only
    test_loss_record = []  # 040202 after only
    train_acc_record = []  # 040202 after only
    acc_record = []
    epoch = 0
    mode = args.m
    if mode == 1:
        pt_files = glob.glob(os.path.join(model_save_path, '*.pth'))
        ft_files = glob.glob(os.path.join(model_save_path, '*ft*.pth'))
        checkpoint_files = list(set(pt_files).difference(set(ft_files)))
        if len(checkpoint_files) > 0:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            print('load parameters from ' + latest_checkpoint)
            state = torch.load(latest_checkpoint)
            if load_backbone_only == True:
                model_dict = model.state_dict()
                pt_dict = state['model']
                for k, v in pt_dict.items():
                    if 'backbone' in k:
                        model_dict.update({k: v})
                        # print(k)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state['model'])
            train_loss_record = state['train_loss_record']  # 040202 after only
            test_loss_record = state['test_loss_record']  # 040202 after only
            train_acc_record = state['train_acc_record']  # 040202 after only
            acc_record = state['acc_record']
            epoch = state['epoch']
        else:
            print('0 saves found, initialize parameters')
    else:
        print('initialize parameters')
    criterion = nn.MSELoss()
    return model, train_loss_record, test_loss_record, train_acc_record, acc_record, epoch, criterion, max_n_class

# 模型训练
def train(model,      # 模型
          criterion,  # 损失函数
          optimizer,  # 优化器
          train_loader, test_loader,
          train_loss_record, test_loss_record,
          train_acc_record, acc_record,
          epoch, n_epoch,
          ft=''
          ):
    
    model.to(device)
    start_time = time.time()
    while epoch < n_epoch:
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for _, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, gt = torch.max(labels, 1)
            total = total + gt.size(0)
            correct = correct + (predicted == gt).sum().item()
            loss = criterion(outputs, labels)
            epoch_loss = epoch_loss + loss.item()
            loss.backward()
            optimizer.step()
        optimizer = lr_scheduler(optimizer, epoch, lr_decay_epoch=100)
        train_acc = 100. * float(correct) / float(total)
        train_acc_record.append(train_acc)
        train_loss_record.append(epoch_loss)
        print('epoch %d/%d, loss: %.5f' % (epoch + 1, n_epoch, epoch_loss), end='')
        print('  train accuracy: %.3f' % train_acc, end='')

        torch.cuda.empty_cache()

        model.eval()
        epoch_loss = 0
        correct = 0
        total = 0
        for _, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, gt = torch.max(labels, 1)
            total = total + gt.size(0)
            correct = correct + (predicted == gt).sum()
            loss = criterion(outputs, labels)
            epoch_loss = epoch_loss + loss.item()
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)
        test_loss_record.append(epoch_loss)
        print('  test accuracy: %.3f' % acc, end='')
        print('  estimated remaining time: %.3fh' % ((time.time() - start_time) / (epoch + 1) / 3600 * (n_epoch - epoch - 1)))
        epoch += 1

        if epoch % 10 == 0 and epoch > 0:
            print('saving model: ' + model_save_path + model_name + ft + '_{}.pth'.format(epoch))
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'train_loss_record': train_loss_record,  # 040202 after only
                'test_loss_record': test_loss_record,  # 040202 after only
                'train_acc_record': train_acc_record,  # 040202 after only
                'acc_record': acc_record
            }
            if not os.path.isdir(model_save_path):
                os.mkdir(model_save_path)
            torch.save(state, model_save_path + model_name + ft + '_{}.pth'.format(epoch))

# 测试
def evaluate(
    model,      # 模型
    test_loader # 测试数据
    ):
    
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        _, gt = torch.max(labels, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == gt).sum()
    print('test accuracy: %.3f' % (100. * float(correct) / float(total)))


if __name__ == '__main__':
    number = '042602'
    ds_name = 'dg'

    # dg: dvs-gesture
    # nm: n-mnist
    # ct: n-caltech101
    # cf: cifar10-dvs

    m_name = 'sres4'
    # sres: 4, 5, 6, 7, 18
    # scnn: 0, 1, 2, 3, 4
    n_step = 100
    dt = args.t * 1000  # temporal resolution, in us

    model_name = '{}_{}_{}_{}c'.format(number, ds_name, m_name, args.c)
    batch_size = 40
    ds = 4  # spatial resolution
    n_class = args.c
    n_epoch = 400
    learning_rate = (10 ** (-args.r))
    model_save_path = './models_save/' + model_name + '/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 16

    # print(device)
    if args.f == 0:
        print('== train ==')
        model, train_loss_record, test_loss_record, train_acc_record, acc_record, epoch, criterion, n_class_test = \
            get_model(device, ds_name, m_name)
        
        train_loader = get_data_loader(dataset_name=ds_name, group_name='train', n_step=n_step, n_class=n_class, ds=ds,
                                       dt=dt, batch_size=batch_size, num_workers=num_workers)
        test_loader = get_data_loader(dataset_name=ds_name, group_name='test', n_step=n_step, n_class=n_class_test, ds=ds,
                                      dt=dt, batch_size=batch_size, num_workers=num_workers)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train(model, criterion, optimizer, train_loader, test_loader, train_loss_record, test_loss_record,
              train_acc_record, acc_record, epoch=epoch, n_epoch=n_epoch)

    elif args.f == 1:
        print('== evaluate ==')
        model, _, _, _, _, _, _, n_class_test = get_model(device, ds_name, m_name)

        test_loader = get_data_loader(dataset_name=ds_name, group_name='test', n_step=n_step, n_class=n_class_test,
                                      ds=ds, dt=dt, batch_size=batch_size, num_workers=num_workers)
        evaluate(model, test_loader)

    elif args.f == 2:
        print('== fine-tune ==')
        ft_ds_name = 'dg'
        n_class_ft = 0
        if ft_ds_name == 'dg':
            n_class_ft = 11
        elif ft_ds_name == 'nm':
            n_class_ft = 10
        elif ft_ds_name == 'ct':
            n_class_ft = 100
        elif ft_ds_name == 'cf':
            n_class_ft = 10
        model, train_loss_record, test_loss_record, train_acc_record, acc_record, _, criterion, _\
            = get_model(device, ft_ds_name, m_name, load_backbone_only=True)

        train_loader = get_data_loader(dataset_name=ft_ds_name, group_name='train', n_step=n_step, n_class=n_class_ft,
                                       ds=ds, dt=dt, batch_size=batch_size, num_workers=num_workers)
        test_loader = get_data_loader(dataset_name=ft_ds_name, group_name='test', n_step=n_step, n_class=n_class_ft,
                                      ds=ds, dt=dt, batch_size=batch_size, num_workers=num_workers)
        for param in model.backbone.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        train(model, criterion, optimizer, train_loader, test_loader, train_loss_record, test_loss_record,
              train_acc_record, acc_record, epoch=0, n_epoch=200, ft='_ft_{}'.format(ft_ds_name))
