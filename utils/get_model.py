from data_loader.dvsgesture_dataset import *
from data_loader.nmnist_dataset import *
from data_loader.caltech_dataset import *
from data_loader.cifar_dataset import *
from models.scnn import *
from models.sresnet import *

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

def get_model(
        device,  # 加载到设备
        n_class_target,  # 分类种类
        model_name,  # 模型名称
        weight_name,  # 权重名称
        load_param=False,  # load pre-trained weights or not
        load_backbone_only=False  # 加载模式
):
    """

    Args:
        device: device to load the model
        n_class_target: num of classes to classify
        model_name: name of model
        weight_name: name of pre-trained weights
        load_param: load weights or not
        load_backbone_only: load backbone only or not

    Returns: model, train_loss_record, test_loss_record, train_acc_record, acc_record, epoch

    """
    model = Model(device=device, n_class=n_class_target, m_name=model_name)
    train_loss_record = []  # 040202 after only
    test_loss_record = []  # 040202 after only
    train_acc_record = []  # 040202 after only
    acc_record = []  # test_acc_record
    epoch = 0  # current epoch

    weight_save_path = './models_save/' + model_name + '/'

    if load_param is True:
        pt_weights = glob.glob(os.path.join(weight_save_path, '*.pth'))
        ft_weights = glob.glob(os.path.join(weight_save_path, '*ft*.pth'))
        pt_weights = list(set(pt_weights).difference(set(ft_weights)))
        if len(pt_weights) > 0:
            latest_pt_weight = max(pt_weights, key=os.path.getctime)
            print('utils.get_model.get_model: load weights from ' + latest_pt_weight)
            state = torch.load(latest_pt_weight)
            if load_backbone_only is True:
                model_dict = model.state_dict()
                pt_dict = state['model']
                for k, v in pt_dict.items():
                    if 'backbone' in k:
                        model_dict.update({k: v})
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state['model'])
            train_loss_record = state['train_loss_record']  # 040202 after only
            test_loss_record = state['test_loss_record']  # 040202 after only
            train_acc_record = state['train_acc_record']  # 040202 after only
            acc_record = state['acc_record']
            epoch = state['epoch']
        else:
            print('utils.get_model.get_model: 0 pt-weights found, initialize weights')
    else:
        print('utils.get_model.get_model: initialize weights')

    return model, train_loss_record, test_loss_record, train_acc_record, acc_record, epoch
