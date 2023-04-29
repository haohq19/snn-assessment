import argparse
from data_loader.dvsgesture_dataset import *
from data_loader.caltech_dataset import *
from models.scnn import *
from models.sresnet import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='snn evaluation')
parser.add_argument('-g', help='Number of GPU to use', default=0, type=int)
parser.add_argument('-t', help='Duration of one time slice (ms)', default=50, type=int)
parser.add_argument('-r', help='Learning rate', default=4, type=int)
parser.add_argument('-c', help='Number of classes', default=100, type=int)
parser.add_argument('-m', help='Mode, 0: init paras, 1: load paras', default=1, type=int)
parser.add_argument('-f', help='Train, 0: train, 1: evaluate, 2 fine-tune, 3 feat-eval', default=2, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.g)


class Model(nn.Module):

    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device
        self.backbone = SCNN2(device=self.device)

    def forward(self, input):
        output = self.backbone(input)
        return output

    def mem_update(self, _func, _x, _mem, _spike):
        _mem = _mem * decay * (1 - _spike) + _func(_x)
        _spike = self.spike_func(_mem)
        return _mem, _spike



def get_model(device, index, mode):

    model = Model(device=device)
    pt_files = glob.glob(os.path.join(model_save_path, '*.pth'))
    ft_files = glob.glob(os.path.join(model_save_path, '*ft*.pth'))
    pt_files = list(set(pt_files).difference(set(ft_files)))
    if mode == 1:
        pt_files = ft_files
    if len(pt_files) > 0 and len(pt_files) > index:
        pt_files.sort(key=os.path.getctime)
        # print(pt_files[index], end='')
        state = torch.load(pt_files[index])
        model_dict = model.state_dict()
        pt_dict = state['model']
        for k, v in pt_dict.items():
            if 'backbone' in k:
                model_dict.update({k: v})
        model.load_state_dict(model_dict)
    else:
        print('0 saves found, initialize parameters')
    return model


if __name__ == '__main__':
    number = '040702'
    ds_name = 'dg'
    n_classes = []
    md_name = 'scnn2'
    if ds_name == 'dg':
        n_classes = [11, 9, 7, 5, 3, 0]
    elif ds_name == 'nm':
        n_classes = []
    elif ds_name == 'ct':
        n_classes = []
    elif ds_name == 'cf':
        n_classes = [10, 8, 6, 4, 2, 0]

    for n_class in n_classes:
        model_name = '{}_{}_{}_{}c'.format(number, ds_name, md_name, n_class)
        model_save_path = './models_save/' + model_name + '/'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_model(device, -1, 0)

        params = torch.nn.utils.parameters_to_vector(model.backbone.parameters()).detach().cpu().numpy()
        mean = np.mean(params)
        abs_mean = np.mean(abs(params))
        std = np.std(params)

        print(" n_class: %.2d" % n_class, " mean: %.6f" % mean, " abs_mean: %.6f" % abs_mean, " std: %.6f" % std)



