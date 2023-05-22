import argparse
import os
import re
import numpy as np
from utils.get_data import *
from utils.get_model import *
from LogME.LogME import *

parser = argparse.ArgumentParser(description='snn assessment')
parser.add_argument('--gpu', help='Number of GPU to use', default=7, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


def map(model, train_loader, device):
    model.to(device)
    model.eval()
    outputs = []
    labels = []
    outs = []
    for _, (input, label) in enumerate(train_loader):

        label = np.argmax(label.numpy(), axis=1)
        output = model(input.to(device)).detach().cpu().numpy()
        out = np.mean(output, axis=2)
        outputs.append(output)
        outs.append(out)
        labels.append(label)

    return outputs




if __name__ == '__main__':
    number = '042510'
    dataset_name = 'dg'

    # dg: dvs-gesture
    # nm: n-mnist
    # ct: n-caltech101
    # cf: cifar10-dvs

    model_name = 'scnn1'

    # sres: 4, 5, 6, 7, 18
    # scnn: 0, 1, 2, 3, 4

    n_step = 100
    dt = 10 * 1000  # temporal resolution, in us
    cls = 11

    weight_name = '{}_{}_{}_{}c'.format(number, dataset_name, model_name, cls)
    classifier_name = 'lc'
    batch_size = 40
    ds = 4  # spatial resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 16

    ft_dataset_name = 'dg'
    train_loader, n_class_total = get_data(
        dataset_name=ft_dataset_name,
        group_name='train',
        n_step=n_step,
        n_class=0,
        ds=ds,
        dt=dt,
        batch_size=batch_size,
        num_workers=num_workers)

    model, train_loss_record, test_loss_record, train_acc_record, acc_record, _ = get_model(device,
                                                                                            n_class_total,
                                                                                            model_name,
                                                                                            weight_name,
                                                                                            load_param=1,
                                                                                            load_backbone_only=False,
                                                                                            classifier_name=classifier_name
                                                                                            )

    map(model=model, train_loader=train_loader, device=device)
