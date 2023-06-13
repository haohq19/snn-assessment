import argparse
from utils.get_data import *
from utils.get_model import *
from utils.optimizer import *
from models.criterion import *

parser = argparse.ArgumentParser(description='eval model')
parser.add_argument('--gpu', help='Number of GPU to use', default=0, type=int)
parser.add_argument('--dt', help='Duration of one time slice (ms)', default=10, type=int)
parser.add_argument('--cls', help='Number of classes', default=11, type=int)
parser.add_argument('--ld', help='Mode, 0: init params, 1: load params', default=0, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


# 模型训练
def eval(model,  # 模型
         criterion,  # 损失函数
         test_loader,
         device,
         ):

    model.to(device)


    model.eval()
    test_set_loss = 0
    correct = 0
    total = 0

    for _, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        out = outputs.detach().cpu().numpy()
        _, predicted = torch.max(outputs, 1)
        _, gt = torch.max(labels, 1)
        total = total + gt.size(0)
        correct = correct + (predicted == gt).sum().item()
        loss = criterion(outputs, labels)
        test_set_loss = test_set_loss + loss.item()

    acc = 100. * float(correct) / float(total)
    print('accuracy: %.3f' % acc)
    print('loss: %.3f' % test_set_loss)



if __name__ == '__main__':
    number = '050801'
    dataset_name = 'dg'
    ft_dataset_name = 'dg'

    # dg: dvs-gesture
    # nm: n-mnist
    # ct: n-caltech101
    # cf: cifar10-dvs

    model_name = 'sres7'

    # sres: 4, 5, 6, 7, 18
    # scnn: 0, 1, 2, 3, 4

    n_step = 100
    dt = args.dt * 1000  # temporal resolution, in us

    weight_name = '{}_{}_{}_{}c'.format(number, dataset_name, model_name, args.cls)
    classifier_name = 'lc'
    batch_size = 40
    ds = 4  # spatial resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 16

    print('evaluate: eval models')
    train_loader, n_class_total = get_data(dataset_name=ft_dataset_name,
                                           group_name='train',
                                           n_step=n_step,
                                           n_class=args.cls,
                                           ds=ds,
                                           dt=dt,
                                           batch_size=batch_size,
                                           num_workers=num_workers)

    test_loader, _ = get_data(dataset_name=ft_dataset_name,
                              group_name='test',
                              n_step=n_step,
                              n_class=n_class_total,
                              ds=ds,
                              dt=dt,
                              batch_size=batch_size,
                              num_workers=num_workers)

    model, _, _, _, _, _ = get_model(device=device,
                                     n_class_target=n_class_total,
                                     model_name=model_name,
                                     weight_name=weight_name,
                                     load_param=1,
                                     classifier_name=classifier_name
                                     )

    eval(model=model,
         criterion=MemLoss(),
         test_loader=test_loader,
         device=device
         )


