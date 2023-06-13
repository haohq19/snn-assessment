import argparse
from tools.get_data import *
from tools.get_model import get_model
from tools.optimizer import *
from tools.get_timestamp import *
from models.criterion import *
from config.config import *

parser = argparse.ArgumentParser(description='train model')
parser.add_argument('--gpu', help='Number of GPU to use', default=0, type=int)
parser.add_argument('--dt', help='Temporal resolution of one frame (ms)', default=10, type=int)
parser.add_argument('--ds', help='Spatial resolution of one frame (pixel)', default=4, type=int)
parser.add_argument('--learning_rate', help='Learning rate', default=4, type=int)
parser.add_argument('--tr_ds', help='Name of the train dataset', default='', type=str)
parser.add_argument('--ts_ds', help='Name of the test dataset', default='', type=str)
parser.add_argument('--ft_ds', help='Name of the fine-tune dataset', default='', type=str)
parser.add_argument('--num_instances', help='Number of instances in the dataset', default=11, type=int)
parser.add_argument('--ld_backbone', help='if load backbone weights', default=0, type=int)
parser.add_argument('--ld_head', help='if load head weights', default=0, type=int)
parser.add_argument('--mode', help='run mode, 0: train, 1: fine-tune', default=0, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


# 模型训练
def train(model,  # 模型
          criterion,  # 损失函数
          scheduler,  # 优化器
          train_loader,
          test_loader,
          train_loss_record,
          test_loss_record,
          train_acc_record,
          acc_record,
          epoch,
          n_epoch,
          weight_name,
          device,
          ft_dataset_name=''
          ):

    model.to(device)
    start_time = time.time()
    weight_save_path = './models_save/' + weight_name + '/'
    if ft_dataset_name != '':
        weight_name += '_ft_{}'.format(ft_dataset_name)

    while epoch < n_epoch:
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        optimizer = scheduler.step(epoch)

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
        print('  estimated remaining time: %.3fh' % (
                    (time.time() - start_time) / (epoch + 1) / 3600 * (n_epoch - epoch - 1)))
        epoch += 1

        if epoch % 10 == 0 and epoch > 0:
            print('saving model: ' + weight_save_path + weight_name + '_{}.pth'.format(epoch))
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'train_loss_record': train_loss_record,  # 040202 after only
                'test_loss_record': test_loss_record,  # 040202 after only
                'train_acc_record': train_acc_record,  # 040202 after only
                'acc_record': acc_record
            }
            if not os.path.isdir(weight_save_path):
                os.mkdir(weight_save_path)
            torch.save(state, weight_save_path + weight_name + '_{}.pth'.format(epoch))


if __name__ == '__main__':
    config = Config()
    timestamp = get_timestamp()
    print(timestamp)
    train_dataset_name = args.tr_ds
    test_dataset_name = args.ts_ds
    finetune_dataset_name = args.ft_ds

    # dg: dvs-gesture
    # nm: n-mnist
    # ct: n-caltech101
    # cf: cifar10-dvs

    model_name = 'sres7'

    # sres: 4, 5, 6, 7, 18
    # scnn: 0, 1, 2, 3, 4


    n_step = 100
    dt = args.dt * 1000  # temporal resolution, in us


    weight_name = '{}_{}_{}_{}c'.format(timestamp, model_name, dataset_name, args.cls)
    classifier_name = 'rc'
    batch_size = 40
    ds = 4  # spatial resolution
    n_epoch = 400
    learning_rate = (10 ** (-args.lr))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 16

    # print(device)
    if args.ft == 0:
        print('train: train models')
        train_loader = get_data(config=config.config['train'])
        n_instances_total = get_instances_num(config.config['train'])
        test_loader = get_data(config=config.config['test'])

        model = get_model(config=config.config['model'], device=device)                                                                                                    )
        train_loss_record, test_loss_record, train_acc_record, acc_record, epoch = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = Scheduler(optimizer, learning_rate)
        train(model=model,
              criterion=nn.MSELoss(),
              scheduler=scheduler,
              train_loader=train_loader,
              test_loader=test_loader,
              train_loss_record=train_loss_record,
              test_loss_record=test_loss_record,
              train_acc_record=train_acc_record,
              acc_record=acc_record,
              epoch=epoch,
              n_epoch=n_epoch,
              weight_name=weight_name,
              device=device
              )


    elif args.ft == 1:
        print('train: fine-tune models')
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
        test_loader, _ = get_data(
            dataset_name=ft_dataset_name,
            group_name='test',
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
                                                                                                load_param=args.ld,
                                                                                                load_backbone_only=True,
                                                                                                classifier_name=classifier_name
                                                                                                )
        for param in model.backbone.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        scheduler = Scheduler(optimizer, learning_rate)
        train(
            model=model,
            criterion=nn.MSELoss(),
            scheduler=scheduler,
            train_loader=train_loader,
            test_loader=test_loader,
            train_loss_record=train_loss_record,
            test_loss_record=test_loss_record,
            train_acc_record=train_acc_record,
            acc_record=acc_record,
            epoch=0,
            n_epoch=200,
            weight_name=weight_name,
            device=device,
            ft_dataset_name=ft_dataset_name
        )
