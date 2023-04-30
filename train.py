import argparse
from utils.get_data import *
from utils.get_model import *
from utils.optimizer import *
from models.criterion import *

parser = argparse.ArgumentParser(description='train model')
parser.add_argument('--gpu', help='Number of GPU to use', default=7, type=int)
parser.add_argument('--dt', help='Duration of one time slice (ms)', default=10, type=int)
parser.add_argument('--lr', help='Learning rate', default=4, type=int)
parser.add_argument('--cls', help='Number of classes', default=11, type=int)
parser.add_argument('--ld', help='Mode, 0: init params, 1: load params', default=1, type=int)
parser.add_argument('--ft', help='Train, 0: train, 1: fine-tune', default=1, type=int)
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
    """

    Args:
        model: model
        criterion:
        scheduler:
        train_loader:
        test_loader:
        train_loss_record:
        test_loss_record:
        train_acc_record:
        acc_record:
        epoch:
        n_epoch:
        weight_name:
        device:
        ft_dataset_name:

    Returns:

    """
    model.to(device)
    start_time = time.time()
    weight_save_path = './models_save/' + weight_name + '/'
    if ft_dataset_name != '':
        weight_name += '_{}'.format(ft_dataset_name)

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
    dt = args.dt * 1000  # temporal resolution, in us

    weight_name = '{}_{}_{}_{}c'.format(number, dataset_name, model_name, args.cls)
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
        train_loader, n_class_total = get_data(dataset_name=dataset_name,
                                               group_name='train',
                                               n_step=n_step,
                                               n_class=args.cls,
                                               ds=ds,
                                               dt=dt,
                                               batch_size=batch_size,
                                               num_workers=num_workers)

        test_loader, _ = get_data(dataset_name=dataset_name,
                                  group_name='test',
                                  n_step=n_step,
                                  n_class=n_class_total,
                                  ds=ds,
                                  dt=dt,
                                  batch_size=batch_size,
                                  num_workers=num_workers)

        model, train_loss_record, test_loss_record, train_acc_record, acc_record, epoch= get_model(device=device,
                                                                                                   n_class_target=n_class_total,
                                                                                                   model_name=model_name,
                                                                                                   weight_name=weight_name,
                                                                                                   load_param=args.ld,
                                                                                                   classifier_name=classifier_name
                                                                                                   )


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
        ft_dataset_name = 'cf'
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
