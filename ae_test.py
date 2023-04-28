import argparse
from data_loader.dvsgesture_dataset import *
from data_loader.nmnist_dataset import *
from data_loader.caltech_dataset import *
from data_loader.cifar_dataset import *
from models.autoencoder import *

parser = argparse.ArgumentParser(description='snn evaluation')
parser.add_argument('-g', help='Number of GPU to use', default=0, type=int)
parser.add_argument('-c', help='Number of classes', default=0, type=int)
parser.add_argument('-m', help='Mode, 0: init params, 1: load params', default=0, type=int)
parser.add_argument('-f', help='Train, 0: train, 1: evaluate, 2 fine-tune', default=0, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.g)

    
def lr_scheduler(_optimizer, _epoch, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if _epoch % lr_decay_epoch == 0 and _epoch > 0:
        for param_group in _optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.3
    return _optimizer

if __name__ == '__main__':
    
    n_step = 100
    dt = 10 * 1000  # temporal resolution, in us

    bs = 40
    ds = 4  # spatial resolution
    n_class = args.c
    n_epoch = 400
    learning_rate = (10 ** (-6))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 16
    size = [2, int(128 / ds), int(128 / ds)]
    dataset = DvsGestureDataset(n_step=n_step, n_class=n_class, group_name='train', size=size, ds=ds, dt=dt)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=False)
    model = AutoEncoder0(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 0
    loss_record = []
    model.to(device)
    start_time = time.time()
    
    while epoch < n_epoch:
        
        model.train()
        epoch_loss = 0
        for _, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            loss = model(inputs)
            epoch_loss = epoch_loss + loss.item()
            loss.backward()
            optimizer.step()
        optimizer = lr_scheduler(optimizer, epoch, lr_decay_epoch=100)
        loss_record.append(epoch_loss)
        print('epoch %d/%d, loss: %.5f' % (epoch + 1, n_epoch, epoch_loss), end='')
        print('  estimated remaining time: %.3fh' % ((time.time() - start_time) / (epoch + 1) / 3600 * (n_epoch - epoch - 1)))
        epoch += 1

