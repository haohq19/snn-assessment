
class Scheduler():

    def __init__(self, optimizer, lr, decay=0.3, lr_decay_epoch=100):
        """

        Args:
            optimizer: optimizer to scheduler the parameters
            lr: initial learning rate
            decay: decay rate
            lr_decay_epoch: epoch each learning rate decay
        """
        self.optimizer = optimizer
        self.lr = lr
        self.decay = decay
        self.lr_decay_epoch = lr_decay_epoch

    def step(self, epoch):
        lr = self.lr * self.decay ** int(epoch / self.lr_decay_epoch)
        if epoch % 10 == 0 and epoch > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return self.optimizer