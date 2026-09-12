import torch.optim.lr_scheduler as _Scheduler


class PolyLRScheduler:
    """
    Polynomial learning rate decay:
        lr = initial_lr * (1 - current_epoch / max_epochs) ** exponent
    """

    def __init__(self, optimizer, max_epochs: int, exponent: float = 0.9, initial_lr: float = 0.01):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.exponent = exponent
        self.initial_lr = initial_lr
        self.current_epoch = 0

    def step(self, epoch: int = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        progress = min(self.current_epoch / max(1, self.max_epochs), 1.0)
        lr = self.initial_lr * (1.0 - progress) ** self.exponent

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]
