import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required


class GD(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0):
        self.lr = lr
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(GD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GD, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha = weight_decay)
                p.data.add_(d_p , alpha = -group['lr'])   #p.data -= d_p * group['lr'];p.data.add_(d_p , alpha = -group['lr'])
        return loss

    def fix_siz(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha = weight_decay)
                p.data.add_(d_p , alpha = -group['lr'])   #p.data -= d_p * group['lr'];p.data.add_(d_p , alpha = -group['lr'])
        return loss


    def step_fold(self,c_l,c_s, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p ,ci,c in zip(group['params'],c_l.values(),c_s.values()):
                if p.grad is None:
                    continue
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                p.grad.data = p.grad.data.to(device)
                c.data = c.data.to(device)
                ci.data = ci.data.to(device)
                d_p = p.grad.data+c.data-ci.data

                p.data=p.data-d_p.data*group['lr']

        return loss

    def adjust_learning_rate(self, round_i):
        raise BaseException("Deleted.")

        lr = self.lr * (0.5 ** (round_i // 30))
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def soft_decay_learning_rate(self):
        raise BaseException("Deleted.")
        self.lr *= 0.99
        for param_group in self.param_groups:
            param_group['lr'] = self.lr

    def inverse_prop_decay_learning_rate(self, round_i):
        for param_group in self.param_groups:
            param_group['lr'] = self.lr/(round_i+1)

    def set_lr(self, lr):
        raise BaseException("Deleted.")
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        return self.param_groups[0]['lr']


class LrdGD(GD):
    def __init__(self, params, lr=required, weight_decay=0):
        raise BaseException("Deleted.")
        super(LrdGD, self).__init__(params, lr, weight_decay)

    def step(self, lr, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-lr, d_p)
        return loss
