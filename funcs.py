import torch
import torch.nn.functional as F
from models import *

def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y/T, dim=1), F.softmax(teacher_scores/T, dim=1)) * (T*T * 2. * alpha)\
           + F.cross_entropy(y, labels) * (1. - alpha)


def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_no_params(net, verbose=True):

    params = net.state_dict()
    tot= 0
    conv_tot = 0
    for p in params:
        no = params[p].view(-1).__len__()
        tot += no
        if 'bn' not in p:
            if verbose:
                print('%s has %d params' % (p,no))
        if 'conv' in p:
            conv_tot += no

    if verbose:
        print('Net has %d conv params' % conv_tot)
        print('Net has %d params in total' % tot)
    return tot


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



