import torch


def euclidean_dist(x, y):
    """

    :param torch.Tensor x:
    :param torch.Tensor y:
    :rtype: torch.Tensor
    :return:  dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).view(m, 1)
    yy = torch.pow(y, 2).sum(1, keepdim=True).view(n, 1).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())


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
