import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(output, target):
    return torch.nn.CrossEntropyLoss(output, target)
