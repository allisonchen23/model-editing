import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(logits, target):
    loss = torch.nn.CrossEntropyLoss()
    return loss(logits, target)
