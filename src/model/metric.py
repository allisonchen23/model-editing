import torch


def accuracy(output, target):
    '''
    Return accuracy
    Arg(s):
        output : N x C torch.tensor
            logit outputs of model
        target : N x 1 torch.tensor
            integer labels
    Returns:
        float : accuracy of predictions

    '''
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
