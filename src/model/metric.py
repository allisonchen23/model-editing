import torch
import numpy as np
from sklearn.metrics import confusion_matrix

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

def _accuracy(output, target):
    print("helper acc")
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return torch.tensor([correct]), torch.tensor([len(target)])


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def per_class_accuracy(output, target):
    '''
    Return the accuracy of each class

    Arg(s):
        output : B x C torch.tensor or np.array
            model outputs (pre-softmax)
        target : B-dim torch.tensor or np.array
            integer binary ground truth labels

    Returns:
        C x 1 np.array of per class accuracy
    '''
    # Convert to numpy arrays
    if torch.is_tensor(output):
        output = output.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    # Make confusion matrix (rows are true, columns are predicted)
    n_classes = output.shape[1]
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    cmat = confusion_matrix(
        target,
        pred,
        labels=[i for i in range(n_classes)])

    # Get counts
    pred_counts = np.diagonal(cmat)
    target_counts = np.sum(cmat, axis=1)

    # Nan occurs if no target counts. In these cases, set those classes to 0
    return np.nan_to_num(pred_counts / target_counts)

def precision_recall_f1(output, target):
    '''
    Given outputs and targets, calculate per-class precision

    Arg(s):
        output : B x C torch.tensor or np.array
            model outputs in logits
        target : B-dim torch.tensor or np.array
            ground truth target classes

    Returns:
        precisions, recalls, f1s
            3-tuple of dict
                dict {int : float} of class to precision/recall/f1
    '''
    # Move off of gpu and convert to numpy if necessary
    if torch.is_tensor(output):
        output = output.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    unique_labels = np.unique(target)

    precisions = {}
    recalls = {}
    f1s = {}

    for label in unique_labels:
        TP = np.sum(np.where(((output == label) & (target == label)), 1, 0))
        FP = np.sum(np.where(((output == label) & (target != label)), 1, 0))
        FN = np.sum(np.where(((output != label) & (target == label)), 1, 0))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)

        precisions[label] = precision
        recalls[label] = recall
        f1s[label] = f1

    return precisions, recalls, f1s

# def per_class_counts(output, target):
#     '''
#     Return the number of predicted and true examples per class

#     Arg(s):
#         output : B x C x 1 torch.tensor
#             model outputs (pre-softmax)
#         target : B x 1 torch.tensor
#             integer binary ground truth labels
#     '''
#     with torch.no_grad():
#         n_classes = output.shape[1]
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         cmat = confusion_matrix(
#             target.cpu().numpy(),
#             pred.cpu().numpy(),
#             labels=[i for i in range(10)])  # rows are true, columns are predicted
#         # Convert back to torch
#         cmat = torch.from_numpy(cmat)
#         pred_counts = torch.diag(cmat)
#         target_counts = torch.sum(cmat, dim=1)

#     return torch.stack([pred_counts, target_counts], dim=0)  # 2 x C torch.tensor