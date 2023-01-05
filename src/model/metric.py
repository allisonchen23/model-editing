import torch
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

def compute_metrics(metric_fns, prediction, target, unique_labels=None):
    '''
    Given list of metric functions, calculate metrics for given predictions and targets
    Arg(s):
        metric_fns : list[functions]
            list of metrics to compute
        prediction : N-length np.array or torch.tensor
            model predictions
        target : N-length np.array or torch.tensor
            ground truth values
        unique_labels : list[int] or C-length np.array
            sequence of unique labels

    Returns:
        metrics dict{str : np.array}
    '''

    # Create list of metric function names
    metric_fn_names = set()
    for fn in metric_fns:
        metric_fn_names.add(fn.__name__)

    # Data structure to store calculated metrics
    metrics = {}

    # Convert tensors -> arrays if necessary
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    # Obtain unique labels if not provided
    if unique_labels is None:
        n_classes = np.unique(target).shape[0]  # assumes all classes are in target and go from 0 to n_classes - 1
        unique_labels = [i for i in range(n_classes)]
    else:
        n_classes = len(unique_labels)

    # Make confusion matrix (rows are true, columns are predicted)
    assert prediction.shape[0] == target.shape[0]
    cmat = confusion_matrix(
        target,
        prediction,
        labels=unique_labels)

    # Calculate TP, TN, FP, and FN for each class
    total = np.sum(cmat)
    TPs = np.diag(cmat)
    FPs = np.sum(cmat, axis=1) - TPs
    FNs = np.sum(cmat, axis=0) - TPs
    TNs = total - (TPs + FPs + FNs)

    # store whether or not we want to calculate f1
    calculate_f1 = False
    for metric_fn in metric_fns:
        metric_name = metric_fn.__name__
        if metric_name == "f1":
            calculate_f1 = True
            continue

        metric = metric_fn(
            TPs=TPs,
            TNs=TNs,
            FPs=FPs,
            FNs=FNs)

        metrics[metric_name] = metric

    if calculate_f1:
        metrics['f1'] = f11(
            precisions=metrics['precision'],
            recalls=metrics['recall'])

    return metrics


def accuracy1(TPs, TNs, FPs, FNs):
    '''
    Given true positives, true negatives, false positives, and false negatives,
        calculate accuracy overall

    Arg(s):
        TPs : C-length np.array
            True positives for each class
        TNs : C-length np.array
            True negatives for each class
        FPs : C-length np.array
            False positives for each class
        FNs : C-length np.array
            False negatives for each class
    Returns
        accuracy : float
            (TP + TN) / (TP + TN + FP + FN)
    '''

    return np.sum(TPs + TNs) / np.sum(TPs + TNs + FPs + FNs)

def per_class_accuracy1(TPs, TNs, FPs, FNs):
    '''
    Given true positives, true negatives, false positives, and false negatives,
        calculate per class accuracy

    Arg(s):
        TPs : C-length np.array
            True positives for each class
        TNs : C-length np.array
            True negatives for each class
        FPs : C-length np.array
            False positives for each class
        FNs : C-length np.array
            False negatives for each class
    Returns
        per_class_accuracies : C-length np.array
            per class accuracy = (TP + TN) / (TP + FP + TN + FN)
    '''
    return (TPs + TNs) / (TPs + TNs + FPs + FNs)

def precision1(TPs, TNs, FPs, FNs):
    '''
    Given true positives, true negatives, false positives, and false negatives,
        calculate per class precision

    Arg(s):
        TPs : C-length np.array
            True positives for each class
        TNs : C-length np.array
            True negatives for each class
        FPs : C-length np.array
            False positives for each class
        FNs : C-length np.array
            False negatives for each class
    Returns
        precisions : C-length np.array
            precision = TP / (TP + FP)
    '''
    return TPs / (TPs + FPs)

def recall1(TPs, TNs, FPs, FNs):
    '''
    Given true positives, true negatives, false positives, and false negatives,
        calculate per class recall

    Arg(s):
        TPs : C-length np.array
            True positives for each class
        TNs : C-length np.array
            True negatives for each class
        FPs : C-length np.array
            False positives for each class
        FNs : C-length np.array
            False negatives for each class
    Returns
        recall : C-length np.array
            recall = TP / (TP + FN)
    '''
    return TPs / (TPs + FNs)

def f11(precisions, recalls):
    '''
    Given precision and recall for each class,
        calculate f1 score per class

    Arg(s):
        precisions : C-length np.array
            precisions for each class
        recalls : C-length np.array
            recalls for each class

    Returns:
        f1s : C-length np.array
            f1 = 2 * precision * recall / (precision + recall)
    '''
    return 2 * precisions * recalls / (precisions + recalls)

def accuracy(prediction, target):
    '''
    Return accuracy
    Arg(s):
        prediction : N x 1 torch.tensor
            logit outputs of model
        target : N x 1 torch.tensor
            integer labels
    Returns:
        float : accuracy of predictions

    '''

    assert len(prediction.shape) == 1, "Prediction must be 1-dim array, received {}-shape array.".format(prediction.shape)
    assert len(target.shape) == 1, "Target must be 1-dim array, received {}-shape array.".format(target.shape)

    # Convert to numpy arrays
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    correct = np.sum(prediction == target)
    return correct / len(target)


def accuracy_outputs(output, target):
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

def per_class_accuracy(prediction, target, unique_labels=None):
    '''
    Return the accuracy of each class

    Arg(s):
        prediction : B-dim torch.tensor or np.array
            model prediction (post-argmax)
        target : B-dim torch.tensor or np.array
            integer binary ground truth labels
        unique_labels : list[int]
            can specify expected labels
    Returns:
        C x 1 np.array of per class accuracy
    '''
    # Convert to numpy arrays
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    if unique_labels is None:
        n_classes = np.unique(target).shape[0]  # assumes all classes are in target and go from 0 to n_classes - 1
        unique_labels = [i for i in range(n_classes)]
    else:
        n_classes = len(unique_labels)
    # Make confusion matrix (rows are true, columns are predicted)
    assert prediction.shape[0] == target.shape[0]
    cmat = confusion_matrix(
        target,
        prediction,
        labels=unique_labels)

    # Get counts
    pred_counts = np.diagonal(cmat)
    target_counts = np.sum(cmat, axis=1)

    # Nan occurs if no target counts. In these cases, set those classes to 0
    return np.nan_to_num(pred_counts / target_counts)

def per_class_accuracy_outputs(output, target):
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

def recall(prediction, target, unique_labels=None):
    '''
    Calculate per class recall using sklearn library

    Arg(s):
        prediction : B-dim torch.tensor or np.array
            model prediction
        target : B-dim torch.tensor or np.array
            ground truth target classes
        unique_labels : list[int]
            can specify expected labels

    Returns:
        recall : np.array
    '''
     # Move off of gpu and convert to numpy if necessary
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    return recall_score(
        y_true=target,
        y_pred=prediction,
        labels=unique_labels,
        average=None)

def precision(prediction, target, unique_labels=None):
    '''
    Calculate per class precision using sklearn library

    Arg(s):
        prediction : B-dim torch.tensor or np.array
            model prediction
        target : B-dim torch.tensor or np.array
            ground truth target classes
        unique_labels : list[int]
            can specify expected labels

    Returns:
        precision : np.array
    '''
    # Move off of gpu and convert to numpy if necessary
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    return precision_score(
        y_true=target,
        y_pred=prediction,
        labels=unique_labels,
        average=None)

def f1(prediction, target, unique_labels=None):
    '''
    Calculate per class f1 using sklearn library

    Arg(s):
        prediction : B-dim torch.tensor or np.array
            model prediction
        target : B-dim torch.tensor or np.array
            ground truth target classes
        unique_labels : list[int]
            can specify expected labels

    Returns:
        f1 : np.array
    '''
    # Move off of gpu and convert to numpy if necessary
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    return f1_score(
        y_true=target,
        y_pred=prediction,
        labels=unique_labels,
        average=None)


def precision_recall_f1(prediction, target, unique_labels=None):
    '''
    Given outputs and targets, calculate per-class precision

    Arg(s):
        prediction : B-dim torch.tensor or np.array
            model prediction
        target : B-dim torch.tensor or np.array
            ground truth target classes
        unique_labels : list[int]
            can specify expected labels

    Returns:
        (precisions, recalls, f1s)
            3-tuple of lists
    '''
    # Move off of gpu and convert to numpy if necessary
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    if unique_labels is None:
        n_classes = np.unique(target).shape[0]  # assumes all classes are in target and go from 0 to n_classes - 1
        unique_labels = [i for i in range(n_classes)]

    precisions = []
    recalls = []
    f1s = []

    # Calculate precision, recall, f1 for each class
    for label in unique_labels:
        # Need True Positives, False Positives, and False Negatives
        TP = np.sum(np.where(((prediction == label) & (target == label)), 1, 0))
        FP = np.sum(np.where(((prediction == label) & (target != label)), 1, 0))
        FN = np.sum(np.where(((prediction != label) & (target == label)), 1, 0))

        # Calculate metrics
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)

        # Store in respective lists
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return precisions, recalls, f1s
