import torch
from sklearn.metrics import confusion_matrix


class Metrics():
    def __init__(self, metric_fns):

        self.total_metrics = {}
        self.metric_fns = metric_fns

        for metric_fn in metric_fns:
            name = metric_fn.__name__
            if name == 'accuracy' or \
            name == 'top_k_acc':
                self.total_metrics[name] = 0
            elif name == 'per_class_counts':
                self.total_metrics[name] = []
            else:
                raise ValueError("Unsupported metric {}".format(name))

    def update(self, output, target):
        for metric_fn in self.metric_fns:
            metric = metric_fn(output, target)
            name = metric_fn.__name__
            if name == 'accuracy' or \
            name == 'top_k_acc':
                self.total_metrics[name] += metric
            elif name == 'per_class_counts':
                self.total_metrics[name].append(metric)

    def get_total_metrics(self):
        # Calculate per class accuracy
        if 'per_class_counts' in self.total_metrics:
            per_class_counts = self.total_metrics['per_class_counts']
            # Stack across batches
            per_class_counts = torch.stack(per_class_counts, dim=0)  # N x 2 x C where N is number of batches
            # Sum across batches
            per_class_counts = torch.sum(per_class_counts, dim=0)  # 2 x C
            # Obtain counts for predictions and targets
            pred_per_class_counts = per_class_counts[0, ...]
            target_per_class_counts = per_class_counts[1, ...]
            # Calculate and store accuracy
            per_class_acc = pred_per_class_counts / target_per_class_counts
            self.total_metrics['per_class_acc'] = per_class_acc
        total_metrics = self.total_metrics.copy()
        total_metrics.pop('per_class_counts')
        return total_metrics

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


def per_class_counts(output, target):
    '''
    Return the number of predicted and true examples per class

    Arg(s):
        output : B x C x 1 torch.tensor
            model outputs (pre-softmax)
        target : B x 1 torch.tensor
            integer binary ground truth labels
    '''
    with torch.no_grad():
        n_classes = output.shape[1]
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        cmat = confusion_matrix(
            target.cpu().numpy(),
            pred.cpu().numpy(),
            labels=[i for i in range(10)])  # rows are true, columns are predicted
        # Convert back to torch
        cmat = torch.from_numpy(cmat)
        pred_counts = torch.diag(cmat)
        target_counts = torch.sum(cmat, dim=1)

    return torch.stack([pred_counts, target_counts], dim=0)  # 2 x C torch.tensor