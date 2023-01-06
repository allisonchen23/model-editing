import sys
sys.path.insert(0, 'src/model')
import metric
import numpy as np
from mlxtend.evaluate import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score

# Test per_class_accuracy()
OUTPUTS = np.array([
    [0.3, 0.5, 0.7],  # 2
    [0.2, 0.1, 0.1],  # 0
    [0.4, 0.3, 0.2],  # 0
    [0.1, 0.5, 0.4],  # 1
    [0.1, 0.1, 0.2],  # 2
    ])
'''
def test_per_class_accuracy_from_predictions():
    predictions = np.argmax(OUTPUTS, axis=1)
    print(predictions)
    # 100% accuracy for each class
    target1 = np.array([2, 0, 0, 1, 2])

    pca1 = metric.per_class_accuracy_from_predictions(predictions, target1)
    assert (pca1 == np.ones(3)).all()

    # Degenerate case
    # # 0% accuracy for each class, not all target labels used
    # target2 = np.array([1, 1, 1, 0, 1])
    # pca2 = metric.per_class_accuracy(predictions, target2)
    # print(type(pca2))
    # print(pca2)
    # print(pca2 == np.array([0.0, 0.0, 0.0]))
    # assert (pca2 == np.array([0.0, 0.0, 0.0])).all()

    # Some correct, some not
    target3 = np.array([2, 1, 0, 1, 0])
    pca3 = metric.per_class_accuracy_from_predictions(predictions, target3)
    assert(pca3 == np.array([0.5, 0.5, 1.0])).all()

    # Some correct, some not again
    target4 = np.array([0, 2, 1, 1, 1])
    pca4 = metric.per_class_accuracy_from_predictions(predictions, target4)
    print(pca4)
    assert (pca4 == np.array([0.0, 1/3, 0.0])).all()

    # All incorrect
    target5 = np.array([0, 1, 1, 2, 0])
    pca5 = metric.per_class_accuracy_from_predictions(predictions, target5)
    assert(pca5 == np.array([0.0, 0.0, 0.0])).all()

# def test_per_class_accuracy_outputs():

    # 100% accuracy for each class
    target1 = np.array([2, 0, 0, 1, 2])
    pca1 = metric.per_class_accuracy_outputs(OUTPUTS, target1)
    assert (pca1 == np.ones(3)).all()

    # 0% accuracy for each class, not all target labels used
    target2 = np.array([1, 1, 1, 0, 1])
    pca2 = metric.per_class_accuracy_outputs(OUTPUTS, target2)
    assert (pca2 == np.array([0.0, 0.0, 0.0])).all()

    # Some correct, some not
    target3 = np.array([2, 1, 0, 1, 0])
    pca3 = metric.per_class_accuracy_outputs(OUTPUTS, target3)
    assert(pca3 == np.array([0.5, 0.5, 1.0])).all()

    # Some correct, some not again
    target4 = np.array([1, 1, 1, 1, 1])
    pca4 = metric.per_class_accuracy_outputs(OUTPUTS, target4)
    assert (pca4 == np.array([0.0, 0.2, 0.0])).all()
'''


def test_metrics():
    '''
    Test overall and per class accuracy comparing against mlxtend library
    Test precision, recall, and f1 comparing against sklearn
    '''
    y_preds = [
        [1, 0, 0, 0, 1, 2, 0, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 2, 2, 2, 0, 0, 0]]

    y_target = [0, 0, 0, 1, 1, 1, 2, 2, 2]

    for y_pred in y_preds:
        metrics = metric.compute_metrics(
            metric_fns=[
                metric.accuracy,
                metric.per_class_accuracy,
                metric.recall,
                metric.precision,
                metric.f1],
            prediction=np.array(y_pred),
            target=np.array(y_target))
        print(metrics)
        # Compare overall accuracy
        mlxtend_accuracy = accuracy_score(y_target, y_pred)
        assert mlxtend_accuracy == metrics['accuracy']


        # Compare per class accuracy
        n_classes = np.unique(np.array(y_target)).shape[0]
        for i in range(n_classes):
            mlxtend_per_class_accuracy = accuracy_score(
                y_target,
                y_pred,
                method='binary',
                pos_label=i)
            # print("{}: {}".format(i, mlxtend_per_class_accuracy))
            assert mlxtend_per_class_accuracy== metrics['per_class_accuracy'][i]

        # Compare precision
        sklearn_precision = precision_score(
            y_true=y_target,
            y_pred=y_pred,
            average=None)
        print(sklearn_precision)
        assert (sklearn_precision == metrics['precision']).all()


def test_precision_recall_f1_from_predictions():
    labels = []
    n_data = 25
    n_classes = 5
    for i in range(n_data):
        labels.append(i // n_classes)
    labels = np.array(labels)
    # labels = [0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5]

    # Test case 1
    output = np.array([0 for i in range(25)])
    precision, recall, f1 = metric.precision_recall_f1_from_predictions(prediction=output, target=labels)
    sk_recall = metric.recall_from_predictions(prediction=output, target=labels)
    sk_precision = metric.precision_from_predictions(prediction=output, target=labels)
    sk_f1 = metric.f1_from_predictions(prediction=output, target=labels)
    print(recall, sk_recall)
    print(precision, sk_precision)
    print(f1, sk_f1)
    assert precision[0] == 0.2
    assert np.all(np.isnan(precision[1:]))
    assert recall == [1, 0, 0, 0, 0]
    assert f1[0] == (0.4 / 1.2)
    assert np.all(np.isnan(f1[1:]))

    # Test case 2
    output = np.concatenate([[0, 1, 2, 3, 4] for i in range(n_classes)], axis=0)
    precision, recall, f1 = metric.precision_recall_f1_from_predictions(prediction=output, target=labels)
    sk_recall = metric.recall_from_predictions(prediction=output, target=labels)
    sk_precision = metric.precision_from_predictions(prediction=output, target=labels)
    sk_f1 = metric.f1_from_predictions(prediction=output, target=labels)
    print(recall, sk_recall)
    print(precision, sk_precision)
    print(f1, sk_f1)

    assert precision == [0.2 for i in range(n_classes)]
    assert recall == [0.2 for i in range(n_classes)]
    assert f1 == [(2 * 0.2 * 0.2 / (0.4)) for i in range(n_classes)]


if __name__ == "__main__":

    test_metrics()