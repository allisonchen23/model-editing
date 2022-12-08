import sys
sys.path.insert(0, 'src/model')
import metric
import numpy as np

# Test per_class_accuracy()
PREDS = np.array([
    [0.3, 0.5, 0.7],  # 2
    [0.2, 0.1, 0.1],  # 0
    [0.4, 0.3, 0.2],  # 0
    [0.1, 0.5, 0.4],  # 1
    [0.1, 0.1, 0.2],  # 2
    ])
def test_per_class_accuracy():

    # 100% accuracy for each class
    target1 = np.array([2, 0, 0, 1, 2])
    pca1 = metric.per_class_accuracy(PREDS, target1)
    assert (pca1 == np.ones(3)).all()

    # 0% accuracy for each class, not all target labels used
    target2 = np.array([1, 1, 1, 0, 1])
    pca2 = metric.per_class_accuracy(PREDS, target2)
    assert (pca2 == np.array([0.0, 0.0, 0.0])).all()

    # Some correct, some not
    target3 = np.array([2, 1, 0, 1, 0])
    pca3 = metric.per_class_accuracy(PREDS, target3)
    assert(pca3 == np.array([0.5, 0.5, 1.0])).all()

    # Some correct, some not again
    target4 = np.array([1, 1, 1, 1, 1])
    pca4 = metric.per_class_accuracy(PREDS, target4)
    assert (pca4 == np.array([0.0, 0.2, 0.0])).all()


if __name__ == "__main__":
    test_per_class_accuracy()