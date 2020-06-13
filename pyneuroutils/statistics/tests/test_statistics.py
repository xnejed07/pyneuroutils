import unittest
from types import SimpleNamespace

from pyneuroutils.statistics import *
from pyneuroutils.datasets.diabetes import *
from pyneuroutils.datasets.iris import *



class Test(unittest.TestCase):
    def setUp(self):
        self.iris = SimpleNamespace()
        self.diabetes = SimpleNamespace()

        self.iris.data, self.iris.targets = iris_dataset().get_full_dataset()
        self.iris.probs = iris_classifier(self.iris.data)

        self.diabetes.data, self.diabetes.targets = diabetes_dataset().get_full_dataset()
        self.diabetes.probs = diabetes_classifier(self.diabetes.data)

    def test_binarize_2class(self):
        x = label_binarize(self.diabetes.targets)

    def test_binarize_multi_class(self):
        x = label_binarize(self.iris.targets)

    def test_confusion_matrix(self):
        y = confusion_matrix_argmax(self.iris.probs, self.iris.targets)
        stop = 1

    def test_kappa_score(self):
        y = kappa_score_argmax(self.iris.probs, self.iris.targets)

    def test_f1_score(self):
        y = f1_score_argmax(self.iris.probs, self.iris.targets)

    def test_auroc(self):
        y = auroc_score(self.iris.probs, self.iris.targets)

    def test_auprc(self):
        y = auprc_score(self.iris.probs, self.iris.targets)

    def test_binary_f1_score(self):
        y = f1_score_binary(self.diabetes.probs[:, 1], self.diabetes.targets, threshold=0.5)

    def test_random_classifier(self):
        random_binary_classifier(1000, 0.10)


if __name__ == '__main__':
    unittest.main()
