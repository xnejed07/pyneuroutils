import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


class Statistics(object):
    @staticmethod
    def label_binarize(targets):
        classes = np.unique(targets)
        classes = np.sort(classes)
        if len(classes) > 2:
            return label_binarize(y=targets, classes=classes, neg_label=0, pos_label=1, sparse_output=False)
        else:
            targets = np.expand_dims(targets, axis=1)
            y = np.concatenate([1 - targets, targets], axis=1)
            return y

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        return np.array(confusion_matrix(y_true=y_true, y_pred=y_pred))

    @staticmethod
    def confusion_matrix_binary(probs, targets, threshold=0.5):
        if probs.ndim != 1:
            raise Exception("Expect shape [n,]")
        y_pred = probs >= threshold
        return Statistics.confusion_matrix(y_true=targets, y_pred=y_pred)

    @staticmethod
    def confusion_matrix_argmax(probs, targets):
        y_pred = np.argmax(probs, axis=1)
        return Statistics.confusion_matrix(y_true=targets, y_pred=y_pred)

    @staticmethod
    def kappa_score(y_true, y_pred):
        return cohen_kappa_score(y1=y_true, y2=y_pred)

    @staticmethod
    def kappa_score_binary(probs, targets, threshold=0.5):
        if probs.ndim != 1:
            raise Exception("Expect shape [n,]")

        y_pred = probs >= threshold
        return Statistics.kappa_score(y_true=targets, y_pred=y_pred)

    @staticmethod
    def kappa_score_argmax(probs, targets):
        y_pred = np.argmax(probs, axis=1)
        return Statistics.kappa_score(y_true=targets, y_pred=y_pred)

    @staticmethod
    def f1_score(y_true, y_pred):
        return f1_score(y_true=y_true, y_pred=y_pred, average=None)

    @staticmethod
    def f1_score_binary(probs, targets, threshold=0.5):
        if probs.ndim != 1:
            raise Exception("Expect shape [n,]")
        y_pred = probs >= threshold
        return Statistics.f1_score(targets, y_pred)

    @staticmethod
    def kappa_score_binary_threshold_sweep(probs, targets, start=0, stop=1, num=101):
        t = np.linspace(start=start, stop=stop, num=num)
        y = np.zeros_like(t)
        for i in range(t.shape[0]):
            y[i] = Statistics.kappa_score_binary(probs=probs, targets=targets, threshold=t[i])
        return t, y

    @staticmethod
    def f1_score_argmax(probs, targets):
        y_pred = np.argmax(probs, axis=1)
        return Statistics.f1_score(targets, y_pred)

    @staticmethod
    def auroc_score(probs, targets):
        y_true = Statistics.label_binarize(targets)
        return roc_auc_score(y_true=y_true, y_score=probs, average=None)

    @staticmethod
    def auprc_score(probs, targets):
        y_true = Statistics.label_binarize(targets)
        return average_precision_score(y_true=y_true, y_score=probs, average=None)

    @staticmethod
    def random_binary_classifier(Nexamples, p):
        """
        :param N: number of observations
        :param p: probability of positive class
        :return:
        """
        target = np.random.binomial(size=Nexamples, n=1, p=p)
        probs = np.random.rand(Nexamples, 1)
        probs = np.concatenate([probs, 1 - probs], axis=1)
        return probs, target

    @staticmethod
    def pseudo_random_binary_classifier(Nexamples, p):
        target = np.random.binomial(size=Nexamples, n=1, p=p)
        probs = np.zeros_like(target, dtype='float')
        for i in range(target.shape[0]):
            probs[i] = np.random.beta(1, 3) if target[i] == 1 else np.random.beta(5, 1)
        probs = np.expand_dims(probs, axis=1)
        probs = np.concatenate([probs, 1 - probs], axis=1)
        return probs, target
