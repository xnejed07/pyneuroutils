import numpy as np
import sklearn.metrics as skm
import sklearn.preprocessing as skp



def label_binarize(targets):
    classes = np.unique(targets)
    classes = np.sort(classes)
    if len(classes) > 2:
        return skp.label_binarize(y=targets, classes=classes, neg_label=0, pos_label=1, sparse_output=False)
    else:
        targets = np.expand_dims(targets, axis=1)
        y = np.concatenate([1 - targets, targets], axis=1)
        return y

def confusion_matrix(y_true, y_pred):
    return np.array(skm.confusion_matrix(y_true=y_true, y_pred=y_pred))

def confusion_matrix_binary(probs, targets, threshold=0.5):
    if probs.ndim != 1:
        raise Exception("Expect shape [n,]")
    y_pred = probs >= threshold
    return confusion_matrix(y_true=targets, y_pred=y_pred)


def confusion_matrix_argmax(probs, targets):
    y_pred = np.argmax(probs, axis=1)
    return confusion_matrix(y_true=targets, y_pred=y_pred)

def kappa_score(y_true, y_pred):
    return skm.cohen_kappa_score(y1=y_true, y2=y_pred)

def kappa_score_binary(probs, targets, threshold=0.5):
    if probs.ndim != 1:
        raise Exception("Expect shape [n,]")

    y_pred = probs >= threshold
    return kappa_score(y_true=targets, y_pred=y_pred)

def kappa_score_argmax(probs, targets):
    y_pred = np.argmax(probs, axis=1)
    return kappa_score(y_true=targets, y_pred=y_pred)

def f1_score(y_true, y_pred):
    return skm.f1_score(y_true=y_true, y_pred=y_pred, average=None)

def f1_score_binary(probs, targets, threshold=0.5):
    if probs.ndim != 1:
        raise Exception("Expect shape [n,]")
    y_pred = probs >= threshold
    return f1_score(targets, y_pred)


def kappa_score_binary_threshold_sweep(probs, targets, start=0, stop=1, num=101):
    t = np.linspace(start=start, stop=stop, num=num)
    y = np.zeros_like(t)
    for i in range(t.shape[0]):
        y[i] = kappa_score_binary(probs=probs, targets=targets, threshold=t[i])
    return t, y


def f1_score_argmax(probs, targets):
    y_pred = np.argmax(probs, axis=1)
    return f1_score(targets, y_pred)

def auroc_score(probs, targets, binarize=True):
    if binarize:
        targets = label_binarize(targets)
    return skm.roc_auc_score(y_true=targets, y_score=probs, average=None)


def auprc_score(probs, targets, binarize=True):
    if binarize:
        targets = label_binarize(targets)
    return skm.average_precision_score(y_true=targets, y_score=probs, average=None)

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

def pseudo_random_binary_classifier(Nexamples, p):
    target = np.random.binomial(size=Nexamples, n=1, p=p)
    probs = np.zeros_like(target, dtype='float')
    for i in range(target.shape[0]):
        probs[i] = np.random.beta(1, 3) if target[i] == 1 else np.random.beta(5, 1)
    probs = np.expand_dims(probs, axis=1)
    probs = np.concatenate([probs, 1 - probs], axis=1)
    return probs, target
