import numpy as np
import torch

def normalize_y_pred(y_pred, thresh=0.5):
    y_pred = [1 if yp >= thresh else 0 for yp in y_pred]
    return np.array(y_pred)


def onehot2label(y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred


def true_positive_multiclass(y_pred, y_true):
    y_pred = normalize_classification_y_pred(y_pred)
    tp = [1 if torch.equal(yp, yt) else 0 for yp, yt in zip(y_pred, y_true)]
    return np.sum(tp)


def normalize_classification_y_pred(y_pred):
    y_pred = [torch.argmax(yp) for yp in y_pred]
    return y_pred


def true_positive(y_true, y_pred, thresh=0.5):
    y_pred = normalize_y_pred(y_pred, thresh)
    cat_y_pred = y_pred[y_true == 1]
    cat_y_true = y_true[y_true == 1]
    tp = [1 for i in range(len(cat_y_true)) if cat_y_pred[i] == 1]
    return np.sum(tp)


def true_negative(y_true, y_pred, thresh = 0.5):
    y_pred = normalize_y_pred(y_pred, thresh)
    cat_y_pred = y_pred[y_true == 0]
    cat_y_true = y_true[y_true == 0]
    tn = [1 for i in range(len(cat_y_true)) if cat_y_pred[i] == 0]
    return np.sum(tn)