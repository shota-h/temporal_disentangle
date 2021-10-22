import numpy as np
import torch


def label2onehot(y_pred, n_class):
    onehot_label = np.zeros((len(y_pred), n_class))
    for i, y in enumerate(y_pred):
        onehot_label[i, y] = 1
    return onehot_label


def onehot2label(y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred


def normalize_y_pred(y_pred, thresh=0.5):
    y_pred = [1 if yp >= thresh else 0 for yp in y_pred]
    return np.array(y_pred)


def true_positive_multiclass(y_pred, y_true):
    if len(y_pred.size()) > 1:
        y_pred = normalize_classification_y_pred(y_pred)
    tp = [1 if torch.equal(yp, yt) else 0 for yp, yt in zip(y_pred, y_true)]
    return np.sum(tp)


def normalize_classification_y_pred(y_pred):
    y_pred = [torch.argmax(yp) for yp in y_pred]
    return y_pred


def get_binary_model_scores(y_pred, y_true, thresh=0.5):
    tp = true_positive(y_pred, y_true, thresh=0.5)
    tn = true_negative(y_pred, y_true, thresh=0.5)
    fp = false_positive(y_pred, y_true, thresh=0.5)
    fn = false_negative(y_pred, y_true, thresh=0.5)
    return tp, tn, fp, fn


def true_positive(y_pred, y_true, thresh=0.5):
    if type(y_true) is torch.Tensor:
        y_true = y_true.to('cpu')
    if type(y_pred) is torch.Tensor:
        y_pred = y_pred.to('cpu')
    y_pred = normalize_y_pred(y_pred, thresh)
    cat_y_pred = y_pred[y_true == 1]
    # cat_y_true = y_true[y_true == 1]
    tp = [1 for i in range(len(cat_y_pred)) if cat_y_pred[i] == 1]
    return np.sum(tp)


def false_positive(y_pred, y_true, thresh=0.5):
    if type(y_true) is torch.Tensor:
        y_true = y_true.to('cpu')
    if type(y_pred) is torch.Tensor:
        y_pred = y_pred.to('cpu')
    y_pred = normalize_y_pred(y_pred, thresh)
    cat_y_pred = y_pred[y_true == 0]
    # cat_y_true = y_true[y_true == 0]
    fp = [1 for i in range(len(cat_y_pred)) if cat_y_pred[i] == 1]
    return np.sum(fp)


def true_negative(y_pred, y_true, thresh = 0.5):
    if type(y_true) is torch.Tensor:
        y_true = y_true.to('cpu')
    if type(y_pred) is torch.Tensor:
        y_pred = y_pred.to('cpu')
    y_pred = normalize_y_pred(y_pred, thresh)
    cat_y_pred = y_pred[y_true == 0]
    # cat_y_true = y_true[y_true == 0]
    tn = [1 for i in range(len(cat_y_pred)) if cat_y_pred[i] == 0]
    return np.sum(tn)


def false_negative(y_pred, y_true, thresh = 0.5):
    if type(y_true) is torch.Tensor:
        y_true = y_true.to('cpu')
    if type(y_pred) is torch.Tensor:
        y_pred = y_pred.to('cpu')
    y_pred = normalize_y_pred(y_pred, thresh)
    cat_y_pred = y_pred[y_true == 1]
    # cat_y_true = y_true[y_true == 0]
    fn = [1 for i in range(len(cat_y_pred)) if cat_y_pred[i] == 0]
    return np.sum(fn)

