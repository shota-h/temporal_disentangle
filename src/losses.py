import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

def negative_entropy_loss(input, small_value=1e-4):
    softmax_input = F.softmax(input, dim=1) + small_value
    w = torch.ones((softmax_input.size(0), softmax_input.size(1))) / softmax_input.size(1)
    w = w.to(device)
    log_input = torch.log(softmax_input)
    weight_log_input = torch.mul(w, log_input)
    neg_entropy = -1 * torch.sum(weight_log_input, dim=1) / weight_log_input.size()[1]
    return torch.mean(neg_entropy)


def make_index_rankingloss(target):
    idx = []
    labels = []
    for i, j in itertools.permutations(range(target), 2):
        idx.append([i, j])
        if target[i] > target[j]:
            labels.append(1)
        elif target[i] < target[j]:
            labels.append(-1)
        else:
            labels.append(0)

    return np.asarray(idx), np.asarray(labels)