import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Fourier_mse(nn.Module):
    def __init__(self, img_w, img_h, mask=False, dm=0, sep=False, mode='hp'):
        super(Fourier_mse, self).__init__()
        if mask:
            if mode == 'hp':
                mv = 0
                self.mask = torch.ones((1, img_w, img_h, 2))
            if mode == 'lp':
                mv = 1
                self.mask = torch.zeros((1, img_w, img_h, 2))
            for u, v in itertools.product(np.arange(-dm, dm), np.arange(-dm, dm)):
                if np.abs(u) + np.abs(v):
                    self.mask[0, img_h//2+u, img_w//2+v, :] = mv
            # self.mask = 1 - self.mask
        else:
            self.mask = torch.ones((1, img_w, img_h, 2))
        self.sep = sep

    def forward(self, inputs, targets):
        sum_loss = 0
        M = self.mask.repeat((inputs.size(0), 1, 1, 1)).to(device)
        for c in range(inputs.size(1)):
            cat_inputs = inputs[:, c, :, :, None]
            comp_inputs = torch.cat((cat_inputs, torch.zeros_like(cat_inputs)), axis=-1)
            cat_targets = targets[:, c, :, :, None]
            comp_targets = torch.cat((cat_targets, torch.zeros_like(cat_targets)), axis=-1)

            ft_inputs = torch.fft(comp_inputs, 2)
            ft_targets = torch.fft(comp_targets, 2)
            masked_inputs = ft_inputs * M
            masked_targets = ft_targets * M
            diff_ft = nn.MSELoss()(masked_inputs, masked_targets)
            sum_loss = sum_loss + diff_ft

        return sum_loss / 3

    def predict(self, inputs):
        M = self.mask.repeat((inputs.size(0), 1, 1))
        dst = []
        for c in range(inputs.size(1)):
            cat_inputs = inputs[:, c, :, :, None]
            comp_inputs = torch.cat((cat_inputs, torch.zeros_like(cat_inputs)), axis=-1)
            ft_inputs = torch.fft(comp_inputs, 2)
            buff = M * ft_inputs
            dst.append(buff)

        return dst_real


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=0):
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