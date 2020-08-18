import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Fourier_mse(nn.Module):
    def __init__(self, img_w, img_h, mask=False, dm=0, sep=False, mode='hp'):
        super(Fourier_mse, self).__init__()
        self.mask = torch.ones((1, img_w, img_h, 2))
        if mask:
            # for u, v in itertools.product(np.arange(-dm, dm), np.arange(-dm, dm)):
            #     # if np.abs(u) + np.abs(v):
                #    self.mask[0, img_h//2+u, img_w//2+v, :] = 0
            self.mask[:, :dm, :dm, :] = 0
            self.mask[:, -dm:, :dm, :] = 0
            self.mask[:, dm:, -dm:, :] = 0
            self.mask[:, -dm:, -dm:, :] = 0
            # self.mask[:, img_h//2-dm:img_h//2+dm, img_w//2-dm:img_w//2+dm, :] = 0
            if mode == 'lp':
                self.mask = 1 - self.mask
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
            # masked_inputs = ft_inputs * M
            masked_targets = ft_targets * M
            diff_ft = nn.MSELoss()(ft_inputs, masked_targets)
            sum_loss = sum_loss + diff_ft

        return sum_loss / 3

    def predict(self, inputs):
        M = self.mask.repeat((inputs.size(0), 1, 1, 1))
        dst = []
        masked_dst = []
        for c in range(inputs.size(1)):
            cat_inputs = inputs[:, c, :, :, None]
            comp_inputs = torch.cat((cat_inputs, torch.zeros_like(cat_inputs)), axis=-1)
            ft_inputs = torch.fft(comp_inputs, 2)
            dst.append(ft_inputs)
            buff = M * ft_inputs
            masked_dst.append(buff)

        return dst, masked_dst


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
    # weight_log_input = w * log_input
    # weight_log_input = torch.mul(w, log_input)
    neg_entropy = -1 * torch.sum(log_input, dim=1) / log_input.size()[1]
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

def loss_vae(recon_x, x, mu1, mu2, logvar1, logvar2):
    BCE = F.binary_cross_entropy(recon_x.view(recon_x.size(0), -1), x.view(x.size(0), -1), reduction='sum') / recon_x.size(0)
    # BCE = F.binary_cross_entropy(recon_x.view(recon_x.size(0), -1), xview(recon_x.size(0), -1), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD1 = torch.mean(-0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=-1))
    KLD2 = torch.mean(-0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=-1))
    return BCE + KLD1 + KLD2