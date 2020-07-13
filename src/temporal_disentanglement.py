import os
import sys
import json
import itertools
import time
import argparse
import h5py
import cv2
import numpy as np
import random as rn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib.animation as animation
from functools import partial
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import torch
from torch.nn.modules import activation
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.utils.data import DataLoader
import torchvision.models as models
import tensorboardX as tbx

# import archs
# from data_handling import numpy_pop, GetHDF5dataset, GetHDF5dataset_MultiLabel

SEED = 1
torch.manual_seed(SEED)
rn.seed(SEED)
np.random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
current_path = './'

def load_model(in_dpath, model):
    return model.load_state_dict(torch.load(in_dpath))


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
    

def negative_entropy_loss(input):
    small_value = 1e-4
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


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = nn.init.xavier_normal_(m.weight.data)
            # m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias)
            # m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data = nn.init.xavier_normal_(m.weight.data)
            # m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias)
            # m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias)
            # m.weight.data.normal_(0, 0.02)
            # m.bias.data.zero_()


def statistical_augmentation(features):
    outs = []
    for h0 in features:
        e_h0 = torch.normal(0, 1, size=h0.size()).to(device)
        std_h0 = torch.std(h0, dim=0)
        new_h0 = h0 + torch.mul(e_h0, std_h0)
        outs.append(new_h0)
    return outs

class TDAE(nn.Module):
    def __init__(self, n_class=3, ksize=3):
        super().__init__()
        # self.model = models.vgg16(num_classes=n_class, pretrained=False)
        # self.enc = models.resnet18(pretrained=False)
        # self.enc = nn.Sequential(*list(self.enc.children())[:8])
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128, 64, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.enc = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        self.subnet_conv_t1 = nn.Sequential(nn.Conv2d(64, 32, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.subnet_t1 = nn.Sequential(nn.Linear(in_features=8*8*64, out_features=256),
                                    nn.ReLU())
        self.subnet_p1 = nn.Sequential(nn.Linear(in_features=8*8*64, out_features=256),
                                    nn.ReLU())
        self.subnet_t2 = nn.Sequential(nn.Linear(in_features=256, out_features=8*8*64),
                                    nn.ReLU())
        self.subnet_p2 = nn.Sequential(nn.Linear(in_features=256, out_features=8*8*64),
                                    nn.ReLU())
        self.classifier_t = nn.Linear(in_features=256, out_features=n_class)
        # self.classifier_p = nn.Linear(in_features=64, out_features=n_class)

        # self.deconv1 = nn.Sequential(nn.Linear(in_features=64*2, out_features=392),
        #                             nn.ReLU())
        # self.deconv2 = torch.nn.Upsample(scale_factor=2,mode='nearest')

        self.dec_fc1 = nn.Sequential(nn.Linear(in_features=256*2, out_features=8*8*64),
                                    nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 128, 2, stride=2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(16, 3, 2, stride=2),
                                nn.Sigmoid())
        self.dec = nn.Sequential(self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5)

        # self.classifier_t = nn.Linear(in_features=64, out_features=n_class)
        # self.classifier_p = nn.Linear(in_features=64, out_features=n_class)
        initialize_weights(self)
        
    def forward(self, input):
        h0 = self.enc(input)
        h0 = torch.reshape(h0, (h0.size(0), -1))
        # h0 = F.avg_pool2d(h0, kernel_size=h0.size()[2])
        # t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        t0 = self.subnet_t1(h0)
        p0 = self.subnet_p1(h0)
        class_preds = self.classifier_t(t0)
        class_preds_adv = self.classifier_t(p0)
        # t1 = self.subnet_t2(t0)
        # p1 = self.subnet_p2(p0)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        # t1 = torch.reshape(t1, (t1.size(0), 64, 8, 8))
        # p1 = torch.reshape(t1, (p1.size(0), 64, 8, 8))
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, 8, 8))
        # concat_enc = torch.cat((t1, p1), dim=1)
        rec = self.dec(concat_h0)
        return class_preds, class_preds_adv, rec

        # p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        # t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        # p0 = p0.view(p0.size(0), 64)
        # t0 = t0.view(t0.size(0), 64)
        # self.p0 = self.fn_p(p0)
        # self.t0 = self.fn_t(t0)
        # self.augs = statistical_augmentation([self.p0, self.t0])

        # pred_p = self.classifier_p(self.p0)
        # pred_t = self.classifier_t(self.t0)

        # re_enc = torch.cat([self.p0, self.t0], dim=1)
        # aug_re_enc = torch.cat([self.augs[0], self.augs[1]], dim=1)

        # dec = self.dec_fn(re_enc)
        # dec = dec.view(-1, 8, 7, 7)
        # dec = self.decoder(dec)

        # aug_dec = self.dec_fn(aug_re_enc)
        # aug_dec = aug_dec.view(-1, 8, 7, 7)
        # aug_dec = self.decoder(aug_dec)
        # return dec, pred_t, pred_p, aug_dec

    def hidden_output(self, input):
        h0 = self.enc(input)
        h0 = torch.reshape(h0, (h0.size(0), -1))
        # h0 = F.avg_pool2d(h0, kernel_size=h0.size()[2])
        # t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        t0 = self.subnet_t1(h0)
        p0 = self.subnet_p1(h0)
        return t0, p0
        class_preds = self.classifier_t(t0)
        class_preds_adv = self.classifier_t(p0)
        # t1 = self.subnet_t2(t0)
        # p1 = self.subnet_p2(p0)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        # t1 = torch.reshape(t1, (t1.size(0), 64, 8, 8))
        # p1 = torch.reshape(t1, (p1.size(0), 64, 8, 8))
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, 8, 8))
        # concat_enc = torch.cat((t1, p1), dim=1)
        rec = self.dec(concat_h0)

        h0 = self.enc(input)
        h0 = torch.reshape(h0, (h0.size(0), -1))
        # h0 = F.avg_pool2d(h0, kernel_size=h0.size()[2])
        # t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        t0 = self.subnet_t1(h0)
        p0 = self.subnet_p1(h0)
        
        # h0 = self.enc(input)
        # p0 = self.subnet_p(h0)
        # t0 = self.subnet_t(h0)

        # p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        # t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        # p0 = p0.view(p0.size(0), 64)
        # t0 = t0.view(t0.size(0), 64)
        # self.p0 = self.fn_p(p0)
        # self.t0 = self.fn_t(t0)
        # pred_p = self.classifier_p(self.p0)
        # pred_t = self.classifier_t(self.t0)

        return t0, p0


class D2AE(nn.Module):
    def __init__(self, n_class, ksize=3):
        super().__init__()
        # self.model = models.vgg16(num_classes=n_class, pretrained=False)
        self.enc = models.resnet18(pretrained=False)
        self.enc = nn.Sequential(*list(self.enc.children())[:8])
        
        self.conv_p_1 = nn.Conv2d(512, 256, ksize, padding=(ksize-1)//2)
        self.conv_p_2 = nn.Conv2d(256, 128, ksize, padding=(ksize-1)//2)
        self.conv_p_3 = nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2)
        self.subnet_p = nn.Sequential(self.conv_p_1,
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    self.conv_p_2,
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    self.conv_p_3,
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.fn_p = nn.Sequential(nn.Linear(in_features=64, out_features=64),
                                nn.ReLU())
        self.conv_t_1 = nn.Conv2d(512, 256, ksize, padding=(ksize-1)//2)
        self.conv_t_2 = nn.Conv2d(256, 128, ksize, padding=(ksize-1)//2)
        self.conv_t_3 = nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2)
        self.subnet_t = nn.Sequential(self.conv_t_1,
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    self.conv_t_2,
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    self.conv_t_3,
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.fn_t = nn.Sequential(nn.Linear(in_features=64, out_features=64),
                                nn.ReLU())

        self.dec_fn = nn.Sequential(nn.Linear(in_features=64*2, out_features=392),
                                    nn.ReLU())
        self.dec_upsampling = torch.nn.Upsample(scale_factor=2,mode='nearest')
        # self.dec_deconv1 = nn.ConvTranspose2d(8, 128, 2, stride=2)
        # self.dec_deconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # self.dec_deconv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # self.dec_deconv4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        # self.dec_deconv5 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        # self.dec_deconv1 = nn.ConvTranspose2d(2, 8, 2, stride=2)
        # self.dec_deconv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        # self.dec_deconv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # self.dec_deconv4 = nn.ConvTranspose2d(32, 3, 2, stride=2)

        # self.dec_conv1 = nn.Conv2d(2, 8, ksize, padding=(ksize-1)//2)
        # self.dec_conv2 = nn.Conv2d(8, 16, ksize, padding=(ksize-1)//2)
        # self.dec_conv3 = nn.Conv2d(16, 32, ksize, padding=(ksize-1)//2)
        # self.dec_conv4 = nn.Conv2d(32, 3, ksize, padding=(ksize-1)//2)

        self.dec_conv1 = nn.Conv2d(8, 128, ksize, padding=(ksize-1)//2)
        self.dec_conv1_1 = nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2)
        self.dec_conv1_2 = nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2)
        self.dec_conv2 = nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2)
        self.dec_conv2_1 = nn.Conv2d(64, 64, ksize, padding=(ksize-1)//2)
        self.dec_conv2_2 = nn.Conv2d(64, 64, ksize, padding=(ksize-1)//2)
        self.dec_conv3 = nn.Conv2d(64, 32, ksize, padding=(ksize-1)//2)
        self.dec_conv3_1 = nn.Conv2d(32, 32, ksize, padding=(ksize-1)//2)
        self.dec_conv3_2 = nn.Conv2d(32, 32, ksize, padding=(ksize-1)//2)
        self.dec_conv4 = nn.Conv2d(32, 16, ksize, padding=(ksize-1)//2)
        self.dec_conv4_1 = nn.Conv2d(16, 16, ksize, padding=(ksize-1)//2)
        self.dec_conv4_2 = nn.Conv2d(16, 16, ksize, padding=(ksize-1)//2)
        self.dec_conv5 = nn.Conv2d(16, 3, ksize, padding=(ksize-1)//2)
        self.dec_conv5_1 = nn.Conv2d(3, 3, ksize, padding=(ksize-1)//2)
        self.dec_conv5_2 = nn.Conv2d(3, 3, ksize, padding=(ksize-1)//2)

        # self.decoder = nn.Sequential(self.dec_deconv1,
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU(),
        #                             self.dec_deconv2,
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU(),
        #                             self.dec_deconv3,
        #                             nn.BatchNorm2d(32),
        #                             nn.ReLU(),
        #                             self.dec_deconv4,
        #                             nn.BatchNorm2d(16),
        #                             nn.ReLU(),
        #                             self.dec_deconv5,
        #                             nn.Sigmoid())
        self.decoder = nn.Sequential(self.dec_upsampling,
                                    self.dec_conv1, nn.BatchNorm2d(128), nn.ReLU(),
                                    self.dec_conv1_1, nn.BatchNorm2d(128), nn.ReLU(),
                                    self.dec_conv1_2, nn.BatchNorm2d(128), nn.ReLU(),
                                    self.dec_upsampling,
                                    self.dec_conv2, nn.BatchNorm2d(64), nn.ReLU(),
                                    self.dec_conv2_1, nn.BatchNorm2d(64), nn.ReLU(),
                                    self.dec_conv2_2, nn.BatchNorm2d(64), nn.ReLU(),
                                    self.dec_upsampling,
                                    self.dec_conv3, nn.BatchNorm2d(32), nn.ReLU(),
                                    self.dec_conv3_1, nn.BatchNorm2d(32), nn.ReLU(),
                                    self.dec_conv3_2, nn.BatchNorm2d(32), nn.ReLU(),
                                    self.dec_upsampling,
                                    self.dec_conv4, nn.BatchNorm2d(16), nn.ReLU(),
                                    self.dec_conv4_1, nn.BatchNorm2d(16), nn.ReLU(),
                                    self.dec_conv4_2, nn.BatchNorm2d(16), nn.ReLU(),
                                    self.dec_upsampling,
                                    # self.dec_conv5, nn.Sigmoid(),
                                    # self.dec_conv5_1, nn.Sigmoid(),
                                    self.dec_conv5, nn.Sigmoid())

        self.classifier_t = nn.Linear(in_features=64, out_features=n_class)
        self.classifier_p = nn.Linear(in_features=64, out_features=n_class)
        initialize_weights(self)
        
    def forward(self, input):
        h0 = self.enc(input)
        p0 = self.subnet_p(h0)
        t0 = self.subnet_t(h0)

        p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        p0 = p0.view(p0.size(0), 64)
        t0 = t0.view(t0.size(0), 64)
        self.p0 = self.fn_p(p0)
        self.t0 = self.fn_t(t0)
        self.augs = statistical_augmentation([self.p0, self.t0])

        pred_p = self.classifier_p(self.p0)
        pred_t = self.classifier_t(self.t0)

        re_enc = torch.cat([self.p0, self.t0], dim=1)
        aug_re_enc = torch.cat([self.augs[0], self.augs[1]], dim=1)

        dec = self.dec_fn(re_enc)
        dec = dec.view(-1, 8, 7, 7)
        dec = self.decoder(dec)

        aug_dec = self.dec_fn(aug_re_enc)
        aug_dec = aug_dec.view(-1, 8, 7, 7)
        aug_dec = self.decoder(aug_dec)
        return dec, pred_t, pred_p, aug_dec

    def hidden_output_p(self, input):
        h0 = self.enc(input)
        p0 = self.subnet_p(h0)
        t0 = self.subnet_t(h0)

        p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        p0 = p0.view(p0.size(0), 64)
        t0 = t0.view(t0.size(0), 64)
        self.p0 = self.fn_p(p0)
        self.t0 = self.fn_t(t0)
        pred_p = self.classifier_p(self.p0)
        pred_t = self.classifier_t(self.t0)

        return pred_p


def main():
    def output_reconst_img(x0, rec_x0, path):
            x0 = x0.detach().numpy()
            x0 = np.transpose(x0, (1, 2, 0))
            rec_x0 = rec_x0.detach().to('cpu').numpy()
            rec_x0 = np.transpose(rec_x0, (1, 2, 0))
            x0 = np.append(x0, rec_x0, axis=1)
            x0 = cv2.cvtColor(x0, cv2.COLOR_BGR2RGB)
            cv2.imwrite(path, np.uint8(255*x0))


    label_dict = {'part_label': 3,
                    'mayo_label': 5}
    from torchsummary import summary
    writer = tbx.SummaryWriter('./test_disentangle/learning_process')
    labels = ['mayo_label']
    n_class = 5
    model = D2AE(n_class)
    model.to(device)
    # summary(model, (3,224,224))

    dataloader = GetHDF5dataset_MultiLabel('./dataset/colon_DirSplit.hdf5', label_name=labels, n_sample=-1)
    
    x0 = dataloader.samples
    x0 = np.transpose(x0, (0, 3, 1, 2))
    t = dataloader.targets
    
    n_sample = len(x0)
    idx = list(range(0, n_sample))
    np.random.seed(SEED)
    np.random.shuffle(idx)
    x0 = x0[idx]
    
    t_dict = {}
    for l in range(len(labels)):
        t_dict[l] = t[l, :] - 1

    for k in t_dict.keys():
        t_dict[k] = t_dict[k][idx]
    del dataloader
    data_pairs = torch.utils.data.TensorDataset(torch.from_numpy(x0).float(),
                                                torch.from_numpy(t_dict[0]).long())
    ratio = 0.8
    n_sample = len(data_pairs)
    train_size = int(n_sample*ratio)
    val_size = n_sample - train_size
    
    # train_set, val_set = torch.utils.data.random_split(data_pairs, [train_size, val_size])
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    train_size = int(n_sample * ratio)
    val_size = n_sample - train_size
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, n_sample))

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # criterion_adv = nn.NLLLoss()
    criterion_pred = nn.CrossEntropyLoss()
    criterion_rec = nn.MSELoss()
    params_adv = list(model.classifier_p.parameters())
    params = list(model.parameters())
    # optim_adv = optim.Adam(params_adv, lr=1e-4)
    # optimizer = optim.Adam(params, lr=1e-4)
    optim_adv = optim.SGD(params_adv, lr=0.1)
    optimizer = optim.SGD(params, lr=0.1)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler_adv = StepLR(optim_adv, step_size=10, gamma=0.1)
    
    n_epochs = 200
    model.train()
    # l_rec = 1.81e-5
    l_rec = 1
    l_adv = 1e-1
    for epoch in range(n_epochs):
        losses = []
        accs_p = []
        accs_t = []
        for iter, (in_data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            model.zero_grad()
            in_data /= 255
            rec, pred_t, pred_p, aug_rec = model(in_data.to(device))
            # grads = torch.autograd.grad(outputs=, inputs=, create_graph=True)
            # rank_idx, rank_label = make_index_rankingloss(target.detach().to('cpu').numpy())

            loss_i = criterion_pred(pred_t.to(device), target.to(device))
            loss_h = l_adv * criterion_pred(pred_p.to(device), target.to(device))
            loss_adv = l_adv * negative_entropy_loss(pred_p.to(device))
            # loss_rec = l_rec * criterion_rec(rec.to(device), in_data.to(device))
            # loss_rec_aug = l_rec * criterion_rec(aug_rec.to(device), in_data.to(device))
            # loss = loss_i + loss_h + loss_rec + loss_rec_aug
            loss = loss_i + loss_h
            
            loss.backward(retain_graph=True)
            model.classifier_p.zero_grad()
            optimizer.step()
            optimizer.zero_grad()

            loss_adv.backward(retain_graph=True)
            optim_adv.step()
            optimizer.zero_grad()

            with torch.no_grad():
                rec, pred_t, pred_p, aug_rec = model(in_data.to(device))
            y_true = target.to('cpu')
            pred_p = pred_p.detach().to('cpu')
            pred_t = pred_t.detach().to('cpu')
            tp_p = true_positive_multiclass(pred_p, y_true)
            tp_t = true_positive_multiclass(pred_t, y_true)
            acc_p = tp_p / len(in_data)
            acc_t = tp_t / len(in_data)

            accs_p.append(acc_p)
            accs_t.append(acc_t)
            # losses.append((loss+loss_adv).item())
            losses.append(loss.item())

        # output_reconst_img(in_data[0], rec[0], '{}/test_disentangle/reconst.png'.format(current_path))

        print('epoch:{}, loss:{}, acc_p:{}, acc_t:{}'.format(epoch+1, np.mean(losses), np.mean(accs_p), np.mean(accs_t)))
        writer.add_scalar('data/train_loss', np.mean(np.array(losses)), (epoch + 1))
        writer.add_scalar('data/train_acc_p', np.mean(np.array(accs_p)), (epoch + 1))
        writer.add_scalar('data/train_acc_t', np.mean(np.array(accs_t)), (epoch + 1))

        if (epoch + 1) % 10 == 0:
            # x0 = in_data[0].detach().numpy()
            # x0 = np.transpose(x0, (1, 2, 0))
            # rec_x0 = rec[0].detach().to('cpu').numpy()
            # rec_x0 = np.transpose(rec_x0, (1, 2, 0))
            # x0 = np.append(x0, rec_x0, axis=1)
            # x0 = cv2.cvtColor(x0, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('{}/test_disentangle/reconst.png'.format(current_path), np.uint8(255*x0))
            conf_mat_p = confusion_matrix(y_true, onehot2label(pred_p))
            conf_mat_t = confusion_matrix(y_true, onehot2label(pred_t))
            print(conf_mat_p)
            print(conf_mat_t)

            with torch.no_grad():
                val_loss = []
                acc_p = 0
                acc_t = 0
                for iter, (in_data, target) in enumerate(val_loader):
                    in_data /= 255
                    rec, pred_t, pred_p, aug_rec = model(in_data.to(device))

                    loss_i = criterion_pred(pred_t.to(device), target.to(device))
                    loss_h = l_adv * criterion_pred(pred_p.to(device), target.to(device))
                    loss_adv = l_adv * negative_entropy_loss(pred_p.to(device))
                    loss_rec = l_rec * criterion_rec(rec.to(device), in_data.to(device))
                    loss_rec_aug = l_rec * criterion_rec(aug_rec.to(device), in_data.to(device))
                    loss = loss_i + loss_h + loss_rec + loss_rec_aug

                    y_true = target.to('cpu')
                    pred_p = pred_p.detach().to('cpu')
                    pred_t = pred_t.detach().to('cpu')
                    tp_p = true_positive_multiclass(pred_p, y_true)
                    tp_t = true_positive_multiclass(pred_t, y_true)
                    val_loss.append(loss.item())
                    acc_p += tp_p
                    acc_t += tp_t

            print('epoch:{}, loss:{}, acc_p:{}, acc_t:{}'.format(epoch+1, np.mean(val_loss), acc_p/(n_sample-train_size), acc_t/(n_sample-train_size)))
            writer.add_scalar('data/val_loss', np.mean(val_loss), (epoch + 1))
            writer.add_scalar('data/val_acc_p', acc_p/(n_sample-train_size), (epoch + 1))
            writer.add_scalar('data/val_acc_t', acc_t/(n_sample-train_size), (epoch + 1))
        
        # model.eval()
        # if (epoch + 1) % 10 == 0:
        #     val_losses = []
        #     val_accs = []
        #     with torch.no_grad():
        #         for iter, (in_data, out_data) in enumerate(val_loader):
        #             y = model(in_data.to(device))
        #             loss = criterion(y.float().to(device), out_data.float().to(device))
        #             val_losses.append(loss.item())
        #             y_true = out_data.to('cpu').numpy()
        #             y_pred = y.detach().to('cpu').numpy()
        #             tn = true_negative(y_true.reshape(-1), y_pred.reshape(-1))
        #             tp = true_positive(y_true.reshape(-1), y_pred.reshape(-1))
        #             acc = (tn + tp)/len(y_true)
        #             val_accs.append(acc)

        #     writer.add_scalar('data/val_loss', np.mean(np.array(val_losses)), (epoch + 1))
        #     writer.add_scalar('data/val_acc', np.mean(np.array(val_accs)), (epoch + 1))
        #     print('epoch [{}/{}], val loss:{:.6f}, val acc:{:.4f}'.format(epoch+1, n_epochs, np.asarray(val_losses).mean(), np.asarray(val_accs).mean()))
    writer.close()
    torch.save(model.state_dict(), '{}/temp_disentangle/model_param.json'.format(current_path))

def train_TDAE():
    with h5py.File('data/toy_data.hdf5') as f:
        srcs = []
        targets1 = []
        targets2 = []
        for group_key in f.keys():
            group = group_key
            for parent_key in f[group].keys():
                parent_group = '{}/{}'.format(group, parent_key)
                src = []
                target1 = []
                target2 = []
                for child_key in f[parent_group].keys():
                    child_group = '{}/{}'.format(parent_group, child_key)
                    src.append(f[child_group][()])
                    target1.append(f[child_group].attrs['part'])
                    target2.append(f[child_group].attrs['color'])
                srcs.extend(src)
                targets1.extend(target1)
                targets2.extend(target2)
    srcs = np.asarray(srcs)
    srcs = srcs / srcs.max()
    srcs = np.transpose(srcs, (0, 3, 1, 2))
    targets1 = np.asarray(targets1)
    targets2 = np.asarray(targets2)
    srcs = torch.from_numpy(srcs).float()
    model = TDAE()
    model = model.to(device)
    # srcs = srcs[:10]
    # targets1 = targets1[:10]
    data_pairs = torch.utils.data.TensorDataset(srcs,
                                                torch.from_numpy(targets1).long())
    ratio = 0.8
    n_sample = len(data_pairs)
    train_size = int(n_sample*ratio)
    val_size = n_sample - train_size
    
    # train_set, val_set = torch.utils.data.random_split(data_pairs, [train_size, val_size])
    # train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    train_size = int(n_sample * ratio)
    val_size = n_sample - train_size
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, n_sample))

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    # criterion_adv = nn.NLLLoss()
    criterion_classifier = nn.CrossEntropyLoss()
    criterion_reconst = nn.MSELoss()
    # params_adv = list(model.subnet_p1.parameters()) + list(model.enc.parameters())
    # params_adv = list(model.subnet_p1.parameters()) + list(model.classifier_t.parameters())
    params_adv = list(model.subnet_p1.parameters()) + list(model.enc.parameters())
    params = list(model.parameters())
    # optim_adv = optim.Adam(params_adv, lr=1e-4)
    optim_adv = optim.Adam(params_adv)
    optimizer = optim.Adam(params)
    # optim_adv = optim.SGD(params, lr=0.001)
    # optimizer = optim.SGD(params, lr=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler_adv = StepLR(optim_adv, step_size=10, gamma=0.1)
    
    n_epochs = 1000
    model.train()
    # l_rec = 1.81e-5
    l_rec = 1
    l_adv = 1e-1
    for epoch in range(n_epochs):
        losses = []
        accs_p = []
        accs_t = []
        Loss = []
        CLoss = []
        Acc = 0
        Acc_adv = 0
        for ite, (in_data, target) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            optim_adv.zero_grad()
            model.zero_grad()
            preds, preds_adv, reconst = model(in_data.to(device))
            # with torch.no_grad():
            #     h0_t, h0_p = model.hidden_output_p(in_data.to(device))
            # preds_adv = model.classifier_t(h0_p)

            # grads = torch.autograd.grad(outputs=, inputs=, create_graph=True)
            # rank_idx, rank_label = make_index_rankingloss(target.detach().to('cpu').numpy())

            loss_classifier = criterion_classifier(preds.to(device), target.to(device))
            # loss_h = l_adv * criterion_pred(pred_p.to(device), target.to(device))
            # loss_classifier_adv = criterion_classifier(preds_adv.to(device), target.to(device))
            loss_reconst = criterion_reconst(reconst.to(device), in_data.to(device))
            loss_adv = negative_entropy_loss(preds_adv.to(device))
            # loss_rec_aug = l_rec * criterion_rec(aug_rec.to(device), in_data.to(device))
            # loss = loss_i + loss_h + loss_rec + loss_rec_aug
            # loss = loss_classifier + loss_reconst
            loss = loss_classifier + loss_reconst + 1000*loss_adv
            # loss_classifier.backward()
            # loss_reconst.backward()
            loss.backward()
            optimizer.step()
            # model.classifier_p.zero_grad()
            
            # optimizer.zero_grad()
            # print(loss_adv.item())
            # preds, preds_adv, reconst = model(in_data.to(device))
            # optim_adv.zero_grad()
            # loss_adv.backward()
            # optim_adv.step()
            # loss_adv.backward(retain_graph=True)

            Loss.append(loss.item())
            CLoss.append(loss_classifier.item())
            
            y_true = target.to('cpu')
            preds = preds.detach().to('cpu')
            preds_adv = preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            Acc_adv += true_positive_multiclass(preds_adv, y_true)

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                preds, preds_adv, reconst = model(in_data.to(device))
                np_input = in_data[0].detach().to('cpu')
                np_reconst = reconst[0].detach().to('cpu')
                fig = plt.figure(figsize=(16*2, 9))
                ax = fig.add_subplot(1, 2, 1)
                ax.imshow(np.transpose(np_input, (1,2,0)))
                ax = fig.add_subplot(1, 2, 2)
                ax.imshow(np.transpose(np_reconst, (1,2,0)))
                fig.savefig('results/{:04d}.png'.format(epoch+1))
                plt.close(fig)
        print('epoch: {} loss: {} classifier loss: {} Acc: {}, Acc_adv: {}'.format(epoch+1, np.mean(Loss), np.mean(CLoss), Acc/len(train_set), Acc_adv/len(train_set)))

if __name__ == '__main__':
    with h5py.File('./data/colon.hdf5', 'r') as f:
        key_list = list(f.keys())
        m_count = []
        p_count = []
        for k in key_list:
            mayo_label = f[k].attrs['mayo_label']
            part_label = f[k].attrs['part_label']
            nmayo_label = np.array([1 if cat <= 2 else 2 for cat in mayo_label])
            print('M', mayo_label)
            print('N', nmayo_label)
            print('P', part_label)
            print()
            # sys.exit()
            dm = mayo_label[1:] - mayo_label[:-1]
            dp = part_label[1:] - part_label[:-1]
            m_count.append(np.sum(dm != 0)/(len(mayo_label)-1))
            p_count.append(np.sum(dp != 0)/(len(part_label)-1))
    m_hist, m_bins = np.histogram(m_count, bins=np.arange(0, 1.1, .1))
    p_hist, p_bins = np.histogram(p_count, bins=np.arange(0, 1.1, .1))
    print(m_bins[1:])
    print(m_hist)
    print(p_hist)
    print(np.cumsum(m_hist)/np.sum(m_hist))
    print(np.cumsum(p_hist)/np.sum(p_hist))
    sys.exit()
    train_TDAE()
    pass
    # main()
