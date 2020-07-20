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
from torchvision.utils import make_grid
import tensorboardX as tbx
# from torch.utils.tensorboard import SummaryWriter


from __init__ import clean_directory 

SEED = 1
torch.manual_seed(SEED)
rn.seed(SEED)
np.random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, n_class1=3, n_class2=2, ksize=3, d2ae_flag=False, img_w=256, img_h=256):
        super().__init__()
        if d2ae_flag:
            n_class2 = n_class1
        self.img_h, self.img_w = img_h, img_w
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
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        self.enc = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        self.subnet_conv_t1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.subnet_conv_t2 = nn.Sequential(nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.subnet_conv_p1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.subnet_conv_p2 = nn.Sequential(nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.subnet_t1 = nn.Sequential(nn.Linear(in_features=64, out_features=256),
                                    nn.ReLU())
        self.subnet_p1 = nn.Sequential(nn.Linear(in_features=64, out_features=256),
                                    nn.ReLU())
                
        # self.subnet_t2 = nn.Sequential(nn.Linear(in_features=256, out_features=8*8*64),
        #                             nn.ReLU())
        # self.subnet_p2 = nn.Sequential(nn.Linear(in_features=256, out_features=8*8*64),
        #                             nn.ReLU())
        self.classifier_main = nn.Linear(in_features=256, out_features=n_class1)
        self.classifier_sub = nn.Linear(in_features=256, out_features=n_class2)

        # self.deconv1 = nn.Sequential(nn.Linear(in_features=64*2, out_features=392),
        #                             nn.ReLU())
        # self.deconv2 = torch.nn.Upsample(scale_factor=2,mode='nearest')

        self.dec_fc1 = nn.Sequential(nn.Linear(in_features=256*2, out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*64),
                                    nn.ReLU())
        # self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 2, stride=2),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())
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
        
    def forward_train_like_D2AE(self, input):
        h0 = self.enc(input)
        t0 = self.subnet_conv_t1(h0)
        p0 = self.subnet_conv_p1(h0)
        t0 = self.subnet_conv_t2(t0)
        p0 = self.subnet_conv_p2(p0)
        t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        t0 = torch.reshape(t0, (t0.size(0), -1))
        p0 = torch.reshape(p0, (p0.size(0), -1))
        t0 = self.subnet_t1(t0)
        p0 = self.subnet_p1(p0)
        p0_no_grad = p0.clone().detach()
        class_main_preds = self.classifier_main(t0)
        class_sub_preds = self.classifier_sub(p0)
        class_sub_preds_adv = self.classifier_sub(p0_no_grad)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_w//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return class_main_preds, class_sub_preds, class_sub_preds_adv, rec

    def forward(self, input):
        h0 = self.enc(input)
        # h0 = torch.reshape(h0, (h0.size(0), -1))
        t0 = self.subnet_conv_t1(h0)
        p0 = self.subnet_conv_p1(h0)
        t0 = self.subnet_conv_t2(t0)
        p0 = self.subnet_conv_p2(p0)
        t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        t0 = torch.reshape(t0, (t0.size(0), -1))
        p0 = torch.reshape(p0, (p0.size(0), -1))
        t0 = self.subnet_t1(t0)
        p0 = self.subnet_p1(p0)
        class_main_preds = self.classifier_main(t0)
        class_sub_preds = self.classifier_main(p0)
        class_main_preds_adv = self.classifier_main(p0)
        class_sub_preds_adv = self.classifier_main(t0)
        # class_preds_adv = self.classifier_t(p0)
        # t1 = self.subnet_t2(t0)

        # p1 = self.subnet_p2(p0)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        # t1 = torch.reshape(t1, (t1.size(0), 64, 8, 8))
        # p1 = torch.reshape(t1, (p1.size(0), 64, 8, 8))
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        # concat_enc = torch.cat((t1, p1), dim=1)
        rec = self.dec(concat_h0)
        # return class_main_preds, class_preds_adv, rec
        return class_main_preds, class_sub_preds, class_main_preds_adv, class_sub_preds_adv, rec

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
        t0 = self.subnet_conv_t1(h0)
        p0 = self.subnet_conv_p1(h0)
        t0 = self.subnet_conv_t2(t0)
        p0 = self.subnet_conv_p2(p0)
        t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        t0 = torch.reshape(t0, (t0.size(0), -1))
        p0 = torch.reshape(p0, (p0.size(0), -1))
        t0 = self.subnet_t1(t0)
        p0 = self.subnet_p1(p0)
        return t0, p0

    def reconst(self, input):
        h0 = self.enc(input)
        t0 = self.subnet_conv_t1(h0)
        p0 = self.subnet_conv_p1(h0)
        t0 = self.subnet_conv_t2(t0)
        p0 = self.subnet_conv_p2(p0)
        t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        t0 = torch.reshape(t0, (t0.size(0), -1))
        p0 = torch.reshape(p0, (p0.size(0), -1))
        t0 = self.subnet_t1(t0)
        p0 = self.subnet_p1(p0)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def shuffle_reconst(self, input, idx1, idx2):
        h0 = self.enc(input)
        t0 = self.subnet_conv_t1(h0)
        p0 = self.subnet_conv_p1(h0)
        t0 = self.subnet_conv_t2(t0)
        p0 = self.subnet_conv_p2(p0)
        t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        t0 = torch.reshape(t0, (t0.size(0), -1))
        p0 = torch.reshape(p0, (p0.size(0), -1))
        t0 = self.subnet_t1(t0)
        p0 = self.subnet_p1(p0)
        concat_h0 = torch.cat((t0[idx1], p0[idx2]), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec


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

def train_TDAE(data_path='data/toy_data.hdf5'):
    if 'toy_data' in data_path:
        img_w = 256
        img_h = 256
    else:
        img_w = 224
        img_h = 224
    out_fig_dpath = './reports/figure'
    out_param_dpath = './reports/param'
    out_board_dpath = './reports/runs'
    clean_directory(out_fig_dpath)
    clean_directory(out_param_dpath)
    clean_directory(out_board_dpath)
    # clean_directory(out_board_dpath)
    d2ae_flag = False
    writer = tbx.SummaryWriter(out_board_dpath)

    with h5py.File(data_path, 'r') as f:
        srcs = []
        targets1 = []
        targets2 = []
        for group_key in f.keys():
            group = group_key
            # print(group)
            # return
            for parent_key in f[group].keys():
                parent_group = '{}/{}'.format(group, parent_key)
                src = []
                target1 = []
                target2 = []
                for child_key in f[parent_group].keys():
                    child_group = '{}/{}'.format(parent_group, child_key)
                    src.append(f[child_group][()])
                    target1.append(f[child_group].attrs['part'])
                    target2.append(f[child_group].attrs['mayo'])
                srcs.extend(src)
                targets1.extend(target1)
                targets2.extend(target2)
    srcs = np.asarray(srcs)
    srcs = srcs / srcs.max()
    srcs = np.transpose(srcs, (0, 3, 1, 2))
    targets1 = np.asarray(targets1)
    targets2 = np.asarray(targets2)
    srcs = torch.from_numpy(srcs).float()
    model = TDAE(n_class1=3, n_class2=5, d2ae_flag = d2ae_flag, img_h=img_h, img_w=img_w)
    model = model.to(device)
    # srcs = srcs[:10]
    # targets1 = targets1[:10]
    data_pairs = torch.utils.data.TensorDataset(srcs,
                                                torch.from_numpy(targets1).long(),
                                                torch.from_numpy(targets2).long())
    ratio = 0.6
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
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # criterion_adv = nn.NLLLoss()
    criterion_classifier = nn.CrossEntropyLoss()
    criterion_reconst = nn.MSELoss()
    params = list(model.parameters())
    # optim_adv = optim.Adam(params_adv, lr=1e-4)
    params_adv = list(model.classifier_sub.parameters())
    optim_adv = optim.Adam(params_adv)
    optimizer = optim.Adam(params)
    # optim_adv = optim.SGD(params, lr=0.001)
    # optimizer = optim.SGD(params, lr=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler_adv = StepLR(optim_adv, step_size=10, gamma=0.1)
    
    n_epochs = 300
    model.train()
    best_loss = np.inf
    # l_rec = 1.81e-5
    l_adv = 1.0e-0
    l_recon = 1.0e-0 * 2
    for epoch in range(n_epochs):
        losses = []
        accs_p = []
        accs_t = []
        Loss, RecLoss, CLoss, CLoss_sub = [], [], [], []
        Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0
        for ite, (in_data, target, sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            if d2ae_flag:
                optim_adv.zero_grad()
                preds, sub_preds_adv, sub_preds, reconst = model.forward_train_like_D2AE(in_data.to(device))
                loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                loss_reconst.backward(retain_graph=True)
                loss_classifier_main = criterion_classifier(preds.to(device), target.to(device))
                loss_classifier_main.backward(retain_graph=True)
                loss_sub_adv = l_adv * negative_entropy_loss(sub_preds_adv.to(device).to(device))
                loss_sub_adv.backward(retain_graph=True)
                model.classifier_sub.zero_grad()
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
                preds, sub_preds_adv, sub_preds, reconst = model.forward_train_like_D2AE(in_data.to(device))
                loss_classifier_sub = l_adv * criterion_classifier(sub_preds_adv.to(device), target.to(device))
                loss_classifier_sub.backward(retain_graph=True)
                optim_adv.step()
                loss = loss_classifier_main + loss_classifier_sub + loss_sub_adv + loss_reconst
            else:
                preds, sub_preds, preds_adv, sub_preds_adv, reconst = model(in_data.to(device))
                loss_reconst = l_recon*criterion_reconst(reconst.to(device), in_data.to(device))
                loss_reconst.backward(retain_graph=True)
                loss_adv = l_adv*negative_entropy_loss(sub_preds.to(device))
                loss_adv.backward(retain_graph=True)
                model.classifier_main.zero_grad()
                loss_classifier_main = criterion_classifier(preds.to(device), target.to(device))
                loss_classifier_main.backward(retain_graph=True)
                # loss_classifier_sub = criterion_classifier(sub_preds.to(device), sub_target.to(device))
                # loss_adv = negative_entropy_loss(preds_adv.to(device))
                # loss_sub_adv = negative_entropy_loss(sub_preds_adv.to(device))
                # loss = loss_classifier_main + loss_classifier_sub + 0*loss_adv + 0*loss_sub_adv + loss_reconst
                # loss = loss_classifier_main + loss_classifier_sub + 0*loss_adv + 0*loss_sub_adv + loss_reconst
                loss = loss_classifier_main + loss_adv + loss_reconst
                # loss.backward()
                optimizer.step()

            Loss.append(loss.item())
            RecLoss.append(loss_reconst.item())
            CLoss.append(loss_classifier_main.item())
            CLoss_sub.append(loss_adv.item())
            
            y_true = target.to('cpu')
            sub_y_true = sub_target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = sub_preds.detach().to('cpu')
            # preds_adv = preds_adv.detach().to('cpu')
            # sub_preds_adv = sub_preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, y_true)
            # Acc_adv += true_positive_multiclass(preds_adv, y_true)
            # sub_Acc_adv += true_positive_multiclass(sub_preds_adv, sub_y_true)

        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}, Acc_adv: {}, sub Acc_adv: {}'.format(epoch+1, np.mean(Loss), Acc/len(train_set), sub_Acc/len(train_set), Acc_adv/len(train_set), sub_Acc_adv/len(train_set)))
        writer.add_scalar('summarize loss',
            np.mean(Loss), epoch)
        writer.add_scalar('rec loss',
            np.mean(RecLoss), epoch)
        writer.add_scalar('classifier loss',
            np.mean(CLoss), epoch)
        writer.add_scalar('Adv loss',
            np.mean(CLoss_sub), epoch)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for in_data, target1, target2 in train_loader:
                    reconst = model.reconst(in_data.to(device))
                    np_input = in_data[0].detach().to('cpu')
                    np_reconst = reconst[0].detach().to('cpu')
                    # fig = plt.figure(figsize=(16*2, 9))
                    # ax = fig.add_subplot(1, 2, 1)
                    # ax.set_title('{}:{}'.format(target1[0].detach().to('cpu'), target2[0].detach().to('cpu')))
                    # ax.imshow(np.transpose(np_input, (1,2,0)))
                    # ax = fig.add_subplot(1, 2, 2)
                    # ax.set_title('{}:{}'.format(target1[0].detach().to('cpu'), target2[0].detach().to('cpu')))
                    # ax.imshow(np.transpose(np_reconst, (1,2,0)))
                    # fig.savefig('{}/{:04d}.png'.format(out_fig_dpath, epoch+1))
                    # plt.close(fig)
                    img_grid = make_grid(torch.stack([np_input, np_reconst]))
                    writer.add_image('test'.format(epoch+1), img_grid)
                    break
                val_losses = []
                for in_data, target, sub_target in val_loader:
                    if d2ae_flag:
                        preds, sub_preds_adv, sub_preds, reconst = model.forward_train_like_D2AE(in_data.to(device))
                        loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                        loss_classifier_main = criterion_classifier(preds.to(device), target.to(device))
                        loss_classifier_sub = l_adv * criterion_classifier(sub_preds_adv.to(device), target.to(device))
                        loss_sub_adv = l_adv * negative_entropy_loss(sub_preds_adv.to(device).to(device))
                        val_loss = loss_reconst + loss_classifier_main + loss_sub_adv + loss_classifier_sub
                    else:
                        preds, sub_preds, preds_adv, sub_preds_adv, reconst = model(in_data.to(device))
                        val_loss_reconst = criterion_reconst(reconst.to(device), in_data.to(device))
                        val_loss_classifier_main = criterion_classifier(preds.to(device), target.to(device))
                        val_loss_adv = negative_entropy_loss(sub_preds.to(device))
                        # val_loss_classifier_sub = criterion_classifier(sub_preds.to(device), sub_target.to(device))
                        # val_loss_adv = negative_entropy_loss(preds_adv.to(device))
                        # val_loss_sub_adv = negative_entropy_loss(sub_preds_adv.to(device))
                        val_loss = val_loss_classifier_main + val_loss_adv + val_loss_reconst
                        # val_loss = val_loss_classifier_main + val_loss_classifier_sub + 0*val_loss_adv + 0*val_loss_sub_adv + val_loss_reconst

                    val_losses.append(val_loss.item())
                print('epoch: {} val loss: {}'.format(epoch+1, np.mean(val_losses)))
                if best_loss > np.mean(val_losses):
                    best_loss = np.mean(val_losses)
                    torch.save(model.state_dict(), '{}/TDAE_test_bestparam.json'.format(out_param_dpath))

    torch.save(model.state_dict(), '{}/TDAE_test_param.json'.format(out_param_dpath))
    writer.close()


def val_TDAE(data_path='data/toy_data.hdf5'):
    if 'toy_data' in data_path:
        img_w = 256
        img_h = 256
    else:
        img_w = 224
        img_h = 224

    d2ae_flag = False
    with h5py.File(data_path, 'r') as f:
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
    model = TDAE(n_class1=3, n_class2=5, d2ae_flag = d2ae_flag, img_h=img_h, img_w=img_w)
    model.load_state_dict(torch.load('./reports/param/TDAE_test_param.json'))
    model = model.to(device)
    data_pairs = torch.utils.data.TensorDataset(srcs,
                                                torch.from_numpy(targets1).long(),
                                                torch.from_numpy(targets2).long())
    ratio = 0.6
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

    # _ = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    # _ = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)

    cat_val_set = val_set
    with torch.no_grad():
        model.eval()
        for n_iter, (inputs, targets1, target2) in enumerate(val_loader):
            reconst = model.reconst(inputs.to(device))
            s_reconst = model.shuffle_reconst(inputs.to(device), [0, 1], [1, 0])
            np_input0 = inputs[0].detach().to('cpu')
            np_input1 = inputs[1].detach().to('cpu')
            np_reconst0 = reconst[0].detach().to('cpu')
            np_reconst1 = reconst[1].detach().to('cpu')
            s_np_reconst0 = s_reconst[0].detach().to('cpu')
            s_np_reconst1 = s_reconst[1].detach().to('cpu')
            fig = plt.figure(figsize=(16*2, 9*2))
            ax = fig.add_subplot(2, 3, 1)
            ax.set_title('1')
            ax.imshow(np.transpose(np_input0, (1,2,0)))
            ax = fig.add_subplot(2, 3, 2)
            ax.set_title('1')
            ax.imshow(np.transpose(np_reconst0, (1,2,0)))
            ax = fig.add_subplot(2, 3, 3)
            ax.set_title('1')
            ax.imshow(np.transpose(s_np_reconst0, (1,2,0)))
            ax = fig.add_subplot(2, 3, 4)
            ax.set_title('2')
            ax.imshow(np.transpose(np_input1, (1,2,0)))
            ax = fig.add_subplot(2, 3, 5)
            ax.set_title('2')
            ax.imshow(np.transpose(np_reconst1, (1,2,0)))
            ax = fig.add_subplot(2, 3, 6)
            ax.set_title('2')
            ax.imshow(np.transpose(s_np_reconst1, (1,2,0)))
            fig.savefig('reports/sample{:04d}.png'.format(n_iter))
            plt.close(fig)
            if n_iter > 10:
                break

def data_review():
    with h5py.File('./data/colon.hdf5', 'r') as f:
        with h5py.File('./data/colon_renew.hdf5', 'w') as f_out:
            key = list(f.keys())
            df = pd.read_csv('./data/colon_data2label.csv')
            # flag = False
            for k in key:
                cat_df = df[df.sequence_num == int(k)]
                for i, fname in enumerate(cat_df.filename):
                    # print(f[k].attrs['mayo_label'][i])
                    p = f[k].attrs['part_label'][i]
                    m = f[k].attrs['mayo_label'][i]
                    f_out.create_dataset(name='img/{}/{}'.format(k, fname), data=f[k][i])
                    f_out['img/{}/{}'.format(k, fname)].attrs['part'] = p-1
                    f_out['img/{}/{}'.format(k, fname)].attrs['mayo'] = m-1
    
if __name__ == '__main__':
    if os.path.exists('./data/colon_renew.hdf5') is False:
        data_review()
    d = './data/colon_renew.hdf5'
    train_TDAE()
    val_TDAE()
    sys.exit()
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
    pass
    # main()
