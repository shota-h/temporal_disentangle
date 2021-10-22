import os
import sys
import copy
import json
import itertools
import time
import datetime
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
import seaborn as sns
from functools import partial
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import torch
from torch.nn.modules import activation
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.utils import make_grid
from torchvision.datasets import SVHN, MNIST
from torchvision import transforms

import tensorboardX as tbx

from losses import TripletLoss, negative_entropy_loss, Fourier_mse, loss_vae, MetricLoss
from metrics import true_positive_multiclass, true_positive, true_negative
from __init__ import clean_directory, SetIO
from data_handling import get_triplet_flatted_data_with_idx, random_label_replace
from archs import TDAE_VAE
from archs import TDAE_VAE_fullsuper_disentangle
from archs import SemiSelfClassifier

SEED = 1
torch.manual_seed(SEED)
rn.seed(SEED)
np.random.seed(SEED)
gpu_devices = ','.join([str(id) for id in range(torch.cuda.device_count())])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def scalars2summary(writer, tags, vals, epoch):
    summaries = []
    for val, tag in zip(vals, tags):
        writer.add_scalar(tag, val, epoch)


def args2pandas(args):
    dict_args = copy.copy(vars(args))
    for k in dict_args.keys():
        dict_args[k] = [dict_args[k]]
    return pd.DataFrame.from_dict(dict_args)


def get_character_dataset():
    mnist_train = MNIST("./data/MNIST", train=True, download=True, transform=transforms.ToTensor())
    # mnist_test = MNIST("MNIST", train=False, download=True, transform=transforms.ToTensor())
    SVHN_train = SVHN("./data/SVHN", split='train', download=True, transform=transforms.ToTensor())
    
    print(mnist_train.data.max(), mnist_train.data.min(), mnist_train.data.size())
    print(SVHN_train.data.max(), SVHN_train.data.min(), SVHN_train.data.shape)
    print(mnist_train.targets.max(), mnist_train.targets.min())
    print(SVHN_train.labels.max(), SVHN_train.labels.min())
    # SVHN_test = SVHN("SVHN", split='test', download=True, transform=transforms.ToTensor())

    # train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)


def argparses():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--ndeconv', type=int, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--dm', type=int, default=0)
    parser.add_argument('--T1', type=int, default=100)
    parser.add_argument('--T2', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--data', type=str, default='toy')
    parser.add_argument('--param', type=str, default='best')
    parser.add_argument('--ex', type=str, default=None)
    parser.add_argument('--classifier', type=float, default=1)
    parser.add_argument('--rec', type=float, default=1e-3)
    parser.add_argument('--adv', type=float, default=1.0)
    parser.add_argument('--tri', type=float, default=1e-2)
    parser.add_argument('--ratio', type=float, default=0.0)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--triplet', action='store_true')
    parser.add_argument('--single_triplet', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--rev', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--adapt', action='store_true')
    parser.add_argument('--adapt_alpha', action='store_true')
    parser.add_argument('--semi', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--use_pseudo', action='store_true')
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--dual', action='store_true')
    parser.add_argument('--labeldecomp', action='store_false')
    parser.add_argument('--channels', type=int, nargs='+', default=[3,16,32,64,128])
    return parser.parse_args()


class PairwiseSampler(Dataset):
    def __init__(self, X, u):
        self.X = X
        self.u = u

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx_i):
        x_i = self.X[idx_i]
        u_i = self.u[idx_i]

        indices = list(range(len(self.u)))
        indices.remove(idx_i)
        idx_j = np.random.choice(indices)

        x_j = self.X[idx_j]
        u_j = self.u[idx_j]

        return x_i, x_j, u_i, u_j


class TensorTransforms():
    def __init__(self, methods, p):
        self.methods = []
        for m in methods:
            self.methods.append(m(p))
        
    def __call__(self, tensors):
        trans_tensors = []
        for tensor in tensors:
            for m in self.methods:
                tensor = m(tensor)
            trans_tensors.append(tensor)
        return torch.stack(trans_tensors)


class vflip():
    """Flips tensor vertically.
    """
    def __init__(self, p):
        self.p = p
    def __call__(self, tensor):
        if rn.random() <= self.p:
            return tensor.flip(1)
        else:
            return tensor


class hflip():
    """Flips tensor horizontally.
    """
    def __init__(self, p):
        self.p = p
    def __call__(self, tensor):
        if rn.random() <= self.p:
            return tensor.flip(2)
        else:
            return tensor


class rotate():
    def __init__(self, p):
        self.p = p
    def __call__(self, tensor):
        if rn.random() <= self.p:
            return tensor.rot90(rn.randint(1,3), [1,2])
        else:
            return tensor


def statistical_augmentation(features):
    outs = []
    for h0 in features:
        e_h0 = torch.normal(0, 1, size=h0.size()).to(device)
        std_h0 = torch.std(h0, dim=0)
        new_h0 = h0 + torch.mul(e_h0, std_h0)
        outs.append(new_h0)
    return outs


def dynamic_weight_average(prev_loss, current_loss, temp=1.0):
    loss_ratio_dict = {}
    weight_dict = {}
    sum_loss = 0
    for k in prev_loss.keys():
        loss_ratio_dict[k] = current_loss[k] / prev_loss[k]
        sum_loss += np.exp(loss_ratio_dict[k] / temp)
    for k in loss_ratio_dict.keys():
        weight_dict[k] = len(list(loss_ratio_dict.keys())) * np.exp(loss_ratio_dict[k] / temp) / sum_loss

    return weight_dict


def validate_linearclassifier(X_train, Y_train, X_tests, Y_tests):
    logreg = LogisticRegression(penalty='l2', solver="sag")
    logreg.fit(X_train, Y_train)
    train_score = logreg.score(X_train, Y_train)
    test_scores = []
    for X_test, Y_test in zip(X_tests, Y_tests):
        test_scores.append(logreg.score(X_test, Y_test))
    return train_score, test_scores

def get_outputpath():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_VAE_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_VAE_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'huge' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_VAE_huge' 
        data_path = './data/huge_toy_data.hdf5'
    elif 'Huge' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_VAE_HUGE' 
        data_path = './data/Huge_toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_VAE_colon'
        data_path = './data/colon_renew.hdf5'
    
    if not(args.ex is None):
        ex = args.ex
    else:
        if args.triplet:
            ex = 'prop'
        else:
            ex = 'exist'
            
    return img_w, img_h, out_source_dpath, data_path, ex


def pseudo_label_update(model, inputs, idx=1):
    with torch.no_grad():
        model.eval()
        pred_pseudo_labels = model.predict_label(inputs)
    return pred_pseudo_labels[idx].to('cpu')


def train_fixmatch():
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    out_source_dpath += 'SemiSelf'
    out_source_dpath = os.path.join(out_source_dpath, ex)

    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs, _, _ = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    else:
        src, targets1, targets2, idxs, _, _ = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
        
    ratio = [0.7, 0.2, 0.1]
    n_sample = len(src)
    train_size = int(n_sample*ratio[0])
    val_size = int(n_sample*ratio[1])
    test_size = n_sample - train_size - val_size
    # train_set, val_set = torch.utils.data.random_split(data_pairs, [train_size, val_size])
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))
    
    true_targets1 = copy.deepcopy(targets1)
    true_targets2 = copy.deepcopy(targets2)
    if args.semi:
        fix_indices = torch.cat([torch.ones(train_size), torch.zeros(val_size+test_size)])
        targets2 = random_label_replace(targets2, ratio=args.ratio, value=-2, fix_indices=fix_indices, seed=2)
        pseudo_label = torch.randint(low=0, high=torch.unique(true_targets2).size(0), size=targets2.size())
        pseudo_label[targets2 != -2] = -2
        if args.dual:
            targets1 = random_label_replace(targets1, ratio=args.ratio, value=-2, fix_indices=fix_indices, seed=2)
            pseudo_label1 = torch.randint(low=0, high=torch.unique(true_targets1).size(0), size=targets1.size())
            pseudo_label1[targets1 != -2] = -2

    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)
    
    model = SemiSelfClassifier(n_classes=[torch.unique(true_targets1).size(0), torch.unique(true_targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)
    if args.retrain:
        model.load_state_dict(torch.load('{}/param/SemiSelf_bestparam.json'.format(out_source_dpath)))
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_board_dpath = '{}/re_runs'.format(out_source_dpath)
        out_condition_dpath = '{}/re_condition'.format(out_source_dpath)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_board_dpath = '{}/runs'.format(out_source_dpath)
        out_condition_dpath = '{}/condition'.format(out_source_dpath)

    clean_directory(out_param_dpath)
    clean_directory(out_board_dpath)
    clean_directory(out_condition_dpath)
    writer = tbx.SummaryWriter(out_board_dpath)
    model = model.to(device)

    ngpus = 1
    if args.multi:
        ngpus = torch.cuda.device_count()    
        model = nn.DataParallel(model, device_ids=[g for g in range(ngpus)])

    params = list(model.parameters())
    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    train_size1 = train_set[:][3][train_set[:][3] != -2].size(0)
    train_size2 = train_set[:][4][train_set[:][4] != -2].size(0)

    train_loader = DataLoader(train_set, batch_size=args.batch*ngpus, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch*ngpus, shuffle=False)

    # criterion_adv = nn.NLLLoss()        
    
    n_epochs = args.epoch
    best_epoch, best_loss = 0, np.inf
    best_semi_epoch, best_semi_loss = 0, np.inf


def val_SemiSelf(zero_padding=False):
    torch.manual_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    out_source_dpath += 'SemiSelf'
    args = argparses()

    out_source_dpath = out_source_dpath + '/' + ex
    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs, _, _ = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    else:
        src, targets1, targets2, idxs, _, _ = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)

    # true_targets2 = copy.deepcopy(targets2)
    # if args.semi:
    #     targets2 = random_label_replace(targets2, ratio=args.ratio, value=-2)
    #     pseudo_label = torch.randint(low=0, high=torch.unique(true_targets2).size(0), size=targets2.size())
    #     pseudo_label[targets2 != -2] = -2

    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)

    if args.retrain:
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_val_dpath = '{}/re_val_{}'.format(out_source_dpath, args.param)
        out_fig_dpath = '{}/re_fig_{}'.format(out_source_dpath, args.param)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_val_dpath = '{}/val_{}'.format(out_source_dpath, args.param)
        out_fig_dpath = '{}/fig_{}'.format(out_source_dpath, args.param)
    clean_directory(out_fig_dpath)
    clean_directory(out_val_dpath)

    model = SemiSelfClassifier(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)

    if args.param == 'best':
        model.load_state_dict(torch.load('{}/SemiSelf_bestparam.json'.format(out_param_dpath)))
    else:
        model.load_state_dict(torch.load('{}/SemiSelf_lastparam.json'.format(out_param_dpath)))
    model = model.to(device)

    ratio = [0.7, 0.2, 0.1]
    n_sample = len(data_pairs)
    train_size = int(n_sample*ratio[0])
    val_size = int(n_sample*ratio[1])
    test_size = n_sample - (train_size + val_size)
    
    # train_set, val_set = torch.utils.data.random_split(data_pairs, [train_size, val_size])
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

    with torch.no_grad():
        for first, loader in zip(['train', 'val'], [train_loader, val_loader]):
            model.eval()
            X1, X2, Y1, Y2 = [], [], [], []
            Prob1, Prob2 = [], []
            for n_iter, (idx, _, _, target, sub_target) in enumerate(loader):
                (mu1, mu2) = model.hidden_output(src[idx].to(device))
                prob1, prob2 = model.predict_proba(src[idx].to(device))
                
                mu1 = mu1.detach().to('cpu').numpy()
                mu2 = mu2.detach().to('cpu').numpy()
                X1.extend(mu1)
                X2.extend(mu2)
                Y1.extend(target.detach().to('cpu').numpy())
                Y2.extend(sub_target.detach().to('cpu').numpy())
        
            X1 = np.asarray(X1)
            X2 = np.asarray(X2)
            Y1 = np.asarray(Y1)
            Y2 = np.asarray(Y2) 
            markers = ['.', 'x']
            colors1 = ['blue', 'orange', 'magenta']
            colors1 = colors1[:len(np.unique(Y1))]
            colors2 = ['r', 'g', 'y', 'm', 'c']
            colors2 = colors2[:len(np.unique(Y2))]
            if args.rev:
                buff = colors1
                colors1 = colors2
                colors2 = buff

            if first == 'train':
                logreg_main2main = LogisticRegression(penalty='l2', solver="sag")
                logreg_main2main.fit(X1, Y1)
                logreg_main2sub = LogisticRegression(penalty='l2', solver="sag")
                logreg_main2sub.fit(X1, Y2)
                logreg_sub2main = LogisticRegression(penalty='l2', solver="sag")
                logreg_sub2main.fit(X2, Y1)
                logreg_sub2sub = LogisticRegression(penalty='l2', solver="sag")
                logreg_sub2sub.fit(X2, Y2)

            for Y in [Y1, Y2]:
                for u in np.unique(Y):
                    print(u, ':', len(Y[Y==u]), '/', len(Y))
            pred_Y11 = logreg_main2main.predict(X1)
            pred_Y12 = logreg_main2sub.predict(X1)
            pred_Y21 = logreg_sub2main.predict(X2)
            pred_Y22 = logreg_sub2sub.predict(X2)
        
            for (X, ex) in zip([X1, X2], ['main', 'sub']):
                if ex == 'main':
                    pred_Y1 = pred_Y11
                    pred_Y2 = pred_Y12
                elif ex == 'sub':
                    pred_Y1 = pred_Y21
                    pred_Y2 = pred_Y22
                rn.seed(SEED)
                np.random.seed(SEED)
                tsne = TSNE(n_components=2, random_state=SEED)
                Xt = tsne.fit_transform(X)
                fig = plt.figure(figsize=(16*2, 9))
                for ia, (Y, co) in enumerate(zip([Y1, Y2], [colors1, colors2])): 
                    ax = fig.add_subplot(1,2,ia+1)
                    for iy, k in enumerate(np.unique(Y)):
                        ax.scatter(x=Xt[Y==k,0], y=Xt[Y==k,1], c=co[iy], alpha=0.5, marker='.')
                    ax.set_aspect('equal', 'datalim')
                fig.savefig('{}/{}_hidden_features_{}.png'.format(out_fig_dpath, first, ex))
                plt.close(fig)

                fig = plt.figure(figsize=(16*2, 9))
                for ia, (Y, co) in enumerate(zip([pred_Y1, pred_Y2], [colors1, colors2])): 
                    ax = fig.add_subplot(1,2,ia+1)
                    for iy, k in enumerate(np.unique(Y)):
                        ax.scatter(x=Xt[Y==k,0], y=Xt[Y==k,1], c=co[iy], alpha=0.5, marker='.')
                    ax.set_aspect('equal', 'datalim')
                fig.savefig('{}/{}_hidden_features_{}_Classifier{}.png'.format(out_fig_dpath, first, ex, ex))
                plt.close(fig)
                continue
                fig = plt.figure(figsize=(6*2, 6))
                ax = fig.add_subplot(1,2,1)
                ax.hist(np.mean(X, axis=0))
                ax = fig.add_subplot(1,2,2)
                ax.hist(np.std(X, axis=0))
                fig.savefig('{}/{}_dist_{}.png'.format(out_fig_dpath, first, ex))
                plt.close(fig)
                Xn = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 
                print(Xn.shape)
                print(Xn.max())
                print(Xn.min())
                pca = PCA()
                pca.fit(Xn)
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(1,1,1)
                ratio = np.append([0], np.cumsum(pca.explained_variance_ratio_))
                pca_idx = np.where(ratio >= 0.99)[0]
                ax.plot(ratio)
                fig.savefig('{}/{}_ratio_{}.png'.format(out_fig_dpath, first, ex))
                plt.close(fig)
                
                pca_feature = pca.transform(Xn)
                fig = plt.figure(figsize=(16*2, 9))
                for ia, (Y, co) in enumerate(zip([Y1, Y2], [colors1, colors2])):
                    ax = fig.add_subplot(1,2,ia+1)
                    for iy, k in enumerate(np.unique(Y)):
                        ax.scatter(pca_feature[Y==k, 0], pca_feature[Y==k, 1], alpha=0.5, c=co[iy], marker='.')
                        ax.set_aspect('equal', 'datalim')
                fig.savefig('{}/{}_pca_{}.png'.format(out_fig_dpath, first, ex))
                plt.close(fig)
                
                cs = ['dim{}'.format(d+1) for d in range(5)]
                df = pd.DataFrame(pca_feature[:, :5], columns=cs)
                for tag, Y, co in zip(['Y1', 'Y2'], [Y1, Y2], [colors1, colors2]):
                    tar = ['Class{}'.format(y) for y in Y]
                    df['target'] = tar
                    palette_dict = {}
                    for t, c in zip(np.unique(tar), co):
                        palette_dict[t] = c
                    sns.pairplot(df, hue='target', diag_kind='hist', vars=cs, palette=palette_dict).savefig('{}/{}_PairPlot_{}_{}.png'.format(out_fig_dpath, first, ex, tag))
                    plt.close()
                
                for tag, Y, co in zip(['PredY1', 'PredY2'], [pred_Y1, pred_Y2], [colors1, colors2]):
                    tar = ['Class{}'.format(y) for y in Y]
                    df['target'] = tar
                    palette_dict = {}
                    for t, c in zip(np.unique(tar), co):
                        palette_dict[t] = c
                    sns.pairplot(df, hue='target', diag_kind='hist', vars=cs, palette=palette_dict).savefig('{}/{}_PairPlot_{}_{}_pred.png'.format(out_fig_dpath, first, ex, tag))
                    plt.close()

                cat_feature = pca_feature[:, :pca_idx[0]+1]
                tsne = TSNE(n_components=2, random_state=SEED)
                cat_tsne_feature = tsne.fit_transform(cat_feature)
                fig = plt.figure(figsize=(16*2, 9))
                for ia, (Y, co) in enumerate(zip([Y1, Y2], [colors1, colors2])):
                    ax = fig.add_subplot(1,2,ia+1)
                    for iy, k in enumerate(np.unique(Y)):
                        ax.scatter(cat_tsne_feature[Y==k, 0], cat_tsne_feature[Y==k, 1], alpha=0.5, c=co[iy], marker='.')
                        ax.set_aspect('equal', 'datalim')
                fig.savefig('{}/{}_decomp_tsne_{}.png'.format(out_fig_dpath, first, ex))
                plt.close(fig)

                fig = plt.figure(figsize=(16*2, 9))
                for ia, (Y, co) in enumerate(zip([pred_Y1, pred_Y2], [colors1, colors2])):
                    ax = fig.add_subplot(1,2,ia+1)
                    for iy, k in enumerate(np.unique(Y)):
                        ax.scatter(cat_tsne_feature[Y==k, 0], cat_tsne_feature[Y==k, 1], alpha=0.5, c=co[iy], marker='.')
                        ax.set_aspect('equal', 'datalim')
                fig.savefig('{}/{}_decomp_tsne_Classifier{}.png'.format(out_fig_dpath, first, ex))
                plt.close(fig)


def test_SemiSelf():
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    args = argparses()
    out_source_dpath += 'SemiSelf'
    out_source_dpath = out_source_dpath + '/' + ex
    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    else:
        src, targets1, targets2, idxs, seq_ids, img_paths = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
        # src, targets1, targets2, idxs, seq_ids, img_paths = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)

    ratio = [0.7, 0.2, 0.1]
    n_sample = len(src)
    train_size = int(n_sample*ratio[0])
    val_size = int(n_sample*ratio[1])
    test_size = n_sample - (train_size + val_size)
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))
    test_indices = list(range(train_size+val_size, n_sample))
    true_targets1 = copy.deepcopy(targets1)
    true_targets2 = copy.deepcopy(targets2)
    test_seq_id = [seq_ids[i] for i in test_indices]
    if args.semi:
        fix_indices = torch.cat([torch.ones(train_size), torch.zeros(val_size+test_size)])
        discarded_targets2 = random_label_replace(targets2, ratio=args.ratio, value=-2, fix_indices=fix_indices, seed=2)
        pseudo_label = torch.randint(low=0, high=torch.unique(true_targets2).size(0), size=targets2.size())
        pseudo_label[discarded_targets2 != -2] = -2
        if args.dual:
            discarded_targets1 = random_label_replace(targets1, ratio=args.ratio, value=-2, fix_indices=fix_indices, seed=2)
            pseudo_label1 = torch.randint(low=0, high=torch.unique(true_targets1).size(0), size=targets1.size())
            pseudo_label1[discarded_targets1 != -2] = -2
    # iters = 0
    # for indices, k in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
    #     for loc, mayo, sid, img, p in zip(targets1[indices], targets2[indices], [seq_ids[i] for i in indices], src[indices], [img_paths[ind] for ind in indices]):
    #         img = img.numpy()
    #         img = np.transpose(img, (1, 2, 0))
    #         img = img[:,:,::-1]

    #         opath = os.path.join('./reports', 'SeqSamples_{}'.format(k), p.rsplit('/', 1)[0])
    #         if sid == '2':
    #             print(opath)
    #         if os.path.exists(opath) is False:
    #             print(opath)
    #             os.makedirs(opath)
    #         cv2.imwrite('./reports/SeqSamples_{}/{}'.format(k, p), img*255)

    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)
    
    if args.retrain:
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_test_dpath = '{}/re_test_{}'.format(out_source_dpath, args.param)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_test_dpath = '{}/test_{}'.format(out_source_dpath, args.param)
    # clean_directory(out_test_dpath)
    
    model = SemiSelfClassifier(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)
    # model = SemiSelfClassifier(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)
    # model = TDAE_VAE_fullsuper_disentangle(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_set = torch.utils.data.dataset.Subset(data_pairs, test_indices)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    if args.param == 'best':
        model.load_state_dict(torch.load('{}/SemiSelf_bestparam.json'.format(out_param_dpath)))
    else:
        model.load_state_dict(torch.load('{}/SemiSelf_param_lastparam.json'.format(out_param_dpath)))
    model = model.to(device)
    for ntag in ['all', 'semi']:
        X1_dict = {}
        X2_dict = {}
        Y1_dict = {}
        Y2_dict = {}
        pY1_dict = {}
        pY2_dict = {}
        if ntag == 'all':
            model.load_state_dict(torch.load('{}/SemiSelf_bestparam.json'.format(out_param_dpath)))
        elif ntag == 'semi':
            model.load_state_dict(torch.load('{}/SemiSelf_semi_bestparam.json'.format(out_param_dpath)))
        model = model.to(device)
        conf_mat_dict1 = {}
        conf_mat_dict2 = {}
        for gl, pl in itertools.product(range(torch.unique(targets1).size(0)), range(torch.unique(targets1).size(0))):
            conf_mat_dict1['{}to{}'.format(gl, pl)] = []
        for gl, pl in itertools.product(range(torch.unique(targets2).size(0)), range(torch.unique(targets2).size(0))):
            conf_mat_dict2['{}to{}'.format(gl, pl)] = []
        with torch.no_grad():
            for k, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
                model.eval()
                eval_dict = {}
                eval_dict['path'] = []
                eval_dict['loc_label'] = []
                eval_dict['mayo_label'] = []
                eval_dict['mayo_discard_label'] = []
                eval_dict['loc_pred'] = []
                eval_dict['mayo_pred'] = []
                eval_dict['seq'] = []
                if args.dual:
                    eval_dict['part_discard_label'] = []
                    
                X1, X2, Y1, Y2 = [], [], [], []
                pY1, pY2 = [], []
                Target = []
                H0 = []
                H1 = []
                for n_iter, (idx, _, _, target, sub_target) in enumerate(loader):
                    cat_paths = [img_paths[iidx] for iidx in idx]
                    (mu1, mu2) = model.hidden_output(src[idx].to(device))
                    mu1 = mu1.detach().to('cpu').numpy()
                    mu2 = mu2.detach().to('cpu').numpy()
                    pred_y1, pred_y2 = model.predict_label(src[idx].to(device))
                    X1.extend(mu1)
                    X2.extend(mu2)
                    target_np = target.detach().to('cpu').numpy()
                    sub_target_np = sub_target.detach().to('cpu').numpy()
                    dis_sub_target_np = discarded_targets2[idx].detach().to('cpu').numpy()
                    pred_np = pred_y1.detach().to('cpu').numpy()
                    sub_pred_np = pred_y2.detach().to('cpu').numpy()
                    eval_dict['path'].extend(cat_paths)
                    eval_dict['loc_label'].extend(target_np)
                    eval_dict['mayo_label'].extend(sub_target_np)
                    eval_dict['mayo_discard_label'].extend(dis_sub_target_np)
                    if args.dual:
                        dis_main_target_np = discarded_targets1[idx].detach().to('cpu').numpy()
                        eval_dict['part_discard_label'].extend(dis_main_target_np)
                    eval_dict['loc_pred'].extend(pred_np)
                    eval_dict['mayo_pred'].extend(sub_pred_np)
                    eval_dict['seq'].extend([c.split('/')[1] for c in cat_paths])
                    Y1.extend(target.detach().to('cpu').numpy())
                    Y2.extend(sub_target.detach().to('cpu').numpy())
                    pY1.extend(pred_y1.detach().to('cpu').numpy())
                    pY2.extend(pred_y2.detach().to('cpu').numpy())

                df = pd.DataFrame.from_dict(eval_dict)
                df.to_csv('{}/eachPredicted_{}.csv'.format(out_test_dpath, k))
                X1_dict[k] = np.asarray(X1)
                X2_dict[k] = np.asarray(X2)
                Y1_dict[k] = np.asarray(Y1)
                Y2_dict[k] = np.asarray(Y2)
                pY1_dict[k] = np.asarray(pY1)
                pY2_dict[k] = np.asarray(pY2)
                conf_mat_label1 = confusion_matrix(Y1, pY1)
                conf_mat_label2 = confusion_matrix(Y2, pY2)
                for gl, pl in itertools.product(range(torch.unique(targets1).size(0)), range(torch.unique(targets1).size(0))):
                    conf_mat_dict1['{}to{}'.format(gl, pl)].append(conf_mat_label1[gl, pl])
                for gl, pl in itertools.product(range(torch.unique(targets2).size(0)), range(torch.unique(targets2).size(0))):
                    conf_mat_dict2['{}to{}'.format(gl, pl)].append(conf_mat_label2[gl, pl])

            df = pd.DataFrame.from_dict(conf_mat_dict1)
            df.to_csv('{}/conf_mat_loc.csv'.format(out_test_dpath))
            df = pd.DataFrame.from_dict(conf_mat_dict2)
            df.to_csv('{}/conf_mat_mayo.csv'.format(out_test_dpath))

        # return
        score_dict_withPseudo = {}
        for indices, k in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
            pred_y1 = torch.from_numpy(pY1_dict[k])
            pred_y2 = torch.from_numpy(pY2_dict[k])
            target1_acc = true_positive_multiclass(pred_y1, targets1[indices])
            cat_target2 = discarded_targets2[indices]
            cat_full_target2 = true_targets2[indices]
            remain_target2_acc = true_positive_multiclass(pred_y2[cat_target2!=-2], cat_target2[cat_target2!=-2])
            discarded_target2_acc = true_positive_multiclass(pred_y2[cat_target2==-2], cat_full_target2[cat_target2==-2])
            if args.dual:
                cat_full_target1 = true_targets1[indices]
                cat_target1 = discarded_targets1[indices]
                discarded_target1_acc = true_positive_multiclass(pred_y1[cat_target1==-2], cat_full_target1[cat_target1==-2])
            target2_acc = true_positive_multiclass(pred_y2, targets2[indices])
            if k == 'train':
                score_dict_withPseudo['main'] = [target1_acc/len(indices)]
                score_dict_withPseudo['sub'] = [target2_acc/len(indices)]
                score_dict_withPseudo['remain_sub'] = [remain_target2_acc/len(cat_target2[cat_target2!=-2])]
                score_dict_withPseudo['pseudo_sub'] = [discarded_target2_acc/len(cat_target2[cat_target2==-2])]
                if args.dual:
                    score_dict_withPseudo['pseudo_main'] = [discarded_target1_acc/len(cat_target1[cat_target1==-2])]
            else:
                score_dict_withPseudo['main'].append(target1_acc/len(indices))
                score_dict_withPseudo['sub'].append(target2_acc/len(indices))
                score_dict_withPseudo['remain_sub'].append(remain_target2_acc/len(cat_target2[cat_target2!=-2]))
                score_dict_withPseudo['pseudo_sub'].append(discarded_target2_acc/len(cat_target2[cat_target2==-2]))
                if args.dual:
                    score_dict_withPseudo['pseudo_main'].append(discarded_target1_acc/len(cat_target1[cat_target1==-2]))
        print(score_dict_withPseudo)
        df = pd.DataFrame.from_dict(score_dict_withPseudo)
        df.to_csv('{}/ResultNN_withPseudo_{}.csv'.format(out_test_dpath, ntag))

        tag = ['main2main', 'main2sub', 'sub2main', 'sub2sub']
        score_dict = {}
        score_nn = {}
        for k in ['train', 'val', 'test']:
            pred1 = np.sum(pY1_dict[k] == Y1_dict[k])
            pred2 = np.sum(pY2_dict[k] == Y2_dict[k])
            if k == 'train':
                score_nn['main'] = [pred1 / len(pY1_dict[k])]
                score_nn['sub'] = [pred2 / len(pY2_dict[k])]
            else:
                score_nn['main'].append(pred1 / len(pY1_dict[k]))
                score_nn['sub'].append(pred2 / len(pY2_dict[k]))
        df = pd.DataFrame.from_dict(score_nn)
        df.to_csv('{}/ResultNN_{}.csv'.format(out_test_dpath, ntag))

        for itag, (X_dict, Y_dict) in enumerate(itertools.product([X1_dict, X2_dict], [Y1_dict, Y2_dict])):
            for k in ['train', 'val', 'test']:
                if k == 'train':
                    logreg = LogisticRegression(solver="sag", max_iter=200)
                    logreg.fit(X_dict[k], Y_dict[k])
                    score_reg = logreg.score(X_dict[k], Y_dict[k])
                    score_dict[tag[itag]] = [score_reg]
                else:
                    score_reg = logreg.score(X_dict[k], Y_dict[k])
                    score_dict[tag[itag]].append(score_reg)
                # if tag[itag] == 'sub2main':
                #     print(logreg.predict_proba(X_dict[k]))
                # l = logreg.predict_proba(X_dict[k])
                # p = np.argmax(l, axis=1)
                
        df = pd.DataFrame.from_dict(score_dict)
        df.to_csv('{}/LinearReg.csv'.format(out_test_dpath, ntag))


def confirm_seq():
    torch.manual_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    args = argparses()

    out_source_dpath = out_source_dpath + '/' + ex
    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path)
    else:
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path)
    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)
    if args.retrain:
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_fig_dpath = '{}/re_conf'.format(out_source_dpath)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_fig_dpath = '{}/conf'.format(out_source_dpath)
    clean_directory(out_fig_dpath)

    if args.full:
        model = TDAE_VAE_fullsuper_disentangle(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)
    else:
        model = TDAE_VAE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)

    ratio = [0.7, 0.2, 0.1]
    n_sample = len(data_pairs)
    train_size = int(n_sample*ratio[0])
    
    # train_set, val_set = torch.utils.data.random_split(data_pairs, [train_size, val_size])
    train_indices = list(range(0, train_size))
    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    
    markers = ['.', 'x']
    colors1 = ['blue', 'orange', 'purple']
    colors2 = ['r', 'g']
    if args.rev:
        buff = colors1
        colors1 = colors2
        colors2 = buff
        
    with torch.no_grad():
        start = 10
        end_e = 100
        ds = 10
        score_dict = {}
        score_dict['main2main'] = []
        score_dict['main2sub'] = []
        score_dict['sub2main'] = []
        score_dict['sub2sub'] = []
        for iters in range(start, end_e+ds, ds):
            model.load_state_dict(torch.load('{}/TDAE_param_e{:04}.json'.format(out_param_dpath, iters)))
            model.to(device)
            for first, loader in zip(['train'], [train_loader]):
                model.eval()
                X1, X2, Y1, Y2 = [], [], [], []
                for n_iter, (idx, _, _, target, sub_target) in enumerate(loader):
                    (mu1, mu2) = model.hidden_output(src[idx].to(device))
                    mu1 = mu1.detach().to('cpu').numpy()
                    mu2 = mu2.detach().to('cpu').numpy()
                    X1.extend(mu1)
                    X2.extend(mu2)
                    Y1.extend(target.detach().to('cpu').numpy())
                    Y2.extend(sub_target.detach().to('cpu').numpy())
            
                X1 = np.asarray(X1)
                X2 = np.asarray(X2)
                Y1 = np.asarray(Y1)
                Y2 = np.asarray(Y2)
                logreg_main2main = LogisticRegression(penalty='l2', solver="sag")
                logreg_main2main.fit(X1, Y1)
                logreg_main2sub = LogisticRegression(penalty='l2', solver="sag")
                logreg_main2sub.fit(X1, Y2)
                logreg_sub2main = LogisticRegression(penalty='l2', solver="sag")
                logreg_sub2main.fit(X2, Y1)
                logreg_sub2sub = LogisticRegression(penalty='l2', solver="sag")
                logreg_sub2sub.fit(X2, Y2)
                for reg, X, Y, k in zip([logreg_main2main, logreg_main2sub, logreg_sub2main, logreg_sub2sub], [X1, X1, X2, X2], [Y1, Y2, Y1, Y2], ['main2main', 'main2sub', 'sub2main', 'sub2sub']):
                    score_dict[k].append(reg.score(X, Y))
                continue
                for X, ex in zip([X1, X2], ['main', 'sub']):
                    rn.seed(SEED)
                    np.random.seed(SEED)
                    tsne = TSNE(n_components=2, random_state=SEED)
                    Xt = tsne.fit_transform(X)
                    fig = plt.figure(figsize=(16*2, 9))
                    for ia, (Y, co) in enumerate(zip([Y1, Y2], [colors1, colors2])): 
                        ax = fig.add_subplot(1,2,ia+1)
                        for iy, k in enumerate(np.unique(Y)):
                            ax.scatter(x=Xt[Y==k,0], y=Xt[Y==k,1], marker='.', alpha=0.5, c=co[iy])
                        ax.set_aspect('equal', 'datalim')
                    fig.savefig('{}/{}_hidden_features_{}_iter{:04}.png'.format(out_fig_dpath, first, ex, iters))
                    plt.close(fig)

                    fig = plt.figure(figsize=(6*2, 6))
                    ax = fig.add_subplot(1,2,1)
                    ax.hist(np.mean(X, axis=0))
                    ax = fig.add_subplot(1,2,2)
                    ax.hist(np.std(X, axis=0))
                    fig.savefig('{}/{}_dist_{}_iter{:04}.png'.format(out_fig_dpath, first, ex, iters))
                    plt.close(fig)

                    Xn = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 
                    pca = PCA()
                    pca.fit(Xn)
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(1,1,1)
                    ratio = np.append([0], np.cumsum(pca.explained_variance_ratio_))
                    ax.plot(ratio)
                    fig.savefig('{}/{}_ratio_{}_iter{:04}.png'.format(out_fig_dpath, first, ex, iters))
                    plt.close(fig)
                    
                    pca_feature = pca.transform(Xn)
                    fig = plt.figure(figsize=(16*2, 9))
                    for ia, (Y, co) in enumerate(zip([Y1, Y2], [colors1, colors2])):
                        ax = fig.add_subplot(1,2,ia+1)
                        for iy, k in enumerate(np.unique(Y)):
                            ax.scatter(pca_feature[Y==k, 0], pca_feature[Y==k, 1], alpha=0.5, marker='.', c=co[iy])
                            ax.set_aspect('equal', 'datalim')
                    fig.savefig('{}/{}_pca_{}_iter{:04}.png'.format(out_fig_dpath, first, ex, iters))
                    plt.close(fig)
                    cs = ['dim{}'.format(d+1) for d in range(5)]
                    df = pd.DataFrame(pca_feature[:, :5], columns=cs)
                    for tag, Y, co in zip(['Y1', 'Y2'], [Y1, Y2], [colors1, colors2]):
                        tar = ['Class{}'.format(y) for y in Y]
                        df['target'] = tar
                        sns.pairplot(df, hue='target', diag_kind='hist', vars=cs, palette={'Class0':co[0], 'Class1':co[1]}).savefig('{}/{}_PairPlot_{}_{}_iter{:04}.png'.format(out_fig_dpath, first, ex, tag, iters))
                        plt.close()
                    pca_idx = np.where(ratio >= 0.99)[0]
                    cat_feature = pca_feature[:, :pca_idx[0]+1]
                    tsne = TSNE(n_components=2, random_state=SEED)
                    cat_tsne_feature = tsne.fit_transform(cat_feature)
                    fig = plt.figure(figsize=(16*2, 9))
                    for ia, (Y, co) in enumerate(zip([Y1, Y2], [colors1, colors2])):
                        ax = fig.add_subplot(1,2,ia+1)
                        for iy, k in enumerate(np.unique(Y)):
                            ax.scatter(cat_tsne_feature[Y==k, 0], cat_tsne_feature[Y==k, 1], alpha=0.5, marker='.', c=co[iy])
                            ax.set_aspect('equal', 'datalim')
                    fig.savefig('{}/{}_decomp_tsne_{}_iter{:04}.png'.format(out_fig_dpath, first, ex, iters))
                    plt.close(fig)

    df = pd.DataFrame.from_dict(score_dict)
    df.to_csv('{}/LinearReg.csv'.format(out_fig_dpath))


def main():
    args = argparses()
    print(args)

    if args.train:
        train_fixmatch()
    if args.val:
        val_fixmatch()
    if args.test:
        test_fixmatch()
    
if __name__ == '__main__':
    # with SetIO('./out.log'):
    main()
