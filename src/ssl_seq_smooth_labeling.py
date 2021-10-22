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
from multiprocessing import Pool, cpu_count
from scipy.stats import entropy
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.semi_supervised import LabelPropagation
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

from losses import TripletLoss, negative_entropy_loss, Fourier_mse, loss_vae, MetricLoss, cross_entropy_with_soft_label
from metrics import true_positive_multiclass, true_positive, true_negative, label2onehot
from __init__ import clean_directory, SetIO
from data_handling import get_triplet_flatted_data_with_idx, random_label_replace, get_sequence_splitted_data_with_const
from archs import SemiSelfClassifier
from label_propagation import HMN
from constrained_pseudo_labeling import get_pseudo_labeling_with_mip, get_pseudo_soft_labeling_with_qp
from TwoSampler import *

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
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--unlabeled_batch', type=int, default=32)
    parser.add_argument('--ndeconv', type=int, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--dm', type=int, default=0)
    parser.add_argument('--T1', type=int, default=100)
    parser.add_argument('--T2', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--data', type=str)
    parser.add_argument('--param', type=str, default='best')
    parser.add_argument('--ex', type=str, default=None)
    parser.add_argument('--classifier', type=float, default=1e+0)
    parser.add_argument('--c1', type=float, default=1e+0)
    parser.add_argument('--c2', type=float, default=1e+0)
    parser.add_argument('--adv', type=float, default=1e-1)
    parser.add_argument('--ratio', type=float, default=0.0)
    parser.add_argument('--smooth', type=float, default=0.0)
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--adapt_weight', action='store_true')
    parser.add_argument('--adapt_alpha', action='store_true')
    parser.add_argument('--semi', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--dual', action='store_true')
    parser.add_argument('--labeldecomp', action='store_false')
    parser.add_argument('--soft', action='store_true')
    parser.add_argument('--seq_base', action='store_true')
    parser.add_argument('--temp', action='store_true')
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
    out_source_dpath = './reports/SSL_ConstLabeling' 
    out_source_dpath = os.path.join(out_source_dpath, args.data)
    
    if 'toy' in args.data:
        img_w, img_h = 256, 256
        data_path = './data/toy_data.hdf5'
    elif 'huge' in args.data:
        img_w, img_h = 256, 256
        data_path = './data/huge_toy_data.hdf5'
    elif 'Huge' in args.data:
        img_w, img_h = 256, 256
        data_path = './data/Huge_toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        data_path = './data/colon_renew.hdf5'
    
    if not(args.ex is None):
        ex = args.ex
    else:
        if args.semi:
            ex = 'SSL'
            if args.smooth > 0:
                if args.soft:
                    ex = ex + 'soft_prop_smooth{}'.format(args.smooth)
                else:
                    ex = ex + 'prop_smooth{}'.format(args.smooth)
            else:
                if args.soft:
                    ex = ex + 'soft_nonConst'
                else:
                    ex = ex + 'nonConst'
        else:
            ex = 'FSL'
            
        ex = ex + '_r{}'.format(args.ratio*10)
        
        if args.adapt_alpha:
            ex = ex + '_T1{}T2{}_alpha{}'.format(args.T1, args.T2, args.alpha)
            
        if args.adv > 0:
            ex = ex + '_withDRL_adv{}'.format(args.adv)
        else:
            ex = ex + '_woDRL'
        
        if args.c1 <= 0:
            ex = ex + '_wMayo'
        if args.c2 <= 0:
            ex = ex + '_wPart'
            
    return img_w, img_h, out_source_dpath, data_path, ex


def pseudo_label_update(model, inputs, idx=1):
    with torch.no_grad():
        model.eval()
        pred_pseudo_labels = model.predict_label(inputs)
    return pred_pseudo_labels[idx].to('cpu')


def get_sequence_based_constraints(sample_num):
    inds = list(range(sample_num))
    const = []
    for ind1, ind2 in zip(inds[:-1], inds[1:]):
        if ind1 + 1 == ind2:
            const.append([ind1, ind2])

    return const

def get_discarded_label_for_ssl(targets1, targets2, id_dict):
    
    args = argparses()
    seq_ids = list(id_dict.keys())
    rn.Random(SEED).shuffle(seq_ids)

    ratio = [0.7, 0.2, 0.1]
    seq_num = len(seq_ids)
    train_seq_num = int(seq_num * ratio[0])
    val_seq_num = int(seq_num * ratio[1])
    test_seq_num = seq_num - (train_seq_num + val_seq_num)

    seq_id_dict = {'train': seq_ids[:train_seq_num], 'val': seq_ids[train_seq_num:train_seq_num+val_seq_num], 'test': seq_ids[train_seq_num+val_seq_num:]}
    sample_ids = {}
    for tag, seq_ids in seq_id_dict.items():
        sample_ids[tag] = []
        for cat_seq_id in seq_ids:
            sample_ids[tag].extend(id_dict[cat_seq_id])

    (train_size, val_size, test_size) = (len(sample_ids[k]) for k in ['train', 'val', 'test'])
    n_sample = train_size + val_size + test_size
    
    true_targets1 = copy.deepcopy(targets1)
    true_targets2 = copy.deepcopy(targets2)

    if args.ratio > 0.0:
        if args.seq_base:
            shuffled_ids = list(range(len(seq_id_dict['train'])))
            rn.Random(SEED).shuffle(shuffled_ids)
            discard_seq_ids = [seq_id_dict['train'][s_ids] for s_ids in shuffled_ids[:int(len(shuffled_ids)*args.ratio)]]
            discard_ids = []
            for cat_seq_id in discard_seq_ids:
                discard_ids.extend(id_dict[cat_seq_id])
            targets1[discard_ids] = -2
            targets2[discard_ids] = -2
        else:
            fix_indices = torch.zeros(n_sample)
            fix_indices[sample_ids['train']] = 1
            targets1 = random_label_replace(targets1, ratio=args.ratio, value=-2, fix_indices=fix_indices, seed=2)
            targets2 = random_label_replace(targets2, ratio=args.ratio, value=-2, fix_indices=fix_indices, seed=2)
    
    if args.semi:
        if args.soft:
            pseudo_label1 = torch.zeros((targets1.size(0), torch.unique(true_targets1).size(0)))
            pseudo_label2 = torch.zeros((targets2.size(0), torch.unique(true_targets2).size(0)))
        else:
            pseudo_label1 = torch.randint(low=0, high=torch.unique(true_targets1).size(0), size=targets1.size())
            pseudo_label1[targets1 != -2] = -2

            pseudo_label2 = torch.randint(low=0, high=torch.unique(true_targets2).size(0), size=targets2.size())
            pseudo_label2[targets2 != -2] = -2
    else:
        if args.soft:
            pseudo_label1 = torch.zeros((targets1.size(0), torch.unique(true_targets1).size(0)))
            pseudo_label2 = torch.zeros((targets2.size(0), torch.unique(true_targets2).size(0)))
        else:
            pseudo_label1 = torch.randint(low=0, high=torch.unique(true_targets1).size(0), size=targets1.size())
            pseudo_label1[targets1 != -2] = -2
            pseudo_label2 = torch.randint(low=0, high=torch.unique(true_targets2).size(0), size=targets2.size())
            pseudo_label2[targets2 != -2] = -2
            
    return (true_targets1, targets1, pseudo_label1), (true_targets2, targets2, pseudo_label2), seq_id_dict, sample_ids

def train_model():
    torch.manual_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    out_source_dpath = os.path.join(out_source_dpath, ex)
    
    src, targets1, targets2, id_dict, img_paths = get_sequence_splitted_data_with_const(data_path, label_decomp=args.labeldecomp)
    
    # Discard Label based on the ratio between labeled unlabeled ratio args.ratio 
    (true_targets1, targets1, pseudo_label1), (true_targets2, targets2, pseudo_label2), seq_ids, sample_ids = get_discarded_label_for_ssl(targets1, targets2, id_dict)

    model = SemiSelfClassifier(n_classes=[torch.unique(true_targets1).size(0), torch.unique(true_targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)

    if args.retrain:
        model.load_state_dict(torch.load('{}/param/model_bestparam.json'.format(out_source_dpath)))
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
    ngpus = 1

    if args.multi:
        ngpus = torch.cuda.device_count()    
        model = nn.DataParallel(model, device_ids=[g for g in range(ngpus)])
    model = model.to(device)
    params = list(model.parameters())
    
    data_pairs = torch.utils.data.TensorDataset(src, targets1, targets2, pseudo_label1, pseudo_label2)
    if args.semi is False:
        sample_ids['train'] = [i for i in sample_ids['train'] if targets1[i] != -2]
        
    train_set = torch.utils.data.dataset.Subset(data_pairs, sample_ids['train'])
    val_set = torch.utils.data.dataset.Subset(data_pairs, sample_ids['val'])
    test_set = torch.utils.data.dataset.Subset(data_pairs, sample_ids['test'])


    train_size1 = train_set[:][1][train_set[:][1] != -2].size(0)
    train_size2 = train_set[:][2][train_set[:][2] != -2].size(0)

    labeled_indexes = torch.cat((torch.nonzero(targets1[sample_ids['train']]!=-2), torch.nonzero(targets2[sample_ids['train']]!=-2)))
    labeled_indexes = torch.unique(labeled_indexes).detach().numpy()

    unlabeled_indexes = torch.cat((torch.nonzero(targets1[sample_ids['train']]==-2), torch.nonzero(targets2[sample_ids['train']]==-2)))
    unlabeled_indexes = torch.unique(unlabeled_indexes).detach().numpy()
    
    if args.semi:
        batch_sampler = TwoStreamBatchSampler(labeled_indexes, unlabeled_indexes, args.batch*ngpus, args.unlabeled_batch*ngpus)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler)
        train_loader_compute_accuracy = DataLoader(train_set, batch_size=args.batch*ngpus, shuffle=False)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch*ngpus)
        train_loader_compute_accuracy = DataLoader(train_set, batch_size=args.batch*ngpus, shuffle=False)

    val_loader = DataLoader(val_set, batch_size=args.batch*ngpus, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch*ngpus, shuffle=False)

    # criterion_adv = nn.NLLLoss()
    criterion_classifier = nn.CrossEntropyLoss(ignore_index=-2)
    # kldiv = nn.KLDivLoss(reduction='sum')
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    optimizer = optim.Adam(params)
    # optimizer = optim.SGD(params, lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    n_epochs = args.epoch
    best_epoch, best_loss = 0, np.inf
    best_semi_epoch, best_semi_loss = 0, np.inf

    l_adv, l_c = args.adv, args.classifier
    train_keys = ['loss/train_all', 'loss/train_classifier_main', 'loss/train_classifier_sub', 'loss/train_adv_main', 'loss/train_adv_sub', 'loss/train_no_grad_main', 'loss/train_no_grad_sub', 'loss/train_pseudo_adv', 'loss/train_pseudo_classifier']
    val_keys = ['loss/val_all', 'loss/val_classifier_main', 'loss/val_classifier_sub', 'loss/val_adv_main', 'loss/val_adv_sub', 'loss/val_no_grad_main', 'loss/val_no_grad_sub']

    weight_keys = ['classifier_main', 'classifier_sub', 'adv_main', 'adv_sub']
    w_args = [args.classifier*args.c1, args.classifier*args.c2, args.adv, args.adv]
        
    weight_dict = {}
    current_loss_dict = {}
    prev_loss_dict = {}
    weight_change = {}
    for k, w in zip(weight_keys, w_args):
        weight_dict[k] = w
        weight_change[k] = [w]
        current_loss_dict[k] = 0

    T1, T2 = args.T1, args.T2
    semi_alpha = 0.1
    mytrans = TensorTransforms(methods=[vflip, hflip, rotate], p=0.3)
    for epoch in range(n_epochs):
        time_start_epoch = time.perf_counter()
        Acc, sub_Acc  = 0, 0
        loss_dict = {}
        for k in train_keys:
            loss_dict[k] = []
        if args.adapt_alpha:
            if epoch + 1 > T1:
                if epoch + 1 < T2:
                    semi_alpha = (epoch + 1 - T1)/(T2 - T1) * args.alpha
                else:
                    semi_alpha = args.alpha

        for iters, (x, target, sub_target, p_target, p_sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            x = mytrans(x)
            preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, z1, z2 = model.forward(x.to(device))
                
            loss_classifier_main = weight_dict['classifier_main'] * criterion_classifier(preds.to(device), target.to(device))
            loss_classifier_sub = weight_dict['classifier_sub'] * criterion_classifier(sub_preds.to(device), sub_target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            loss_classifier_sub.backward(retain_graph=True)
            if (epoch + 1 > T1) and args.semi:
                if args.soft:
                    loss_pseudo_classifier1 = semi_alpha * weight_dict['classifier_main'] * cross_entropy_with_soft_label(preds[target==-2], p_target[target==-2].to(device))
                    loss_pseudo_classifier2 = semi_alpha * weight_dict['classifier_sub'] * cross_entropy_with_soft_label(sub_preds[sub_target==-2], p_sub_target[sub_target==-2].to(device))
                    # loss_pseudo_classifier1 = semi_alpha * weight_dict['classifier_main'] * kldiv(logsoftmax(preds[target==-2]), p_target[target==-2].to(device))
                    # loss_pseudo_classifier2 = semi_alpha * weight_dict['classifier_sub'] * kldiv(logsoftmax(sub_preds[sub_target==-2]), p_sub_target[sub_target==-2].to(device))
                    loss_pseudo_classifier = loss_pseudo_classifier1 + loss_pseudo_classifier2
                    loss_pseudo_classifier.backward(retain_graph=True)
                else:
                    loss_pseudo_classifier1 = semi_alpha * weight_dict['classifier_main'] * criterion_classifier(preds.to(device), p_target.to(device))
                    loss_pseudo_classifier2 = semi_alpha * weight_dict['classifier_sub'] * criterion_classifier(sub_preds.to(device), p_sub_target.to(device))
                    loss_pseudo_classifier = loss_pseudo_classifier1 + loss_pseudo_classifier2
                    loss_pseudo_classifier.backward(retain_graph=True)
            else:
                loss_pseudo_classifier = torch.Tensor([0])

            if args.adv > 0:
                loss_adv_main = weight_dict['adv_main'] * negative_entropy_loss(preds_adv.to(device))
                loss_adv_sub = weight_dict['adv_sub'] * negative_entropy_loss(sub_preds_adv.to(device))
                loss_adv_main.backward(retain_graph=True)
                loss_adv_sub.backward(retain_graph=True)

                model.disentangle_classifiers[0].zero_grad()
                model.disentangle_classifiers[1].zero_grad()
            
                loss_main_no_grad = weight_dict['adv_main'] * criterion_classifier(preds_adv_no_grad.to(device), sub_target.to(device))
                loss_sub_no_grad = weight_dict['adv_sub'] * criterion_classifier(sub_preds_adv_no_grad.to(device), target.to(device))
                loss_main_no_grad.backward(retain_graph=True)
                loss_sub_no_grad.backward(retain_graph=True)
                if args.semi and (epoch+1 > T1):
                    if args.soft:
                        loss_pseudo_label1 = semi_alpha * weight_dict['adv_main'] * cross_entropy_with_soft_label(preds_adv_no_grad[sub_target==-2], p_sub_target[sub_target==-2].to(device))
                        loss_pseudo_label2 = semi_alpha * weight_dict['adv_sub'] * cross_entropy_with_soft_label(sub_preds_adv_no_grad[target==-2], p_target[target==-2].to(device))
                        # loss_pseudo_label1 = semi_alpha * weight_dict['adv_main'] * kldiv(logsoftmax(preds_adv_no_grad[sub_target==-2]), p_sub_target[sub_target==-2].to(device))
                        # loss_pseudo_label2 = semi_alpha * weight_dict['adv_sub'] * kldiv(logsoftmax(sub_preds_adv_no_grad[target==-2]), p_target[target==-2].to(device))
                        loss_pseudo_label = loss_pseudo_label1 + loss_pseudo_label2
                        loss_pseudo_label.backward(retain_graph=True)
                    else:
                        loss_pseudo_label1 = semi_alpha * weight_dict['adv_main'] * criterion_classifier(preds_adv_no_grad.to(device), p_sub_target.to(device))
                        loss_pseudo_label2 = semi_alpha * weight_dict['adv_sub'] * criterion_classifier(sub_preds_adv_no_grad.to(device), p_target.to(device))
                        loss_pseudo_label = loss_pseudo_label1 + loss_pseudo_label2
                        loss_pseudo_label.backward(retain_graph=True)
                else:
                    loss_pseudo_label = torch.Tensor([0])
            
            else:
                loss_adv_main  = torch.Tensor([0])
                loss_adv_sub  = torch.Tensor([0])
                loss_main_no_grad  = torch.Tensor([0])
                loss_sub_no_grad  = torch.Tensor([0])
                loss_pseudo_label  = torch.Tensor([0])
            
            optimizer.step()
            loss = loss_classifier_main + loss_classifier_sub + loss_adv_main + loss_adv_sub + loss_main_no_grad + loss_sub_no_grad + loss_pseudo_label + loss_pseudo_classifier

            if args.adapt_weight:
                current_loss_dict['classifier_main'] += loss_classifier_main.item() + loss_classifier_sub.item()
                current_loss_dict['classifier_sub'] += loss_classifier_main.item() + loss_classifier_sub.item()
                current_loss_dict['adv_main'] += loss_adv_main.item() + loss_main_no_grad.item()
                current_loss_dict['adv_sub'] += loss_adv_sub.item() + loss_sub_no_grad.item()

            for k, val in zip(train_keys, [loss, loss_classifier_main, loss_classifier_sub, loss_adv_main, loss_adv_sub, loss_main_no_grad, loss_sub_no_grad, loss_pseudo_label, loss_pseudo_classifier]):
                loss_dict[k].append(val.item())
            
            y_true = target.to('cpu')
            sub_y_true = sub_target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = sub_preds.detach().to('cpu')
            Acc += true_positive_multiclass(preds[y_true!=-2], y_true[y_true!=-2])
            sub_Acc += true_positive_multiclass(sub_preds[sub_y_true!=-2], sub_y_true[sub_y_true!=-2])
                    
        for k in loss_dict.keys():
            loss_dict[k] = np.mean(loss_dict[k])

        print('-'*10)
        print('epoch: {:04} loss: {:.5} \nAcc: {:.5} sub Acc: {:.5}'.format(epoch+1, loss_dict['loss/train_all'], Acc/train_size1, sub_Acc/train_size2))

        summary = scalars2summary(writer=writer,
                            tags=list(loss_dict.keys()), 
                            vals=list(loss_dict.values()), epoch=epoch+1)
        
        train_acc1 = 0
        train_acc2 = 0
        with torch.no_grad():
            model.eval()
            task1_size, task2_size = 0, 0
            for iters, (x, target, sub_target, p_target, p_sub_target) in enumerate(train_loader_compute_accuracy):
                preds, sub_preds, _, _, _, _, _, _ = model.forward(x.to(device))
                y1_true = target.to('cpu')
                y2_true = sub_target.to('cpu')
                preds = preds.detach().to('cpu')
                sub_preds = sub_preds.detach().to('cpu')
                train_acc1 += true_positive_multiclass(preds[y1_true!=-2], y1_true[y1_true!=-2])
                train_acc2 += true_positive_multiclass(sub_preds[y2_true!=-2], y2_true[y2_true!=-2])

            train_acc1 = train_acc1 / train_size1
            train_acc2 = train_acc2 / train_size2
            summary = scalars2summary(writer=writer,
                            tags=['acc/train_main', 'acc/train_sub'], 
                            vals=[train_acc1, train_acc2], epoch=epoch+1)

        if (epoch + 1) % args.step == 0:
            print('Validation Step')
            with torch.no_grad():
                model.eval()
                val_loss_dict = {}
                for k in val_keys:
                    val_loss_dict[k] = []
                val_Acc, val_sub_Acc = 0, 0
                for v_i, (x, target, sub_target, p_target, p_sub_target) in enumerate(val_loader):
                    preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, z1, z2 = model.forward(x.to(device))
                    val_loss_classifier_main = weight_dict['classifier_main'] * criterion_classifier(preds.to(device), target.to(device))
                    val_loss_classifier_sub = weight_dict['classifier_sub'] * criterion_classifier(sub_preds.to(device), sub_target.to(device))
                    if args.adv > 0:
                        val_loss_adv_main = weight_dict['adv_main'] * negative_entropy_loss(preds_adv.to(device))
                        val_loss_adv_sub = weight_dict['adv_sub'] * negative_entropy_loss(sub_preds_adv.to(device))
                        val_loss_no_grad_main = weight_dict['adv_main'] * criterion_classifier(preds_adv_no_grad.to(device), sub_target.to(device))
                        val_loss_no_grad_sub = weight_dict['adv_sub'] * criterion_classifier(sub_preds_adv_no_grad.to(device), target.to(device))
                    else:
                        val_loss_adv_main  = torch.Tensor([0])
                        val_loss_adv_sub  = torch.Tensor([0])
                        val_loss_no_grad_main  = torch.Tensor([0])
                        val_loss_no_grad_sub  = torch.Tensor([0])

                    val_loss = val_loss_classifier_main + val_loss_classifier_sub + val_loss_adv_main + val_loss_adv_sub

                    for k, val in zip(val_keys, [val_loss, val_loss_classifier_main, val_loss_classifier_sub, val_loss_adv_main, val_loss_adv_sub, val_loss_no_grad_main, val_loss_no_grad_sub]):
                        val_loss_dict[k].append(val.item())

                    y_true = target.to('cpu')
                    sub_y_true = sub_target.to('cpu')
                    preds = preds.detach().to('cpu')
                    sub_preds = sub_preds.detach().to('cpu')
                    val_Acc += true_positive_multiclass(preds, y_true)
                    val_sub_Acc += true_positive_multiclass(sub_preds, sub_y_true)
                
                for k in val_loss_dict.keys():
                    val_loss_dict[k] = np.mean(val_loss_dict[k])

                summary = scalars2summary(writer=writer, 
                                        tags=list(val_loss_dict.keys()), 
                                        vals=list(val_loss_dict.values()), epoch=epoch+1)

                summary = scalars2summary(writer=writer, 
                                        tags=['acc/val_main', 'acc/val_sub'], 
                                        vals=[val_Acc / len(val_set), val_sub_Acc/len(val_set)], epoch=epoch+1)

                print('val loss: {:.5} Acc: {:.5} sub Acc: {:.5}'.format(val_loss_dict['loss/val_all'], val_Acc / len(val_set), val_sub_Acc/len(val_set)))
                
                # if args.multi:
                #     torch.save(model.module.state_dict(), '{}/model_param_e{:04}.json'.format(out_param_dpath, epoch+1))
                # else:
                #     torch.save(model.state_dict(), '{}/model_param_e{:04}.json'.format(out_param_dpath, epoch+1))

                if best_loss > val_loss_dict['loss/val_all']:
                    best_epoch = epoch + 1
                    best_loss = val_loss_dict['loss/val_all']
                    if args.multi:
                        torch.save(model.module.state_dict(), '{}/model_bestparam.json'.format(out_param_dpath))
                    else:
                        torch.save(model.state_dict(), '{}/model_bestparam.json'.format(out_param_dpath))
                
                if ((epoch + 1) > T1) and (best_semi_loss > val_loss_dict['loss/val_all']) and args.semi:
                    best_semi_epoch = epoch + 1
                    best_semi_loss = val_loss_dict['loss/val_all']
                    if args.multi:
                        torch.save(model.module.state_dict(), '{}/model_semi_bestparam.json'.format(out_param_dpath))
                    else:
                        torch.save(model.state_dict(), '{}/model_semi_bestparam.json'.format(out_param_dpath))
                
        # Pseudo Labeling Step
        if (epoch + 1) >= T1 and args.semi:
            print('Pseudo Labeling Step')
            time_start = time.perf_counter()
            if args.smooth <= 0:
                with torch.no_grad():
                    model.eval()
                    for train_seq_key in seq_ids['train']:
                        cat_id = id_dict[train_seq_key]
                        if args.soft:
                            pred1, pred2 = model.predict_proba(src[cat_id].to(device))
                            pseudo_label1[cat_id] = pred1.detach().to('cpu').float().clone()
                            pseudo_label2[cat_id] = pred2.detach().to('cpu').float().clone()
                        else:
                            pred1, pred2 = model.predict_label(src[cat_id].to(device))
                            pseudo_label1[cat_id] = pred1.detach().to('cpu').long().clone()
                            pseudo_label2[cat_id] = pred2.detach().to('cpu').long().clone()
                            pseudo_label1[targets1 != -2] = -2
                            pseudo_label2[targets2 != -2] = -2
            else:
                with torch.no_grad():
                    model.eval()
                    inputs1, inputs2 = [], []
                    cat_ids = []
                    for train_seq_key in seq_ids['train']:
                        cat_id = id_dict[train_seq_key]
                        cat_ids.append(cat_id)
                        const = get_sequence_based_constraints(len(cat_id))
                        pred1, pred2 = model.predict_proba(src[cat_id].to(device))

                        label1 = targets1[cat_id].clone().numpy()
                        pred1 = pred1.detach().to('cpu').numpy()
                        pred1[label1!=-2] = label2onehot(label1[label1!=-2], torch.unique(true_targets1).size(0))

                        label2 = targets2[cat_id].clone().numpy()
                        pred2 = pred2.detach().to('cpu').numpy()
                        pred2[label2!=-2] = label2onehot(label2[label2!=-2], torch.unique(true_targets2).size(0))

                        inputs1.append((pred1, const, args.smooth))
                        inputs2.append((pred2, const, args.smooth))

                n_core = cpu_count()
                print('Optimization now', '.'*10)
                print('Core num:', n_core)
                ps = []
                for c_coeff, inputs in zip([args.c1, args.c2], [inputs1, inputs2]):
                    # if c_coeff <= 0:
                    #     continue
                    if args.soft:
                        ps.append([get_pseudo_soft_labeling_with_qp(cat_inputs) for cat_inputs in inputs])
                        # ps.append(p.map(get_pseudo_soft_labeling_with_qp, inputs))
                    else:
                        p = Pool(n_core)
                        ps.append(p.map(get_pseudo_labeling_with_mip, inputs))
                        p.close()
                        p.terminate()
                        
                print('Optimization end', '.'*10)
                if args.soft:
                    for p1, p2, cat_id in zip(ps[0], ps[1], cat_ids):
                        pseudo_label1[cat_id] = torch.Tensor(p1).float()
                        pseudo_label2[cat_id] = torch.Tensor(p2).float()
                else:
                    for p1, p2, cat_id in zip(ps[0], ps[1], cat_ids):
                        pseudo_label1[cat_id] = torch.Tensor(p1).long()
                        pseudo_label1[targets1 != -2] = -2
                        pseudo_label2[cat_id] = torch.Tensor(p2).long()
                        pseudo_label2[targets2 != -2] = -2
    
            time_stop = time.perf_counter()
            print('Comp. Time: {:.3}'.format(time_stop-time_start))
        
        # Compute Acciracy of Unlabeled Sample
        if args.ratio > 0 and args.semi:
            pseudo_accuracy = 0
            pseudo_tpr_list = []
            for p_label, t_label, dis_label in zip([pseudo_label1, pseudo_label2], [true_targets1, true_targets2], [targets1, targets2]):
                cat_pseudo = p_label[sample_ids['train']]
                cat_dis_target = dis_label[sample_ids['train']]
                cat_target = t_label[sample_ids['train']]
                pseudo_size = len(cat_pseudo[cat_dis_target==-2])
                pseudo_tp_count = true_positive_multiclass(cat_pseudo[cat_dis_target==-2], cat_target[cat_dis_target==-2])
                pseudo_tpr_list.append(pseudo_tp_count/pseudo_size)
            
            print('PseudoMain ACC: {}'.format(pseudo_tpr_list[0]))
            print('PseudoSub ACC: {}'.format(pseudo_tpr_list[1]))
            summary = scalars2summary(writer=writer,
                    tags=['acc/PseudoAcc_main'], 
                    vals=[pseudo_tpr_list[0]], epoch=epoch+1)
            summary = scalars2summary(writer=writer,
                    tags=['acc/PseudoAcc_sub'], 
                    vals=[pseudo_tpr_list[1]], epoch=epoch+1)

        time_stop_epoch = time.perf_counter()
        print('Comp. Time: {:.3} per epoch'.format(time_stop_epoch-time_start_epoch))
        
        if (epoch + 1) == T1 and args.ratio > 0 and args.semi:
            model.load_state_dict(torch.load('{}/model_bestparam.json'.format(out_param_dpath)))

    if args.multi:
        torch.save(model.module.state_dict(), '{}/model_lastparam.json'.format(out_param_dpath))
    else:
        torch.save(model.state_dict(), '{}/model_lastparam.json'.format(out_param_dpath))
    
    args.best_epoch = best_epoch
    args.best_loss = best_loss
    args.best_semi_loss = best_semi_loss
    df = args2pandas(args)
    df.to_csv('{}/condition.csv'.format(out_condition_dpath))
    writer.close()
    if args.adapt_weight:
        df = pd.DataFrame.from_dict(weight_change)
        df.to_csv('{}/WeightChange.csv'.format(out_condition_dpath))


def val_model(zero_padding=False):
    torch.manual_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    out_source_dpath = os.path.join(out_source_dpath, ex)
    out_source_dpath = './reports/SSL_ConstLabeling' 
    out_source_dpath = os.path.join(out_source_dpath, args.data)
    out_source_dpath = os.path.join(out_source_dpath, 'Part/06r/softprop_smooth1.0_withSSL_woDRL_r6.0_T1100T2400_alpha1.0_woMayoLabel')
    src, targets1, targets2, id_dict, img_paths = get_sequence_splitted_data_with_const(data_path, label_decomp=args.labeldecomp)    
    (true_targets1, targets1, pseudo_label1), (true_targets2, targets2, pseudo_label2), seq_ids, sample_ids = get_discarded_label_for_ssl(targets1, targets2, id_dict)

    model = SemiSelfClassifier(n_classes=[torch.unique(true_targets1).size(0), torch.unique(true_targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)
    
    if args.retrain:
        model.load_state_dict(torch.load('{}/param/model_bestparam.json'.format(out_source_dpath)))
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_board_dpath = '{}/re_runs'.format(out_source_dpath)
        out_condition_dpath = '{}/re_condition'.format(out_source_dpath)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_board_dpath = '{}/runs'.format(out_source_dpath)
        out_condition_dpath = '{}/condition'.format(out_source_dpath)
        out_fig_dpath = '{}/figs'.format(out_source_dpath)
        
    ngpus = 1
    if args.multi:
        ngpus = torch.cuda.device_count()    
        model = nn.DataParallel(model, device_ids=[g for g in range(ngpus)])
    model = model.to(device)
    params = list(model.parameters())

    data_pairs = torch.utils.data.TensorDataset(src, targets1, targets2, pseudo_label1, pseudo_label2)
    train_set = torch.utils.data.dataset.Subset(data_pairs, sample_ids['train'])
    val_set = torch.utils.data.dataset.Subset(data_pairs, sample_ids['val'])
    test_set = torch.utils.data.dataset.Subset(data_pairs, sample_ids['test'])

    train_size1 = train_set[:][1][train_set[:][1] != -2].size(0)
    train_size2 = train_set[:][2][train_set[:][2] != -2].size(0)

    train_loader = DataLoader(train_set, batch_size=args.batch*ngpus, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch*ngpus, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch*ngpus, shuffle=False)

    if args.param == 'best':
        model.load_state_dict(torch.load('{}/model_bestparam.json'.format(out_param_dpath)))
    else:
        model.load_state_dict(torch.load('{}/model_lastparam.json'.format(out_param_dpath)))
    model = model.to(device)

    with torch.no_grad():
        for first, loader in zip(['train', 'val'], [train_loader, val_loader]):
            model.eval()
            if first == 'train':
                continue
            X1, X2, Y1, Y2 = [], [], [], []
            Prob1, Prob2 = [], []
            Pred1, Pred2 = [], []
            
            seq_key = seq_ids[first]
            cat_ids = []
            for cat_seq_key in seq_key:
                cat_id = id_dict[cat_seq_key]
                cat_ids.append(cat_id)
                prob1, prob2 = model.predict_proba(src[cat_id].to(device))
                pred1, pred2 = model.predict_label(src[cat_id].to(device))
                Y1.append(targets1[cat_id].clone().numpy())
                Y2.append(targets2[cat_id].clone().numpy())

                Prob1.append(prob1.detach().to('cpu').numpy())
                Prob2.append(prob2.detach().to('cpu').numpy())
                Pred1.append(pred1.detach().to('cpu').numpy())
                Pred2.append(pred2.detach().to('cpu').numpy())
            
            Prob1 = np.asarray(Prob1)
            Prob2 = np.asarray(Prob2)
            Pred1 = np.asarray(Pred1)
            Pred2 = np.asarray(Pred2)
            Y1 = np.asarray(Y1)
            Y2 = np.asarray(Y2)
            
            entro1 = []
            for prob in Prob1:
                e1 = np.array([entropy(p) for p in prob])
                entro1.append(np.mean(e1))
            entro1 = np.array(entro1)
            sort_ind = np.argsort(entro1)
            sort_entro1 = entro1[sort_ind]
            
            sort_y1 = [Y1[ind] for ind in sort_ind]
            sort_pred1 = [Pred1[ind] for ind in sort_ind]
            accs = []
            for p, y in zip(sort_pred1, sort_y1):
                acc = [1 if p==l else 0 for p, l in zip(p, y)]
                print(acc)
                accs.append(np.mean(acc))
            acu_acc = []
            score = 0
            for a in accs:
                score += a
                acu_acc.append(score/(len(acu_acc)+1))
            print(acu_acc)
            plt.plot(accs)
            plt.savefig('./acc_part_best_seq.png')

            return                


            
            
            for n_iter, (x, target, sub_target, p_target, p_sub_target) in enumerate(loader):
                (mu1, mu2) = model.hidden_output(x.to(device))
                prob1, prob2 = model.predict_proba(x.to(device))
                pred1, pred2 = model.predict_label(x.to(device))
                Prob1.extend(prob1.detach().to('cpu').numpy())
                Prob2.extend(prob2.detach().to('cpu').numpy())
                Pred1.extend(pred1.detach().to('cpu').numpy())
                Pred2.extend(pred2.detach().to('cpu').numpy())
                mu1 = mu1.detach().to('cpu').numpy()
                mu2 = mu2.detach().to('cpu').numpy()
                X1.extend(mu1)
                X2.extend(mu2)
                Y1.extend(target.detach().to('cpu').numpy())
                Y2.extend(sub_target.detach().to('cpu').numpy())
                continue

            Prob1 = np.asarray(Prob1)
            Prob2 = np.asarray(Prob2)
            Pred1 = np.asarray(Pred1)
            Pred2 = np.asarray(Pred2)
            Y1 = np.asarray(Y1)
            Y2 = np.asarray(Y2)
            
            entro1 = np.array([entropy(p) for p in Prob1])
            entro2 = np.array([entropy(p) for p in Prob2])
            sort_ind = np.argsort(entro2)
            sort_entro1 = entro2[sort_ind]
            print(sort_entro1)
            sort_y1 = Y2[sort_ind]
            sort_pred1 = Pred2[sort_ind]
            acc = [1 if p==l else 0 for p, l in zip(sort_pred1, sort_y1)]
            acu_acc = []
            score = 0
            for a in acc:
                score += a
                acu_acc.append(score/(len(acu_acc)+1))
            plt.plot(acu_acc)
            plt.savefig('./acu_part_mayo_last.png')
            return
            
            X1 = np.asarray(X1)
            X2 = np.asarray(X2)
            Y1 = np.asarray(Y1)
            Y2 = np.asarray(Y2) 
            markers = ['.', 'x']
            colors1 = ['blue', 'orange', 'magenta']
            colors1 = colors1[:len(np.unique(Y1))]
            colors2 = ['r', 'g', 'y', 'm', 'c']
            colors2 = colors2[:len(np.unique(Y2))]

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


def test_model():
    torch.manual_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    out_source_dpath = os.path.join(out_source_dpath, ex)
    
    src, targets1, targets2, id_dict, img_paths = get_sequence_splitted_data_with_const(data_path, label_decomp=args.labeldecomp)

    (true_targets1, targets1, pseudo_label1), (true_targets2, targets2, pseudo_label2), seq_ids, sample_ids = get_discarded_label_for_ssl(targets1, targets2, id_dict)
    model = SemiSelfClassifier(n_classes=[torch.unique(true_targets1).size(0), torch.unique(true_targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)
    
    if args.retrain:
        model.load_state_dict(torch.load('{}/param/model_bestparam.json'.format(out_source_dpath)))
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_board_dpath = '{}/re_runs'.format(out_source_dpath)
        out_condition_dpath = '{}/re_condition'.format(out_source_dpath)
        out_test_dpath = '{}/re_test_{}'.format(out_source_dpath, args.param)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_board_dpath = '{}/runs'.format(out_source_dpath)
        out_condition_dpath = '{}/condition'.format(out_source_dpath)
        out_test_dpath = '{}/test_{}'.format(out_source_dpath, args.param)
    clean_directory(out_test_dpath)

    ngpus = 1
    if args.multi:
        ngpus = torch.cuda.device_count()    
        model = nn.DataParallel(model, device_ids=[g for g in range(ngpus)])
    model = model.to(device)
    params = list(model.parameters())

    data_pairs = torch.utils.data.TensorDataset(src, targets1, targets2, pseudo_label1, pseudo_label2, torch.Tensor(list(range(src.size(0)))))
    
    # if args.semi is False:
    #     sample_ids['train'] = [i for i in sample_ids['train'] if targets1[i] != -2]
    
    train_set = torch.utils.data.dataset.Subset(data_pairs, sample_ids['train'])
    val_set = torch.utils.data.dataset.Subset(data_pairs, sample_ids['val'])
    test_set = torch.utils.data.dataset.Subset(data_pairs, sample_ids['test'])

    train_size1 = train_set[:][1][train_set[:][1] != -2].size(0)
    train_size2 = train_set[:][2][train_set[:][2] != -2].size(0)

    train_loader = DataLoader(train_set, batch_size=args.batch*ngpus, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch*ngpus, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch*ngpus, shuffle=False)

    if args.param == 'best':
        model.load_state_dict(torch.load('./reports/SSL_ConstLabeling/colon/FSL_r0.0_woDRL/param/model_bestparam.json'))
        # model.load_state_dict(torch.load('{}/model_bestparam.json'.format(out_param_dpath)))
    else:
        model.load_state_dict(torch.load('{}/model_lastparam.json'.format(out_param_dpath)))
    model = model.to(device)
    tags = ['all']
    if args.semi:
        tags = tags + ['semi']
    for ntag in tags:
        X1_dict = {}
        X2_dict = {}
        Y1_dict = {}
        Y2_dict = {}
        pY1_dict = {}
        pY2_dict = {}
        if ntag == 'all':
            if args.param == 'best':
                model.load_state_dict(torch.load('./reports/SSL_ConstLabeling/colon/FSL_r0.0_woDRL/param/model_bestparam.json'))
                # model.load_state_dict(torch.load('{}/model_bestparam.json'.format(out_param_dpath)))
            else:
                model.load_state_dict(torch.load('{}/model_lastparam.json'.format(out_param_dpath)))
        elif ntag == 'semi':
            if args.param == 'best':
                model.load_state_dict(torch.load('{}/model_semi_bestparam.json'.format(out_param_dpath)))
            else:
                model.load_state_dict(torch.load('{}/model_semi_lastparam.json'.format(out_param_dpath)))
        model = model.to(device)
        conf_mat_dict1 = {}
        conf_mat_dict2 = {}
        for gl, pl in itertools.product(range(torch.unique(true_targets1).size(0)), range(torch.unique(true_targets1).size(0))):
            conf_mat_dict1['{}to{}'.format(gl, pl)] = []
        for gl, pl in itertools.product(range(torch.unique(true_targets2).size(0)), range(torch.unique(true_targets2).size(0))):
            conf_mat_dict2['{}to{}'.format(gl, pl)] = []

        with torch.no_grad():
            smoothed_preds_dict1 = {}
            smoothed_preds_dict2 = {}
            # smooth_paras = [0 0.5, 1.0, 10]
            smooth_paras = [0.001, 0.01, 0.1, 1, 10]
            smooth_paras = []
            labels_dict1 = {}
            labels_dict2 = {}
            for k, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
                seq_key = seq_ids[k]
                model.eval()
                eval_dict = {}
                eval_dict['path'] = []
                eval_dict['loc_label'] = []
                eval_dict['mayo_label'] = []
                eval_dict['mayo_discard_label'] = []
                eval_dict['loc_discard_label'] = []
                eval_dict['loc_pred'] = []
                eval_dict['mayo_pred'] = []
                eval_dict['seq'] = []

                print('temporal smootheness Step', 'data:', k)
                # smoothed_result1 = {}
                # smoothed_result2 = {}
                for smooth_para in smooth_paras:
                    # smoothed_result1[k+str(smooth_para)] = 0
                    # smoothed_result2[k+str(smooth_para)] = 0
                    # smoothed_preds_dict1[k+str(smooth_para)] = []
                    # smoothed_preds_dict2[k+str(smooth_para)] = []
                    inputs1, inputs2 = [], []
                    labels1, labels2 = [], []
                    cat_ids = []
                    for cat_seq_key in seq_key:
                        cat_id = id_dict[cat_seq_key]
                        cat_ids.append(cat_id)
                        const = get_sequence_based_constraints(len(cat_id))
                        pred1, pred2 = model.predict_proba(src[cat_id].to(device))
                        if smooth_para == smooth_paras[0]:
                            label1 = targets1[cat_id].clone().numpy()
                            labels1.extend(label1)
                            label2 = targets2[cat_id].clone().numpy()
                            labels2.extend(label2)
                        pred1 = pred1.detach().to('cpu').numpy()
                        pred2 = pred2.detach().to('cpu').numpy()

                        inputs1.append((pred1, const, smooth_para))
                        inputs2.append((pred2, const, smooth_para))

                    if smooth_para == smooth_paras[0]:
                        labels_dict1[k] = labels1
                        labels_dict2[k] = labels2
                    n_core = cpu_count()
                    ps = []
                    for c_coeff, inputs in zip([args.c1, args.c2], [inputs1, inputs2]):
                        if args.temp:
                            ps.append([get_pseudo_soft_labeling_with_qp(cat_inputs) for cat_inputs in inputs])
                        else:
                            p = Pool(n_core)
                            ps.append(p.map(get_pseudo_labeling_with_mip, inputs))
                            p.close()
                            p.terminate()

                    smoothed_preds1 = []
                    smoothed_preds2 = []
                    # print('origin', np.argmax(inputs1[0][0], axis=1))
                    # print('label', labels1[0])
                    if args.temp:
                        for p1, p2, cat_id in zip(ps[0], ps[1], cat_ids):
                            smoothed_pred1  = torch.argmax(torch.Tensor(p1).float(), dim=1)
                            smoothed_pred2  = torch.argmax(torch.Tensor(p2).float(), dim=1)
                            smoothed_preds1.extend(smoothed_pred1.detach().to('cpu').numpy())
                            smoothed_preds2.extend(smoothed_pred2.detach().to('cpu').numpy())
                            # print('soft smooth', smooth_para, smoothed_pred1)
                            # print('Acc', true_positive_multiclass(torch.Tensor(np.argmax(inputs1[0][0], axis=1)).long(), torch.Tensor(labels1[0]).long()), true_positive_multiclass(smoothed_pred1.long(), torch.Tensor(labels1[0]).long()))
                            # smoothed_result1[k+str(smooth_para)] += true_positive_multiclass(smoothed_pred1.long(), torch.Tensor(labels1[0]).long())
                            # smoothed_result2[k+str(smooth_para)] += true_positive_multiclass(smoothed_pred2.long(), torch.Tensor(labels2[0]).long())
                    else:
                        for p1, p2, cat_id in zip(ps[0], ps[1], cat_ids):
                            smoothed_pred1 = torch.Tensor(p1).long()
                            smoothed_pred2 = torch.Tensor(p2).long()
                            smoothed_preds1.extend(smoothed_pred1.detach().to('cpu').numpy())
                            smoothed_preds2.extend(smoothed_pred2.detach().to('cpu').numpy())
                            # print('smooth', smooth_para, smoothed_pred1)
                            # print('Acc', true_positive_multiclass(torch.Tensor(np.argmax(inputs1[0][0], axis=1)).long(), torch.Tensor(labels1[0]).long()), true_positive_multiclass(smoothed_pred1.long(), torch.Tensor(labels1[0]).long()))
                            # smoothed_result1[k+str(smooth_para)] += true_positive_multiclass(smoothed_pred1.long(), torch.Tensor(labels1[0]).long())
                            # smoothed_result2[k+str(smooth_para)] += true_positive_multiclass(smoothed_pred2.long(), torch.Tensor(labels2[0]).long())
                    smoothed_preds_dict1[k+str(smooth_para)] = smoothed_preds1
                    smoothed_preds_dict2[k+str(smooth_para)] = smoothed_preds2
                    eval_dict['smooth_loc_pred'+str(smooth_para)] = smoothed_preds1
                    eval_dict['smooth_mayo_pred'+str(smooth_para)] = smoothed_preds2
                    
                X1, X2, Y1, Y2 = [], [], [], []
                pY1, pY2 = [], []
                H0 = []
                for n_iter, (x, target, sub_target, p_target, p_sub_target, idx) in enumerate(loader):
                    idx = idx.long().numpy()
                    cat_paths = [img_paths[iidx] for iidx in idx]
                    h0 = model.enc(x.to(device))
                    H0.extend(h0.detach().to('cpu').numpy())
                    
                    (mu1, mu2) = model.hidden_output(x.to(device))
                    mu1 = mu1.detach().to('cpu').numpy()
                    mu2 = mu2.detach().to('cpu').numpy()
                    pred_y1, pred_y2 = model.predict_label(x.to(device))
                    X1.extend(mu1)
                    X2.extend(mu2)
                    pred_np = pred_y1.detach().to('cpu').numpy()
                    sub_pred_np = pred_y2.detach().to('cpu').numpy()
                    
                    target_np = true_targets1[idx].detach().to('cpu').numpy()
                    sub_target_np = true_targets2[idx].detach().to('cpu').numpy()
                    
                    dis_main_target_np = target.detach().to('cpu').numpy()
                    dis_sub_target_np = sub_target.detach().to('cpu').numpy()
                    
                    eval_dict['path'].extend(cat_paths)
                    eval_dict['loc_label'].extend(target_np)
                    eval_dict['mayo_label'].extend(sub_target_np)
                    eval_dict['loc_discard_label'].extend(dis_main_target_np)
                    eval_dict['mayo_discard_label'].extend(dis_sub_target_np)
                    eval_dict['loc_pred'].extend(pred_np)
                    eval_dict['mayo_pred'].extend(sub_pred_np)
                    eval_dict['seq'].extend([c.split('/')[1] for c in cat_paths])
                    Y1.extend(target_np)
                    Y2.extend(sub_target_np)
                    pY1.extend(pred_np)
                    pY2.extend(sub_pred_np)

                df = pd.DataFrame.from_dict(eval_dict)
                if args.temp:
                    df.to_csv('{}/eachPredicted_{}_soft.csv'.format(out_test_dpath, k))
                else:
                    df.to_csv('{}/eachPredicted_{}.csv'.format(out_test_dpath, k))
                H0 = np.array(H0)
                print(H0.shape)
                np.save('{}/HiddenFeatures_{}.npy'.format(out_test_dpath, k), H0)
                if k == 'test':
                    return
                continue
                X1_dict[k] = np.asarray(X1)
                X2_dict[k] = np.asarray(X2)
                Y1_dict[k] = np.asarray(Y1)
                Y2_dict[k] = np.asarray(Y2)
                pY1_dict[k] = np.asarray(pY1)
                pY2_dict[k] = np.asarray(pY2)
                conf_mat_label1 = confusion_matrix(Y1, pY1)
                conf_mat_label2 = confusion_matrix(Y2, pY2)
                for gl, pl in itertools.product(range(torch.unique(true_targets1).size(0)), range(torch.unique(true_targets1).size(0))):
                    conf_mat_dict1['{}to{}'.format(gl, pl)].append(conf_mat_label1[gl, pl])
                for gl, pl in itertools.product(range(torch.unique(true_targets2).size(0)), range(torch.unique(true_targets2).size(0))):
                    conf_mat_dict2['{}to{}'.format(gl, pl)].append(conf_mat_label2[gl, pl])

            df = pd.DataFrame.from_dict(conf_mat_dict1)
            df.to_csv('{}/conf_mat_loc.csv'.format(out_test_dpath))
            df = pd.DataFrame.from_dict(conf_mat_dict2)
            df.to_csv('{}/conf_mat_mayo.csv'.format(out_test_dpath))
        
        score_dict_withPseudo = {}
        for indices, k in zip([sample_ids['train'], sample_ids['val'], sample_ids['test']], ['train', 'val', 'test']):
            pred_y1 = torch.from_numpy(pY1_dict[k])
            pred_y2 = torch.from_numpy(pY2_dict[k])
            cat_full_target2 = true_targets2[indices]
            cat_target2 = targets2[indices]
            remain_target2_acc = true_positive_multiclass(pred_y2[cat_target2!=-2], cat_target2[cat_target2!=-2])
            discarded_target2_acc = true_positive_multiclass(pred_y2[cat_target2==-2], cat_full_target2[cat_target2==-2])

            target1_acc = true_positive_multiclass(pred_y1, true_targets1[indices])
            target2_acc = true_positive_multiclass(pred_y2, true_targets2[indices])

            cat_full_target1 = true_targets1[indices]
            cat_target1 = targets1[indices]
            remain_target1_acc = true_positive_multiclass(pred_y1[cat_target1!=-2], cat_target1[cat_target1!=-2])
            discarded_target1_acc = true_positive_multiclass(pred_y1[cat_target1==-2], cat_full_target1[cat_target1==-2])
            if k == 'train':
                score_dict_withPseudo['main'] = [target1_acc/len(indices)]
                score_dict_withPseudo['sub'] = [target2_acc/len(indices)]
                score_dict_withPseudo['remain_main'] = [remain_target1_acc/len(cat_target1[cat_target1!=-2])]
                score_dict_withPseudo['remain_sub'] = [remain_target2_acc/len(cat_target2[cat_target2!=-2])]
                score_dict_withPseudo['pseudo_main'] = [discarded_target1_acc/len(cat_target1[cat_target1==-2])]
                score_dict_withPseudo['pseudo_sub'] = [discarded_target2_acc/len(cat_target2[cat_target2==-2])]
                for s in smooth_paras:
                    smooth_pred1 = torch.Tensor(smoothed_preds_dict1[k+str(s)]).long()
                    smooth_pred2 = torch.Tensor(smoothed_preds_dict2[k+str(s)]).long()
                    smooth_acc1 = true_positive_multiclass(smooth_pred1, true_targets1[indices])
                    smooth_acc2 = true_positive_multiclass(smooth_pred2, true_targets2[indices])
                    print(k+str(s))
                    print(target1_acc, smooth_acc1)
                    print(target2_acc, smooth_acc2)
                    score_dict_withPseudo['main_smooth'+str(s)] = [smooth_acc1/len(indices)]
                    score_dict_withPseudo['sub_smooth'+str(s)] = [smooth_acc2/len(indices)]
                
            else:
                score_dict_withPseudo['main'].append(target1_acc/len(indices))
                score_dict_withPseudo['sub'].append(target2_acc/len(indices))
                score_dict_withPseudo['remain_main'].append(remain_target1_acc/len(cat_target1[cat_target1!=-2]))
                score_dict_withPseudo['remain_sub'].append(remain_target2_acc/len(cat_target2[cat_target2!=-2]))
                score_dict_withPseudo['pseudo_main'].append(discarded_target1_acc/len(cat_target1[cat_target1==-2]))
                score_dict_withPseudo['pseudo_sub'].append(discarded_target2_acc/len(cat_target2[cat_target2==-2]))
                for s in smooth_paras:
                    # smooth_pred1 = torch.from_numpy(smoothed_preds_dict1[k+str(s)])
                    # smooth_pred2 = torch.from_numpy(smoothed_preds_dict2[k+str(s)])
                    smooth_pred1 = torch.Tensor(smoothed_preds_dict1[k+str(s)]).long()
                    smooth_pred2 = torch.Tensor(smoothed_preds_dict2[k+str(s)]).long()
                    smooth_acc1 = true_positive_multiclass(smooth_pred1, true_targets1[indices])
                    smooth_acc2 = true_positive_multiclass(smooth_pred2, true_targets2[indices])
                    print(k+str(s))
                    print(target1_acc, smooth_acc1)
                    print(target2_acc, smooth_acc2)
                    score_dict_withPseudo['main_smooth'+str(s)].append(smooth_acc1/len(indices))
                    score_dict_withPseudo['sub_smooth'+str(s)].append(smooth_acc2/len(indices))

        df = pd.DataFrame.from_dict(score_dict_withPseudo)
        if args.temp:
            df.to_csv('{}/ResultNN_withPseudo_{}_softConst.csv'.format(out_test_dpath, ntag))
        else:
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
                
        df = pd.DataFrame.from_dict(score_dict)
        df.to_csv('{}/LinearReg.csv'.format(out_test_dpath, ntag))


def read_graphfile(f):
    from scipy.sparse import csr_matrix
    graph_data = np.genfromtxt(f, delimiter=' ', dtype=int)
    row = np.hstack([graph_data[:,0], graph_data[:,1]])
    col = np.hstack([graph_data[:,1], graph_data[:,0]])
    max_nid = np.max(row)
    return csr_matrix((np.ones(len(row)), (row,col)), shape=(max_nid+1,max_nid+1))


def read_labelfile(f):
    label_data = np.genfromtxt(f, delimiter=' ', dtype=int)
    return label_data[:,0],label_data[:,1]


def label_prop():
    G = read_graphfile('./src/sample.edgelist').tolil()
    x,y = read_labelfile('./src/sample.label')
    print(G.toarray())
    print(x)
    print(y)
    clf = HMN(graph=G)
    clf.fit(x,y)
    predicted = clf.predict_proba(np.arange(G.shape[0]))
    print(predicted)
    # label_prop_model = LabelPropagation()
    # G = np.array([[0, 0,0], [], [0,0,1], [0, 0, 0]])
    # clf = HMN(graph=G) # Gは隣接行列（scipy.sparse）
    # x_train = np.array([1])
    # y_train = np.array([1])
    # x_test = np.array([1])
    # clf.fit(x_train, y_train) # x_trainはラベル付きノードIDのリスト、y_trainは対応するノードのラベルのリスト
    # predicted = clf.predict(x_test) # x_testラベルを推定したいノードIDのリスト
    # print(predicted)

def main():
    args = argparses()
    print(args)

    if args.train:
        train_model()
    if args.val:
        val_model()
    if args.test:
        test_model()
    
if __name__ == '__main__':
    # with SetIO('./out.log'):
    main()
