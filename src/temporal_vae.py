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

from losses import TripletLoss, negative_entropy_loss, Fourier_mse, loss_vae
from metrics import true_positive_multiclass, true_positive, true_negative
from __init__ import clean_directory, SetIO
from data_handling import get_triplet_flatted_data_with_idx, random_label_replace
from archs import TDAE_VAE
from archs import TDAE_VAE_fullsuper_disentangle

SEED = 1
torch.manual_seed(SEED)
rn.seed(SEED)
np.random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current_path = './'

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
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--dlim', type=int, default=0)
    parser.add_argument('--ndeconv', type=int, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--dm', type=int, default=0)
    parser.add_argument('--data', type=str, default='toy')
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--param', type=str, default='best')
    parser.add_argument('--fill', type=str, default='hp')
    parser.add_argument('--ex', type=str, default=None)
    parser.add_argument('--classifier', type=float, default=1e-1)
    parser.add_argument('--c1', type=float, default=1)
    parser.add_argument('--c2', type=float, default=1)
    parser.add_argument('--rec', type=float, default=1e-3)
    parser.add_argument('--adv', type=float, default=1e-2)
    parser.add_argument('--tri', type=float, default=1e-3)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--margin', type=float, default=1e-1)
    parser.add_argument('--triplet', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--fou', action='store_true')
    parser.add_argument('--rev', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--adapt', action='store_true')
    parser.add_argument('--semi', action='store_true')
    parser.add_argument('--labeldecomp', action='store_false')
    parser.add_argument('--ngpus', default=1)
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


def train_TDAE_VAE():
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    out_source_dpath = out_source_dpath + '/' + ex

    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    else:
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)

    # out_source_dpath = clean_directory(out_source_dpath, replace=False)

    ratio = [0.7, 0.2, 0.1]
    n_sample = len(data_pairs)
    train_size = int(n_sample*ratio[0])
    val_size = int(n_sample*ratio[1])
    test_size = n_sample - train_size - val_size

    # train_set, val_set = torch.utils.data.random_split(data_pairs, [train_size, val_size])
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)

    model = TDAE_VAE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)
    if args.ngpus > 1:
        g_list = [i for i in range(args.ngpus)]
        model = nn.DataParallel(model, device_ids=g_list)
    model = model.to(device)

    if args.retrain:
        if args.param == 'best':
            model.load_state_dict(torch.load('{}/param/TDAE_test_bestparam.json'.format(out_source_dpath)))
        else:
            model.load_state_dict(torch.load('{}/param/TDAE_test_param.json'.format(out_source_dpath)))
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

    # criterion_adv = nn.NLLLoss()
    criterion_classifier = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=args.margin)
    if args.fou:
        criterion_reconst = Fourier_mse(img_h=img_h, img_w=img_w, mask=True, dm=args.dm, mode=args.fill)
    else:
        criterion_reconst = nn.MSELoss()
    
    criterion_vae = loss_vae
    params = list(model.parameters())
    optimizer = optim.Adam(params)
    # optimizer = optim.SGD(params, lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    Scores_reg = [[], []]
    Scores_reg_adv = [[], []]
    Vals_reg = [[], []]
    n_epochs = args.epoch
    best_loss = np.inf
    best_epoch = 0
    l_adv, l_recon, l_tri, l_c = args.adv, args.rec, args.tri, args.classifier
    train_keys = ['loss/train_all', 'loss/train_classifier_main', 'loss/train_classifier_sub', 'loss/train_adv', 'loss/train_rec', 'loss/train_triplet']
    val_keys = ['loss/val_all', 'loss/val_classifier_main', 'loss/val_classifier_sub', 'loss/val_adv', 'loss/val_rec', 'loss/val_triplet']

    weight_keys = ['classifier', 'adv', 'reconst']
    if args.triplet:
        weight_keys += ['triplet']
    weight_dict = {}
    current_loss_dict = {}
    weight_change = {}
    for k, w in zip(weight_keys, [args.classifier, args.adv, args.rec, args.triplet]):
        weight_dict[k] = w
        weight_change[k] = [w]
        current_loss_dict[k] = 0

    for epoch in range(n_epochs):
        accs_p, acc_t = [], []
        Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0
        loss_dict = {}
        for k in train_keys:
            loss_dict[k] = []

        for ite, (idx, p_idx, n_idx, target, _) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            (preds, sub_preds, preds_adv, reconst, mu1, mu2, logvar1, logvar2) = model.forward(src[idx].to(device))
            if args.triplet:
                (_, _, _, _, p_mu1, p_mu2, _, _) = model.forward(src[p_idx].to(device))
                (_, _, _, _, n_mu1, n_mu2, _, _) = model.forward(src[n_idx].to(device))
                loss_triplet = weight_dict['triplet'] * criterion_triplet(mu2, p_mu2, n_mu2)
                loss_triplet.backward(retain_graph=True)
            else:
                loss_triplet = torch.Tensor([0])

            loss_reconst = weight_dict['reconst'] * criterion_vae(reconst.to(device), src[idx].to(device), mu1, mu2, logvar1, logvar2)
            loss_reconst.backward(retain_graph=True)

            loss_classifier_main = weight_dict['classifier'] * criterion_classifier(preds.to(device), target.to(device))
            loss_classifier_main.backward(retain_graph=True)

            loss_adv = weight_dict['adv'] * negative_entropy_loss(sub_preds.to(device))
            loss_adv.backward(retain_graph=True)

            model.classifiers[1].zero_grad()
            loss_classifier_sub = weight_dict['adv'] * criterion_classifier(preds_adv.to(device), target.to(device))
            loss_classifier_sub.backward(retain_graph=True)
            optimizer.step()

            loss = loss_classifier_main + loss_classifier_sub + loss_adv + loss_reconst + loss_triplet
            for k, val in zip(train_keys, [loss.item(), loss_classifier_main.item(), loss_classifier_sub.item(), loss_adv.item(), loss_reconst.item(), loss_triplet.item()]):
                loss_dict[k].append(val)
            
            y_true = target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, y_true)
        
        for k in loss_dict.keys():
            loss_dict[k] = np.mean(loss_dict[k])
            
        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}'.format(epoch+1,loss_dict['loss/train_all'], Acc/len(train_set), sub_Acc/len(train_set)))
        summary = scalars2summary(writer=writer, 
                                tags=list(loss_dict.keys()), 
                                vals=list(loss_dict.values()), epoch=epoch+1)
        
        if (epoch + 1) % args.step == 0:
            model.eval()
            with torch.no_grad():
                for v_i, (idx, p_idx, n_idx, _, _) in enumerate(train_loader):
                    if v_i == 0:
                        reconst = model.reconst(src[idx].to(device))
                        np_input = src[idx[0]].detach().to('cpu')
                        np_reconst = reconst[0].detach().to('cpu')
                        img_grid = make_grid(torch.stack([np_input, np_reconst]))
                        writer.add_image('test', img_grid, epoch+1)
                        break
                val_loss_dict = {}
                for k in val_keys:
                    val_loss_dict[k] = []

                for v_i, (idx, p_idx, n_idx, target, _) in enumerate(val_loader):
                    (preds, sub_preds, preds_adv, reconst, mu1, mu2, logvar1, logvar2) = model.forward(src[idx].to(device))
                    if args.triplet:
                        (_, _, _, _, p_mu1, p_mu2, _, _) = model.forward(src[p_idx].to(device))
                        (_, _, _, _, n_mu1, n_mu2, _, _) = model.forward(src[n_idx].to(device))
                        val_loss_triplet = weight_dict['triplet'] * criterion_triplet(mu2, p_mu2, n_mu2)
                    else:
                        val_loss_triplet = torch.Tensor([0])

                    val_loss_reconst = weight_dict['reconst'] * criterion_vae(reconst.to(device), src[idx].to(device), mu1, mu2, logvar1, logvar2)
                    val_loss_classifier_main = weight_dict['classifier'] * criterion_classifier(preds.to(device), target.to(device))
                    val_loss_classifier_sub = weight_dict['adv'] * criterion_classifier(preds_adv.to(device), target.to(device))
                    val_loss_adv = weight_dict['adv'] * negative_entropy_loss(sub_preds.to(device))
                    val_loss = val_loss_reconst + val_loss_classifier_main + val_loss_adv + val_loss_classifier_sub + val_loss_triplet
                    
                    for k, val in zip(val_keys, [val_loss, val_loss_classifier_main, val_loss_classifier_sub, val_loss_adv, val_loss_reconst, val_loss_triplet]):
                        val_loss_dict[k].append(val.item())

                for k in val_loss_dict.keys():
                    val_loss_dict[k] = np.mean(val_loss_dict[k])
                print('epoch: {} val loss: {}'.format(epoch+1, val_loss_dict['loss/val_all']))

                summary = scalars2summary(writer=writer, 
                                        tags=list(val_loss_dict.keys()), 
                                        vals=list(val_loss_dict.values()), epoch=epoch+1)

                if args.ngpus > 1:
                    torch.save(model.module.state_dict(), '{}/TDAE_param_e{:04}.json'.format(out_param_dpath, epoch+1))
                else:
                    torch.save(model.state_dict(), '{}/TDAE_param_e{:04}.json'.format(out_param_dpath, epoch+1))
                if best_loss > val_loss_dict['loss/val_all']:
                    best_epoch = epoch + 1
                    best_loss = val_loss_dict['loss/val_all']
                    if args.ngpus > 1:
                        torch.save(model.module.state_dict(), '{}/TDAE_test_bestparam.json'.format(out_param_dpath))
                    else:
                        torch.save(model.state_dict(), '{}/TDAE_test_bestparam.json'.format(out_param_dpath))

    if args.ngpus > 1:
        torch.save(model.module.state_dict(), '{}/TDAE_test_param.json'.format(out_param_dpath))
    else:
        torch.save(model.state_dict(), '{}/TDAE_test_param.json'.format(out_param_dpath))
    
    args.best_epoch = best_epoch
    df = args2pandas(args)
    df.to_csv('{}/condition.csv'.format(out_condition_dpath))
    
    writer.close()


def pseudo_label_update(model, inputs):
    with torch.no_grad():
        model.eval()
        _, pred_pseudo_label = model.predict_label(inputs)
    return pred_pseudo_label.to('cpu')


def train_TDAE_VAE_fullsuper_disentangle():
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    out_source_dpath = os.path.join(out_source_dpath, ex)

    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    else:
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
        
    true_targets2 = copy.deepcopy(targets2)
    if args.semi:
        targets2 = random_label_replace(targets2, ratio=args.ratio, value=-2)
        pseudo_label = torch.randint(low=0, high=torch.unique(true_targets2).size(0), size=targets2.size())
        pseudo_label[targets2 != -2] = -2
    
    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)
        
    model = TDAE_VAE_fullsuper_disentangle(n_classes=[torch.unique(targets1).size(0), torch.unique(true_targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)
    if args.retrain:
        model.load_state_dict(torch.load('{}/param/TDAE_test_bestparam.json'.format(out_source_dpath)))
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
    
    ratio = [0.7, 0.2, 0.1]
    n_sample = len(data_pairs)
    train_size = int(n_sample*ratio[0])
    val_size = int(n_sample*ratio[1])
    test_size = n_sample - train_size - val_size

    # train_set, val_set = torch.utils.data.random_split(data_pairs, [train_size, val_size])
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))
    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)

    # criterion_adv = nn.NLLLoss()
    criterion_classifier = nn.CrossEntropyLoss(ignore_index=-2)
    criterion_triplet = TripletLoss(margin=args.margin)
    if args.fou:
        criterion_reconst = Fourier_mse(img_h=img_h, img_w=img_w, mask=True, dm=args.dm, mode=args.fill)
    else:
        criterion_reconst = nn.MSELoss()
    criterion_vae = loss_vae
    
    params = list(model.parameters())
    optimizer = optim.Adam(params)
    # optimizer = optim.SGD(params, lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    n_epochs = args.epoch
    best_epoch, best_loss = 0, np.inf
    l_adv, l_recon, l_tri, l_c = args.adv, args.rec, args.tri, args.classifier
    train_keys = ['loss/train_all', 'loss/train_classifier_main', 'loss/train_classifier_sub', 'loss/train_adv_main', 'loss/train_adv_sub', 'loss/train_no_grad_main', 'loss/train_no_grad_sub', 'loss/train_rec', 'loss/train_triplet', 'loss/train_pseudo_adv', 'loss/train_pseudo_classifier']
    val_keys = ['loss/val_all', 'loss/val_classifier_main', 'loss/val_classifier_sub', 'loss/val_adv_main', 'loss/val_adv_sub', 'loss/val_no_grad_main', 'loss/val_no_grad_sub', 'loss/val_rec', 'loss/val_triplet', 'loss/val_pseudo_adv', 'loss/val_pseudo_classifier']

    weight_keys = ['classifier_main', 'classifier_sub', 'adv_main', 'adv_sub', 'reconst']
    w_args = [args.classifier, args.classifier, args.adv, args.adv, args.rec]
    if args.triplet:
        weight_keys += ['triplet']
        w_args += [args.tri]
        
    weight_dict = {}
    current_loss_dict = {}
    prev_loss_dict = {}
    weight_change = {}
    for k, w in zip(weight_keys, w_args):
        weight_dict[k] = w
        weight_change[k] = [w]
        current_loss_dict[k] = 0

    for epoch in range(n_epochs):
        Acc, sub_Acc  = 0, 0
        loss_dict = {}
        for k in train_keys:
            loss_dict[k] = []
        
        if args.adapt:
            if epoch > 1:
                prev_dict = {}
                current_dict = {}
                for k in current_loss_dict.keys():
                    if prev_loss_dict[k] > 0 and current_loss_dict[k] > 0:
                        prev_dict[k] = prev_loss_dict[k]
                        current_dict[k] = current_loss_dict[k]
                new_weight_dict = dynamic_weight_average(prev_dict, current_dict)
                for k in new_weight_dict.keys():
                    weight_dict[k] = new_weight_dict[k]
            prev_loss_dict = current_loss_dict
            current_loss_dict = {}
            for k in weight_keys:
                weight_change[k].append(weight_dict[k])
                current_loss_dict[k] = 0

        for iters, (idx, p_idx, n_idx, target, sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, reconst, mu1, mu2, logvar1, logvar2 = model.forward(src[idx].to(device))
            if args.triplet:
                _, _, _, _, _, _, _, p_mu1, p_mu2, _, _ = model.forward(src[p_idx].to(device))
                _, _, _, _, _, _, _, n_mu1, n_mu2, _, _ = model.forward(src[n_idx].to(device))
                # loss_triplet = l_tri * criterion_triplet(mu1, p_mu1, n_mu1)
                loss_triplet = weight_dict['triplet'] * criterion_triplet(mu2, p_mu2, n_mu2)
                loss_triplet.backward(retain_graph=True)
            else:
                loss_triplet = torch.Tensor([0])
                
            if weight_dict['reconst'] > 0:
                loss_reconst = weight_dict['reconst'] * criterion_vae(reconst.to(device), src[idx].to(device), mu1, mu2, logvar1, logvar2)
                loss_reconst.backward(retain_graph=True)
            else:
                loss_reconst = torch.Tensor([0])
            
            loss_classifier_main = weight_dict['classifier_main'] * criterion_classifier(preds.to(device), target.to(device))
            loss_classifier_sub = weight_dict['classifier_sub'] * criterion_classifier(sub_preds.to(device), sub_target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            loss_classifier_sub.backward(retain_graph=True)
            if args.semi:
                loss_pseudo_classifier = 0.1 * weight_dict['classifier_sub'] * criterion_classifier(sub_preds.to(device), pseudo_label[idx].to(device))
                loss_pseudo_classifier.backward(retain_graph=True)
            else:
                loss_pseudo_classifier = torch.Tensor([0])
            
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
            if args.semi:
                loss_pseudo_label = 0.1 * weight_dict['adv_main'] * criterion_classifier(preds_adv_no_grad.to(device), pseudo_label[idx].to(device))
                loss_pseudo_label.backward(retain_graph=True)
            else:
                loss_pseudo_label = torch.Tensor([0])
            
            optimizer.step()
            loss = loss_classifier_main + loss_classifier_sub + loss_adv_main + loss_adv_sub +  loss_reconst + loss_main_no_grad + loss_sub_no_grad + loss_triplet + loss_pseudo_label + loss_pseudo_classifier

            if args.adapt:
                current_loss_dict['classifier_main'] += loss_classifier_main.item() + loss_classifier_sub.item()
                current_loss_dict['classifier_sub'] += loss_classifier_main.item() + loss_classifier_sub.item()
                current_loss_dict['reconst'] += loss_reconst.item()
                current_loss_dict['adv_main'] += loss_adv_main.item() + loss_main_no_grad.item()
                current_loss_dict['adv_sub'] += loss_adv_sub.item() + loss_sub_no_grad.item()
                if args.triplet:
                    current_loss_dict['triplet'] += loss_triplet.item()

            for k, val in zip(train_keys, [loss, loss_classifier_main, loss_classifier_sub, loss_adv_main, loss_adv_sub, loss_main_no_grad, loss_sub_no_grad, loss_reconst, loss_triplet, loss_pseudo_label, loss_pseudo_classifier]):
                loss_dict[k].append(val.item())
            
            y_true = target.to('cpu')
            sub_y_true = true_targets2[idx].to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = sub_preds.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, sub_y_true)
                    
        for k in loss_dict.keys():
            loss_dict[k] = np.mean(loss_dict[k])

        print('epoch: {:04} loss: {:.5} \nAcc: {:.5} sub Acc: {:.5}'.format(epoch+1, loss_dict['loss/train_all'], Acc/len(train_set), sub_Acc/len(train_set)))

        summary = scalars2summary(writer=writer,
                            tags=list(loss_dict.keys()), 
                            vals=list(loss_dict.values()), epoch=epoch+1)
        
        if (epoch + 1) % args.step == 0:
            with torch.no_grad():
                model.eval()
                for v_i, (idx, _, _, _, _) in enumerate(train_loader):
                    if v_i == 0:
                        reconst = model.reconst(src[idx].to(device))
                        np_input = src[idx][0].detach().to('cpu')
                        np_reconst = reconst[0].detach().to('cpu')
                        img_grid = make_grid(torch.stack([np_input, np_reconst]))
                        writer.add_image('train example', img_grid, epoch+1)

                val_loss_dict = {}
                for k in val_keys:
                    val_loss_dict[k] = []
                val_Acc = 0
                val_sub_Acc = 0
                for v_i, (idx, p_idx, n_idx, target, sub_target) in enumerate(val_loader):
                    preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, reconst,  mu1, mu2, logvar1, logvar2 = model.forward(src[idx].to(device))
                    if args.triplet:
                        _, _, _, _, _, _, _,  p_mu1, p_mu2, _, _ = model.forward(src[p_idx].to(device))
                        _, _, _, _, _, _, _,  n_mu1, n_mu2, _, _ = model.forward(src[n_idx].to(device))
                        # val_loss_triplet = criterion_triplet(mu1, p_mu1, n_mu1)
                        val_loss_triplet = weight_dict['triplet'] * criterion_triplet(mu2, p_mu2, n_mu2)
                    else:
                        val_loss_triplet = torch.Tensor([0])
                    
                    if args.semi:
                        val_loss_pseudo_label = 0.1 * weight_dict['adv_main'] * criterion_classifier(preds_adv_no_grad.to(device), pseudo_label[idx].to(device))
                        val_loss_pseudo_classifier = 0.1 * weight_dict['classifier_sub'] * criterion_classifier(sub_preds.to(device), pseudo_label[idx].to(device))
                    else:
                        val_loss_pseudo_label = torch.Tensor([0])
                        val_loss_pseudo_classifier = torch.Tensor([0])
                    if weight_dict['reconst'] > 0:
                        val_loss_reconst = weight_dict['reconst'] * criterion_vae(reconst.to(device), src[idx].to(device), mu1, mu2, logvar1, logvar2)
                    else:
                        val_loss_reconst = torch.Tensor([0])
                    val_loss_classifier_main = weight_dict['classifier_main'] * criterion_classifier(preds.to(device), target.to(device))
                    val_loss_classifier_sub = weight_dict['classifier_sub'] * criterion_classifier(sub_preds.to(device), sub_target.to(device))

                    val_loss_adv_main = weight_dict['adv_main'] * negative_entropy_loss(preds_adv.to(device))
                    val_loss_adv_sub = weight_dict['adv_sub'] * negative_entropy_loss(sub_preds_adv.to(device))
                    val_loss_no_grad_main = weight_dict['adv_main'] * criterion_classifier(preds_adv_no_grad.to(device), sub_target.to(device))
                    val_loss_no_grad_sub = weight_dict['adv_sub'] * criterion_classifier(sub_preds_adv_no_grad.to(device), target.to(device))
                    
                    val_loss = val_loss_reconst + val_loss_classifier_main + val_loss_classifier_sub + val_loss_adv_main + val_loss_no_grad_main + val_loss_classifier_sub + val_loss_adv_sub + val_loss_no_grad_sub + val_loss_triplet + val_loss_pseudo_label + val_loss_pseudo_classifier

                    for k, val in zip(val_keys, [val_loss, val_loss_classifier_main, val_loss_classifier_sub, val_loss_adv_main, val_loss_adv_sub, val_loss_no_grad_main, val_loss_no_grad_sub, val_loss_reconst, val_loss_triplet, val_loss_pseudo_label, val_loss_pseudo_classifier]):
                        val_loss_dict[k].append(val.item())

                    y_true = target.to('cpu')
                    sub_y_true = true_targets2[idx].to('cpu')
                    preds = preds.detach().to('cpu')
                    sub_preds = sub_preds.detach().to('cpu')
                    val_Acc += true_positive_multiclass(preds, y_true)
                    val_sub_Acc += true_positive_multiclass(sub_preds, sub_y_true)
                
                for k in val_loss_dict.keys():
                    val_loss_dict[k] = np.mean(val_loss_dict[k])

                summary = scalars2summary(writer=writer, 
                                        tags=list(val_loss_dict.keys()), 
                                        vals=list(val_loss_dict.values()), epoch=epoch+1)

                print('val loss: {:.5} Acc: {:.5} sub Acc: {:.5}'.format(val_loss_dict['loss/val_all'], val_Acc/len(val_set), val_sub_Acc/len(val_set)))
                torch.save(model.state_dict(), '{}/TDAE_param_e{:04}.json'.format(out_param_dpath, epoch+1))

                if best_loss > val_loss_dict['loss/val_all']:
                    best_epoch = epoch + 1
                    best_loss = val_loss_dict['loss/val_all']
                    torch.save(model.state_dict(), '{}/TDAE_test_bestparam.json'.format(out_param_dpath))
        if args.semi:
            for _, (idx, _, _, _, _) in enumerate(train_loader):
                pseudo_label[idx] = pseudo_label_update(model, src[idx].to(device))
            pseudo_label[targets2 != -2] = -2


    torch.save(model.state_dict(), '{}/TDAE_test_param.json'.format(out_param_dpath))
    
    args.best_epoch = best_epoch
    df = args2pandas(args)
    df.to_csv('{}/condition.csv'.format(out_condition_dpath))
    writer.close()
    if args.adapt:
        df = pd.DataFrame.from_dict(weight_change)
        df.to_csv('{}/WeightChange.csv'.format(out_condition_dpath))


def val_TDAE_VAE(zero_padding=False):
    torch.manual_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    args = argparses()

    out_source_dpath = out_source_dpath + '/' + ex
    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    else:
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
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

    if args.full:
        model = TDAE_VAE_fullsuper_disentangle(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)
    else:
        model = TDAE_VAE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)

    if args.param == 'best':
        model.load_state_dict(torch.load('{}/TDAE_test_bestparam.json'.format(out_param_dpath)))
    else:
        model.load_state_dict(torch.load('{}/TDAE_test_param.json'.format(out_param_dpath)))
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
        model.eval()
        n_s = 10
        for first, loader in zip(['train', 'val'], [train_loader, val_loader]):
            for n_iter, (idx, _, _, t1, t2) in enumerate(val_loader):
                reconst = model.reconst(src[idx[:2]].to(device))
                s_reconst = model.shuffle_reconst(src[idx].to(device), idx1=[0, 1], idx2=[1, 0])
                pad0_reconst = model.fix_padding_reconst(src[idx[:2]].to(device), which_val=0, pad_val=0)
                pad1_reconst = model.fix_padding_reconst(src[idx[:2]].to(device), which_val=1, pad_val=0)
                np_input0 = src[idx[:2]][0].detach().to('cpu')
                np_input1 = src[idx[:2]][1].detach().to('cpu')
                np_reconst0 = reconst[0].detach().to('cpu')
                np_reconst1 = reconst[1].detach().to('cpu')
                s_np_reconst0 = s_reconst[0].detach().to('cpu')
                s_np_reconst1 = s_reconst[1].detach().to('cpu')
                pad0_np_reconst0 = pad0_reconst[0].detach().to('cpu')
                pad0_np_reconst1 = pad0_reconst[1].detach().to('cpu')
                pad1_np_reconst0 = pad1_reconst[0].detach().to('cpu')
                pad1_np_reconst1 = pad1_reconst[1].detach().to('cpu')
                np_t1 = t1[:2].detach().to('cpu')
                np_t2 = t2[:2].detach().to('cpu')
                fig = plt.figure(figsize=(16*4, 9*2))
                for ii, npimg in enumerate([np_input0, np_reconst0, s_np_reconst0, pad0_np_reconst0, pad1_np_reconst0, np_input1, np_reconst1, s_np_reconst1, pad0_np_reconst1, pad1_np_reconst1]):
                    ax = fig.add_subplot(2, 5, ii+1)
                    if ii < 5:
                        ax.set_title('{}:{}'.format(np_t1[0], np_t2[0]))
                    else:
                        ax.set_title('{}:{}'.format(np_t1[1], np_t2[1]))
                    ax.imshow(np.transpose(npimg, (1,2,0)))
                fig.savefig('{}/{}_sample{:04d}.png'.format(out_val_dpath, first, n_iter))
                plt.close(fig)
                if n_iter >= n_s:
                    break
    
    with torch.no_grad():
        for first, loader in zip(['train', 'val'], [train_loader, val_loader]):
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
            markers = ['.', 'x']
            colors1 = ['blue', 'orange', 'magenta']
            colors1 = colors1[:len(np.unique(Y1))]
            colors2 = ['r', 'g', 'y']
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

                fig = plt.figure(figsize=(6*2, 6))
                ax = fig.add_subplot(1,2,1)
                ax.hist(np.mean(X, axis=0))
                ax = fig.add_subplot(1,2,2)
                ax.hist(np.std(X, axis=0))
                fig.savefig('{}/{}_dist_{}.png'.format(out_fig_dpath, first, ex))
                plt.close(fig)

                Xn = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 
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


def test_TDAE_VAE():
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    args = argparses()
    out_source_dpath = out_source_dpath + '/' + ex
    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    else:
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)

    if args.retrain:
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_test_dpath = '{}/re_test_{}'.format(out_source_dpath, args.param)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_test_dpath = '{}/test_{}'.format(out_source_dpath, args.param)
    clean_directory(out_test_dpath)
    
    if args.full:
        model = TDAE_VAE_fullsuper_disentangle(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)
    else:
        model = TDAE_VAE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)

    if args.param == 'best':
        model.load_state_dict(torch.load('{}/TDAE_test_bestparam.json'.format(out_param_dpath)))
    else:
        model.load_state_dict(torch.load('{}/TDAE_test_param.json'.format(out_param_dpath)))
    model = model.to(device)

    ratio = [0.7, 0.2, 0.1]
    n_sample = len(data_pairs)
    train_size = int(n_sample*ratio[0])
    val_size = int(n_sample*ratio[1])
    test_size = n_sample - (train_size + val_size)
    
    # train_set, val_set = torch.utils.data.random_split(data_pairs, [train_size, val_size])
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))
    test_indices = list(range(train_size+val_size, n_sample))

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_set = torch.utils.data.dataset.Subset(data_pairs, test_indices)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    X1_dict = {}
    X2_dict = {}
    Y1_dict = {}
    Y2_dict = {}
    pY1_dict = {}
    pY2_dict = {}
    with torch.no_grad():
        for k, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
            model.eval()
            X1, X2, Y1, Y2 = [], [], [], []
            pY1, pY2 = [], []
            for n_iter, (idx, _, _, target, sub_target) in enumerate(loader):
                (mu1, mu2) = model.hidden_output(src[idx].to(device))
                mu1 = mu1.detach().to('cpu').numpy()
                mu2 = mu2.detach().to('cpu').numpy()
                pred_y1, pred_y2 = model.predict_label(src[idx].to(device))
                X1.extend(mu1)
                X2.extend(mu2)
                Y1.extend(target.detach().to('cpu').numpy())
                Y2.extend(sub_target.detach().to('cpu').numpy())
                pY1.extend(pred_y1.detach().to('cpu').numpy())
                pY2.extend(pred_y2.detach().to('cpu').numpy())
    
            X1_dict[k] = np.asarray(X1)
            X2_dict[k] = np.asarray(X2)
            Y1_dict[k] = np.asarray(Y1)
            Y2_dict[k] = np.asarray(Y2)
            pY1_dict[k] = np.asarray(pY1)
            pY2_dict[k] = np.asarray(pY2)
        
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
    df.to_csv('{}/ResultNN.csv'.format(out_test_dpath))

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
    df.to_csv('{}/LinearReg.csv'.format(out_test_dpath))

    
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
    if args.full:
        if args.mode == 'val':
            val_TDAE_VAE()
            return
        elif args.mode == 'test':
            test_TDAE_VAE()
            return
        elif args.mode == 'train':
            train_TDAE_VAE_fullsuper_disentangle()
            return
        train_TDAE_VAE_fullsuper_disentangle()
        val_TDAE_VAE()
        test_TDAE_VAE()
        return

    if args.mode == 'val':
        print('call val')
        val_TDAE_VAE()
        return
    elif args.mode == 'test':
        print('call test')
        test_TDAE_VAE()
        return
    elif args.mode == 'train':
        print('call train')
        train_TDAE_VAE()
        return
    elif args.mode == 'conf':
        print('call conf')
        confirm_seq()
        return
    print('call train')
    train_TDAE_VAE()
    print('call val')
    val_TDAE_VAE()
    print('call test')
    test_TDAE_VAE()


if __name__ == '__main__':
    # with SetIO('./out.log'):
    main()
