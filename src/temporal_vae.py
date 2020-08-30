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
from data_handling import get_triplet_flatted_data, get_flatted_data, get_triplet_flatted_data_with_idx
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
    parser.add_argument('--rec', type=float, default=1e-3)
    parser.add_argument('--adv', type=float, default=1e-2)
    parser.add_argument('--tri', type=float, default=1e-3)
    parser.add_argument('--margin', type=float, default=1e-1)
    parser.add_argument('--triplet', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--fou', action='store_true')
    parser.add_argument('--d2ae', action='store_true')
    parser.add_argument('--rev', action='store_true')
    parser.add_argument('--full', action='store_true')
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


def train_TDAE_VAE_v2():
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()

    # if args.ex is None:
    #     t = datetime.datetime.now()
    #     t = t.strftime('%H-%M-%S')
    #     out_source_dpath = out_source_dpath + '/' + t
    out_source_dpath = out_source_dpath + '/' + ex

    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path)
    else:
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path)
    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)

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
    for epoch in range(n_epochs):
        accs_p, acc_t = [], []
        Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0
        # Loss, RecLoss, CLoss, CLoss_sub, TriLoss, CSub = [], [], [], [], [], []
        loss_dict = {}
        for k in train_keys:
            loss_dict[k] = []

        for ite, (idx, p_idx, n_idx, target, _) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            losses = []
            (preds, sub_preds, preds_adv, reconst, _, p0_anchor, mu1, mu2, logvar1, logvar2) = model.forward(src[idx].to(device))
            if args.triplet:
                (_, _, _, _, _, p0_pos, _, _, _, _) = model.forward(src[p_idx].to(device))
                (_, _, _, _, _, p0_neg, _, _, _, _) = model.forward(src[n_idx].to(device))
                loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                loss_triplet.backward(retain_graph=True)
            else:
                loss_triplet = torch.Tensor([0])

            loss_reconst = l_recon * criterion_vae(reconst.to(device), src[idx].to(device), mu1, mu2, logvar1, logvar2)
            loss_reconst.backward(retain_graph=True)
            loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device))
            loss_adv.backward(retain_graph=True)
            model.classifiers[1].zero_grad()
            loss_classifier_sub = l_c * criterion_classifier(preds_adv.to(device), target.to(device))
            loss_classifier_sub.backward(retain_graph=True)
            optimizer.step()
            # optimizer.zero_grad()
            # _, _, sub_preds, _ = model.forward(in_data.to(device))
            # optim_adv.step()
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
        # summary = scalars2summary(writer=writer, tags=list(loss_dict.keys()), vals=[np.mean(Loss), np.mean(RecLoss), np.mean(CLoss), np.mean(CLoss_sub), np.mean(TriLoss), np.mean(CSub)], epoch=epoch+1)
        
        if (epoch + 1) % args.step == 0:
            model.eval()
            with torch.no_grad():
                for v_i, (idx, p_idx, n_idx, target1, target2) in enumerate(train_loader):
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

                for v_i, (idx, p_idx, n_idx, target1, target2) in enumerate(val_loader):
                    (preds, sub_preds, preds_adv, reconst, _, p0_anchor, mu1, mu2, logvar1, logvar2) = model.forward(src[idx].to(device))
                    if args.triplet:
                        (_, _, _, _, _, p0_pos, _, _, _, _) = model.forward(src[p_idx].to(device))
                        (_, _, _, _, _, p0_neg, _, _, _, _) = model.forward(src[n_idx].to(device))
                        val_loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                    else:
                        val_loss_triplet = torch.Tensor([0])

                    val_loss_reconst = l_recon * criterion_vae(reconst.to(device), src[idx].to(device), mu1, mu2, logvar1, logvar2)
                    val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target1.to(device))
                    val_loss_classifier_sub = l_adv * criterion_classifier(preds_adv.to(device), target1.to(device))
                    val_loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device))
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


def triplet_train_TDAE_VAE():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_VAE_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_VAE_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_VAE_colon'
        data_path = './data/colon_renew.hdf5'
    else:
        return
    if args.ex is None:
        pass
    else:
        out_source_dpath = out_source_dpath + '/' + args.ex

    if args.rev:
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path)
    else:
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path)
    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)

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
    for epoch in range(n_epochs):
        accs_p, acc_t = [], []
        Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0
        Loss, RecLoss, CLoss, CLoss_sub, TriLoss, CSub = [], [], [], [], [], []
        for ite, (idx, p_idx, n_idx, target, _) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            losses = []
            (preds, sub_preds, preds_adv, reconst, _, p0_anchor, mu1, mu2, logvar1, logvar2), (_, _, _, _, _, p0_pos, _, _, _, _), (_, _, _, _, _, p0_neg, _, _, _, _) = model.forward(src[idx].to(device)), model.forward(src[p_idx].to(device)), model.forward(src[n_idx].to(device))
            loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
            loss_triplet.backward(retain_graph=True)
            loss_reconst = l_recon * criterion_vae(reconst.to(device), src[idx].to(device), mu1, mu2, logvar1, logvar2)
            loss_reconst.backward(retain_graph=True)
            loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device))
            loss_adv.backward(retain_graph=True)
            model.classifiers[1].zero_grad()
            loss_classifier_sub = l_c * criterion_classifier(preds_adv.to(device), target.to(device))
            loss_classifier_sub.backward(retain_graph=True)
            optimizer.step()
            # optimizer.zero_grad()
            # _, _, sub_preds, _ = model.forward(in_data.to(device))
            # optim_adv.step()
            loss = loss_classifier_main + loss_classifier_sub + loss_adv + loss_reconst + loss_triplet

            Loss.append(loss.item())
            RecLoss.append(loss_reconst.item())
            CLoss.append(loss_classifier_main.item())
            CLoss_sub.append(loss_adv.item())
            CSub.append(loss_classifier_sub.item())
            TriLoss.append(loss_triplet.item())
            
            y_true = target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, y_true)

        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}, Acc_adv: {}, sub Acc_adv: {}'.format(epoch+1, np.mean(Loss), Acc/len(train_set), sub_Acc/len(train_set), Acc_adv/len(train_set), sub_Acc_adv/len(train_set)))
        summary = scalars2summary(writer=writer, tags=['loss/train_all', 'loss/train_rec', 'loss/train_classifier', 'loss/train_adv', 'loss/train_triplet', 'loss/train_classifier_sub'], vals=[np.mean(Loss), np.mean(RecLoss), np.mean(CLoss), np.mean(CLoss_sub), np.mean(TriLoss), np.mean(CSub)], epoch=epoch+1)
        
        if (epoch + 1) % args.step == 0:
            model.eval()
            with torch.no_grad():
                X_train = []
                Y_train1 = []
                Y_train2 = []
                for v_i, (idx, p_idx, n_idx, target1, target2) in enumerate(train_loader):
                    if v_i == 0:
                        reconst = model.reconst(src[idx].to(device))
                        np_input = src[idx[0]].detach().to('cpu')
                        np_reconst = reconst[0].detach().to('cpu')
                        img_grid = make_grid(torch.stack([np_input, np_reconst]))
                        writer.add_image('test', img_grid, epoch+1)
                    (_, t0) = model.hidden_output(src[idx].to(device))
                    t0 = t0.detach().to('cpu').numpy()
                    X_train.extend(t0)
                    Y_train1.extend(target1.detach().to('cpu').numpy())
                    Y_train2.extend(target2.detach().to('cpu').numpy())

                X_val, Y_val1, Y_val2 = [], [], []
                val_losses = []
                val_c_loss = []
                val_r_loss = []
                val_a_loss = []
                val_t_loss = []
                val_s_loss = []
                for v_i, (idx, p_idx, n_idx, target1, target2) in enumerate(val_loader):
                    (_, t0) = model.hidden_output(src[idx].to(device))
                    t0 = t0.detach().to('cpu').numpy()
                    X_val.extend(t0)
                    Y_val1.extend(target1.detach().to('cpu').numpy())
                    Y_val2.extend(target2.detach().to('cpu').numpy())
                    (preds, sub_preds, preds_adv, reconst, _, p0_anchor, mu1, mu2, logvar1, logvar2), (_, _, _, _, _, p0_pos, _, _, _, _), (_, _, _, _, _, p0_neg, _, _, _, _) = model.forward(src[idx].to(device)), model.forward(src[p_idx].to(device)), model.forward(src[n_idx].to(device))
                    val_loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                    val_loss_reconst = l_recon * criterion_vae(reconst.to(device), src[idx].to(device), mu1, mu2, logvar1, logvar2)
                    val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target1.to(device))
                    val_loss_classifier_sub = l_adv * criterion_classifier(preds_adv.to(device), target1.to(device))
                    val_loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device))
                    val_loss = val_loss_reconst + val_loss_classifier_main + val_loss_adv + val_loss_classifier_sub + val_loss_triplet
                        
                    val_losses.append(val_loss.item())
                    val_c_loss.append(val_loss_classifier_main.item())
                    val_r_loss.append(val_loss_reconst.item())
                    val_a_loss.append(val_loss_adv.item())
                    val_t_loss.append(val_loss_triplet.item())
                    val_s_loss.append(val_loss_classifier_sub.item())
                
                X_train = np.asarray(X_train)
                Y_train1 = np.asarray(Y_train1)
                Y_train2 = np.asarray(Y_train2)
                X_val = np.asarray(X_val)
                Y_val1 = np.asarray(Y_val1)
                Y_val2 = np.asarray(Y_val2)
                score_train, score_test = validate_linearclassifier(X_train, Y_train1, [X_val], [Y_val1])
                Scores_reg[0].append(score_train)
                Vals_reg[0].append(score_test[0])
                score_train, score_test = validate_linearclassifier(X_train, Y_train2, [X_val], [Y_val2])
                Scores_reg[1].append(score_train)
                Vals_reg[1].append(score_test[0])              

                print('epoch: {} val loss: {}'.format(epoch+1, np.mean(val_losses)))
                summary = scalars2summary(writer=writer,
                    tags=['Reg/Tar1Train', 'Reg/Tar1Val', 'Reg/Tar2Train', 'Reg/Tar2Val', 'loss/val_all', 'loss/val_classifier', 'loss/val_reconst', 'loss/val_adv', 'loss/val_triplet', 'loss/val_classifier_sub'], 
                    vals=[Scores_reg[0][-1], Vals_reg[0][-1], Scores_reg[1][-1], Vals_reg[1][-1], np.mean(val_losses), np.mean(val_c_loss), np.mean(val_r_loss), np.mean(val_a_loss), np.mean(val_t_loss), np.mean(val_s_loss)], epoch=epoch+1)

                # writer.add_scalar('val adv loss',
                #     np.mean(val_a_loss), epoch)

                if best_loss > np.mean(val_losses):
                    best_epoch = epoch + 1
                    best_loss = np.mean(val_losses)
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


def train_TDAE_VAE():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_VAE_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_VAE_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_VAE_colon'
        data_path = './data/colon_renew.hdf5'
    else:
        return
    if not(args.ex is None):
        out_source_dpath = os.path.join(out_source_dpath, args.ex)

    if args.rev:
        src, targets2, targets1 = get_flatted_data(data_path)
    else:
        src, targets1, targets2 = get_flatted_data(data_path)
    data_pairs = torch.utils.data.TensorDataset(src, targets1, targets2)

    model = TDAE_VAE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)

    if args.retrain:
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
    Scores_reg, Scores_reg_adv = [[], []], [[], []]
    Vals_reg = [[], []]
    n_epochs = args.epoch
    best_epoch = 0
    best_loss = np.inf
    l_adv, l_recon, l_tri, l_c = args.adv, args.rec, args.tri, args.classifier
    for epoch in range(n_epochs):
        accs_p, acc_t = [], []
        Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0
        Loss, RecLoss, CLoss, CLoss_sub, CSub = [], [], [], [], []
        for ite, (in_data, target, _) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            losses = []
            preds, sub_preds, preds_adv, reconst, mu1, mu2, logvar1, logvar2 = model.forward(in_data.to(device))
            loss_reconst = l_recon * criterion_vae(reconst.to(device), in_data.to(device), mu1, mu2, logvar1, logvar2)
            loss_reconst.backward(retain_graph=True)
            loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device))
            loss_adv.backward(retain_graph=True)
            model.classifiers[1].zero_grad()
            loss_classifier_sub = l_adv * criterion_classifier(preds_adv.to(device), target.to(device))
            loss_classifier_sub.backward(retain_graph=True)
            optimizer.step()
            loss = loss_classifier_main + loss_classifier_sub + loss_adv + loss_reconst

            Loss.append(loss.item())
            RecLoss.append(loss_reconst.item())
            CLoss.append(loss_classifier_main.item())
            CLoss_sub.append(loss_adv.item())
            CSub.append(loss_classifier_sub.item())
            
            y_true = target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, y_true)

        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}, Acc_adv: {}, sub Acc_adv: {}'.format(epoch+1, np.mean(Loss), Acc/len(train_set), sub_Acc/len(train_set), Acc_adv/len(train_set), sub_Acc_adv/len(train_set)))


        summary = scalars2summary(writer=writer,
                            tags=['loss/train_all', 'loss/train_rec', 'loss/train_classifier', 'loss/train_adv', 'loss/train_classifier_sub'], 
                            vals=[np.mean(Loss), np.mean(RecLoss), np.mean(CLoss), np.mean(CLoss_sub), np.mean(CSub)], epoch=epoch+1)
        # writer.add_scalar('summarize loss',
        #     np.mean(Loss), epoch)
        # writer.add_scalar('rec loss',
        #     np.mean(RecLoss), epoch)
        # writer.add_scalar('classifier loss',
        #     np.mean(CLoss), epoch)
        # writer.add_scalar('Adv loss',
        #     np.mean(CLoss_sub), epoch)
        
        if (epoch + 1) % args.step == 0:
            model.eval()
            with torch.no_grad():
                X_train = []
                Y_train1, Y_train2 = [], []
                for v_i, (in_data, target1, target2) in enumerate(train_loader):
                    if v_i == 0:
                        reconst = model.reconst(in_data.to(device))
                        np_input = in_data[0].detach().to('cpu')
                        np_reconst = reconst[0].detach().to('cpu')
                        img_grid = make_grid(torch.stack([np_input, np_reconst]))
                        writer.add_image('train example', img_grid, epoch+1)
                    (_, t0) = model.hidden_output(in_data.to(device))
                    t0 = t0.detach().to('cpu').numpy()
                    X_train.extend(t0)
                    Y_train1.extend(target1.detach().to('cpu').numpy())
                    Y_train2.extend(target2.detach().to('cpu').numpy())

                X_val, Y_val1, Y_val2 = [], [], []
                val_losses = []
                val_c_loss = []
                val_r_loss = []
                val_a_loss = []
                val_s_loss = []
                for v_i, (in_data, target1, target2) in enumerate(val_loader):
                    (_, t0) = model.hidden_output(in_data.to(device))
                    t0 = t0.detach().to('cpu').numpy()
                    X_val.extend(t0)
                    Y_val1.extend(target1.detach().to('cpu').numpy())
                    Y_val2.extend(target2.detach().to('cpu').numpy())

                    preds, sub_preds, preds_adv, reconst, mu1, mu2, logvar1, logvar2 = model.forward(in_data.to(device))
                    val_loss_reconst = l_recon * criterion_vae(reconst.to(device), in_data.to(device), mu1, mu2, logvar1, logvar2)
                    val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target1.to(device))
                    val_loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device))
                    val_loss_classifier_sub = l_adv * criterion_classifier(preds_adv.to(device), target1.to(device))
                    val_loss = val_loss_reconst + val_loss_classifier_main + val_loss_adv + val_loss_classifier_sub

                    val_losses.append(val_loss.item())
                    val_c_loss.append(val_loss_classifier_main.item())
                    val_r_loss.append(val_loss_reconst.item())
                    val_a_loss.append(val_loss_adv.item())
                    val_s_loss.append(val_loss_classifier_sub.item())
            
                X_train = np.asarray(X_train)
                Y_train1 = np.asarray(Y_train1)
                Y_train2 = np.asarray(Y_train2)
                X_val = np.asarray(X_val)
                Y_val1 = np.asarray(Y_val1)
                Y_val2 = np.asarray(Y_val2)

                score_train, score_test = validate_linearclassifier(X_train, Y_train1, [X_val], [Y_val1])
                Scores_reg[0].append(score_train)
                Vals_reg[0].append(score_test[0])
                score_train, score_test = validate_linearclassifier(X_train, Y_train2, [X_val], [Y_val2])
                Scores_reg[1].append(score_train)
                Vals_reg[1].append(score_test[0])

                summary = scalars2summary(writer=writer, 
                    tags=['Reg/Tar1Train', 'Reg/Tar1Val', 'Reg/Tar2Train', 'Reg/Tar2Val', 'loss/val_all', 'loss/val_classifier', 'loss/val_reconst', 'loss/val_adv', 'loss/val_classifier_sub'], 
                    vals=[Scores_reg[0][-1], Vals_reg[0][-1], Scores_reg[1][-1], Vals_reg[1][-1], np.mean(val_losses), np.mean(val_c_loss), np.mean(val_r_loss), np.mean(val_a_loss), np.mean(val_s_loss)], epoch=epoch+1)

                print('epoch: {} val loss: {}'.format(epoch+1, np.mean(val_losses)))
                writer.add_scalar('val loss',
                    np.mean(val_losses), epoch)
                writer.add_scalar('val classifier loss',
                    np.mean(val_c_loss), epoch)
                writer.add_scalar('val reconst loss',
                    np.mean(val_r_loss), epoch)
                writer.add_scalar('val adv loss',
                    np.mean(val_a_loss), epoch)

                if best_loss > np.mean(val_losses):
                    best_epoch = epoch + 1
                    best_loss = np.mean(val_losses)
                    torch.save(model.state_dict(), '{}/TDAE_test_bestparam.json'.format(out_param_dpath))

    torch.save(model.state_dict(), '{}/TDAE_test_param.json'.format(out_param_dpath))
    
    args.best_epoch = best_epoch
    df = args2pandas(args)
    df.to_csv('{}/condition.csv'.format(out_condition_dpath))
    
    writer.close()


def train_TDAE_VAE_fullsuper_disentangle():
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    
    # if not(args.ex is None):
    out_source_dpath = os.path.join(out_source_dpath, ex)

    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path)
    else:
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path)
    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)
        
    # data_pairs = torch.utils.data.TensorDataset(src, targets1, targets2)
    model = TDAE_VAE_fullsuper_disentangle(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)
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
    Scores_reg, Scores_reg_adv = [[], []], [[], []]
    Vals_reg = [[], []]
    n_epochs = args.epoch
    best_epoch = 0
    best_loss = np.inf
    l_adv, l_recon, l_tri, l_c = args.adv, args.rec, args.tri, args.classifier
    train_keys = ['loss/train_all', 'loss/train_classifier_main', 'loss/train_classifier_sub', 'loss/train_adv_main', 'loss/train_adv_sub', 'loss/train_no_grad_main', 'loss/train_no_grad_sub', 'loss/train_rec', 'loss/train_triplet']
    val_keys = ['loss/val_all', 'loss/val_classifier_main', 'loss/val_classifier_sub', 'loss/val_adv_main', 'loss/val_adv_sub', 'loss/val_no_grad_main', 'loss/val_no_grad_sub', 'loss/val_rec', 'loss/val_triplet']
    for epoch in range(n_epochs):
        accs_p, acc_t = [], []
        Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0
        loss_dict = {}
        for k in train_keys:
            loss_dict[k] = []
        # for ite, (in_data, target, sub_target) in enumerate(train_loader):
        for ite, (idx, p_idx, n_idx, target, sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            losses = []
            preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, reconst, mu1, mu2, logvar1, logvar2 = model.forward(src[idx].to(device))
            if args.triplet:
                _, _, _, _, _, _, _, p_mu1, p_mu2, _, _ = model.forward(src[p_idx].to(device))
                _, _, _, _, _, _, _, n_mu1, n_mu2, _, _ = model.forward(src[n_idx].to(device))
                # loss_triplet = l_tri * criterion_triplet(mu1, p_mu1, n_mu1)
                loss_triplet = l_tri * criterion_triplet(mu2, p_mu2, n_mu2)
                loss_triplet.backward(retain_graph=True)
                LossT.append(loss_triplet.item())
            else:
                loss_triplet = torch.Tensor([0])
                
            loss_reconst = l_recon * criterion_vae(reconst.to(device), in_data.to(device), mu1, mu2, logvar1, logvar2)
            loss_reconst.backward(retain_graph=True)
            loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            loss_classifier_sub = l_c * criterion_classifier(sub_preds.to(device), sub_target.to(device))
            loss_classifier_sub.backward(retain_graph=True)
            loss_adv_main = l_adv * negative_entropy_loss(preds_adv.to(device))
            loss_adv_sub = l_adv * negative_entropy_loss(sub_preds_adv.to(device))
            loss_adv_main.backward(retain_graph=True)
            loss_adv_sub.backward(retain_graph=True)
            model.disentangle_classifiers[0].zero_grad()
            model.disentangle_classifiers[1].zero_grad()
            loss_main_no_grad = l_adv * criterion_classifier(preds_adv_no_grad.to(device), sub_target.to(device))
            loss_main_no_grad.backward(retain_graph=True)
            loss_sub_no_grad = l_adv * criterion_classifier(sub_preds_adv_no_grad.to(device), target.to(device))
            loss_sub_no_grad.backward(retain_graph=True)
            optimizer.step()
            loss = loss_classifier_main + loss_classifier_sub + loss_adv_main + loss_adv_sub +  loss_reconst + loss_main_no_grad + loss_sub_no_grad + loss_train_triplet

            for k, val in zip(train_keys, [loss, loss_classifier_main, loss_classifier_sub, loss_adv_main, loss_adv_sub, loss_main_no_grad, loss_sub_no_grad, loss_reconst, loss_triplet]):
                loss_dict[k].append(val.item())
            
            y_true = target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, y_true)

        for k in loss_dict.keys():
            loss_dict[k] = np.mean(loss_dict[k])

        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}'.format(epoch+1, loss_dict['loss/train_all'], Acc/len(train_set), sub_Acc/len(train_set)))

        summary = scalars2summary(writer=writer,
                            tags=list(loss_dict.keys()), 
                            vals=list(loss_dict.values()), epoch=epoch+1)

        if (epoch + 1) % args.step == 0:
            model.eval()
            with torch.no_grad():
                for v_i, (in_data, target1, target2) in enumerate(train_loader):
                    if v_i == 0:
                        reconst = model.reconst(in_data.to(device))
                        np_input = in_data[0].detach().to('cpu')
                        np_reconst = reconst[0].detach().to('cpu')
                        img_grid = make_grid(torch.stack([np_input, np_reconst]))
                        writer.add_image('train example', img_grid, epoch+1)

                val_loss_dict = {}
                for k in val_keys:
                    val_loss_dict[k] = []

                for v_i, (idx, p_idx, n_idx, target, sub_target) in enumerate(val_loader):
                    preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, reconst,  mu1, mu2, logvar1, logvar2 = model.forward(src[idx].to(device))
                    if args.triplet:
                        _, _, _, _, _, _, _,  p_mu1, p_mu2, _, _ = model.forward(src[p_idx].to(device))
                        _, _, _, _, _, _, _,  n_mu1, n_mu2, _, _ = model.forward(src[n_idx].to(device))
                        # val_loss_triplet = criterion_triplet(mu1, p_mu1, n_mu1)
                        val_loss_triplet = criterion_triplet(mu2, p_mu2, n_mu2)
                    else:
                        val_loss_triplet = torch.Tensor([0])
                        
                    val_loss_reconst = l_recon * criterion_vae(reconst.to(device), in_data.to(device), mu1, mu2, logvar1, logvar2)
                    val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
                    val_loss_classifier_sub = l_c * criterion_classifier(sub_preds.to(device), sub_target.to(device))
                    val_loss_adv_main = l_adv * negative_entropy_loss(preds_adv.to(device))
                    val_loss_adv_sub = l_adv * negative_entropy_loss(sub_preds_adv.to(device))
                    val_loss_no_grad_main = l_adv * criterion_classifier(preds_adv_no_grad.to(device), sub_target.to(device))
                    val_loss_no_grad_sub = l_adv * criterion_classifier(sub_preds_adv_no_grad.to(device), target.to(device))
                    val_loss = val_loss_reconst + val_loss_classifier_main + val_loss_classifier_sub + val_loss_adv_main + val_loss_no_grad_main + val_loss_classifier_sub + val_loss_adv_sub + val_loss_no_grad_sub + val_loss_triplet

                    for k, val in zip(val_keys, [val_loss, val_loss_classifier_main, val_loss_classifier_sub, val_loss_adv_main, val_loss_adv_sub, val_loss_no_grad_main, val_loss_no_grad_sub, val_loss_reconst, val_loss_triplet]):
                        val_loss_dict[k].append(val.item())
                
                for k in val_loss_dict.keys():
                    val_loss_dict[k] = np.mean(val_loss_dict[k])

                summary = scalars2summary(writer=writer, 
                                        tags=list(val_loss_dict.keys()), 
                                        vals=list(val_loss_dict.values()), epoch=epoch+1)

                print('epoch: {} val loss: {}'.format(epoch+1, val_loss_dict['loss/val_all']))
                torch.save(model.state_dict(), '{}/TDAE_param_e{:04}.json'.format(out_param_dpath, epoch+1))
                    
                if best_loss > val_loss_dict['loss/val_all']:
                    best_epoch = epoch + 1
                    best_loss = val_loss_dict['loss/val_all']
                    torch.save(model.state_dict(), '{}/TDAE_test_bestparam.json'.format(out_param_dpath))

    torch.save(model.state_dict(), '{}/TDAE_test_param.json'.format(out_param_dpath))
    
    args.best_epoch = best_epoch
    df = args2pandas(args)
    df.to_csv('{}/condition.csv'.format(out_condition_dpath))
    
    writer.close()


def val_TDAE_VAE(zero_padding=False):
    torch.manual_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    args = argparses()
    
    # if args.ex is None:
    #     pass
    # else:
    out_source_dpath = out_source_dpath + '/' + ex
    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        # srcs, targets2, targets1 = get_flatted_data(data_path)
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path)
    else:
        # srcs, targets1, targets2 = get_flatted_data(data_path)
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path)
    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)
    if args.retrain:
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        # out_val_dpath = '{}/re_val_{}'.format(out_source_dpath, args.param)
        out_fig_dpath = '{}/re_fig_{}'.format(out_source_dpath, args.param)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        # out_val_dpath = '{}/val_{}'.format(out_source_dpath, args.param)
        out_fig_dpath = '{}/fig_{}'.format(out_source_dpath, args.param)
    clean_directory(out_fig_dpath)

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
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)
    
    # with torch.no_grad():
    #     model.eval()
    #     n_s = 50
    #     for first, loader in zip(['train', 'val'], [train_loader, val_loader]):
    #         for n_iter, (idx, _, _, _, _) in enumerate(val_loader):
    #             reconst = model.reconst(src[idx].to(device))
    #             s_reconst = model.shuffle_reconst(src[idx].to(device), idx1=[0, 1], idx2=[1, 0])
    #             np_input0 = src[idx][0].detach().to('cpu')
    #             np_input1 = src[idx][1].detach().to('cpu')
    #             np_reconst0 = reconst[0].detach().to('cpu')
    #             np_reconst1 = reconst[1].detach().to('cpu')
    #             s_np_reconst0 = s_reconst[0].detach().to('cpu')
    #             s_np_reconst1 = s_reconst[1].detach().to('cpu')
    #             pad0_reconst = model.fix_padding_reconst(src[idx].to(device), which_val=0, pad_val=0)
    #             pad0_np_reconst0 = pad0_reconst[0].detach().to('cpu')
    #             pad0_np_reconst1 = pad0_reconst[1].detach().to('cpu')
    #             pad1_reconst = model.fix_padding_reconst(src[idx].to(device), which_val=1, pad_val=0)
    #             pad1_np_reconst0 = pad1_reconst[0].detach().to('cpu')
    #             pad1_np_reconst1 = pad1_reconst[1].detach().to('cpu')
    #             fig = plt.figure(figsize=(16*4, 9*2))
    #             ax = fig.add_subplot(2, 5, 1)
    #             ax.set_title('1')
    #             ax.imshow(np.transpose(np_input0, (1,2,0)))
    #             ax = fig.add_subplot(2, 5, 2)
    #             ax.set_title('1')
    #             ax.imshow(np.transpose(np_reconst0, (1,2,0)))
    #             ax = fig.add_subplot(2, 5, 3)
    #             ax.set_title('1')
    #             ax.imshow(np.transpose(s_np_reconst0, (1,2,0)))
    #             ax = fig.add_subplot(2, 5, 4)
    #             ax.set_title('1')
    #             ax.imshow(np.transpose(pad0_np_reconst0, (1,2,0)))
    #             ax = fig.add_subplot(2, 5, 5)
    #             ax.set_title('1')
    #             ax.imshow(np.transpose(pad1_np_reconst0, (1,2,0)))
    #             ax = fig.add_subplot(2, 5, 6)
    #             ax.set_title('2')
    #             ax.imshow(np.transpose(np_input1, (1,2,0)))
    #             ax = fig.add_subplot(2, 5, 7)
    #             ax.set_title('2')
    #             ax.imshow(np.transpose(np_reconst1, (1,2,0)))
    #             ax = fig.add_subplot(2, 5, 8)
    #             ax.set_title('2')
    #             ax.imshow(np.transpose(s_np_reconst1, (1,2,0)))
    #             ax = fig.add_subplot(2, 5, 9)
    #             ax.set_title('2')
    #             ax.imshow(np.transpose(pad0_np_reconst1, (1,2,0)))
    #             ax = fig.add_subplot(2, 5, 10)
    #             ax.set_title('2')
    #             ax.imshow(np.transpose(pad1_np_reconst1, (1,2,0)))
    #             fig.savefig('{}/{}_sample{:04d}.png'.format(out_val_dpath, first, n_iter))
    #             plt.close(fig)
    #             if n_iter >= n_s:
    #                 break
    
    markers = ['.', 'x']
    colors1 = ['blue', 'orange']
    colors2 = ['r', 'g']
    if args.rev:
        buff = colors1
        colors1 = colors2
        colors2 = buff
        
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
            for (X, ex) in zip([X1, X2], ['main', 'sub']):
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
                # df = pd.DataFrame(pca_feature[:, pca_idx[0]+1:pca_idx[0]+6], columns=cs)
                for tag, Y, co in zip(['Y1', 'Y2'], [Y1, Y2], [colors1, colors2]):
                    tar = ['Class{}'.format(y) for y in Y]
                    df['target'] = tar
                    palette_dict = {}
                    for t, c in zip(np.unique(tar), co):
                        palette_dict[t] = c
                    sns.pairplot(df, hue='target', diag_kind='hist', vars=cs, palette=palette_dict).savefig('{}/{}_PairPlot_{}_{}.png'.format(out_fig_dpath, first, ex, tag))
                    
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


def test_TDAE_VAE_fullsuper_disentangle():
    img_w, img_h, out_source_dpath, data_path = get_outputpath()
    args = argparses()

    if args.ex is None:
        pass
    else:
        out_source_dpath = out_source_dpath + '/' + args.ex

    if args.retrain:
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_test_dpath = '{}/re_test_{}'.format(out_source_dpath, args.param)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_test_dpath = '{}/test_{}'.format(out_source_dpath, args.param)
    clean_directory(out_test_dpath)

    d2ae_flag = False
    if args.rev:
        srcs, targets2, targets1 = get_flatted_data(data_path)
    else:
        srcs, targets1, targets2 = get_flatted_data(data_path)
    data_pairs = torch.utils.data.TensorDataset(srcs, targets1, targets2)
    
    # model = TDAE_out(n_class1=torch.unique(targets1).size(0), n_class2=5, d2ae_flag = d2ae_flag, img_h=img_h, img_w=img_w)
    model = TDAE_VAE_fullsuper_disentangle(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)

    if args.param == 'best':
        model.load_state_dict(torch.load('{}/TDAE_test_bestparam.json'.format(out_param_dpath)))
    else:
        model.load_state_dict(torch.load('{}/TDAE_test_param.json'.format(out_param_dpath)))
    model = model.to(device)

    ratio = [0.7, 0.2, 0.1]
    n_sample = len(data_pairs)
    train_size = int(n_sample*ratio[0])
    val_size = int(n_sample*ratio[1])
    test_size = n_sample - train_size - val_size
    
    # train_set, val_set = torch.utils.data.random_split(data_pairs, [train_size, val_size])
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))
    test_indices = list(range(train_size+val_size, n_sample))

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_set = torch.utils.data.dataset.Subset(data_pairs, test_indices)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    with torch.no_grad():
        model.eval()
        X1, X2, Y1, Y2 = [], [], [], []
        for loader in [train_loader, val_loader, test_loader]:
            X_train1, X_train2, Y_train1, Y_train2 = [], [], [], []
            for n_iter, (inputs, targets1, targets2) in enumerate(loader):
                (h0, t0) = model.hidden_output(inputs.to(device))
                h0 = h0.detach().to('cpu').numpy()
                t0 = t0.detach().to('cpu').numpy()
                X_train1.extend(h0)
                X_train2.extend(t0)
                Y_train1.extend(targets1.detach().to('cpu').numpy())
                Y_train2.extend(targets2.detach().to('cpu').numpy())
    
            X_train1 = np.asarray(X_train1)
            X1.append(X_train1)
            X_train2 = np.asarray(X_train2)
            X2.append(X_train2)
            Y_train1 = np.asarray(Y_train1)
            Y1.append(Y_train1)
            Y_train2 = np.asarray(Y_train2)
            Y2.append(Y_train2)
        
        logreg = LogisticRegression(penalty='l2', solver="sag")
        linear_svc = LinearSVC()
        tag = ['main2main', 'main2sub', 'sub2main', 'sub2sub']
        i = 0
        score_dict = {}

        for X, Y in itertools.product([X1, X2], [Y1, Y2]):
            logreg.fit(X[0], Y[0])
            score_reg = logreg.score(X[0], Y[0])
            print(tag[i])
            print('train-------------------------------')
            score_dict[tag[i]] = [score_reg]
            print(score_reg)
            l = logreg.predict_proba(X[0])
            p = np.argmax(l, axis=1)
            p0 = []
            # for u in np.unique(Y):
                # print(u, ':', np.sum(Y[1]==u), np.sum(Y[2]==u))
                # p0.append(p[Y2[0]==u])
            # print(np.sum(Y2[0]==0), np.sum(Y2[0]==1))
            # for p00, u in zip(p0, np.unique(Y)):
            #     print(np.sum(p0==u))
            score_reg = logreg.score(X[1], Y[1])
            print('val-------------------------------')
            print(score_reg)
            score_dict[tag[i]].append(score_reg)
            score_reg = logreg.score(X[2], Y[2])
            print('test-------------------------------')
            print(score_reg)
            score_dict[tag[i]].append(score_reg)
            i += 1
        df = pd.DataFrame.from_dict(score_dict)
        df.to_csv('{}/LinearReg.csv'.format(out_test_dpath))


def test_TDAE_VAE():
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    args = argparses()
    # if args.ex is None:
    #     pass
    # else:
    out_source_dpath = out_source_dpath + '/' + ex
    if args.rev:
        out_source_dpath = out_source_dpath + '_rev'
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path)
    else:
        src, targets1, targets2, idxs = get_triplet_flatted_data_with_idx(data_path)
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
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_set = torch.utils.data.dataset.Subset(data_pairs, test_indices)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    X1_dict = {}
    X2_dict = {}
    Y1_dict = {}
    Y2_dict = {}
    with torch.no_grad():
        for k, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
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
    
            X1_dict[k] = np.asarray(X1)
            X2_dict[k] = np.asarray(X2)
            Y1_dict[k] = np.asarray(Y1)
            Y2_dict[k] = np.asarray(Y2)
        
    tag = ['main2main', 'main2sub', 'sub2main', 'sub2sub']
    score_dict = {}
    for itag, (X_dict, Y_dict) in enumerate(itertools.product([X1_dict, X2_dict], [Y1_dict, Y2_dict])):
        for k in ['train', 'val', 'test']:
            if k == 'train':
                logreg = LogisticRegression(penalty='l2', solver="sag")
                logreg.fit(X_dict[k], Y_dict[k])
                score_reg = logreg.score(X_dict[k], Y_dict[k])
                score_dict[tag[itag]] = [score_reg]
            else:
                score_reg = logreg.score(X_dict[k], Y_dict[k])
                score_dict[tag[itag]].append(score_reg)
            if tag[itag] == 'sub2main':
                print(logreg.predict_proba(X_dict[k]))
            print(score_dict)
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
        # srcs, targets2, targets1 = get_flatted_data(data_path)
        src, targets2, targets1, idxs = get_triplet_flatted_data_with_idx(data_path)
    else:
        # srcs, targets1, targets2 = get_flatted_data(data_path)
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
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    
    markers = ['.', 'x']
    colors1 = ['blue', 'orange', 'purple']
    colors2 = ['r', 'g']
    if args.rev:
        buff = colors1
        colors1 = colors2
        colors2 = buff
        
    with torch.no_grad():
        for iters in range(50, 350, 50):
            model.load_state_dict(torch.load('{}/TDAE_param_e{:04}.json'.format(out_param_dpath, iters)))
            model.to(device)
            for first, loader in zip(['train'], [train_loader]):
                model.eval()
                X1, X2, Y1, Y2 = [], [], [], []
                # for n_iter, (inputs, targets1, targets2) in enumerate(train_loader):
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
        # val_TDAE_VAE_fullsuper_disentangle()
        test_TDAE_VAE()
        # test_TDAE_VAE_fullsuper_disentangle()
        return

    if args.mode == 'val':
        val_TDAE_VAE()
        return
    elif args.mode == 'test':
        test_TDAE_VAE()
        return
    elif args.mode == 'train':
        train_TDAE_VAE_v2()
        return
    elif args.mode == 'conf':
        confirm_seq()
        return
    print('call train')
    train_TDAE_VAE_v2()
    print('call val')
    val_TDAE_VAE()
    print('call test')
    test_TDAE_VAE()


if __name__ == '__main__':
    # with SetIO('./out.log'):
    main()
