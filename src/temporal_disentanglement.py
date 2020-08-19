import os
import sys
import copy
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
from archs import TDAE_out, TDAE, base_classifier, TDAE_VAE
from archs import TestNet as TDAE_D2AE
from archs import TestNet_v2 as TDAE_D2AE_v2

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
    parser.add_argument('--epoch', type=int, default=300)
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
    parser.add_argument('--classifier', type=float, default=1e-0)
    parser.add_argument('--rec', type=float, default=1e-0)
    parser.add_argument('--adv', type=float, default=1e-1)
    parser.add_argument('--tri', type=float, default=1e-2)
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--triplet', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--fou', action='store_true')
    parser.add_argument('--d2ae', action='store_true')
    parser.add_argument('--rev', action='store_true')
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


def test_classifier():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq'
        data_path='data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy'
        data_path='data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon' 
        data_path='data/colon_renew.hdf5'
    else:
        return

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
    
    model = base_classifier(n_class=torch.unique(targets1).size(0), img_h=img_h, img_w=img_w)
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
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))
    test_indices = list(range(train_size+val_size, n_sample))

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_set = torch.utils.data.dataset.Subset(data_pairs, test_indices)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    score_dict = {}
    with torch.no_grad():
        model.eval()
        for tag, loader, siz in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader], [train_size, val_size, test_size]):
            Acc = 0
            for iter, (in_data, target, _) in enumerate(loader):
                preds = model(in_data.to(device))
                y_true = target.to('cpu')
                preds = preds.detach().to('cpu')
                Acc += true_positive_multiclass(preds, y_true)
            score_dict[tag] = [Acc/siz]
    df = pd.DataFrame.from_dict(score_dict)
    df.to_csv('{}/Score.csv'.format(out_test_dpath))


def train_classifier():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon'
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

    # if args.dlim > 0:
    #     data_pairs = torch.utils.data.TensorDataset(srcs[0][:args.dlim], srcs[1][:args.dlim], srcs[2][:args.dlim], targets1[:args.dlim], targets2[:args.dlim])
    
    model = base_classifier(n_class=torch.unique(targets1).size(0), img_h=img_h, img_w=img_w)

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
    if args.multi:
        g_list = [i for i in range(args.ngpus)]
        model = nn.DataParallel(model, device_ids=g_list)
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
    print(len(train_loader))
    # criterion_adv = nn.NLLLoss()
    criterion_classifier = nn.CrossEntropyLoss()
    params = list(model.parameters())
    optimizer = optim.Adam(params)
    n_epochs = args.epoch
    best_epoch = 0
    best_loss = np.inf
    for epoch in range(n_epochs):
        Loss = []
        for ite, (in_data, target, _) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            preds = model.forward(in_data.to(device))
            loss_classifier_main = criterion_classifier(preds.to(device), target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            optimizer.step()
            loss = loss_classifier_main
            Loss.append(loss.item())
            
        print('epoch: {} loss: {}'.format(epoch+1, np.mean(Loss)))

        summary = scalars2summary(writer=writer,
                            tags=['loss/train_all'], 
                            vals=[np.mean(Loss)], epoch=epoch+1)
        
        if (epoch + 1) % args.step == 0:
            model.eval()
            with torch.no_grad():
                X_val, Y_val1, Y_val2 = [], [], []
                val_losses = []
                for v_i, (in_data, target1, target2) in enumerate(val_loader):
                    preds = model.forward(in_data.to(device))
                    val_loss_classifier_main = criterion_classifier(preds.to(device), target1.to(device))
                    val_loss = val_loss_classifier_main
                    val_losses.append(val_loss.item())

                summary = scalars2summary(writer=writer, 
                    tags=['loss/val_all'], 
                    vals=[np.mean(val_losses)], epoch=epoch+1)

                print('epoch: {} val loss: {}'.format(epoch+1, np.mean(val_losses)))

                if best_loss > np.mean(val_losses):
                    best_epoch = epoch + 1
                    best_loss = np.mean(val_losses)
                    if args.multi:
                        torch.save(model.module.state_dict(), '{}/TDAE_test_bestparam.json'.format(out_param_dpath))
                    else:
                        torch.save(model.state_dict(), '{}/TDAE_test_bestparam.json'.format(out_param_dpath))
    if args.multi:
        torch.save(model.module.state_dict(), '{}/TDAE_test_param.json'.format(out_param_dpath))
    else:
        torch.save(model.state_dict(), '{}/TDAE_test_param.json'.format(out_param_dpath))
    args.best_epoch = best_epoch
    df = args2pandas(args)
    df.to_csv('{}/condition.csv'.format(out_condition_dpath))
    
    writer.close()


def triplet_train_TDAE():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon'
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

    model = TDAE_D2AE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)

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

    params = list(model.parameters())
    params_adv = list(model.classifiers[1].parameters())
    optim_adv = optim.Adam(params_adv)
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
        for ite, (idx, p_idx, n_idx, target, sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            losses = []
            if args.d2ae:
                (preds, sub_preds, preds_adv, reconst, _, p0_anchor), (_, _, _, _, _, p0_pos), (_, _, _, _, _, p0_neg) = model(src[idx].to(device)), model(src[p_idx].to(device)), model(src[n_idx].to(device))
                # (_, p0_anchor), (_, p0_pos), (_, p0_neg) = model.hidden_output(src[idx].to(device)), model.hidden_output(src[p_idx].to(device)), model.hidden_output(src[n_idx].to(device))
                loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                loss_triplet.backward(retain_graph=True)
                loss_reconst = l_recon * criterion_reconst(reconst.to(device), src[idx].to(device))
                loss_reconst.backward(retain_graph=True)
                loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
                loss_classifier_main.backward(retain_graph=True)
                loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device).to(device))
                loss_adv.backward(retain_graph=True)
                model.classifiers[1].zero_grad()
                loss_classifier_sub = l_adv * criterion_classifier(preds_adv.to(device), target.to(device))
                loss_classifier_sub.backward(retain_graph=True)
                optimizer.step()
                loss = loss_classifier_main + loss_classifier_sub + loss_adv + loss_reconst + loss_triplet
                
            else:
                (_, p0_anchor), (_, p0_pos), (_, p0_neg) = model.hidden_output(src[idx].to(device)), model.hidden_output(src[p_idx].to(device)), model.hidden_output(src[n_idx].to(device))
                loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                loss_triplet.backward(retain_graph=True)
                losses.append(loss_triplet)

                preds, preds_adv, reconst = model(src[idx].to(device))
                loss_reconst = l_recon * criterion_reconst(reconst.to(device), src[idx].to(device))
                loss_reconst.backward(retain_graph=True)
                losses.append(loss_reconst)
                
                loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                loss_adv.backward(retain_graph=True)
                model.classifiers[0].zero_grad()
                losses.append(loss_adv)
                
                loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
                loss_classifier_main.backward(retain_graph=True)
                losses.append(loss_classifier_main)

                optimizer.step()
                loss = 0
                for cat_loss in losses:
                    loss += cat_loss

            Loss.append(loss.item())
            RecLoss.append(loss_reconst.item())
            CLoss.append(loss_classifier_main.item())
            CLoss_sub.append(loss_adv.item())
            CSub.append(loss_classifier_sub.item())
            TriLoss.append(loss_triplet.item())
            
            y_true = target.to('cpu')
            sub_y_true = sub_target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, y_true)

        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}, Acc_adv: {}, sub Acc_adv: {}'.format(epoch+1, np.mean(Loss), Acc/len(train_set), sub_Acc/len(train_set), Acc_adv/len(train_set), sub_Acc_adv/len(train_set)))
        summary = scalars2summary(writer=writer, tags=['loss/train_all', 'loss/train_rec', 'loss/train_classifier', 'loss/train_adv', 'loss/train_triplet', 'loss/train_classifier_sub'], vals=[np.mean(Loss), np.mean(RecLoss), np.mean(CLoss), np.mean(CLoss_sub), np.mean(TriLoss), np.mean(CSub)], epoch=epoch+1)
        # writer.add_summary(summary, epoch+1)
        # writer.add_scalar('summarize loss',
        #     np.mean(Loss), epoch)
        # writer.add_scalar('rec loss',
        #     np.mean(RecLoss), epoch)
        # writer.add_scalar('classifier loss',
        #     np.mean(CLoss), epoch)
        # writer.add_scalar('Adv loss',
        #     np.mean(CLoss_sub), epoch)
        # writer.add_scalar('Triplet loss',
        #     np.mean(TriLoss), epoch)
        
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
                    if args.d2ae:
                        (preds, sub_preds, preds_adv, reconst, _, p0_anchor), (_, _, _, _, _, p0_pos), (_, _, _, _, _, p0_neg) = model(src[idx].to(device)), model(src[p_idx].to(device)), model(src[n_idx].to(device))
                        # (_, p0_anchor), (_, p0_pos), (_, p0_neg) = model.hidden_output(src[idx].to(device)), model.hidden_output(src[p_idx].to(device)), model.hidden_output(src[n_idx].to(device))
                        val_loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                        # preds, sub_preds_adv, sub_preds, reconst = model(src[idx].to(device))
                        val_loss_reconst = l_recon * criterion_reconst(reconst.to(device), src[idx].to(device))
                        val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target1.to(device))
                        val_loss_classifier_sub = l_adv * criterion_classifier(preds_adv.to(device), target1.to(device))
                        val_loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device))
                        val_loss = val_loss_reconst + val_loss_classifier_main + val_loss_adv + val_loss_classifier_sub + val_loss_triplet
                    else:
                        (_, p0_anchor), (_, p0_pos), (_, p0_neg) = model.hidden_output(src[idx].to(device)), model.hidden_output(src[p_idx].to(device)), model.hidden_output(src[n_idx].to(device))
                        val_loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                        preds, preds_adv, reconst = model(src[idx].to(device))
                        val_loss_reconst = l_recon * criterion_reconst(reconst.to(device), src[idx].to(device))
                        val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target1.to(device))
                        val_loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                        val_loss = val_loss_classifier_main + val_loss_reconst + val_loss_triplet + val_loss_adv
                        
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
    
    
def triplet_train_TDAE_fullsuper():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon'
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

    if args.d2ae:
        model = TDAE_D2AE_v2(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels, triplet=args.triplet)

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

    params = list(model.parameters())
    params_adv = list(model.classifiers[1].parameters())
    optim_adv = optim.Adam(params_adv)
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
        Loss, RecLoss, CLoss, CLoss_sub, TriLoss = [], [], [], [], []
        for ite, (idx, p_idx, n_idx, target, sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            losses = []
            (preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, reconst, _, p0_anchor), (_, _, _, _, _, p0_pos), (_, _, _, _, _, p0_neg) = model.forward(src[idx].to(device)), model.forward(src[p_idx].to(device)), model.forward(src[n_idx].to(device))
            loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
            loss_triplet.backward(retain_graph=True)
            loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            loss_classifier_sub = l_c * criterion_classifier(sub_preds.to(device), sub_target.to(device))
            loss_classifier_sub.backward(retain_graph=True)
            
            loss_reconst = l_recon * criterion_reconst(reconst.to(device), src[idx].to(device))
            loss_reconst.backward(retain_graph=True)
            
            loss_sub_adv = l_adv * negative_entropy_loss(sub_preds_adv.to(device).to(device))
            loss_sub_adv.backward(retain_graph=True)
            loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device).to(device))
            loss_adv.backward(retain_graph=True)
            if args.multi:
                for n_net in range(2):
                    model.module.disentangle_classifiers[n_net].zero_grad()
            else:
                for n_net in range(2):
                    model.disentangle_classifiers[n_net].zero_grad()
                    
            loss_classifier_main_no_grad = l_adv * criterion_classifier(preds_adv_no_grad.to(device), sub_target.to(device))
            loss_classifier_main_no_grad.backward(retain_graph=True)
            loss_classifier_sub_no_grad = l_adv * criterion_classifier(sub_preds_adv_no_grad.to(device), target.to(device))
            loss_classifier_sub_no_grad.backward(retain_graph=True)
            optimizer.step()
            loss = loss_classifier_main + loss_classifier_sub + loss_adv + loss_reconst + loss_classifier_sub_no_grad + loss_classifier_main_no_grad + loss_sub_adv + loss_triplet

            Loss.append(loss.item())
            RecLoss.append(loss_reconst.item())
            CLoss.append(loss_classifier_main.item())
            CLoss_sub.append(loss_adv.item())
            TriLoss.append(loss_triplet.item())
            
            y_true = target.to('cpu')
            sub_y_true = sub_target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, y_true)

        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}, Acc_adv: {}, sub Acc_adv: {}'.format(epoch+1, np.mean(Loss), Acc/len(train_set), sub_Acc/len(train_set), Acc_adv/len(train_set), sub_Acc_adv/len(train_set)))
        summary = scalars2summary(writer=writer, tags=['loss/train_all', 'loss/train_rec', 'loss/train_classifier', 'loss/train_adv', 'loss/train_triplet'], vals=[np.mean(Loss), np.mean(RecLoss), np.mean(CLoss), np.mean(CLoss_sub), np.mean(TriLoss)], epoch=epoch+1)
        # writer.add_summary(summary, epoch+1)
        # writer.add_scalar('summarize loss',
        #     np.mean(Loss), epoch)
        # writer.add_scalar('rec loss',
        #     np.mean(RecLoss), epoch)
        # writer.add_scalar('classifier loss',
        #     np.mean(CLoss), epoch)
        # writer.add_scalar('Adv loss',
        #     np.mean(CLoss_sub), epoch)
        # writer.add_scalar('Triplet loss',
        #     np.mean(TriLoss), epoch)
        
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
                for v_i, (idx, p_idx, n_idx, target1, target2) in enumerate(val_loader):
                    (_, t0) = model.hidden_output(src[idx].to(device))
                    t0 = t0.detach().to('cpu').numpy()
                    X_val.extend(t0)
                    Y_val1.extend(target1.detach().to('cpu').numpy())
                    Y_val2.extend(target2.detach().to('cpu').numpy())
                    if args.d2ae:
                        (preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, reconst, _, p0_anchor), (_, _, _, _, _, p0_pos), (_, _, _, _, _, p0_neg) = model.forward(src[idx].to(device)), model.forward(src[p_idx].to(device)), model.forward(src[n_idx].to(device))
                        val_loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                        (preds, preds_adv, sub_preds, sub_preds_adv, reconst, _, p0_anchor) = model.forward(src[idx].to(device))
                        preds, sub_preds_adv, sub_preds, reconst = model(src[idx].to(device))
                        val_loss_reconst = l_recon * criterion_reconst(reconst.to(device), src[idx].to(device))
                        
                        val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target1.to(device))
                        val_loss_classifier_sub = l_c * criterion_classifier(sub_preds.to(device), target2.to(device))
                        val_loss_adv = l_adv * negative_entropy_loss(sub_preds_adv.to(device))
                        val_loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                        val_loss_classifier_main_no_grad = l_adv * criterion_classifier(preds_adv_no_grad.to(device), target2.to(device))
                        val_loss_classifier_sub_no_grad = l_adv * criterion_classifier(sub_preds_adv_no_grad.to(device), target2.to(device))
                        val_loss = val_loss_reconst + val_loss_classifier_main + val_loss_adv + val_loss_classifier_sub + val_loss_triplet + val_loss_classifier_main_no_grad + val_loss_classifier_sub_no_grad
                    else:
                        (_, p0_anchor), (_, p0_pos), (_, p0_neg) = model.hidden_output(src[idx].to(device)), model.hidden_output(src[p_idx].to(device)), model.hidden_output(src[n_idx].to(device))
                        val_loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                        preds, preds_adv, reconst = model(src[idx].to(device))
                        val_loss_reconst = l_recon * criterion_reconst(reconst.to(device), src[idx].to(device))
                        val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target1.to(device))
                        val_loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                        val_loss = val_loss_classifier_main + val_loss_reconst + val_loss_triplet + val_loss_adv
                        
                    val_losses.append(val_loss.item())
                    val_c_loss.append(val_loss_classifier_main.item())
                    val_r_loss.append(val_loss_reconst.item())
                    val_a_loss.append(val_loss_adv.item())
                    val_t_loss.append(val_loss_triplet.item())
                
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
                    tags=['Reg/Tar1Train', 'Reg/Tar1Val', 'Reg/Tar2Train', 'Reg/Tar2Val', 'loss/val_all', 'loss/val_classifier', 'loss/val_reconst', 'loss/val_adv', 'loss/val_triplet'], 
                    vals=[Scores_reg[0][-1], Vals_reg[0][-1], Scores_reg[1][-1], Vals_reg[1][-1], np.mean(val_losses), np.mean(val_c_loss), np.mean(val_r_loss), np.mean(val_a_loss), np.mean(val_t_loss)], epoch=epoch+1)

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
    
    
def train_TDAE_v2():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon'
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

    # if args.dlim > 0:
    #     data_pairs = torch.utils.data.TensorDataset(srcs[0][:args.dlim], srcs[1][:args.dlim], srcs[2][:args.dlim], targets1[:args.dlim], targets2[:args.dlim])
    if args.d2ae:
        model = TDAE_D2AE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)
    else:
        model = TDAE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)

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

    params = list(model.parameters())
    params_adv = list(model.classifiers[1].parameters())
    optim_adv = optim.Adam(params_adv)
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
            if args.d2ae:
                preds, sub_preds, preds_adv, reconst = model.forward(in_data.to(device))
                loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                loss_reconst.backward(retain_graph=True)
                loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
                loss_classifier_main.backward(retain_graph=True)
                loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device).to(device))
                loss_adv.backward(retain_graph=True)
                model.classifiers[1].zero_grad()
                loss_classifier_sub = l_adv * criterion_classifier(preds_adv.to(device), target.to(device))
                loss_classifier_sub.backward(retain_graph=True)
                optimizer.step()
                loss = loss_classifier_main + loss_classifier_sub + loss_adv + loss_reconst
                
            else:
                preds, preds_adv, reconst = model(in_data.to(device))
                loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                loss_reconst.backward(retain_graph=True)
                losses.append(loss_reconst)
                
                loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                loss_adv.backward(retain_graph=True)
                model.classifiers[0].zero_grad()
                losses.append(loss_adv)
                
                loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
                loss_classifier_main.backward(retain_graph=True)
                losses.append(loss_classifier_main)

                optimizer.step()
                loss = 0
                for cat_loss in losses:
                    loss += cat_loss

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

                    if args.d2ae:
                        preds, sub_preds, preds_adv, reconst = model.forward(in_data.to(device))
                        val_loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                        val_loss_classifier_main = l_adv * criterion_classifier(preds.to(device), target1.to(device))
                        val_loss_classifier_sub = l_adv * criterion_classifier(preds_adv.to(device), target1.to(device))
                        val_loss_adv = l_adv * negative_entropy_loss(sub_preds.to(device))
                        val_loss = val_loss_reconst + val_loss_classifier_main + val_loss_adv + val_loss_classifier_sub
                    else:
                        preds, preds_adv, reconst = model(in_data.to(device))
                        val_loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                        val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target1.to(device))
                        val_loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                        val_loss = val_loss_classifier_main + val_loss_reconst + val_loss_adv
                        
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


def train_TDAE_fullsuper():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon'
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

    # if args.dlim > 0:
    #     data_pairs = torch.utils.data.TensorDataset(srcs[0][:args.dlim], srcs[1][:args.dlim], srcs[2][:args.dlim], targets1[:args.dlim], targets2[:args.dlim])
    
    if args.d2ae:
        model = TDAE_D2AE_v2(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)
    else:
        model = TDAE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)

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

    params = list(model.parameters())
    params_adv = list(model.classifiers[1].parameters())
    optim_adv = optim.Adam(params_adv)
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
        Loss, RecLoss, CLoss, CLoss_sub = [], [], [], []
        CLoss_adv, CLoss_sub_adv = [], []
        for ite, (in_data, target, sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            losses = []
            if args.d2ae:
                preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, reconst = model.forward(in_data.to(device))
                loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                loss_reconst.backward(retain_graph=True)
                loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
                loss_classifier_main.backward(retain_graph=True)
                loss_classifier_sub = l_c * criterion_classifier(sub_preds.to(device), sub_target.to(device))
                loss_classifier_sub.backward(retain_graph=True)
                
                loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                loss_adv.backward(retain_graph=True)
                loss_sub_adv = l_adv * negative_entropy_loss(sub_preds_adv.to(device))
                loss_sub_adv.backward(retain_graph=True)
                
                model.disentangle_classifiers[0].zero_grad()
                model.disentangle_classifiers[1].zero_grad()
                loss_classifier_main_no_grad = l_adv * criterion_classifier(preds_adv_no_grad.to(device), sub_target.to(device))
                loss_classifier_main_no_grad.backward(retain_graph=True)
                loss_classifier_sub_no_grad = l_adv * criterion_classifier(sub_preds_adv_no_grad.to(device), target.to(device))
                loss_classifier_sub_no_grad.backward(retain_graph=True)
                optimizer.step()
                loss = loss_classifier_main + loss_classifier_sub + loss_adv + loss_reconst + loss_classifier_sub_no_grad + loss_classifier_main_no_grad + loss_sub_adv

            Loss.append(loss.item())
            RecLoss.append(loss_reconst.item())
            CLoss.append(loss_classifier_main.item())
            CLoss_sub.append(loss_classifier_sub.item())
            CLoss_adv.append(loss_adv.item())
            CLoss_sub_adv.append(loss_sub_adv.item())
            
            y_true = target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, y_true)

        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}, Acc_adv: {}, sub Acc_adv: {}'.format(epoch+1, np.mean(Loss), Acc/len(train_set), sub_Acc/len(train_set), Acc_adv/len(train_set), sub_Acc_adv/len(train_set)))


        summary = scalars2summary(writer=writer,
                            tags=['loss/train_all', 'loss/train_rec', 'loss/train_classifier', 'loss/train_sub', 'loss/train_main_adv', 'loss/train_sub_adv'], 
                            vals=[np.mean(Loss), np.mean(RecLoss), np.mean(CLoss), np.mean(CLoss_sub), np.mean(CLoss_adv), np.mean(CLoss_sub_adv)], epoch=epoch+1)
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
                for v_i, (in_data, target1, target2) in enumerate(val_loader):
                    (_, t0) = model.hidden_output(in_data.to(device))
                    t0 = t0.detach().to('cpu').numpy()
                    X_val.extend(t0)
                    Y_val1.extend(target1.detach().to('cpu').numpy())
                    Y_val2.extend(target2.detach().to('cpu').numpy())

                    if args.d2ae:
                        preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, reconst = model.forward(in_data.to(device))
                        val_loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                        val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target1.to(device))
                        val_loss_classifier_sub = l_c * criterion_classifier(sub_preds.to(device), target2.to(device))
                        
                        val_loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                        val_loss_sub_adv = l_adv * negative_entropy_loss(sub_preds_adv.to(device))
                        val_loss_classifier_main_no_grad = l_adv * criterion_classifier(preds_adv_no_grad.to(device), target2.to(device))
                        val_loss_classifier_sub_no_grad = l_adv * criterion_classifier(sub_preds_adv_no_grad.to(device), target1.to(device))
                        
                        val_loss = val_loss_reconst + val_loss_classifier_main + val_loss_classifier_sub + val_loss_adv + val_loss_sub_adv +  val_loss_classifier_main_no_grad + val_loss_classifier_sub_no_grad
                        
                    val_losses.append(val_loss.item())
                    val_c_loss.append(val_loss_classifier_main.item())
                    val_r_loss.append(val_loss_reconst.item())
                    val_a_loss.append(val_loss_adv.item())
                
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
                    tags=['Reg/Tar1Train', 'Reg/Tar1Val', 'Reg/Tar2Train', 'Reg/Tar2Val', 'loss/val_all', 'loss/val_classifier', 'loss/val_reconst', 'loss/val_adv'], 
                    vals=[Scores_reg[0][-1], Vals_reg[0][-1], Scores_reg[1][-1], Vals_reg[1][-1], np.mean(val_losses), np.mean(val_c_loss), np.mean(val_r_loss), np.mean(val_a_loss)], epoch=epoch+1)

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


def train_TDAE():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon'
        data_path = './data/colon_renew.hdf5'
    else:
        return

    if args.ex is None:
        pass
    else:
        out_source_dpath = out_source_dpath + '/' + args.ex

    d2ae_flag = False
    if args.rev:
        srcs, targets2, targets1 = get_triplet_flatted_data(data_path)
    else:
        srcs, targets1, targets2 = get_triplet_flatted_data(data_path)
    
    model = TDAE_out(n_class1=torch.unique(targets1).size(0), n_class2=torch.unique(targets2).size(0), d2ae_flag = d2ae_flag, img_h=img_h, img_w=img_w)
    if args.dlim > 0:
        data_pairs = torch.utils.data.TensorDataset(srcs[0][:args.dlim], srcs[1][:args.dlim], srcs[2][:args.dlim], targets1[:args.dlim], targets2[:args.dlim])
    else:
        data_pairs = torch.utils.data.TensorDataset(srcs[0], srcs[1], srcs[2], targets1, targets2)
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
    if args.triplet:
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch*2, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)

    # criterion_adv = nn.NLLLoss()
    criterion_classifier = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=args.margin)
    if args.fou:
        criterion_reconst = Fourier_mse(img_h=img_h, img_w=img_w, mask=True, dm=args.dm, mode=args.fill)
    else:
        criterion_reconst = nn.MSELoss()

    params = list(model.parameters())
    # optim_adv = optim.Adam(params_adv, lr=1e-4)
    params_adv = list(model.classifier_sub.parameters())
    optim_adv = optim.Adam(params_adv)
    optimizer = optim.Adam(params)
    # optimizer = optim.SGD(params, lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    Scores_reg = [[], []]
    Scores_reg_adv = [[], []]
    Vals_reg = [[], []]
    Vals_svm = [[], []]
    n_epochs = args.epoch
    best_loss = np.inf
    best_epoch = 0
    l_adv = args.adv
    l_recon = args.rec
    l_tri = args.tri
    l_c = args.classifier
    for epoch in range(n_epochs):
        accs_p, acc_t = [], []
        Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0
        Loss, RecLoss, CLoss, CLoss_sub, TriLoss = [], [], [], [], []
        for ite, (in_data, p_in_data, n_in_data, target, sub_target) in enumerate(train_loader):
        # for ite, (in_data, target, sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            losses = []
            if d2ae_flag:
                continue
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
                # with torch.no_grad():
                #     h0_anchor, h0_pos, h0_neg = model.enc(in_data.to(device)), model.enc(p_in_data.to(device)), model.enc(n_in_data.to(device))
                #     h0_anchor, h0_pos, h0_neg = Variable(h0_anchor), Variable(h0_pos), Variable(h0_neg)
                # p0_anchor, p0_pos, p0_neg = model.subnets_p(h0_anchor.to(device)), model.subnets_p(h0_pos.to(device)), model.subnets_p(h0_neg.to(device))
                if args.triplet:
                    (_, p0_anchor), (_, p0_pos), (_, p0_neg) = model.hidden_output(in_data.to(device)), model.hidden_output(p_in_data.to(device)), model.hidden_output(n_in_data.to(device))
                    loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                    loss_triplet.backward(retain_graph=True)
                    losses.append(loss_triplet)

                else:
                    loss_triplet = torch.tensor(0)

                preds, preds_adv, reconst = model(in_data.to(device))
                loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                loss_reconst.backward(retain_graph=True)
                losses.append(loss_reconst)
                
                loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                loss_adv.backward(retain_graph=True)
                model.classifier_main.zero_grad()
                losses.append(loss_adv)
                
                loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
                loss_classifier_main.backward(retain_graph=True)
                losses.append(loss_classifier_main)

                optimizer.step()
                loss = 0
                for cat_loss in losses:
                    loss += cat_loss

            Loss.append(loss.item())
            RecLoss.append(loss_reconst.item())
            CLoss.append(loss_classifier_main.item())
            CLoss_sub.append(loss_adv.item())
            TriLoss.append(loss_triplet.item())
            
            y_true = target.to('cpu')
            sub_y_true = sub_target.to('cpu')
            preds = preds.detach().to('cpu')
            sub_preds = preds_adv.detach().to('cpu')
            Acc += true_positive_multiclass(preds, y_true)
            sub_Acc += true_positive_multiclass(sub_preds, y_true)

        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}, Acc_adv: {}, sub Acc_adv: {}'.format(epoch+1, np.mean(Loss), Acc/len(train_set), sub_Acc/len(train_set), Acc_adv/len(train_set), sub_Acc_adv/len(train_set)))
        writer.add_scalar('summarize loss',
            np.mean(Loss), epoch)
        writer.add_scalar('rec loss',
            np.mean(RecLoss), epoch)
        writer.add_scalar('classifier loss',
            np.mean(CLoss), epoch)
        writer.add_scalar('Adv loss',
            np.mean(CLoss_sub), epoch)
        writer.add_scalar('Triplet loss',
            np.mean(TriLoss), epoch)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                X_train = []
                Y_train1 = []
                Y_train2 = []
                for v_i, (in_data, p_in_data, n_in_data, target1, target2) in enumerate(train_loader):
                    if v_i == 0:
                        reconst = model.reconst(in_data.to(device))
                        np_input = in_data[0].detach().to('cpu')
                        np_reconst = reconst[0].detach().to('cpu')
                        img_grid = make_grid(torch.stack([np_input, np_reconst]))
                        writer.add_image('test', img_grid, epoch+1)
                    (_, t0) = model.hidden_output(in_data.to(device))
                    t0 = t0.detach().to('cpu').numpy()
                    X_train.extend(t0)
                    Y_train1.extend(target1.detach().to('cpu').numpy())
                    Y_train2.extend(target2.detach().to('cpu').numpy())

                X_val, Y_val1, Y_val2 = [], [], []
                for v_i, (in_data, p_in_data, n_in_data, target1, target2) in enumerate(val_loader):
                    (_, t0) = model.hidden_output(in_data.to(device))
                    t0 = t0.detach().to('cpu').numpy()
                    X_val.extend(t0)
                    Y_val1.extend(target1.detach().to('cpu').numpy())
                    Y_val2.extend(target2.detach().to('cpu').numpy())
                
                X_train = np.asarray(X_train)
                Y_train1 = np.asarray(Y_train1)
                Y_train2 = np.asarray(Y_train2)
                X_val = np.asarray(X_val)
                Y_val1 = np.asarray(Y_val1)
                Y_val2 = np.asarray(Y_val2)
                logreg = LogisticRegression(penalty='l2', solver="sag")
                for it, (Y_train, Y_adv, Y_val) in enumerate(zip([Y_train1, Y_train2], [Y_train2, Y_train1], [Y_val1, Y_val2])):
                    logreg.fit(X_train, Y_train)
                    score_reg = logreg.score(X_train, Y_train)
                    Scores_reg[it].append(score_reg)

                    score_reg = logreg.score(X_val, Y_val)
                    Vals_reg[it].append(score_reg)

                    score_reg = logreg.score(X_train, Y_adv)
                    Scores_reg_adv[it].append(score_reg)

                writer.add_scalar('Tar1 Reg Score',
                    np.mean(Scores_reg[0][-1]), epoch)
                writer.add_scalar('Tar2 Reg Score',
                    np.mean(Scores_reg[1][-1]), epoch)
                writer.add_scalar('Tar1 Reg adv Score',
                    np.mean(Scores_reg_adv[0][-1]), epoch)
                writer.add_scalar('Tar2 Reg adv Score',
                    np.mean(Scores_reg_adv[1][-1]), epoch)
                writer.add_scalar('Tar1 Reg Val',
                    np.mean(Vals_reg[0][-1]), epoch)
                writer.add_scalar('Tar2 Reg Val',
                    np.mean(Vals_reg[1][-1]), epoch)

                val_losses = []
                val_c_loss = []
                val_r_loss = []
                val_a_loss = []
                for in_data, p_in_data, n_in_data, target, _ in val_loader:
                    if d2ae_flag:
                        preds, sub_preds_adv, sub_preds, reconst = model.forward_train_like_D2AE(in_data.to(device))
                        loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                        loss_classifier_main = criterion_classifier(preds.to(device), target.to(device))
                        loss_classifier_sub = l_adv * criterion_classifier(sub_preds_adv.to(device), target.to(device))
                        loss_sub_adv = l_adv * negative_entropy_loss(sub_preds_adv.to(device).to(device))
                        val_loss = loss_reconst + loss_classifier_main + loss_sub_adv + loss_classifier_sub
                    else:
                        if args.triplet:
                            (_, p0_anchor), (_, p0_pos), (_, p0_neg) = model.hidden_output(in_data.to(device)), model.hidden_output(p_in_data.to(device)), model.hidden_output(n_in_data.to(device))
                            loss_triplet = l_tri * criterion_triplet(p0_anchor, p0_pos, p0_neg)
                        else:
                            loss_triplet = torch.tensor(0)
                        preds, preds_adv, reconst = model(in_data.to(device))
                        val_loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                        val_loss_classifier_main = l_c * criterion_classifier(preds.to(device), target.to(device))
                        val_loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                        val_loss = val_loss_classifier_main + val_loss_reconst + loss_triplet + val_loss_adv
                        
                    val_c_loss.append(val_loss_classifier_main.item())
                    val_r_loss.append(val_loss_reconst.item())
                    val_losses.append(val_loss.item())
                    val_a_loss.append(val_loss_adv.item())

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


def val_TDAE(zero_padding=False):
    torch.manual_seed(SEED)
    rn.seed(SEED)
    np.random.seed(SEED)
    
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq'
        data_path='data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy'
        data_path='data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon' 
        data_path='data/colon_renew.hdf5'
    else:
        return
    if args.ex is None:
        pass
    else:
        out_source_dpath = out_source_dpath + '/' + args.ex
    if args.retrain:
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_val_dpath = '{}/re_val_{}'.format(out_source_dpath, args.param)
        out_fig_dpath = '{}/re_fig_{}'.format(out_source_dpath, args.param)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_val_dpath = '{}/val_{}'.format(out_source_dpath, args.param)
        out_fig_dpath = '{}/fig_{}'.format(out_source_dpath, args.param)
    clean_directory(out_val_dpath)
    clean_directory(out_fig_dpath)

    d2ae_flag = False
    if args.rev:
        srcs, targets2, targets1 = get_flatted_data(data_path)
    else:
        srcs, targets1, targets2 = get_flatted_data(data_path)
    data_pairs = torch.utils.data.TensorDataset(srcs, targets1, targets2)

    if args.d2ae:
        model = TDAE_D2AE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)
    else:
        model = TDAE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)
    # model = TDAE_out(n_class1=torch.unique(targets1).size(0), n_class2=torch.unique(targets2).size(0), d2ae_flag = d2ae_flag, img_h=img_h, img_w=img_w)
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

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)
    
    cat_val_set = val_set
    if zero_padding:
        with torch.no_grad():
            model.eval()
            for n_iter, (inputs, targets1, target2) in enumerate(train_loader):
                reconst = model.reconst(inputs.to(device))
                s_reconst = model.fix_padding_reconst(inputs.to(device), which_val=1, pad_val=0)
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
                fig.savefig('{}/train_sample{:04d}_zero_pad.png'.format(out_val_dpath, n_iter))
                plt.close(fig)
                if n_iter >= 10:
                    return

    with torch.no_grad():
        model.eval()
        for n_iter, (inputs, targets1, target2) in enumerate(val_loader):
            reconst = model.reconst(inputs.to(device))
            s_reconst = model.shuffle_reconst(inputs.to(device), idx1=[0, 1], idx2=[1, 0])
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
            fig.savefig('{}/val_sample{:04d}.png'.format(out_val_dpath, n_iter))
            plt.close(fig)
            if n_iter >= 50:
                break

        for n_iter, (inputs, targets1, target2) in enumerate(train_loader):
            reconst = model.reconst(inputs.to(device))
            s_reconst = model.shuffle_reconst(inputs.to(device), idx1=[0, 1], idx2=[1, 0])
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
            fig.savefig('{}/train_sample{:04d}.png'.format(out_val_dpath, n_iter))
            plt.close(fig)
            if n_iter >= 50:
                break
    
    with torch.no_grad():
        model.eval()
        X_train1, X_train2, Y_train1, Y_train2 = [], [], [], []
        train_hf = []
        for n_iter, (inputs, targets1, targets2) in enumerate(train_loader):
            (h0, t0) = model.hidden_output(inputs.to(device))
            h0 = h0.detach().to('cpu').numpy()
            t0 = t0.detach().to('cpu').numpy()
            # ht = np.append(h0, t0, axis=0)
            X_train1.extend(h0)
            X_train2.extend(t0)
            Y_train1.extend(targets1.detach().to('cpu').numpy())
            Y_train2.extend(targets2.detach().to('cpu').numpy())
    
        X_train1 = np.asarray(X_train1)
        X_train2 = np.asarray(X_train2)
        Y_train1 = np.asarray(Y_train1)
        Y_train2 = np.asarray(Y_train2)
        rn.seed(SEED)
        np.random.seed(SEED)
        tsne = TSNE(n_components=2, random_state=SEED)
        X_train1 = tsne.fit_transform(X_train1)
        X_train2 = tsne.fit_transform(X_train2)

        picK_idx = [2*i+j for i in range(20) for j in [0, 1]]
        fig = plt.figure(figsize=(16*2, 9))
        ax = fig.add_subplot(1,2,1)
        for k in np.unique(Y_train1):
            ax.scatter(x=X_train1[Y_train1==k,0], y=X_train1[Y_train1==k,1], marker='.', alpha=0.5)
        ax.scatter(x=X_train1[picK_idx,0], y=X_train1[picK_idx,1], color='red', marker='x')
        for txt in picK_idx:
            ax.annotate(txt, (X_train1[txt, 0], X_train1[txt, 1]))
        ax.set_aspect('equal', 'datalim')
        ax = fig.add_subplot(1,2,2)
        for k in np.unique(Y_train2):
            ax.scatter(x=X_train1[Y_train2==k,0], y=X_train1[Y_train2==k,1], marker='.', alpha=0.5)
        ax.scatter(x=X_train1[picK_idx,0], y=X_train1[picK_idx,1], color='red', marker='x')
        for txt in picK_idx:
            ax.annotate(txt, (X_train1[txt, 0], X_train1[txt, 1]))
        ax.set_aspect('equal', 'datalim')
        fig.savefig('{}/train_hidden_features_main.png'.format(out_fig_dpath))
        plt.close(fig)

        fig = plt.figure(figsize=(16*2, 9))
        ax = fig.add_subplot(1,2,1)
        for k in np.unique(Y_train1):
            ax.scatter(x=X_train2[Y_train1==k,0], y=X_train2[Y_train1==k,1], marker='.', alpha=0.5)
        ax.scatter(x=X_train2[picK_idx,0], y=X_train2[picK_idx,1], color='red', marker='x')
        for txt in picK_idx:
            ax.annotate(txt, (X_train2[txt, 0], X_train2[txt, 1]))
        ax.set_aspect('equal', 'datalim')
        ax = fig.add_subplot(1,2,2)
        for k in np.unique(Y_train2):
            ax.scatter(x=X_train2[Y_train2==k,0], y=X_train2[Y_train2==k,1], marker='.', alpha=0.5)
        ax.scatter(x=X_train2[picK_idx,0], y=X_train2[picK_idx,1], color='red', marker='x')
        for txt in picK_idx:
            ax.annotate(txt, (X_train2[txt, 0], X_train2[txt, 1]))
        ax.set_aspect('equal', 'datalim')
        fig.savefig('{}/train_hidden_features_sub.png'.format(out_fig_dpath))
        plt.close(fig)

        X1, X2, Y1, Y2 = [], [], [], []
        train_hf = []
        for n_iter, (inputs, targets1, targets2) in enumerate(val_loader):
            (h0, t0) = model.hidden_output(inputs.to(device))
            h0 = h0.detach().to('cpu').numpy()
            t0 = t0.detach().to('cpu').numpy()
            X1.extend(h0)
            X2.extend(t0)
            Y1.extend(targets1.detach().to('cpu').numpy())
            Y2.extend(targets2.detach().to('cpu').numpy())
    
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        Y1 = np.asarray(Y1)
        Y2 = np.asarray(Y2)
        rn.seed(SEED)
        np.random.seed(SEED)
        tsne = TSNE(n_components=2, random_state=SEED)
        X1 = tsne.fit_transform(X1)
        X2 = tsne.fit_transform(X2)

        fig = plt.figure(figsize=(16*2, 9))
        ax = fig.add_subplot(1,2,1)
        for k in np.unique(Y1):
            ax.scatter(x=X2[Y1==k,0], y=X2[Y1==k,1], marker='.', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        ax = fig.add_subplot(1,2,2)
        for k in np.unique(Y2):
            ax.scatter(x=X2[Y2==k,0], y=X2[Y2==k,1], marker='.', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        fig.savefig('{}/val_hidden_features_sub.png'.format(out_fig_dpath))
        plt.close(fig)

        fig = plt.figure(figsize=(16*2, 9))
        ax = fig.add_subplot(1,2,1)
        for k in np.unique(Y1):
            ax.scatter(x=X1[Y1==k,0], y=X1[Y1==k,1], marker='.', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        ax = fig.add_subplot(1,2,2)
        for k in np.unique(Y2):
            ax.scatter(x=X1[Y2==k,0], y=X1[Y2==k,1], marker='.', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        fig.savefig('{}/val_hidden_features_main.png'.format(out_fig_dpath))
        plt.close(fig)


def test_TDAE():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq'
        data_path='data/toy_data_freq_shape.hdf5'
    elif 'toy' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy'
        data_path='data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon' 
        data_path='data/colon_renew.hdf5'
    else:
        return

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
    if args.d2ae:
        model = TDAE_D2AE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)
    else:
        model = TDAE(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w, n_decov=args.ndeconv, channels=args.channels)
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


def main():
    # if os.path.exists('./data/colon_renew.hdf5') is False:
    #     data_review()
    # get_character_dataset()
    # return
    args = argparses()
    print(args)
    if args.mode == 'train':
        if args.triplet:
            triplet_train_TDAE()
            return
        # train_TDAE_fullsuper()
        train_TDAE_v2()
    elif args.mode == 'val':
        val_TDAE()
    elif args.mode == 'test':
        test_TDAE()
    else:
        if args.triplet:
            triplet_train_TDAE()
        else:
            train_TDAE_v2()
            # train_TDAE_fullsuper()
        val_TDAE()
        test_TDAE()

    
if __name__ == '__main__':
    # with SetIO('./out.log'):
    main()
