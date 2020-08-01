import os
import sys
import json
import copy
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
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.utils import make_grid
import tensorboardX as tbx

from losses import TripletLoss, negative_entropy_loss
from metrics import true_positive_multiclass, true_positive, true_negative
from __init__ import clean_directory 
from data_handling import get_triplet_flatted_data, get_flatted_data
from archs import CrossDisentangleNet

SEED = 1
torch.manual_seed(SEED)
rn.seed(SEED)
np.random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current_path = './'

def load_model(in_dpath, model):
    return model.load_state_dict(torch.load(in_dpath))


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

def argparses():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--data', type=str, default='toy')
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--param', type=str, default='best')
    parser.add_argument('--ex', type=str, default=None)
    parser.add_argument('--classifier1', type=float, default=1e-0)
    parser.add_argument('--classifier2', type=float, default=1e-0)
    parser.add_argument('--rec', type=float, default=1e-0)
    parser.add_argument('--adv1', type=float, default=1e-1)
    parser.add_argument('--adv2', type=float, default=1e-1)
    parser.add_argument('--tri', type=float, default=1e-2)
    parser.add_argument('--triplet', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--dm', type=int, default=0)
    parser.add_argument('--fill', type=str, default='hp')
    parser.add_argument('--fou', action='store_true')
    return parser.parse_args()


def main():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/Cross_freq'
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy_data' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/Cross_toy' 
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/Cross_colon' 
        data_path = './data/colon_renew.hdf5'
    else:
        return
    if args.ex is None:
        pass
    else:
        out_source_dpath = out_source_dpath + '/' + args.ex

    srcs, targets1, targets2 = get_flatted_data(data_path)
    data_pairs = torch.utils.data.TensorDataset(srcs, targets1, targets2)
    model = CrossDisentangleNet(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w)

    if args.retrain:
        model.load_state_dict(torch.load('{}/param/test_param.json'.format(out_source_dpath)))
        out_fig_dpath = '{}/re_figure'.format(out_source_dpath)
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_board_dpath = '{}/re_runs'.format(out_source_dpath)
        out_condition_dpath = '{}/re_condition'.format(out_source_dpath)
    else:
        out_fig_dpath = '{}/figure'.format(out_source_dpath)
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_board_dpath = '{}/runs'.format(out_source_dpath)
        out_condition_dpath = '{}/condition'.format(out_source_dpath)
    clean_directory(out_fig_dpath)
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

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    criterion_classifier = nn.CrossEntropyLoss()
    if args.fou:
        criterion_reconst = Fourier_mse(img_h=img_h, img_w=img_w, mask=True, dm=args.dm, mode=args.fill)
    else:
        criterion_reconst = nn.MSELoss()
    params = list(model.parameters())
    optimizer = optim.Adam(params)
    # optim_adv = optim.SGD(params, lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    n_epochs = args.epoch
    best_loss = np.inf
    l_adv1 = args.adv1
    l_adv2 = args.adv2
    l_recon = args.rec
    l_c1 = args.classifier1
    l_c2 = args.classifier2
    best_epoch = 0
    for epoch in range(n_epochs):
        accs_p, acc_t = [], []
        Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0
        Loss, RecLoss, CLoss, CLoss_sub, TriLoss = [], [], [], [], []
        for ite, (in_data, target, sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            losses = []
            preds_main, preds_sub, adv_preds_main, adv_preds_sub, reconst = model(in_data.to(device))
            loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
            loss_reconst.backward(retain_graph=True)
            losses.append(loss_reconst)

            loss_main_adv = l_adv1 * negative_entropy_loss(adv_preds_main.to(device))
            loss_main_adv.backward(retain_graph=True)
            model.classifier_main.zero_grad()
            losses.append(loss_main_adv)
            
            loss_sub_adv = l_adv2 * negative_entropy_loss(adv_preds_sub.to(device))
            loss_sub_adv.backward(retain_graph=True)
            model.classifier_sub.zero_grad()
            losses.append(loss_sub_adv)
            
            loss_classifier_main = l_c1 * criterion_classifier(preds_main.to(device), target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            losses.append(loss_classifier_main)

            loss_classifier_sub = l_c2 * criterion_classifier(preds_sub.to(device), sub_target.to(device))
            loss_classifier_sub.backward(retain_graph=True)
            losses.append(loss_classifier_sub)
            loss = 0
            for cat_loss in losses:
                loss += cat_loss
            optimizer.step()

            Loss.append(loss.item())
            RecLoss.append(loss_reconst.item())
            CLoss.append(loss_classifier_main.item())
            CLoss_sub.append(loss_classifier_sub.item())
            
            y_true = target.to('cpu')
            sub_y_true = sub_target.to('cpu')
            preds_main = preds_main.detach().to('cpu')
            preds_sub = preds_sub.detach().to('cpu')
            adv_preds_main = adv_preds_main.detach().to('cpu')
            adv_preds_sub = adv_preds_sub.detach().to('cpu')
            Acc += true_positive_multiclass(preds_main, y_true)
            sub_Acc += true_positive_multiclass(preds_sub, sub_y_true)
            Acc_adv += true_positive_multiclass(adv_preds_main, y_true)
            sub_Acc_adv += true_positive_multiclass(adv_preds_sub, sub_y_true)

        print('epoch: {} loss: {} \nAcc: {} sub Acc: {}, Acc_adv: {}, sub Acc_adv: {}'.format(epoch+1, np.mean(Loss), Acc/len(train_set), sub_Acc/len(train_set), Acc_adv/len(train_set), sub_Acc_adv/len(train_set)))
        writer.add_scalar('summarize loss',
            np.mean(Loss), epoch+1)
        writer.add_scalar('rec loss',
            np.mean(RecLoss), epoch+1)
        writer.add_scalar('classifier loss',
            np.mean(CLoss), epoch+1)
        writer.add_scalar('sub classifier loss',
            np.mean(CLoss_sub), epoch+1)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for in_data, target1, target2 in train_loader:
                    reconst = model.reconst(in_data.to(device))
                    np_input = in_data[0].detach().to('cpu')
                    np_reconst = reconst[0].detach().to('cpu')
                    img_grid = make_grid(torch.stack([np_input, np_reconst]))
                    writer.add_image('test', img_grid, epoch+1)
                    break
                val_losses = []
                val_main_loss = []
                val_sub_loss = []
                val_rec_loss = []
                Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0

                for in_data, target, sub_target in val_loader:
                    preds_main, preds_sub, adv_preds_main, adv_preds_sub, reconst = model(in_data.to(device))
                    loss_reconst = l_recon * criterion_reconst(reconst.to(device), in_data.to(device))
                    loss_main_adv = l_adv1 * negative_entropy_loss(adv_preds_main.to(device))
                    loss_sub_adv = l_adv2 * negative_entropy_loss(adv_preds_sub.to(device))
                    loss_classifier_main = l_c1 * criterion_classifier(preds_main.to(device), target.to(device))
                    loss_classifier_sub = l_c2 * criterion_classifier(preds_sub.to(device), sub_target.to(device))
                    val_loss = loss_reconst + loss_classifier_main + loss_classifier_sub + loss_main_adv + loss_sub_adv
                    val_losses.append(val_loss.item())
                    val_main_loss.append(loss_classifier_main.item())
                    val_sub_loss.append(loss_classifier_sub.item())
                    val_rec_loss.append(loss_reconst.item())

                    y_true = target.to('cpu')
                    sub_y_true = sub_target.to('cpu')
                    preds_main = preds_main.detach().to('cpu')
                    preds_sub = preds_sub.detach().to('cpu')
                    adv_preds_main = adv_preds_main.detach().to('cpu')
                    adv_preds_sub = adv_preds_sub.detach().to('cpu')
                    Acc += true_positive_multiclass(preds_main, y_true)
                    sub_Acc += true_positive_multiclass(preds_sub, sub_y_true)
                    Acc_adv += true_positive_multiclass(adv_preds_main, y_true)
                    sub_Acc_adv += true_positive_multiclass(adv_preds_sub, sub_y_true)


                print('---------------------------------------')
                print('epoch: {} val loss: {} \nAcc: {} sub Acc: {}, Acc_adv: {}, sub Acc_adv: {}'.format(epoch+1, np.mean(val_losses), Acc/len(val_set), sub_Acc/len(val_set), Acc_adv/len(val_set), sub_Acc_adv/len(val_set)))
                print('---------------------------------------')

                writer.add_scalar('val loss',
                    np.mean(val_losses), epoch+1)
                writer.add_scalar('val reconst loss',
                    np.mean(val_rec_loss), epoch+1)
                writer.add_scalar('val main loss',
                    np.mean(val_main_loss), epoch+1)
                writer.add_scalar('val sub loss',
                    np.mean(val_sub_loss), epoch+1)
                writer.add_scalar('val main acc',
                    Acc/len(val_set), epoch+1)
                writer.add_scalar('val sub acc',
                    sub_Acc/len(val_set), epoch+1)

                if best_loss > np.mean(val_losses):
                    best_loss = np.mean(val_losses)
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), '{}/test_bestparam.json'.format(out_param_dpath))

    torch.save(model.state_dict(), '{}/test_param.json'.format(out_param_dpath))

    dict_args = copy.copy(vars(args))
    dict_args['best_epoch'] = best_epoch
    for k in dict_args.keys():
        dict_args[k] = [dict_args[k]]
    df = pd.DataFrame.from_dict(dict_args)
    df.to_csv('{}/condition.csv'.format(out_condition_dpath))

    writer.close()


def validate(data_path='data/toy_data.hdf5'):
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/Cross_freq' 
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy_data' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/Cross_toy'
        data_path = './data/toy_data.hdf5'
    elif 'colon' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/Cross_colon'
        data_path = './data/colon_renew.hdf5'
    else:
        return
    if args.ex is None:
        pass
    else:
        out_source_dpath = out_source_dpath + '/' + args.ex

    if args.retrain:
        out_param_dpath = '{}/re_param'.format(out_source_dpath)
        out_fig_dpath = '{}/re_fig_{}'.format(out_source_dpath, args.param)
        out_val_dpath = '{}/re_val_{}'.format(out_source_dpath, args.param)
    else:
        out_param_dpath = '{}/param'.format(out_source_dpath)
        out_fig_dpath = '{}/fig_{}'.format(out_source_dpath, args.param)
        out_val_dpath = '{}/val_{}'.format(out_source_dpath, args.param)
    clean_directory(out_val_dpath)
    clean_directory(out_fig_dpath)

    srcs, targets1, targets2 = get_triplet_flatted_data(data_path)    
    model = CrossDisentangleNet(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w)
    if args.param == 'best':
        model.load_state_dict(torch.load('{}/test_bestparam.json'.format(out_param_dpath)))
    else:
        model.load_state_dict(torch.load('{}/test_param.json'.format(out_param_dpath)))
        
    model = model.to(device)

    data_pairs = torch.utils.data.TensorDataset(srcs[0], targets1, targets2)
    ratio = [0.7, 0.2, 0.1]
    n_sample = len(data_pairs)
    train_size = int(n_sample*ratio[0])
    val_size = int(n_sample*ratio[1])
    test_size = n_sample - train_size - val_size
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size+val_size))
    test_indices = list(range(train_size, train_size+val_size))

    train_set = torch.utils.data.dataset.Subset(data_pairs, train_indices)
    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    test_set = torch.utils.data.dataset.Subset(data_pairs, test_indices)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)
    test_loader = DataLoader(val_set, batch_size=2, shuffle=True)

    cat_val_set = val_set
    with torch.no_grad():
        model.eval()
        for n_iter, (inputs, targets1, target2) in enumerate(test_loader):
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
            fig.savefig('{}/sample{:04d}.png'.format(out_val_dpath, n_iter))
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

        fig = plt.figure(figsize=(16*2, 9))
        ax = fig.add_subplot(1,2,1)
        for k in np.unique(Y_train1):
            ax.scatter(x=X_train1[Y_train1==k,0], y=X_train1[Y_train1==k,1], marker='.', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        ax = fig.add_subplot(1,2,2)
        for k in np.unique(Y_train2):
            ax.scatter(x=X_train1[Y_train2==k,0], y=X_train1[Y_train2==k,1], marker='.', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        fig.savefig('{}/train_hidden_features_main.png'.format(out_fig_dpath))
        plt.close(fig)

        fig = plt.figure(figsize=(16*2, 9))
        ax = fig.add_subplot(1,2,1)
        for k in np.unique(Y_train1):
            ax.scatter(x=X_train2[Y_train1==k,0], y=X_train2[Y_train1==k,1], marker='.', alpha=0.5)
        ax.set_aspect('equal', 'datalim')
        ax = fig.add_subplot(1,2,2)
        for k in np.unique(Y_train2):
            ax.scatter(x=X_train2[Y_train2==k,0], y=X_train2[Y_train2==k,1], marker='.', alpha=0.5)
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

def test():
    args = argparses()
    if 'freq' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_freq'
        data_path = './data/toy_data_freq_shape.hdf5'
    elif 'toy_data' in args.data:
        img_w, img_h = 256, 256
        out_source_dpath = './reports/TDAE_toy'
        data_path = './data/toy_data.hdf5'
    elif 'toy_data' in args.data:
        img_w, img_h = 224, 224
        out_source_dpath = './reports/TDAE_colon' 
        data_path = './data/colon_renew.hdf5'
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

    if args.rev:
        srcs, targets2, targets1 = get_triplet_flatted_data(data_path)
    else:
        srcs, targets1, targets2 = get_triplet_flatted_data(data_path)
    
    model = CrossDisentangleNet(n_classes=[torch.unique(targets1).size(0), torch.unique(targets2).size(0)], img_h=img_h, img_w=img_w)
    if args.param == 'best':
        model.load_state_dict(torch.load('{}/TDAE_test_bestparam.json'.format(out_param_dpath)))
    else:
        model.load_state_dict(torch.load('{}/TDAE_test_param.json'.format(out_param_dpath)))
    model = model.to(device)

    data_pairs = torch.utils.data.TensorDataset(srcs[0], targets1, targets2)
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

        logreg.fit(X1[0], Y1[0])
        # linear_svc.fit(X1[0], Y1[0])
        l = logreg.predict_proba(X1[1])
        plt.hist(np.max(l, axis=1), np.arange(0, 1.01, 0.01))
        plt.ylim([0, 100])
        plt.savefig('./test.png')
        plt.close()

        logreg.fit(X2[0], Y1[0])
        # linear_svc.fit(X2[0], Y1[0])
        l = logreg.predict_proba(X2[1])
        plt.hist(np.max(l, axis=1), np.arange(0, 1.01, 0.01))
        plt.ylim([0, 100])
        plt.savefig('./test_1.png')
        plt.close()

        logreg.fit(X1[0], Y2[0])
        # linear_svc.fit(X1[0], Y2[0])
        l = logreg.predict_proba(X1[1])
        plt.hist(np.max(l, axis=1), np.arange(0, 1.01, 0.01))
        plt.ylim([0, 100])
        plt.savefig('./test_tub1.png')
        plt.close()

        logreg.fit(X2[0], Y2[0])
        # linear_svc.fit(X2[0], Y1[0])
        score_reg = logreg.score(X2[0], Y2[0])
        print('train-------------------------------')
        print(score_reg)
        print('-------------------------------')

        score_reg = logreg.score(X2[1], Y2[1])
        print('val-------------------------------')
        print(score_reg)
        print('-------------------------------')
        l = logreg.predict_proba(X2[1])
        plt.hist(np.max(l, axis=1), np.arange(0, 1.01, 0.01))
        plt.ylim([0, 100])
        plt.savefig('./test_tub2.png')
        plt.close()
        return

    
if __name__ == '__main__':
    # if os.path.exists('./data/colon_renew.hdf5') is False:
    #     data_review()
    args = argparses()
    if args.mode == 'train':
        main()
    elif args.mode == 'val':
        validate()
    elif args.mode == 'test':
        test()
    else:
        main()
        validate()
    print(args)
