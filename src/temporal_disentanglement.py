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
import tensorboardX as tbx

from losses import TripletLoss, negative_entropy_loss
from metrics import true_positive_multiclass, true_positive, true_negative
from __init__ import clean_directory 
from data_handling import get_triplet_flatted_data, get_flatted_data
from archs import TDAE_out

SEED = 1
torch.manual_seed(SEED)
rn.seed(SEED)
np.random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current_path = './'


def argparses():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--data', type=str, default='toy')
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--ex', type=str, default=None)
    parser.add_argument('--classifier', type=float, default=1e-0)
    parser.add_argument('--rec', type=float, default=1e-0)
    parser.add_argument('--adv', type=float, default=1e-1)
    parser.add_argument('--tri', type=float, default=1e-2)
    parser.add_argument('--triplet', action='store_true')
    return parser.parse_args()


def load_model(in_dpath, model):
    return model.load_state_dict(torch.load(in_dpath))


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
    args = argparses()

    if 'toy' in data_path:
        img_w = 256
        img_h = 256
        out_source_dpath = './reports/TDAE_toy' 
    else:
        img_w = 224
        img_h = 224
        out_source_dpath = './reports/TDAE_colon'

    if args.ex is None:
        pass
    else:
        out_source_dpath = out_source_dpath + '/' + args.ex

    out_fig_dpath = '{}/figure'.format(out_source_dpath)
    out_param_dpath = '{}/param'.format(out_source_dpath)
    out_board_dpath = '{}/runs'.format(out_source_dpath)
    clean_directory(out_fig_dpath)
    clean_directory(out_param_dpath)
    clean_directory(out_board_dpath)
    d2ae_flag = False
    writer = tbx.SummaryWriter(out_board_dpath)

    srcs, targets1, targets2 = get_triplet_flatted_data(data_path)
    data_pairs = torch.utils.data.TensorDataset(srcs[0], srcs[1], srcs[2], targets1, targets2)
    model = TDAE_out(n_class1=3, n_class2=5, d2ae_flag = d2ae_flag, img_h=img_h, img_w=img_w)
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
    criterion_reconst = nn.MSELoss()
    criterion_triplet = TripletLoss()
    params = list(model.parameters())
    # optim_adv = optim.Adam(params_adv, lr=1e-4)
    params_adv = list(model.classifier_sub.parameters())
    optim_adv = optim.Adam(params_adv)
    optimizer = optim.Adam(params)
    # optimizer = optim.SGD(params, lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    Scores_reg = [[], []]
    Scores_svm = [[], []]
    Scores_reg_adv = [[], []]
    Scores_svm_adv = [[], []]
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
                linear_svc = LinearSVC()
                for it, (Y_train, Y_adv, Y_val) in enumerate(zip([Y_train1, Y_train2], [Y_train2, Y_train1], [Y_val1, Y_val2])):
                    logreg.fit(X_train, Y_train)
                    linear_svc.fit(X_train, Y_train)
                    score_reg = logreg.score(X_train, Y_train)
                    score_svm = linear_svc.score(X_train, Y_train)
                    Scores_reg[it].append(score_reg)
                    Scores_svm[it].append(score_svm)

                    score_reg = logreg.score(X_val, Y_val)
                    score_svm = linear_svc.score(X_val, Y_val)
                    Vals_reg[it].append(score_reg)
                    Vals_svm[it].append(score_svm)

                    score_reg = logreg.score(X_train, Y_adv)
                    score_svm = linear_svc.score(X_train, Y_adv)
                    Scores_reg_adv[it].append(score_reg)
                    Scores_svm_adv[it].append(score_svm)
                    # 予測　
                    # Y_pred = logreg.predict(X_train)
                    # Y_pred = linear_svc.predict(X_train)

                    # スコア
                writer.add_scalar('Tar1 Reg Score',
                    np.mean(Scores_reg[0][-1]), epoch)
                writer.add_scalar('Tar2 Reg Score',
                    np.mean(Scores_reg[1][-1]), epoch)
                writer.add_scalar('Tar1 Reg adv Score',
                    np.mean(Scores_reg_adv[0][-1]), epoch)
                writer.add_scalar('Tar2 Reg adv Score',
                    np.mean(Scores_reg_adv[1][-1]), epoch)
                writer.add_scalar('Tar1 SVM Score',
                    np.mean(Scores_svm[0][-1]), epoch)
                writer.add_scalar('Tar2 SVM Score',
                    np.mean(Scores_svm[1][-1]), epoch)
                writer.add_scalar('Tar1 SVM adv Score',
                    np.mean(Scores_svm_adv[0][-1]), epoch)
                writer.add_scalar('Tar2 SVM adv Score',
                    np.mean(Scores_svm_adv[1][-1]), epoch)
                writer.add_scalar('Tar1 Reg Val',
                    np.mean(Vals_reg[0][-1]), epoch)
                writer.add_scalar('Tar1 SVM Val',
                    np.mean(Vals_svm[0][-1]), epoch)
                writer.add_scalar('Tar2 Reg Val',
                    np.mean(Vals_reg[1][-1]), epoch)
                writer.add_scalar('Tar2 SVM Val',
                    np.mean(Vals_svm[1][-1]), epoch)

                val_losses = []
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
                        val_loss_classifier_main = criterion_classifier(preds.to(device), target.to(device))
                        val_loss_adv = l_adv * negative_entropy_loss(preds_adv.to(device))
                        val_loss = val_loss_classifier_main + val_loss_reconst + loss_triplet
                        
                    val_losses.append(val_loss.item())

                print('epoch: {} val loss: {}'.format(epoch+1, np.mean(val_losses)))
                writer.add_scalar('val loss',
                    np.mean(val_losses), epoch)

                if best_loss > np.mean(val_losses):
                    best_epoch = epoch + 1
                    best_loss = np.mean(val_losses)
                    torch.save(model.state_dict(), '{}/TDAE_test_bestparam.json'.format(out_param_dpath))

    torch.save(model.state_dict(), '{}/TDAE_test_param.json'.format(out_param_dpath))
    
    dict_args = copy.copy(vars(args))
    dict_args['best_epoch'] = best_epoch
    for k in dict_args.keys():
        dict_args[k] = [dict_args[k]]
    df = pd.DataFrame.from_dict(dict_args)
    df.to_csv('{}/condition.csv'.format(out_source_dpath))
    
    writer.close()


def val_TDAE(data_path='data/toy_data.hdf5'):
    args = argparses()
    if 'toy_data' in data_path:
        img_w = 256
        img_h = 256
        out_source_dpath = './reports/TDAE_toy' 
    else:
        img_w = 224
        img_h = 224
        out_source_dpath = './reports/TDAE_colon' 
    if args.ex is None:
        pass
    else:
        out_source_dpath = out_source_dpath + '/' + args.ex
        
    out_val_dpath = '{}/val'.format(out_source_dpath)
    clean_directory(out_val_dpath)

    d2ae_flag = False
    model = TDAE_out(n_class1=3, n_class2=5, d2ae_flag = d2ae_flag, img_h=img_h, img_w=img_w)
    model.load_state_dict(torch.load('{}/param/TDAE_test_bestparam.json'.format(out_source_dpath)))
    model = model.to(device)
    srcs, targets1, targets2 = get_triplet_flatted_data(data_path)

    data_pairs = torch.utils.data.TensorDataset(srcs[0], targets1, targets2)
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
    train_loader = DataLoader(train_set, batch_size=2, shuffle=False)
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
        fig.savefig('{}/train_hidden_features_main.png'.format(out_source_dpath))
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
        fig.savefig('{}/train_hidden_features_sub.png'.format(out_source_dpath))
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
        fig.savefig('{}/val_hidden_features_sub.png'.format(out_source_dpath))
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
        fig.savefig('{}/val_hidden_features_main.png'.format(out_source_dpath))
        plt.close(fig)


    
if __name__ == '__main__':
    # if os.path.exists('./data/colon_renew.hdf5') is False:
    #     data_review()
    args = argparses()
    if args.data == 'toy':
        if args.mode == 'train':
            train_TDAE()
        elif args.mode == 'val':
            val_TDAE()
        else:
            train_TDAE()
            val_TDAE()
    elif args.data == 'colon':
        d = './data/colon_renew.hdf5'
        if args.mode == 'train':
            train_TDAE(d)
        elif args.mode == 'val':
            val_TDAE(d)
        else:
            train_TDAE(d)
            val_TDAE(d)

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
