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
from archs import SingleClassifier
from vat import VATLoss

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
    parser.add_argument('--data', type=str, default='toy')
    parser.add_argument('--param', type=str, default='best')
    parser.add_argument('--ex', type=str, default=None)
    parser.add_argument('--classifier', type=float, default=1)
    parser.add_argument('--ratio', type=float, default=0.0)
    parser.add_argument('--semi', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--labeldecomp', action='store_false')
    parser.add_argument('--channels', type=int, nargs='+', default=[3,16,32,64,128])
    parser.add_argument('--xi', type=float, default=10.0, metavar='XI',
                        help='hyperparameter of VAT (default: 0.1)')
    parser.add_argument('--eps', type=float, default=1.0, metavar='EPS',
                        help='hyperparameter of VAT (default: 1.0)')
    parser.add_argument('--ip', type=int, default=1, metavar='IP',
                        help='hyperparameter of VAT (default: 1)')
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


def train():
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    out_source_dpath += 'SemiSelf'
    out_source_dpath = os.path.join(out_source_dpath, ex)

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

    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)
    
    model = SingleClassifier()
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
    print((targets1==-2).nonzero().size(), targets1.size())
    print((targets2==-2).nonzero().size(), targets2.size())

    train_loader = DataLoader(train_set, batch_size=args.batch*ngpus, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch*ngpus, shuffle=False)

    # criterion_adv = nn.NLLLoss()
    criterion_classifier = nn.CrossEntropyLoss(ignore_index=-2)
    criterion_vat = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)

    optimizer = optim.Adam(params)

    n_epochs = args.epoch
    best_epoch, best_loss, best_acc = 0, np.inf, 0
    best_semi_epoch, best_semi_loss, best_semi_acc = 0, np.inf, 0
    train_keys = ['loss/train_all', 'loss/train_classifier', 'loss/train_vat']
    val_keys = ['loss/val_all', 'loss/val_classifier']
    mytrans = TensorTransforms(methods=[vflip, hflip, rotate], p=0.3)
    for epoch in range(n_epochs):
        Acc, sub_Acc  = 0, 0
        loss_dict = {}
        for k in train_keys:
            loss_dict[k] = []

        for iters, (idx, p_idx, n_idx, target, sub_target) in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            x = mytrans(src[idx])
            sub_preds = model.forward(x.to(device))
            
            loss_classifier = criterion_classifier(sub_preds.to(device), sub_target.to(device))
            loss_vat = criterion_vat(model, x[sub_target==-2].to(device))
            # loss_vat = criterion_vat(model, x.to(device))
            
            loss = loss_vat
            loss = loss_classifier + loss_vat
            loss.backward()
            optimizer.step()

            for k, val in zip(train_keys, [loss, loss_classifier, loss_vat]):
                loss_dict[k].append(val.item())
            
            sub_y_true = sub_target.to('cpu')
            sub_preds = sub_preds.detach().to('cpu')
            sub_Acc += true_positive_multiclass(sub_preds[sub_y_true!=-2], sub_y_true[sub_y_true!=-2])
                    
        for k in loss_dict.keys():
            loss_dict[k] = np.mean(loss_dict[k])

        print('epoch: {:04} loss: {:.5} sub Acc: {:.5}'.format(epoch+1, loss_dict['loss/train_all'], sub_Acc/train_size2))

        summary = scalars2summary(writer=writer,
                            tags=list(loss_dict.keys()), 
                            vals=list(loss_dict.values()), epoch=epoch+1)
        
        train_acc2 = 0
        with torch.no_grad():
            model.eval()
            task1_size, task2_size = 0, 0
            for iters, (idx, _, _, target, sub_target) in enumerate(train_loader):
                sub_preds = model.forward(src[idx].to(device))
                y2_true = sub_target.to('cpu')
                sub_preds = sub_preds.detach().to('cpu')
                train_acc2 += true_positive_multiclass(sub_preds[y2_true!=-2], y2_true[y2_true!=-2])

            train_acc2 = train_acc2 / train_size2
            summary = scalars2summary(writer=writer,
                            tags=['acc/train_sub'], 
                            vals=[train_acc2], epoch=epoch+1)

        if (epoch + 1) % args.step == 0:
            with torch.no_grad():
                model.eval()
                val_loss_dict = {}
                for k in val_keys:
                    val_loss_dict[k] = []
                val_Acc, val_sub_Acc = 0, 0
                for v_i, (idx, p_idx, n_idx, target, sub_target) in enumerate(val_loader):
                    sub_preds = model.forward(src[idx].to(device))

                    val_loss_classifier = criterion_classifier(sub_preds.to(device), sub_target.to(device))
        
                    val_loss = val_loss_classifier

                    for k, val in zip(val_keys, [val_loss, val_loss_classifier]):
                        val_loss_dict[k].append(val.item())

                    sub_y_true = sub_target.to('cpu')
                    sub_preds = sub_preds.detach().to('cpu')
                    val_sub_Acc += true_positive_multiclass(sub_preds, sub_y_true)
                
                for k in val_loss_dict.keys():
                    val_loss_dict[k] = np.mean(val_loss_dict[k])

                summary = scalars2summary(writer=writer, 
                                        tags=list(val_loss_dict.keys()), 
                                        vals=list(val_loss_dict.values()), epoch=epoch+1)

                summary = scalars2summary(writer=writer, 
                                        tags=['acc/val_sub'], 
                                        vals=[val_sub_Acc/len(val_set)], epoch=epoch+1)

                print('val loss: {:.5} sub Acc: {:.5}'.format(val_loss_dict['loss/val_all'], val_sub_Acc/len(val_set)))
                
                if val_sub_Acc/len(val_set) >= best_acc:
                    best_acc = val_sub_Acc/len(val_set)
                    torch.save(model.state_dict(), '{}/SemiSelf_bestparam_acc.json'.format(out_param_dpath))
                    
                if best_loss > val_loss_dict['loss/val_all']:
                    best_epoch = epoch + 1
                    best_loss = val_loss_dict['loss/val_all']
                    if args.multi:
                        torch.save(model.module.state_dict(), '{}/SemiSelf_bestparam.json'.format(out_param_dpath))
                    else:
                        torch.save(model.state_dict(), '{}/SemiSelf_bestparam.json'.format(out_param_dpath))
            
    if args.multi:
        torch.save(model.module.state_dict(), '{}/SemiSelf_lastparam.json'.format(out_param_dpath))
    else:
        torch.save(model.state_dict(), '{}/SemiSelf_lastparam.json'.format(out_param_dpath))
    
    args.best_epoch = best_epoch
    df = args2pandas(args)
    df.to_csv('{}/condition.csv'.format(out_condition_dpath))
    writer.close()
    if args.adapt:
        df = pd.DataFrame.from_dict(weight_change)
        df.to_csv('{}/WeightChange.csv'.format(out_condition_dpath))
        

def test():
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    args = argparses()
    out_source_dpath += 'SemiSelf'
    out_source_dpath = out_source_dpath + '/' + ex
    src, targets1, targets2, idxs, seq_ids, img_paths = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)

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

    data_pairs = torch.utils.data.TensorDataset(idxs[0], idxs[1], idxs[2], targets1, targets2)
    
    out_param_dpath = '{}/param'.format(out_source_dpath)
    out_test_dpath = '{}/test_{}'.format(out_source_dpath, args.param)
    clean_directory(out_test_dpath)
    
    model = SingleClassifier()

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
    for ntag in ['all', 'acc']:
        X1_dict = {}
        X2_dict = {}
        Y1_dict = {}
        Y2_dict = {}
        pY1_dict = {}
        pY2_dict = {}
        if ntag == 'all':
            model.load_state_dict(torch.load('{}/SemiSelf_bestparam.json'.format(out_param_dpath)))
        elif ntag == 'acc':
            model.load_state_dict(torch.load('{}/SemiSelf_bestparam_acc.json'.format(out_param_dpath)))
            
        model = model.to(device)
        conf_mat_dict2 = {}
        for gl, pl in itertools.product(range(torch.unique(targets2).size(0)), range(torch.unique(targets2).size(0))):
            conf_mat_dict2['{}to{}'.format(gl, pl)] = []
        with torch.no_grad():
            for k, loader in zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]):
                model.eval()
                eval_dict = {}
                eval_dict['path'] = []
                eval_dict['mayo_label'] = []
                eval_dict['mayo_discard_label'] = []
                eval_dict['mayo_pred'] = []
                eval_dict['seq'] = []
                    
                X1, X2, Y1, Y2 = [], [], [], []
                pY1, pY2 = [], []
                    
                for n_iter, (idx, _, _, target, sub_target) in enumerate(loader):
                    cat_paths = [img_paths[iidx] for iidx in idx]
                                        
                    pred_y2 = model.predict_label(src[idx].to(device))
                    sub_target_np = sub_target.detach().to('cpu').numpy()
                    dis_sub_target_np = discarded_targets2[idx].detach().to('cpu').numpy()
                    sub_pred_np = pred_y2.detach().to('cpu').numpy()
                    eval_dict['path'].extend(cat_paths)
                    eval_dict['mayo_label'].extend(sub_target_np)
                    eval_dict['mayo_discard_label'].extend(dis_sub_target_np)
                    eval_dict['mayo_pred'].extend(sub_pred_np)
                    eval_dict['seq'].extend([c.split('/')[1] for c in cat_paths])
                    Y2.extend(sub_target.detach().to('cpu').numpy())
                    pY2.extend(pred_y2.detach().to('cpu').numpy())

                df = pd.DataFrame.from_dict(eval_dict)
                df.to_csv('{}/eachPredicted_{}_{}.csv'.format(out_test_dpath, k, ntag))
                Y2_dict[k] = np.asarray(Y2)
                pY2_dict[k] = np.asarray(pY2)
                conf_mat_label2 = confusion_matrix(Y2, pY2)
                for gl, pl in itertools.product(range(torch.unique(targets2).size(0)), range(torch.unique(targets2).size(0))):
                    conf_mat_dict2['{}to{}'.format(gl, pl)].append(conf_mat_label2[gl, pl])

            df = pd.DataFrame.from_dict(conf_mat_dict2)
            df.to_csv('{}/conf_mat_mayo_{}.csv'.format(out_test_dpath, ntag))

        # return
        score_dict_withPseudo = {}
        for indices, k in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
            pred_y2 = torch.from_numpy(pY2_dict[k])
            cat_target2 = discarded_targets2[indices]
            cat_full_target2 = true_targets2[indices]
            remain_target2_acc = true_positive_multiclass(pred_y2[cat_target2!=-2], cat_target2[cat_target2!=-2])
            discarded_target2_acc = true_positive_multiclass(pred_y2[cat_target2==-2], cat_full_target2[cat_target2==-2])
            target2_acc = true_positive_multiclass(pred_y2, targets2[indices])
            if k == 'train':
                score_dict_withPseudo['sub'] = [target2_acc/len(indices)]
                score_dict_withPseudo['remain_sub'] = [remain_target2_acc/len(cat_target2[cat_target2!=-2])]
                score_dict_withPseudo['pseudo_sub'] = [discarded_target2_acc/len(cat_target2[cat_target2==-2])]
            else:
                score_dict_withPseudo['sub'].append(target2_acc/len(indices))
                score_dict_withPseudo['remain_sub'].append(remain_target2_acc/len(cat_target2[cat_target2!=-2]))
                score_dict_withPseudo['pseudo_sub'].append(discarded_target2_acc/len(cat_target2[cat_target2==-2]))

        df = pd.DataFrame.from_dict(score_dict_withPseudo)
        df.to_csv('{}/ResultNN_withPseudo_{}.csv'.format(out_test_dpath, ntag))

        score_dict = {}
        score_nn = {}
        for k in ['train', 'val', 'test']:
            pred2 = np.sum(pY2_dict[k] == Y2_dict[k])
            if k == 'train':
                score_nn['sub'] = [pred2 / len(pY2_dict[k])]
            else:
                score_nn['sub'].append(pred2 / len(pY2_dict[k]))
        df = pd.DataFrame.from_dict(score_nn)
        df.to_csv('{}/ResultNN_{}.csv'.format(out_test_dpath, ntag))


def output_imgexaple():
    args = argparses()
    img_w, img_h, out_source_dpath, data_path, ex = get_outputpath()
    src, targets1, targets2, idxs, _, _ = get_triplet_flatted_data_with_idx(data_path, label_decomp=args.labeldecomp)
    for u1, u2 in itertools.product(np.unique(targets1), np.unique(targets2)):
        print(u1, u2)
        cat_src = src[np.logical_and((targets2==u2),(targets1==u1))]
        iters = 0
        for img in cat_src:
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:,:,::-1]
            cv2.imwrite('./reports/imgs/test_loc{}_level{}_iter{}.png'.format(u1, u2, iters), img*255)
            iters += 1
            if iters >= 30:
                break

def main():
    args = argparses()
    if args.train:
        train()
    if args.val:
        val()
    if args.test:
        test()
    
if __name__ == '__main__':
    # with SetIO('./out.log'):
    main()
