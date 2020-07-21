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
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--data', type=str, default='toy')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def main(data_path='data/toy_data.hdf5'):
    out_source_dpath = './reports/Cross' 
    if 'toy_data' in data_path:
        img_w = 256
        img_h = 256
    else:
        img_w = 224
        img_h = 224
        out_source_dpath = './reports/Cross_colon' 


    out_fig_dpath = '{}/figure'.format(out_source_dpath)
    out_param_dpath = '{}/param'.format(out_source_dpath)
    out_board_dpath = '{}/runs'.format(out_source_dpath)
    clean_directory(out_fig_dpath)
    clean_directory(out_param_dpath)
    clean_directory(out_board_dpath)
    d2ae_flag = False
    writer = tbx.SummaryWriter(out_board_dpath)

    srcs, targets1, targets2 = get_flatted_data(data_path)
    data_pairs = torch.utils.data.TensorDataset(srcs, targets1, targets2)
    model = CrossDisentangleNet(n_class1=torch.unique(targets1).size(0), n_class2=torch.unique(targets2).size(0), img_h=img_h, img_w=img_w)
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
    criterion_reconst = nn.MSELoss()
    criterion_triplet = TripletLoss()
    params = list(model.parameters())
    optimizer = optim.Adam(params)
    # optim_adv = optim.SGD(params, lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    n_epochs = 300
    best_loss = np.inf
    l_adv = 1.0e-1
    l_recon = 1.0e-0
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

            loss_main_adv = l_adv * negative_entropy_loss(adv_preds_main.to(device))
            loss_main_adv.backward(retain_graph=True)
            model.classifier_main.zero_grad()
            losses.append(loss_main_adv)
            
            loss_sub_adv = l_adv * negative_entropy_loss(adv_preds_sub.to(device))
            loss_sub_adv.backward(retain_graph=True)
            model.classifier_sub.zero_grad()
            losses.append(loss_sub_adv)
            
            loss_classifier_main = criterion_classifier(preds_main.to(device), target.to(device))
            loss_classifier_main.backward(retain_graph=True)
            losses.append(loss_classifier_main)

            loss_classifier_sub = criterion_classifier(preds_sub.to(device), sub_target.to(device))
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
                Acc, Acc_adv, sub_Acc, sub_Acc_adv  = 0, 0, 0, 0

                for in_data, target, sub_target in val_loader:
                    preds_main, preds_sub, adv_preds_main, adv_preds_sub, reconst = model(in_data.to(device))
                    loss_reconst = l_recon*criterion_reconst(reconst.to(device), in_data.to(device))
                    # loss_main_adv = l_adv*negative_entropy_loss(adv_preds_main.to(device))
                    # loss_sub_adv = l_adv*negative_entropy_loss(adv_preds_sub.to(device))
                    loss_classifier_main = criterion_classifier(preds_main.to(device), target.to(device))
                    loss_classifier_sub = criterion_classifier(preds_sub.to(device), sub_target.to(device))
                    val_loss = loss_reconst + loss_classifier_main + loss_classifier_sub
                    val_losses.append(val_loss.item())

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
                writer.add_scalar('val main acc',
                    Acc/len(val_set), epoch+1)
                writer.add_scalar('val sub acc',
                    sub_Acc/len(val_set), epoch+1)

                if best_loss > np.mean(val_losses):
                    best_loss = np.mean(val_losses)
                    torch.save(model.state_dict(), '{}/test_bestparam.json'.format(out_param_dpath))

    torch.save(model.state_dict(), '{}/test_param.json'.format(out_param_dpath))
    writer.close()


def validate(data_path='data/toy_data.hdf5'):
    out_source_dpath = './reports/Cross' 
    if 'toy_data' in data_path:
        img_w = 256
        img_h = 256
    else:
        img_w = 224
        img_h = 224
        out_source_dpath = './reports/Cross_colon' 

    out_val_dpath = '{}/val'.format(out_source_dpath)
    clean_directory(out_val_dpath)

    srcs, targets1, targets2 = get_triplet_flatted_data(data_path)
    model = CrossDisentangleNet(n_class1=torch.unique(targets1).size(0), n_class2=torch.unique(targets2).size(0), img_h=img_h, img_w=img_w)
    model.load_state_dict(torch.load('{}/param/test_bestparam.json'.format(out_source_dpath)))
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

    val_set = torch.utils.data.dataset.Subset(data_pairs, val_indices)
    test_set = torch.utils.data.dataset.Subset(data_pairs, test_indices)
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

    
if __name__ == '__main__':
    # if os.path.exists('./data/colon_renew.hdf5') is False:
    #     data_review()
    arg = argparses()
    if arg.data == 'toy':
        main()
        validate()
    elif arg.data == 'colon':
        d = './data/colon_renew.hdf5'
        main(d)
        validate(d)
    else:
        print(arg)
