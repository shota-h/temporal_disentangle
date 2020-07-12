import os, sys, json, itertools, time, argparse, csv, h5py
import cv2
import numpy as np
import random as rn
import argparse
import pandas as pd
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
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
import tensorboardX as tbx
import archs
import numpy as np
import random as rn
import h5py
from data_handling import numpy_pop, GetHDF5dataset, GetHDF5dataset_MultiLabel

SEED = 1
torch.manual_seed(SEED)
rn.seed(SEED)
np.random.seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
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
    input = F.softmax(input, dim=1) + small_value
    w = torch.ones((input.size()[0], input.size()[1])) / input.size()[1]
    w = w.to(device)
    log_input = torch.log(input)
    log_input = torch.mul(w, log_input)
    neg_entropy = -1 * torch.sum(log_input, dim=1) / log_input.size()[1]
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
    torch.save(model.state_dict(), '{}/test_disentangle/model_param.json'.format(current_path))


if __name__ == '__main__':
    main()
