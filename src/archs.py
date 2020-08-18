import itertools
import torch
from torch.nn.modules import activation
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.utils.data import DataLoader
import torchvision.models as models


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


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return torch.reshape(x, (x.size(0), -1))
        # return x.view(x.size(0), -1)

        # t0 = torch.reshape(t0, (t0.size(0), -1))


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

def base_conv(in_ch, out_ch, ksize, stride=2):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, ksize, stride=stride, padding=(ksize-1)//2),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU())

def base_deconv(in_ch, out_ch):
    return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU())

class TDAE_out(nn.Module):
    def __init__(self, n_class1=3, n_class2=2, ksize=3, d2ae_flag=False, img_w=256, img_h=256, channels=[16, 32, 64, 128], n_decov=2):
        super().__init__()
        if d2ae_flag:
            n_class2 = n_class1
        self.img_h, self.img_w = img_h, img_w
        self.channels = channels
        # self.model = models.vgg16(num_classes=n_class, pretrained=False)
        # self.enc = models.resnet18(pretrained=False)
        # self.enc = nn.Sequential(*list(self.enc.children())[:8])
        self.conv1 = base_conv(3, channels[0], ksize)
        self.conv2 = base_conv(channels[0], channels[1], ksize)
        self.conv3 = base_conv(channels[1], channels[2], ksize)
        self.conv4 = base_conv(channels[2], channels[3], ksize)
        self.conv5 = base_conv(channels[3], channels[3], ksize)
                     
        # self.conv1 = nn.Sequential(nn.Conv2d(3, channels[0], ksize, stride=2, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(channels[0]),
        #                             nn.ReLU())
        # self.conv2 = nn.Sequential(nn.Conv2d(channels[0], channels[1], ksize, stride=2, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(channels[1]),
        #                             nn.ReLU())
        # self.conv3 = nn.Sequential(nn.Conv2d(channels[1], channels[2], ksize, stride=2, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(channels[2]),
        #                             nn.ReLU())
        # self.conv4 = nn.Sequential(nn.Conv2d(64, 128, ksize, stride=2, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        # self.conv5 = nn.Sequential(nn.Conv2d(128, 128, ksize, stride=2, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())

        self.enc = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)

        self.subnet_conv_t1 = base_conv(channels[3], channels[3], ksize, stride=1)
        self.subnet_conv_t2 = base_conv(channels[3], channels[2], ksize, stride=1)
        self.subnet_conv_p1 = base_conv(channels[3], channels[3], ksize, stride=1)
        self.subnet_conv_p2 = base_conv(channels[3], channels[2], ksize, stride=1)

        # self.subnet_conv_t1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        # self.subnet_conv_t2 = nn.Sequential(nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())
        self.subnet_t1 = nn.Sequential(nn.Linear(in_features=64, out_features=256),
                                    nn.ReLU())

        # self.subnet_conv_p1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        # self.subnet_conv_p2 = nn.Sequential(nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())
        self.subnet_p1 = nn.Sequential(nn.Linear(in_features=64, out_features=256),
                                    nn.ReLU())

        self.subnets_t = nn.Sequential(self.subnet_conv_t1,
                                        self.subnet_conv_t2,
                                        nn.AvgPool2d(kernel_size=img_h//(2**5)),
                                        Flatten(),
                                        self.subnet_t1)

        self.subnets_p = nn.Sequential(self.subnet_conv_p1,
                                        self.subnet_conv_p2,
                                        nn.AvgPool2d(kernel_size=img_h//(2**5)),
                                        Flatten(),
                                        self.subnet_p1)

        self.classifier_main = nn.Linear(in_features=256, out_features=n_class1)
        self.classifier_sub = nn.Linear(in_features=256, out_features=n_class2)


        self.dec_fc1 = nn.Sequential(nn.Linear(in_features=256*2, 
                                                out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*channels[2]),
                                                nn.ReLU())
    
        self.deconv1 = base_deconv(channels[2], channels[3])
        self.deconv2 = base_deconv(channels[3], channels[2])
        self.deconv3 = base_deconv(channels[2], channels[1])
        self.deconv4 = base_deconv(channels[1], channels[0])
        self.deconv1_conv1 = base_conv(channels[3], channels[3], ksize, stride=1)
        self.deconv1_conv2 = base_conv(channels[3], channels[3], ksize, stride=1)
        self.deconv2_conv1 = base_conv(channels[2], channels[2], ksize, stride=1)
        self.deconv2_conv2 = base_conv(channels[2], channels[2], ksize, stride=1)
        self.deconv3_conv1 = base_conv(channels[1], channels[1], ksize, stride=1)
        self.deconv3_conv2 = base_conv(channels[1], channels[1], ksize, stride=1)
        self.deconv4_conv1 = base_conv(channels[0], channels[0], ksize, stride=1)
        self.deconv4_conv2 = base_conv(channels[0], channels[0], ksize, stride=1)

        # self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 128, 2, stride=2),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        # self.deconv1_conv1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        # self.deconv1_conv2 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(128),
        #                             nn.ReLU())
        # self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())
        # self.deconv2_conv1 = nn.Sequential(nn.Conv2d(64, 64, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())
        # self.deconv2_conv2 = nn.Sequential(nn.Conv2d(64, 64, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())
        # self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2),
        #                             nn.BatchNorm2d(32),
        #                             nn.ReLU())
        # self.deconv3_conv1 = nn.Sequential(nn.Conv2d(32, 32, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(32),
        #                             nn.ReLU())
        # self.deconv3_conv2 = nn.Sequential(nn.Conv2d(32, 32, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(32),
        #                             nn.ReLU())
        # self.deconv4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2),
        #                             nn.BatchNorm2d(16),
        #                             nn.ReLU())
        # self.deconv4_conv1 = nn.Sequential(nn.Conv2d(16, 16, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(16),
        #                             nn.ReLU())
        # self.deconv4_conv2 = nn.Sequential(nn.Conv2d(16, 16, ksize, padding=(ksize-1)//2),
        #                             nn.BatchNorm2d(16),
        #                             nn.ReLU())
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(channels[0], 3, 2, stride=2),
                                nn.Sigmoid())
        # self.dec = nn.Sequential(self.deconv1,
        #                         self.deconv2,
        #                         self.deconv3,
        #                         self.deconv4,
        #                         self.deconv5)
        # self.dec = nn.Sequential(self.deconv1, self.deconv1_conv1,
        #                         self.deconv2, self.deconv2_conv1,
        #                         self.deconv3, self.deconv3_conv1,
        #                         self.deconv4, self.deconv4_conv1,
        #                         self.deconv5)
        self.dec = nn.Sequential(self.deconv1, self.deconv1_conv1, self.deconv1_conv2,
                                self.deconv2, self.deconv2_conv1, self.deconv2_conv2,
                                self.deconv3, self.deconv3_conv1, self.deconv3_conv2,
                                self.deconv4, self.deconv4_conv1, self.deconv4_conv2,
                                self.deconv5)

        initialize_weights(self)
        
    def forward_train_like_D2AE(self, input):
        h0 = self.enc(input)
        t0 = self.subnet_conv_t1(h0)
        p0 = self.subnet_conv_p1(h0)
        t0 = self.subnet_conv_t2(t0)
        p0 = self.subnet_conv_p2(p0)
        t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        t0 = torch.reshape(t0, (t0.size(0), -1))
        p0 = torch.reshape(p0, (p0.size(0), -1))
        t0 = self.subnet_t1(t0)
        p0 = self.subnet_p1(p0)

        p0_no_grad = p0.clone().detach()
        class_main_preds = self.classifier_main(t0)
        class_sub_preds = self.classifier_sub(p0)
        class_sub_preds_adv = self.classifier_sub(p0_no_grad)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_w//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return class_main_preds, class_sub_preds, class_sub_preds_adv, rec

    def forward(self, input):
        h0 = self.enc(input)
        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        class_main_preds = self.classifier_main(t0)
        class_main_preds_adv = self.classifier_main(p0)
        # class_main_preds_adv = self.classifier_sub(p0)
        # class_sub_preds_adv = self.classifier_sub(t0)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return class_main_preds, class_main_preds_adv, rec

    def predict_label(self, input):
        h0 = self.enc(input)
        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        class_main_preds = self.classifier_main(t0)
        class_main_preds_adv = self.classifier_main(p0)
        return torch.max(class_main_preds, 1), torch.max(class_main_preds_adv, 1)

    def hidden_output(self, input):
        h0 = self.enc(input)
        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        return t0, p0

    def reconst(self, input):
        h0 = self.enc(input)
        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def shuffle_reconst(self, input, idx1, idx2):
        h0 = self.enc(input)
        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        concat_h0 = torch.cat((t0[idx1], p0[idx2]), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def share_enc(self, input):
        t0 = self.subnet_conv_t1(input)
        p0 = self.subnet_conv_p1(input)

        t0 = self.subnet_conv_t2(t0)
        p0 = self.subnet_conv_p2(p0)
        
        t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        
        t0 = torch.reshape(t0, (t0.size(0), -1))
        p0 = torch.reshape(p0, (p0.size(0), -1))
        
        t0 = self.subnet_t1(t0)
        p0 = self.subnet_p1(p0)
        return t0, h0


class CrossDisentangleNet(nn.Module):
    def __init__(self, n_classes, ksize=3, img_w=256, img_h=256, channels=[3, 16, 32, 64, 128], latent_dim=256, n_decov=2):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.latent_dim = latent_dim
        self.channels = channels

        enc_layers = []
        for i in range(len(channels)-1):
            enc_layers.append(base_conv(channels[i], channels[i+1], ksize))
        enc_layers.append(base_conv(channels[-1], channels[-1], ksize))
        self.enc = nn.Sequential(*enc_layers)

        subnets = []
        classifiers = []
        for c in n_classes:        
            subnets.append(self.get_subnet(channel=channels[-1], ksize=ksize, nconv=2))
            classifiers.append(nn.Linear(in_features=latent_dim, out_features=c))
        
        self.subnets = nn.ModuleList(subnets)
        self.classifiers = nn.ModuleList(classifiers)


        self.dec_fc1 = nn.Sequential(nn.Linear(in_features=latent_dim*len(n_classes), 
                                                out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*channels[4]),
                                                nn.ReLU())
        
        dec_layers = [base_deconv(channels[4], channels[4])]
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        for in_c, out_c in zip(channels[::-1][:-1], channels[::-1][1:]):
            if out_c == channels[0]:
                dec_layers.append(nn.ConvTranspose2d(in_c, out_c, 2, stride=2))
                dec_layers.append(nn.Sigmoid())
                break
            dec_layers.append(base_deconv(in_c, out_c))
            for i in range(n_decov):
                dec_layers.append(base_conv(out_c, out_c, ksize, stride=1))
        
        self.dec = nn.Sequential(*dec_layers)
        initialize_weights(self)

    def forward(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        classifier_preds = []
        for i, ii in itertools.product(range(len(self.classifiers)), range(len(output_subnets))):
            classifier_preds.append(self.classifiers[i](output_subnets[ii]))

        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return classifier_preds[0], classifier_preds[3], classifier_preds[1], classifier_preds[2], rec

    def predict_label(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        classifier_preds = []
        for i, ii in itertools.product(range(len(self.classifiers)), range(len(output_subnets))):
            classifier_preds.append(self.classifiers[i](output_subnets[ii]))

        return torch.max(classifier_preds[0], 1), torch.max(classifier_preds[3], 1), torch.max(classifier_preds[1], 1), torch.max(classifier_preds[2], 1)

    def hidden_output(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        return output_subnets

    def reconst(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def shuffle_reconst(self, input, idx1, idx2, shuffle_idx):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        output_subnets = [output_subnets[s] for s in shuffle_idx]
        for i, (ii, idx) in enumerate(zip(range(len(output_subnets)), [idx1, idx2])):
            output_subnets[i] = output_subnets[ii][idx]
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec
    
    def get_subnet(self, channel, ksize, nconv):
        subnet_layers = []
        for i in range(nconv):
            subnet_layers.append(base_conv(channel, channel, ksize, stride=1))
        subnet_layers.append(nn.AvgPool2d(kernel_size=self.img_w//(2**5)))
        subnet_layers.append(Flatten())
        subnet_layers.append(nn.Linear(in_features=channel, out_features=self.latent_dim))
        subnet_layers.append(nn.ReLU())
        return nn.Sequential(*subnet_layers)


class TDAE(nn.Module):
    def __init__(self, n_classes, ksize=3, img_w=256, img_h=256, channels=[3, 16, 32, 64, 128], n_decov=2, latent_dim=256):
        super().__init__()

        self.img_h, self.img_w = img_h, img_w
        self.channels = channels
        self.latent_dim = latent_dim

        enc_layers = []
        for i in range(len(channels)-1):
            enc_layers.append(base_conv(channels[i], channels[i+1], ksize))
        enc_layers.append(base_conv(channels[-1], channels[-1], ksize))
        self.enc = nn.Sequential(*enc_layers)

        subnets = []
        classifiers = []
        for i in range(len(n_classes)):     
            subnets.append(self.get_subnet(channel=channels[-1], ksize=ksize, nconv=2))
            classifiers.append(nn.Linear(in_features=latent_dim, out_features=n_classes[0]))
        self.subnets = nn.ModuleList(subnets)
        self.classifiers = nn.ModuleList(classifiers)

        self.dec_fc1 = nn.Sequential(nn.Linear(in_features=latent_dim*len(n_classes), 
                                    out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*channels[-1]),
                                    nn.ReLU())

        dec_layers = [base_deconv(channels[4], channels[4])]
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        for in_c, out_c in zip(channels[::-1][:-1], channels[::-1][1:]):
            if out_c == channels[0]:
                dec_layers.append(nn.ConvTranspose2d(in_c, out_c, 2, stride=2))
                dec_layers.append(nn.Sigmoid())
                break
            dec_layers.append(base_deconv(in_c, out_c))
            for i in range(n_decov):
                dec_layers.append(base_conv(out_c, out_c, ksize, stride=1))
        
        self.dec = nn.Sequential(*dec_layers)
        initialize_weights(self)

    def forward(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        classifier_preds = []
        for i, ii in itertools.product(range(len(self.classifiers)), range(len(output_subnets))):
            classifier_preds.append(self.classifiers[i](output_subnets[ii]))

        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return classifier_preds[0], classifier_preds[1], classifier_preds[2], rec

    def predict_label(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        classifier_preds = []
        for i, ii in itertools.product(range(len(self.classifiers)), range(len(output_subnets))):
            classifier_preds.append(self.classifiers[i](output_subnets[ii]))
        
        return torch.max(classifier_preds[0], 1), torch.max(classifier_preds[1], 1)

    def hidden_output(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        return output_subnets

    def reconst(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def shuffle_reconst(self, input, idx1, idx2, shuffle_idx=[1, 0]):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        # shuffle_output_subnets = [output_subnets[s] for s in shuffle_idx]
        idx = [idx1, idx2]
        for i in range(len(output_subnets)):
            output_subnets[i] = output_subnets[i][idx[i]]
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def get_subnet(self, channel, ksize, nconv):
        subnet_layers = []
        for i in range(nconv):
            subnet_layers.append(base_conv(channel, channel, ksize, stride=1))
        subnet_layers.append(nn.AvgPool2d(kernel_size=self.img_w//(2**5)))
        subnet_layers.append(Flatten())
        subnet_layers.append(nn.Linear(in_features=channel, out_features=self.latent_dim))
        subnet_layers.append(nn.ReLU())
        return nn.Sequential(*subnet_layers)


class TDAE_D2AE(nn.Module):
    def __init__(self, n_classes, ksize=3, img_w=256, img_h=256, channels=[3, 16, 32, 64, 128], n_decov=2, latent_dim=256):
        super().__init__()

        self.img_h, self.img_w = img_h, img_w
        self.channels = channels
        self.latent_dim = latent_dim

        enc_layers = []
        for i in range(len(channels)-1):
            enc_layers.append(base_conv(channels[i], channels[i+1], ksize))
        enc_layers.append(base_conv(channels[-1], channels[-1], ksize))
        self.enc = nn.Sequential(*enc_layers)

        subnets = []
        classifiers = []
        for i in range(len(n_classes)):     
            subnets.append(self.get_subnet(channel=channels[-1], ksize=ksize, nconv=2))
            classifiers.append(nn.Linear(in_features=latent_dim, out_features=n_classes[0]))
        self.subnets = nn.ModuleList(subnets)
        self.classifiers = nn.ModuleList(classifiers)

        self.dec_fc1 = nn.Sequential(nn.Linear(in_features=latent_dim*len(n_classes), 
                                    out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*channels[-1]),
                                    nn.ReLU())

        dec_layers = [base_deconv(channels[4], channels[4])]
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        for in_c, out_c in zip(channels[::-1][:-1], channels[::-1][1:]):
            if out_c == channels[0]:
                dec_layers.append(nn.ConvTranspose2d(in_c, out_c, 2, stride=2))
                dec_layers.append(nn.Sigmoid())
                break
            dec_layers.append(base_deconv(in_c, out_c))
            for i in range(n_decov):
                dec_layers.append(base_conv(out_c, out_c, ksize, stride=1))
        
        self.dec = nn.Sequential(*dec_layers)
        initialize_weights(self)

    def forward(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        classifier_preds = []
        output_subnets_no_grad = output_subnets[-1].clone().detach()
        classifier_preds.append(self.classifiers[0](output_subnets[0]))
        classifier_preds.append(self.classifiers[1](output_subnets[1]))
        classifier_preds.append(self.classifiers[1](output_subnets_no_grad))
        # for i, ii in itertools.product(range(len(self.classifiers)), range(len(output_subnets))):
        #     classifier_preds.append(self.classifiers[i](output_subnets[ii]))

        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return classifier_preds[0], classifier_preds[1], classifier_preds[2], rec

    def predict_label(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        classifier_preds = []
        for i, ii in itertools.product(range(len(self.classifiers)), range(len(output_subnets))):
            classifier_preds.append(self.classifiers[i](output_subnets[ii]))
        
        return torch.max(classifier_preds[0], 1), torch.max(classifier_preds[1], 1)

    def hidden_output(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        return output_subnets

    def reconst(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def shuffle_reconst(self, input, idx1, idx2, shuffle_idx=[1, 0]):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        # shuffle_output_subnets = [output_subnets[s] for s in shuffle_idx]
        idx = [idx1, idx2]
        for i in range(len(output_subnets)):
            output_subnets[i] = output_subnets[i][idx[i]]
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def get_subnet(self, channel, ksize, nconv):
        subnet_layers = []
        for i in range(nconv):
            subnet_layers.append(base_conv(channel, channel, ksize, stride=1))
        subnet_layers.append(nn.AvgPool2d(kernel_size=self.img_w//(2**5)))
        subnet_layers.append(Flatten())
        subnet_layers.append(nn.Linear(in_features=channel, out_features=self.latent_dim))
        subnet_layers.append(nn.ReLU())
        return nn.Sequential(*subnet_layers)


class TestNet(nn.Module):
    def __init__(self, n_classes, ksize=3, img_w=256, img_h=256, channels=[3, 16, 32, 64, 128], n_decov=2, latent_dim=256, base_net='res', triplet=False):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.channels = channels
        self.latent_dim = latent_dim
        self.triplet = triplet
        subnets = []
        classifiers = []
        dec_layers = []
        if base_net == 'res':
            self.enc = models.resnet18(pretrained=False)
            self.enc = nn.Sequential(*list(self.enc.children())[:8])
            subnet_input_dim = 512
        else:
            enc_layers = []
            for i in range(len(channels)-1):
                enc_layers.append(base_conv(channels[i], channels[i+1], ksize))
            enc_layers.append(base_conv(channels[-1], channels[-1], ksize))
            self.enc = nn.Sequential(*enc_layers)
            subnet_input_dim = channels[-1]

        for i in range(len(n_classes)):     
            subnets.append(self.get_subnet(channel=subnet_input_dim, ksize=ksize, nconv=2))
            classifiers.append(nn.Linear(in_features=latent_dim, out_features=n_classes[0]))

        self.subnets = nn.ModuleList(subnets)
        self.classifiers = nn.ModuleList(classifiers)

        self.dec_fc1 = nn.Sequential(nn.Linear(in_features=latent_dim*len(n_classes), 
                                    out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*channels[-1]),
                                    nn.ReLU())

        dec_layers.append(base_deconv(channels[4], channels[4]))
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        for in_c, out_c in zip(channels[::-1][:-1], channels[::-1][1:]):
            if out_c == channels[0]:
                dec_layers.append(nn.ConvTranspose2d(in_c, out_c, 2, stride=2))
                dec_layers.append(nn.Sigmoid())
                break
            dec_layers.append(base_deconv(in_c, out_c))
            for i in range(n_decov):
                dec_layers.append(base_conv(out_c, out_c, ksize, stride=1))
        
        self.dec = nn.Sequential(*dec_layers)
        initialize_weights(self)

    def forward(self, input, latent=False):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        if latent:
            return output_subnets[0], output_subnets[1]

        output_subnets_no_grad = output_subnets[-1].clone().detach()
        classifier_preds = []
        classifier_preds.append(self.classifiers[0](output_subnets[0]))
        classifier_preds.append(self.classifiers[1](output_subnets[1]))
        classifier_preds.append(self.classifiers[1](output_subnets_no_grad))
        # for i, ii in itertools.product(range(len(self.classifiers)), range(len(output_subnets))):
        #     classifier_preds.append(self.classifiers[i](output_subnets[ii]))
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        if self.triplet:
            return classifier_preds[0], classifier_preds[1], classifier_preds[2], rec, output_subnets[0], output_subnets[1]

        return classifier_preds[0], classifier_preds[1], classifier_preds[2], rec

    def predict_label(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        classifier_preds = []
        for i, ii in itertools.product(range(len(self.classifiers)), range(len(output_subnets))):
            classifier_preds.append(self.classifiers[i](output_subnets[ii]))
        
        return torch.max(classifier_preds[0], 1), torch.max(classifier_preds[1], 1)

    def hidden_output(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        return output_subnets

    def reconst(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def shuffle_reconst(self, input, idx1, idx2, shuffle_idx=[1, 0]):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        idx = [idx1, idx2]
        for i in range(len(output_subnets)):
            output_subnets[i] = output_subnets[i][idx[i]]
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def fix_padding_reconst(self, input, which_val, pad_val):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        pad_tensor = torch.ones_like(output_subnets[which_val]) * pad_val

        output_subnets[which_val] = pad_tensor
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def get_subnet(self, channel, ksize, nconv):
        subnet_layers = []
        for i in range(nconv):
            subnet_layers.append(base_conv(channel, channel, ksize, stride=1))
        subnet_layers.append(nn.AvgPool2d(kernel_size=self.img_w//(2**5)))
        subnet_layers.append(Flatten())
        subnet_layers.append(nn.Linear(in_features=channel, out_features=self.latent_dim))
        subnet_layers.append(nn.ReLU())
        return nn.Sequential(*subnet_layers)
    
    
class TestNet_v2(nn.Module):
    def __init__(self, n_classes, ksize=3, img_w=256, img_h=256, channels=[3, 16, 32, 64, 128], n_decov=2, latent_dim=256, base_net='res', triplet=False):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.channels = channels
        self.latent_dim = latent_dim
        self.triplet = triplet
        subnets = []
        classifiers = []
        disentangle_classifiers = []
        dec_layers = []
        if base_net == 'res':
            self.enc = models.resnet18(pretrained=False)
            self.enc = nn.Sequential(*list(self.enc.children())[:8])
            subnet_input_dim = 512
        else:
            enc_layers = []
            for i in range(len(channels)-1):
                enc_layers.append(base_conv(channels[i], channels[i+1], ksize))
            enc_layers.append(base_conv(channels[-1], channels[-1], ksize))
            self.enc = nn.Sequential(*enc_layers)
            subnet_input_dim = channels[-1]

        for i in range(len(n_classes)):     
            subnets.append(self.get_subnet(channel=subnet_input_dim, ksize=ksize, nconv=2))
            classifiers.append(nn.Linear(in_features=latent_dim, out_features=n_classes[i]))
        
        for i in [1, 0]:
            disentangle_classifiers.append(nn.Linear(in_features=latent_dim, out_features=n_classes[i]))
                 
        self.subnets = nn.ModuleList(subnets)
        self.classifiers = nn.ModuleList(classifiers)
        self.disentangle_classifiers = nn.ModuleList(disentangle_classifiers)

        self.dec_fc1 = nn.Sequential(nn.Linear(in_features=latent_dim*len(n_classes), 
                                    out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*channels[-1]),
                                    nn.ReLU())

        dec_layers.append(base_deconv(channels[4], channels[4]))
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        for in_c, out_c in zip(channels[::-1][:-1], channels[::-1][1:]):
            if out_c == channels[0]:
                dec_layers.append(nn.ConvTranspose2d(in_c, out_c, 2, stride=2))
                dec_layers.append(nn.Sigmoid())
                break
            dec_layers.append(base_deconv(in_c, out_c))
            for i in range(n_decov):
                dec_layers.append(base_conv(out_c, out_c, ksize, stride=1))
        
        self.dec = nn.Sequential(*dec_layers)
        initialize_weights(self)

    def forward(self, input, latent=False):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        if latent:
            return output_subnets[0], output_subnets[1]

        output_subnets_no_grad = []
        classifier_preds = []
        disentangle_classifier_preds = []
        output_subnets_no_grad.append(output_subnets[0].clone().detach())
        output_subnets_no_grad.append(output_subnets[1].clone().detach())
        classifier_preds.append(self.classifiers[0](output_subnets[0]))
        classifier_preds.append(self.classifiers[1](output_subnets[1]))
        
        disentangle_classifier_preds.append(self.disentangle_classifiers[0](output_subnets[0]))
        disentangle_classifier_preds.append(self.disentangle_classifiers[0](output_subnets_no_grad[0]))
        disentangle_classifier_preds.append(self.disentangle_classifiers[1](output_subnets[1]))
        disentangle_classifier_preds.append(self.disentangle_classifiers[1](output_subnets_no_grad[1]))
        
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        if self.triplet:
            return  classifier_preds[0], classifier_preds[1], disentangle_classifier_preds[0],disentangle_classifier_preds[1], disentangle_classifier_preds[2], disentangle_classifier_preds[3], rec, output_subnets[0], output_subnets[1]

        return classifier_preds[0], classifier_preds[1], disentangle_classifier_preds[0],disentangle_classifier_preds[1], disentangle_classifier_preds[2], disentangle_classifier_preds[3], rec

    def predict_label(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))

        classifier_preds = []
        for i, ii in itertools.product(range(len(self.classifiers)), range(len(output_subnets))):
            classifier_preds.append(self.classifiers[i](output_subnets[ii]))
        
        return torch.max(classifier_preds[0], 1), torch.max(classifier_preds[1], 1)

    def hidden_output(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        return output_subnets

    def reconst(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def shuffle_reconst(self, input, idx1, idx2, shuffle_idx=[1, 0]):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        idx = [idx1, idx2]
        for i in range(len(output_subnets)):
            output_subnets[i] = output_subnets[i][idx[i]]
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def fix_padding_reconst(self, input, which_val, pad_val):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        pad_tensor = torch.ones_like(output_subnets[which_val]) * pad_val

        output_subnets[which_val] = pad_tensor
        concat_h0 = torch.cat(output_subnets, dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def get_subnet(self, channel, ksize, nconv):
        subnet_layers = []
        for i in range(nconv):
            subnet_layers.append(base_conv(channel, channel, ksize, stride=1))
        subnet_layers.append(nn.AvgPool2d(kernel_size=self.img_w//(2**5)))
        subnet_layers.append(Flatten())
        subnet_layers.append(nn.Linear(in_features=channel, out_features=self.latent_dim))
        subnet_layers.append(nn.ReLU())
        return nn.Sequential(*subnet_layers)


class base_classifier(nn.Module):
    def __init__(self, n_class, ksize=3, img_w=256, img_h=256, latent_dim=256, base_net='res'):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.latent_dim = latent_dim
        subnets = []
        classifiers = []
        self.enc = models.resnet18(pretrained=False)
        self.enc = nn.Sequential(*list(self.enc.children())[:8])
        subnet_input_dim = 512
        subnets.append(self.get_subnet(channel=subnet_input_dim, ksize=ksize, nconv=2))
        classifiers.append(nn.Linear(in_features=latent_dim, out_features=n_class))
        self.subnets = nn.ModuleList(subnets)
        self.classifiers = nn.ModuleList(classifiers)

    def forward(self, input):
        h0 = self.enc(input)
        output_subnets = self.subnets[0](h0)
        preds = self.classifiers[0](output_subnets)
        return preds

    def get_subnet(self, channel, ksize, nconv):
        subnet_layers = []
        for i in range(nconv):
            subnet_layers.append(base_conv(channel, channel, ksize, stride=1))
        subnet_layers.append(nn.AvgPool2d(kernel_size=self.img_w//(2**5)))
        subnet_layers.append(Flatten())
        subnet_layers.append(nn.Linear(in_features=channel, out_features=self.latent_dim))
        subnet_layers.append(nn.ReLU())
        return nn.Sequential(*subnet_layers)
    
    
class TDAE_VAE(nn.Module):
    def __init__(self, n_classes, ksize=3, img_w=256, img_h=256, channels=[3, 16, 32, 64, 128], n_decov=2, latent_dim=256, base_net='res', triplet=False):
        super(TDAE_VAE, self).__init__()
        self.img_h, self.img_w = img_h, img_w
        self.channels = channels
        self.latent_dim = latent_dim
        self.triplet = triplet
        subnets = []
        classifiers = []
        dec_layers = []
        if base_net == 'res':
            self.enc = models.resnet18(pretrained=False)
            self.enc = nn.Sequential(*list(self.enc.children())[:8])
            subnet_input_dim = 512
        else:
            enc_layers = []
            for i in range(len(channels)-1):
                enc_layers.append(base_conv(channels[i], channels[i+1], ksize))
                
            enc_layers.append(base_conv(channels[-1], channels[-1], ksize))
            self.enc = nn.Sequential(*enc_layers)
            subnet_input_dim = channels[-1]

        mu_nets = []
        logvar_nets = []
        for i in range(len(n_classes)):     
            subnets.append(self.get_subnet(channel=subnet_input_dim, ksize=ksize, nconv=2))
            mu_nets.append(self.get_mu_net())
            logvar_nets.append(self.get_logvar_net())
            classifiers.append(nn.Linear(in_features=self.latent_dim, out_features=n_classes[0]))

        self.subnets = nn.ModuleList(subnets)
        self.classifiers = nn.ModuleList(classifiers)
        self.mu_nets = nn.ModuleList(mu_nets)
        self.logvar_nets = nn.ModuleList(logvar_nets)

        self.dec_fc1 = nn.Sequential(nn.Linear(in_features=latent_dim*len(n_classes), 
                                    out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*channels[-1]),
                                    nn.ReLU())

        dec_layers.append(base_deconv(channels[4], channels[4]))
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        dec_layers.append(base_conv(channels[4], channels[4], ksize, stride=1))
        for in_c, out_c in zip(channels[::-1][:-1], channels[::-1][1:]):
            if out_c == channels[0]:
                dec_layers.append(nn.ConvTranspose2d(in_c, out_c, 2, stride=2))
                dec_layers.append(nn.Sigmoid())
                break
            dec_layers.append(base_deconv(in_c, out_c))
            for i in range(n_decov):
                dec_layers.append(base_conv(out_c, out_c, ksize, stride=1))
        
        self.dec = nn.Sequential(*dec_layers)
        initialize_weights(self)

    def encode(self, input):
        h0 = self.enc(input)
        output_mu_nets = []
        output_logvar_nets = []
        for i in range(len(self.subnets)):
            output_subnet = self.subnets[i](h0)
            output_mu_nets.append(self.mu_nets[i](output_subnet))
            output_logvar_nets.append(self.logvar_nets[i](output_subnet))
        return output_mu_nets[0], output_mu_nets[1], output_logvar_nets[0], output_logvar_nets[1]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z1, z2):
        concat_h0 = torch.cat((z1, z2), dim=1)
        dec_h0 = self.dec_fc1(concat_h0)
        dec_h0 = torch.reshape(dec_h0, (dec_h0.size(0), self.channels[-1], self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(dec_h0)
        return rec

    def forward(self, input, latent=False):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        if latent: return z1, z2

        mu2_no_grad = mu2.clone().detach()
        classifier_preds = []
        classifier_preds.append(self.classifiers[0](mu1))
        classifier_preds.append(self.classifiers[1](mu2))
        classifier_preds.append(self.classifiers[1](mu2_no_grad))
        rec = self.decode(z1, z2)
        if self.triplet:
            return classifier_preds[0], classifier_preds[1], classifier_preds[2], rec, z1, z2, mu1, mu2, logvar1, logvar2

        return classifier_preds[0], classifier_preds[1], classifier_preds[2], rec, mu1, mu2, logvar1, logvar2

    def predict_label(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        zs = [mu1, mu2]

        classifier_preds = []
        for i in range(len(self.classifiers)):
            classifier_preds.append(self.classifiers[i](zs[i]))
        
        return torch.max(classifier_preds[0], 1), torch.max(classifier_preds[1], 1)

    def hidden_output(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        return z1, z2

    def reconst(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)        
        rec = self.decode(z1, z2)
        return rec

    def shuffle_reconst(self, input, idx1, idx2, shuffle_idx=[1, 0]):
        h0 = self.enc(input)
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        zs = [z1, z2]        
        idx = [idx1, idx2]
        for i in range(len(zs)):
            zs[i] = zs[i][idx[i]]
        rec = self.decode(zs[0], zs[1])
        return rec

    def get_subnet(self, channel, ksize, nconv):
        subnet_layers = []
        for i in range(nconv):
            subnet_layers.append(base_conv(channel, channel, ksize, stride=1))
        subnet_layers.append(nn.AvgPool2d(kernel_size=self.img_w//(2**5)))
        subnet_layers.append(Flatten())
        subnet_layers.append(nn.Linear(in_features=channel, out_features=self.latent_dim))
        subnet_layers.append(nn.ReLU())
        return nn.Sequential(*subnet_layers)
    
    def get_mu_net(self):
        mu_net = []
        mu_net.append(nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim))
        # mu_net.append(nn.ReLU())
        return nn.Sequential(*mu_net)

    def get_logvar_net(self):
        logvar_net = []
        logvar_net.append(nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim))
        # logvar_net.append(nn.ReLU())
        return nn.Sequential(*logvar_net)