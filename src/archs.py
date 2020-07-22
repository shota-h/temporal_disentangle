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


class TDAE_out(nn.Module):
    def __init__(self, n_class1=3, n_class2=2, ksize=3, d2ae_flag=False, img_w=256, img_h=256):
        super().__init__()
        if d2ae_flag:
            n_class2 = n_class1
        self.img_h, self.img_w = img_h, img_w
        # self.model = models.vgg16(num_classes=n_class, pretrained=False)
        # self.enc = models.resnet18(pretrained=False)
        # self.enc = nn.Sequential(*list(self.enc.children())[:8])
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        self.enc = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        self.subnet_conv_t1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.subnet_conv_t2 = nn.Sequential(nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.subnet_t1 = nn.Sequential(nn.Linear(in_features=64, out_features=256),
                                    nn.ReLU())

        self.subnet_conv_p1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.subnet_conv_p2 = nn.Sequential(nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
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
                                                out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*64),
                                                nn.ReLU())

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 128, 2, stride=2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(16, 3, 2, stride=2),
                                nn.Sigmoid())
        self.deconv1_conv1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.deconv2_conv1 = nn.Sequential(nn.Conv2d(64, 64, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.deconv3_conv1 = nn.Sequential(nn.Conv2d(32, 32, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.deconv4_conv1 = nn.Sequential(nn.Conv2d(16, 16, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.dec = nn.Sequential(self.deconv1,
                                self.deconv2,
                                self.deconv3,
                                self.deconv4,
                                self.deconv5)
        # self.dec = nn.Sequential(self.deconv1, self.deconv1_conv1,
        #                         self.deconv2, self.deconv2_conv1,
        #                         self.deconv3, self.deconv3_conv1,
        #                         self.deconv4, self.deconv4_conv1,
        #                         self.deconv5)

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
        # t0 = self.subnet_conv_t1(h0)
        # p0 = self.subnet_conv_p1(h0)
        # t0 = self.subnet_conv_t2(t0)
        # p0 = self.subnet_conv_p2(p0)
        # t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        # p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        # t0 = torch.reshape(t0, (t0.size(0), -1))
        # p0 = torch.reshape(p0, (p0.size(0), -1))
        # t0 = self.subnet_t1(t0)
        # p0 = self.subnet_p1(p0)

        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        class_main_preds = self.classifier_main(t0)
        class_sub_preds = self.classifier_main(p0)
        class_main_preds_adv = self.classifier_main(p0)
        class_sub_preds_adv = self.classifier_main(t0)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return class_main_preds, class_sub_preds, class_main_preds_adv, class_sub_preds_adv, rec

    def hidden_output(self, input):
        h0 = self.enc(input)
        # t0 = self.subnet_conv_t1(h0)
        # p0 = self.subnet_conv_p1(h0)

        # t0 = self.subnet_conv_t2(t0)
        # p0 = self.subnet_conv_p2(p0)
        
        # t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        # p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        
        # t0 = torch.reshape(t0, (t0.size(0), -1))
        # p0 = torch.reshape(p0, (p0.size(0), -1))
        
        # t0 = self.subnet_t1(t0)
        # p0 = self.subnet_p1(p0)

        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        return t0, p0

    def reconst(self, input):
        h0 = self.enc(input)
        # t0 = self.subnet_conv_t1(h0)
        # p0 = self.subnet_conv_p1(h0)

        # t0 = self.subnet_conv_t2(t0)
        # p0 = self.subnet_conv_p2(p0)
        
        # t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        # p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        
        # t0 = torch.reshape(t0, (t0.size(0), -1))
        # p0 = torch.reshape(p0, (p0.size(0), -1))
        
        # t0 = self.subnet_t1(t0)
        # p0 = self.subnet_p1(p0)

        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec

    def shuffle_reconst(self, input, idx1, idx2):
        h0 = self.enc(input)
        # t0 = self.subnet_conv_t1(h0)
        # p0 = self.subnet_conv_p1(h0)
        # t0 = self.subnet_conv_t2(t0)
        # p0 = self.subnet_conv_p2(p0)
        # t0 = F.avg_pool2d(t0, kernel_size=t0.size()[2])
        # p0 = F.avg_pool2d(p0, kernel_size=p0.size()[2])
        # t0 = torch.reshape(t0, (t0.size(0), -1))
        # p0 = torch.reshape(p0, (p0.size(0), -1))
        # t0 = self.subnet_t1(t0)
        # p0 = self.subnet_p1(p0)

        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        concat_h0 = torch.cat((t0[idx1], p0[idx2]), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return rec


class CrossDisentangleNet(nn.Module):
    def __init__(self, n_class1=3, n_class2=2, ksize=3, img_w=256, img_h=256):
        super().__init__()
        self.img_h, self.img_w = img_h, img_w
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, ksize, stride=2, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())

        self.enc = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        self.subnet_conv_t1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.subnet_conv_t2 = nn.Sequential(nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.subnet_conv_p1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.subnet_conv_p2 = nn.Sequential(nn.Conv2d(128, 64, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.subnet_t1 = nn.Sequential(nn.Linear(in_features=64, out_features=256),
                                    nn.ReLU())
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
                                                out_features=(self.img_w//(2**5))*(self.img_h//(2**5))*64),
                                                nn.ReLU())

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 128, 2, stride=2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.deconv1_conv1 = nn.Sequential(nn.Conv2d(128, 128, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, stride=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.deconv2_conv1 = nn.Sequential(nn.Conv2d(64, 64, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.deconv3_conv1 = nn.Sequential(nn.Conv2d(32, 32, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, stride=2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.deconv4_conv1 = nn.Sequential(nn.Conv2d(16, 16, ksize, padding=(ksize-1)//2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU())
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(16, 3, 2, stride=2),
                                nn.Sigmoid())
        self.dec = nn.Sequential(self.deconv1, self.deconv1_conv1,
                                self.deconv2, self.deconv2_conv1,
                                self.deconv3, self.deconv3_conv1,
                                self.deconv4, self.deconv4_conv1,
                                self.deconv5)

        initialize_weights(self)

    def forward(self, input):
        h0 = self.enc(input)
        t0 = self.subnets_t(h0)
        p0 = self.subnets_p(h0)
        preds_main = self.classifier_main(t0)
        preds_sub = self.classifier_sub(p0)
        adv_preds_main = self.classifier_main(p0)
        adv_preds_sub = self.classifier_sub(t0)
        concat_h0 = torch.cat((t0, p0), dim=1)
        concat_h0 = self.dec_fc1(concat_h0)
        concat_h0 = torch.reshape(concat_h0, (concat_h0.size(0), 64, self.img_h//(2**5), self.img_w//(2**5)))
        rec = self.dec(concat_h0)
        return preds_main, preds_sub, adv_preds_main, adv_preds_sub, rec

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