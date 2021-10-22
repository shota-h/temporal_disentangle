import copy
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


class TDAE_D2AE(nn.Module):
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
            return  classifier_preds[0], classifier_preds[1], disentangle_classifier_preds[0], disentangle_classifier_preds[1], disentangle_classifier_preds[2], disentangle_classifier_preds[3], rec, output_subnets[0], output_subnets[1]

        return classifier_preds[0], classifier_preds[1], disentangle_classifier_preds[0], disentangle_classifier_preds[1], disentangle_classifier_preds[2], disentangle_classifier_preds[3], rec

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
        return classifier_preds[0], classifier_preds[1], classifier_preds[2], rec, mu1, mu2, logvar1, logvar2

    def predict_label(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        zs = [mu1, mu2]

        classifier_preds = []
        for i in range(len(self.classifiers)):
            classifier_preds.append(self.classifiers[i](zs[i]))
        
        return torch.argmax(classifier_preds[0], 1), torch.argmax(classifier_preds[1], 1)

    def hidden_output(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        # z1 = self.reparameterize(mu1, logvar1)
        # z2 = self.reparameterize(mu2, logvar2)
        return mu1, mu2

    def sampling(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        return z1, z2

    def reconst(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)        
        rec = self.decode(mu1, mu2)
        return rec

    def shuffle_reconst(self, input, idx1, idx2, shuffle_idx=[1, 0]):
        h0 = self.enc(input)
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        zs = copy.deepcopy([mu1, mu2])        
        idx = [idx1, idx2]
        for i in range(len(zs)):
            zs[i] = zs[i][idx[i]]
        rec = self.decode(zs[0], zs[1])
        return rec

    def fix_padding_reconst(self, input, which_val, pad_val):
        h0 = self.enc(input)
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        
        pad_tensor = torch.ones_like(z1) * pad_val
        zs = copy.deepcopy([mu1, mu2])        
        zs[which_val] = pad_tensor
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

        
class TDAE_VAE_fullsuper_disentangle(nn.Module):
    def __init__(self, n_classes, ksize=3, img_w=256, img_h=256, channels=[3, 16, 32, 64, 128], n_decov=2, latent_dim=256, base_net='res', triplet=False):
        super(TDAE_VAE_fullsuper_disentangle, self).__init__()
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

        mu_nets = []
        logvar_nets = []
        for i in range(len(n_classes)):
            subnets.append(self.get_subnet(channel=subnet_input_dim, ksize=ksize, nconv=2))
            mu_nets.append(self.get_mu_net())
            logvar_nets.append(self.get_logvar_net())
            classifiers.append(nn.Linear(in_features=self.latent_dim, out_features=n_classes[i]))
        for i in [1, 0]:
            disentangle_classifiers.append(nn.Linear(in_features=self.latent_dim, out_features=n_classes[i]))

        self.subnets = nn.ModuleList(subnets)
        self.classifiers = nn.ModuleList(classifiers)
        self.disentangle_classifiers = nn.ModuleList(disentangle_classifiers)
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

        mu1_no_grad = mu1.clone().detach()
        mu2_no_grad = mu2.clone().detach()
        classifier_preds = []
        dis_classifier_preds = []
        classifier_preds.append(self.classifiers[0](mu1))
        classifier_preds.append(self.classifiers[1](mu2))
        dis_classifier_preds.append(self.disentangle_classifiers[0](mu1))
        dis_classifier_preds.append(self.disentangle_classifiers[0](mu1_no_grad))
        dis_classifier_preds.append(self.disentangle_classifiers[1](mu2))
        dis_classifier_preds.append(self.disentangle_classifiers[1](mu2_no_grad))
        rec = self.decode(z1, z2)
        return classifier_preds[0], classifier_preds[1], dis_classifier_preds[0], dis_classifier_preds[1], dis_classifier_preds[2], dis_classifier_preds[3], rec, mu1, mu2, logvar1, logvar2

    def predict_label(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        zs = [mu1, mu2]

        classifier_preds = []
        for i in range(len(self.classifiers)):
            classifier_preds.append(self.classifiers[i](zs[i]))
        
        return torch.argmax(classifier_preds[0], 1), torch.argmax(classifier_preds[1], 1)

    def hidden_output(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        # z1 = self.reparameterize(mu1, logvar1)
        # z2 = self.reparameterize(mu2, logvar2)
        return mu1, mu2

    def sampling(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        return z1, z2

    def reconst(self, input):
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)        
        rec = self.decode(mu1, mu2)
        return rec

    def shuffle_reconst(self, input, idx1, idx2, shuffle_idx=[1, 0]):
        h0 = self.enc(input)
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)

        zs = copy.deepcopy([mu1, mu2])        
        idx = [idx1, idx2]
        for i in range(len(zs)):
            zs[i] = zs[i][idx[i]]
        rec = self.decode(zs[0], zs[1])
        return rec

    def fix_padding_reconst(self, input, which_val, pad_val):
        h0 = self.enc(input)
        mu1, mu2, logvar1, logvar2 = self.encode(input)
        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        
        pad_tensor = torch.ones_like(z1) * pad_val
        zs = copy.deepcopy([mu1, mu2])        
        zs[which_val] = pad_tensor
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


class SemiSelfClassifier(nn.Module):
    def __init__(self, n_classes, ksize=3, img_w=256, img_h=256, channels=[3, 16, 32, 64, 128], n_decov=2, latent_dim=256, base_net='res', triplet=False):
        super(SemiSelfClassifier, self).__init__()
        self.img_h, self.img_w = img_h, img_w
        self.channels = channels
        self.latent_dim = latent_dim
        self.triplet = triplet
        subnets = []
        classifiers = []
        disentangle_classifiers = []
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
            classifiers.append(nn.Linear(in_features=self.latent_dim, out_features=n_classes[i]))
        for i in [1, 0]:
            disentangle_classifiers.append(nn.Linear(in_features=self.latent_dim, out_features=n_classes[i]))

        self.subnets = nn.ModuleList(subnets)
        self.classifiers = nn.ModuleList(classifiers)
        self.disentangle_classifiers = nn.ModuleList(disentangle_classifiers)
        initialize_weights(self)

    def encode(self, input):
        h0 = self.enc(input)
        output_subnets = []
        for i in range(len(self.subnets)):
            output_subnets.append(self.subnets[i](h0))
        return output_subnets[0], output_subnets[1]

    def forward(self, input, latent=False):
        z1, z2 = self.encode(input)
        z1_no_grad = z1.clone().detach()
        z2_no_grad = z2.clone().detach()
        classifier_preds = []
        dis_classifier_preds = []
        classifier_preds.append(self.classifiers[0](z1))
        classifier_preds.append(self.classifiers[1](z2))
        dis_classifier_preds.append(self.disentangle_classifiers[0](z1))
        dis_classifier_preds.append(self.disentangle_classifiers[0](z1_no_grad))
        dis_classifier_preds.append(self.disentangle_classifiers[1](z2))
        dis_classifier_preds.append(self.disentangle_classifiers[1](z2_no_grad))
        return classifier_preds[0], classifier_preds[1], dis_classifier_preds[0], dis_classifier_preds[1], dis_classifier_preds[2], dis_classifier_preds[3], z1, z2

    def predict_label(self, input):
        z1, z2 = self.encode(input)
        zs = [z1, z2]

        classifier_preds = []
        for i in range(len(self.classifiers)):
            classifier_preds.append(self.classifiers[i](zs[i]))
        
        return torch.argmax(classifier_preds[0], 1), torch.argmax(classifier_preds[1], 1)

    def get_last_output(self, input):
        z1, z2 = self.encode(input)
        zs = [z1, z2]

        classifier_preds = []
        for i in range(len(self.classifiers)):
            classifier_preds.append(self.classifiers[i](zs[i]))
        
        return classifier_preds[0], classifier_preds[1]
    
    def predict_proba(self, input):
        z1, z2 = self.encode(input)
        zs = [z1, z2]

        classifier_preds = []
        for i in range(len(self.classifiers)):
            classifier_preds.append(torch.nn.Softmax(dim=1)(self.classifiers[i](zs[i])))
        
        return classifier_preds[0], classifier_preds[1]
    
    def hidden_output(self, input):
        z1, z2 = self.encode(input)
        return z1, z2

    def get_subnet(self, channel, ksize, nconv):
        subnet_layers = []
        for i in range(nconv):
            subnet_layers.append(base_conv(channel, channel, ksize, stride=1))
        subnet_layers.append(nn.AvgPool2d(kernel_size=self.img_w//(2**5)))
        subnet_layers.append(Flatten())
        subnet_layers.append(nn.Linear(in_features=channel, out_features=self.latent_dim))
        subnet_layers.append(nn.ReLU())
        return nn.Sequential(*subnet_layers)

    def param_update(self, x, y, weight_dict):
        device = next(self.model.parameters()).device

        preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, z1, z2 = self.forward(x)
        loss_classifier_main = weight_dict['classifier_main'] * criterion_classifier(preds.to(device), target.to(device))
        loss_classifier_sub = weight_dict['classifier_sub'] * criterion_classifier(sub_preds.to(device), sub_target.to(device))
        loss_classifier_main.backward(retain_graph=True)
        loss_classifier_sub.backward(retain_graph=True)
        
        return

    def param_update_withSSL(self, x, y):
        return
    
    
class SingleClassifier(nn.Module):
    def __init__(self, n_class=2, ksize=3, img_w=224, img_h=224, channels=[3, 16, 32, 64, 128], latent_dim=256, base_net='res'):
        super(SingleClassifier, self).__init__()
        self.img_h, self.img_w = img_h, img_w
        self.channels = channels
        self.latent_dim = latent_dim
        subnets = []
        classifiers = []
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

        self.subnet = self.get_subnet(channel=subnet_input_dim, ksize=ksize, nconv=2)
        self.classifier = nn.Linear(in_features=self.latent_dim, out_features=n_class)

        initialize_weights(self)

    def encode(self, input):
        h0 = self.enc(input)
        return self.subnet(h0)

    def forward(self, input, latent=False):
        z0 = self.encode(input)
        return self.classifier(z0)

    def predict_label(self, input):
        z0 = self.encode(input)
        preds = self.classifier(z0)
        
        return torch.argmax(preds, 1)

    def get_last_output(self, input):
        z0 = self.encode(input)
        return self.classifier(z0)
            
    def predict_proba(self, input):
        z0 = self.encode(input)
        outputs = self.classifier(z0)
        return torch.nn.Softmax(dim=1)(outputs)
            
    def hidden_output(self, input):
        z0 = self.encode(input)
        return z0

    def get_subnet(self, channel, ksize, nconv):
        subnet_layers = []
        for i in range(nconv):
            subnet_layers.append(base_conv(channel, channel, ksize, stride=1))
        subnet_layers.append(nn.AvgPool2d(kernel_size=self.img_w//(2**5)))
        subnet_layers.append(Flatten())
        subnet_layers.append(nn.Linear(in_features=channel, out_features=self.latent_dim))
        subnet_layers.append(nn.ReLU())
        return nn.Sequential(*subnet_layers)

    def param_update(self, x, y, weight_dict):
        device = next(self.model.parameters()).device

        preds, sub_preds, preds_adv, preds_adv_no_grad, sub_preds_adv, sub_preds_adv_no_grad, z1, z2 = self.forward(x)
        loss_classifier_main = weight_dict['classifier_main'] * criterion_classifier(preds.to(device), target.to(device))
        loss_classifier_sub = weight_dict['classifier_sub'] * criterion_classifier(sub_preds.to(device), sub_target.to(device))
        loss_classifier_main.backward(retain_graph=True)
        loss_classifier_sub.backward(retain_graph=True)
        
        return

    def param_update_withSSL(self, x, y):
        return