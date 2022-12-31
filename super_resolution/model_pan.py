# model_pan.py
#############################################################
# 
# model: PAN (pixel attention network, from paper "Efficient Image Super-Resolution Using Pixel Attention")
# 
# paper link: https://arxiv.org/abs/2010.01073
#              https://link.springer.com/chapter/10.1007/978-3-030-67070-2_3
# 
# paper info: Hengyuan Zhao, Xiangtao Kong, Jingwen He, Yu Qiao, Chao Dong
#             Efficient Image Super-Resolution Using Pixel Attention
#             In Proceedings of the European Conference on Computer Vision (ECCV) 2020 Workshops. pp.56-72
#
# github link: https://github.com/zhaohengyuan1/PAN
# 
# license info: Not found.
#
#############################################################
# https://github.com/zhaohengyuan1/PAN/blob/master/codes/models/SR_model.py -> train option init
# < How to Use with official paper settings for x4 scale >
#
# model = PAN(scale=4)
#
# loss = torch.nn.L1Loss()
#
# optimizer = torch.optim.Adam(model.parameters()
#                             ,lr = 1e-3
#                             ,weight_decay = 0
#                             ,betas= (0.9, 0.99)
#                             )
#
# number_of_iter = 1000000
#
# info: scheduler with 3 restart. when restart, lr will set to init value. (Below option will goes like ↘↑↘↑↘↑↘)
# scheduler = CosineAnnealingLR_Restart(optimizer = optimizer
#                                      ,T_period  = [250000, 250000, 250000, 250000]    # T_period used before next restart.
#                                                                                       # cosine T_period 옵션. restart 전까지만 유효. restarts 보다 1개 많은 element 필요.
#                                      ,restarts  = [250000, 500000, 750000]            # make cosine step to zero -> make LR init value
#                                                                                       # cos 주기 초기화 iter 값. T_period 보다 1개 적은 element 필요.
#                                      ,weights   = [1, 1, 1]                           # restart init LR multiplier. >1 will make larger initial LR.
#                                                                                       # cos 주기 초기화 시, 시작 LR 배율값. >1 값 사용시, optimizer 설정값보다 큰 LR로 init
#                                      ,eta_min   = 1e-7                                # minimum LR
#                                                                                       # cosine 그래프 최소값.
#                                      )
#
# Patch           = (256, 256)          # HR Patch
# Batch           = 32
# Random_Filp     = True                # horizontal flips
# Random_Rotation = [0, 90, 180, 270]
#
#############################################################
#
# Training option for x4
# https://github.com/zhaohengyuan1/PAN/blob/master/codes/options/train/train_PANx4.yml
# 
# ---[ model ]---
# https://github.com/zhaohengyuan1/PAN/blob/master/codes/models/networks.py
# netG = PAN_arch.PAN(in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=4)
# 
# https://github.com/zhaohengyuan1/PAN/blob/master/codes/models/archs/PAN_arch.py
#

#<<<
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
#import models.archs.arch_util as arch_util #-> https://github.com/zhaohengyuan1/PAN/blob/master/codes/models/archs/arch_util.py#L45

def make_layer(block, n_layers):
    # from models.archs.arch_util
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out
        
class SCPA(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False)
                    )
        
        self.PAConv = PAConv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out


class PAN(nn.Module):
    #def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
    def __init__(self, in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=4):
        super(PAN, self).__init__()
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)
        self.scale = scale
        
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        #self.SCPA_trunk = arch_util.make_layer(SCPA_block_f, nb)
        self.SCPA_trunk = make_layer(SCPA_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk
        
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))
        
        out = self.conv_last(fea)
        
        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out

#>>>
#
#
# ---[ loss ]---
# https://github.com/zhaohengyuan1/PAN/blob/master/codes/models/SR_model.py#L42
# Loss: L1 loss (torch.nn.L1Loss())
# 
# 
# ---[ LR scheduler ]---
# https://github.com/zhaohengyuan1/PAN/blob/master/codes/models/lr_scheduler.py
# CosineAnnealingLR_Restart
#

import math
from collections import Counter
from collections import defaultdict
#import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]



if __name__ == "__main__":
    print("EOF: model_pan.py")