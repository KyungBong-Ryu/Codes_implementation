# modrl_lapar_a.py
#############################################################
# 
# model: LAPAR_A (one of proposed networks from "LAPAR: Linearly-Assembled Pixel-Adaptive Regression Network for Single Image Super-resolution and Beyond")
# 
# paper link: https://papers.nips.cc/paper/2020/file/eaae339c4d89fc102edd9dbdb6a28915-Paper.pdf
#             https://arxiv.org/abs/2105.10422
# 
# paper info: Wenbo Li, Kun Zhou, Lu Qi, Nianjuan Jiang, Jiangbo Lu, Jiaya Jia
#             LAPAR: Linearly-Assembled Pixel-Adaptive Regression Network for Single Image Super-resolution and Beyond
#             Advances in Neural Information Processing Systems, vol.33, 2022.
#
# github link: https://github.com/dvlab-research/Simple-SR/tree/master/exps/LAPAR_A_x4
# 
# license info: MIT license 
#
#############################################################

# Official x2, x4 pretrained weight ("https://drive.google.com/drive/folders/1c-KUEPJl7pHs9btqHYoUJkcMPKViObgJ?usp=sharing") can be loaded with this code.
#
# < How to Use >
# from DLCs.model_lapar_a import Network as LAPAR_A
# model = LAPAR_A(scale=4)
# 
# optimizer = optim.Adam(model.parameters()
#                       ,lr             = 4e-4
#                       ,betas          = (0.9, 0.999)
#                       ,eps            = 1e-8
#                       ,weight_decay   = 0
#                       )
#
# scheduler = CosineAnnealingLR_warmup(optimizer
#                                     ,warm_iter = 2000
#                                     ,warm_factor = 0.1
#                                     ,base_lr = 4e-4
#                                     ,min_lr = 1e-7
#                                     ,t_period = [200000, 400000, 600000]    # total train iter = 600 k
#                                     )
# 
# loss = torch.nn.L1Loss(reduction='mean')
# 
# 
# 
# 
#
# 

import math
import os
import pickle
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
#<<<=============================================================================================== config.py (x4)
# https://github.com/dvlab-research/Simple-SR/blob/master/exps/LAPAR_A_x4/config.py

class Config_LAPAR_A:
    # dataset
    DATASET = edict()
    DATASET.TYPE = 'MixDataset'
    DATASET.DATASETS = ['DIV2K', 'Flickr2K']
    DATASET.SPLITS = ['TRAIN', 'TRAIN']
    DATASET.PHASE = 'train'
    DATASET.INPUT_HEIGHT = 64
    DATASET.INPUT_WIDTH = 64
    #DATASET.SCALE = 4
    DATASET.REPEAT = 1
    DATASET.VALUE_RANGE = 255.0
    DATASET.SEED = 100

    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 32
    DATALOADER.NUM_WORKERS = 8

    # model
    MODEL = edict()
    #MODEL.SCALE = DATASET.SCALE
    MODEL.KERNEL_SIZE = 5
    #MODEL.KERNEL_PATH = '../../kernel/kernel_72_k5.pkl'    # https://github.com/dvlab-research/Simple-SR/blob/master/kernel/kernel_72_k5.pkl
    _kernel_path = os.path.dirname(os.path.abspath(__file__)) + '/model_lapar_a_kernel_72_k5.pkl'
    MODEL.KERNEL_PATH = _kernel_path.replace("\\", "/")
    MODEL.IN_CHANNEL = 3
    MODEL.N_CHANNEL = 32
    MODEL.RES_BLOCK = 4
    MODEL.N_WEIGHT = 72
    MODEL.DOWN = 1
    MODEL.DEVICE = 'cuda'

    # solver
    SOLVER = edict()
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.BASE_LR = 4e-4
    SOLVER.BETA1 = 0.9
    SOLVER.BETA2 = 0.999
    SOLVER.WEIGHT_DECAY = 0
    SOLVER.MOMENTUM = 0
    SOLVER.WARM_UP_ITER = 2000
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.T_PERIOD = [200000, 400000, 600000]
    SOLVER.MAX_ITER = SOLVER.T_PERIOD[-1]

    # initialization
    CONTINUE_ITER = None
    INIT_MODEL = None

    # log and save
    LOG_PERIOD = 20
    SAVE_PERIOD = 10000

    # validation
    VAL = edict()
    VAL.PERIOD = 10000
    VAL.TYPE = 'MixDataset'
    VAL.DATASETS = ['BSDS100']
    VAL.SPLITS = ['VAL']
    VAL.PHASE = 'val'
    VAL.INPUT_HEIGHT = None
    VAL.INPUT_WIDTH = None
    #VAL.SCALE = DATASET.SCALE
    VAL.REPEAT = 1
    VAL.VALUE_RANGE = 255.0
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.SAVE_IMG = False
    VAL.TO_Y = True
    #VAL.CROP_BORDER = VAL.SCALE

config_LAPAR_A = Config_LAPAR_A()

#>>>=============================================================================================== config.py (x4)


#<<<=============================================================================================== lightWeightNet.py
# https://github.com/dvlab-research/Simple-SR/blob/master/utils/modules/lightWeightNet.py

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super(Scale, self).__init__()
        self.scale = Parameter(torch.FloatTensor([init_value]))
    
    def forward(self, x):
        return x * self.scale


class AWRU(nn.Module):
    def __init__(self, nf, kernel_size, wn, act=nn.ReLU(True)):
        super(AWRU, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        
        self.body = nn.Sequential(
            wn(nn.Conv2d(nf, nf, kernel_size, padding=kernel_size//2)),
            act,
            wn(nn.Conv2d(nf, nf, kernel_size, padding=kernel_size//2)),
        )
    
    def forward(self, x):
        res = self.res_scale(self.body(x)) + self.x_scale(x)
        return res


class AWMS(nn.Module):
    def __init__(self, nf, out_chl, wn, act=nn.ReLU(True)):
        super(AWMS, self).__init__()
        self.tail_k3 = wn(nn.Conv2d(nf, nf, 3, padding=3//2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(nf, nf, 5, padding=5//2, dilation=1))
        self.scale_k3 = Scale(0.5)
        self.scale_k5 = Scale(0.5)
        self.fuse = wn(nn.Conv2d(nf, nf, 3, padding=3 // 2))
        self.act = act
        self.w_conv = wn(nn.Conv2d(nf, out_chl, 3, padding=3//2))
    
    def forward(self, x):
        x0 = self.scale_k3(self.tail_k3(x))
        x1 = self.scale_k5(self.tail_k5(x))
        cur_x = x0 + x1
        
        fuse_x = self.act(self.fuse(cur_x))
        out = self.w_conv(fuse_x)
        
        return out


class LFB(nn.Module):
    def __init__(self, nf, wn, act=nn.ReLU(inplace=True)):
        super(LFB, self).__init__()
        self.b0 = AWRU(nf, 3, wn=wn, act=act)
        self.b1 = AWRU(nf, 3, wn=wn, act=act)
        self.b2 = AWRU(nf, 3, wn=wn, act=act)
        self.b3 = AWRU(nf, 3, wn=wn, act=act)
        self.reduction = wn(nn.Conv2d(nf * 4, nf, 3, padding=3//2))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        
    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        res = self.reduction(torch.cat([x0, x1, x2, x3], dim=1))
        
        return self.res_scale(res) + self.x_scale(x)


class WeightNet(nn.Module):
    def __init__(self, scale, config):
        super(WeightNet, self).__init__()
        
        in_chl = config.IN_CHANNEL
        nf = config.N_CHANNEL
        n_block = config.RES_BLOCK
        out_chl = config.N_WEIGHT
        #scale = config.SCALE
        
        act = nn.ReLU(inplace=True)
        wn = lambda x: nn.utils.weight_norm(x)
        
        rgb_mean = torch.FloatTensor([0.4488, 0.4371, 0.4040]).view([1, 3, 1, 1]) 
        self.register_buffer('rgb_mean', rgb_mean)
        
        self.head = nn.Sequential(
            wn(nn.Conv2d(in_chl, nf, 3, padding=3//2)),
            act,
        )
        
        body = []
        for i in range(n_block):
            body.append(LFB(nf, wn=wn, act=act))
        self.body = nn.Sequential(*body)
        
        self.up = nn.Sequential(
            wn(nn.Conv2d(nf, nf * scale ** 2, 3, padding=3//2)),
            act,
            nn.PixelShuffle(upscale_factor=scale)
        )
        
        self.tail = AWMS(nf, out_chl, wn, act=act)

    def forward(self, x):
        x = x - self.rgb_mean
        x = self.head(x)
        x = self.body(x)
        x = self.up(x)
        out = self.tail(x)
        
        return out

#>>>=============================================================================================== lightWeightNet.py


#<<<=============================================================================================== network.py (x2, x3, x4 동일)
# https://github.com/dvlab-research/Simple-SR/blob/master/exps/LAPAR_A_x4/network.py
# from utils.modules.lightWeightNet import WeightNet

class ComponentDecConv(nn.Module):
    def __init__(self, k_path, k_size):
        super(ComponentDecConv, self).__init__()
        
        kernel = pickle.load(open(k_path, 'rb'))
        kernel = torch.from_numpy(kernel).float().view(-1, 1, k_size, k_size)
        self.register_buffer('weight', kernel)
        
    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=None, stride=1, padding=0, groups=1)
        
        return out


class Network(nn.Module):
    
    def __init__(self, scale, config=config_LAPAR_A):
        super(Network, self).__init__()
        
        self.k_size = config.MODEL.KERNEL_SIZE
        #self.s = config.MODEL.SCALE
        self.s = scale  # option (scale factor): 2, 3, 4
        
        self.w_conv = WeightNet(scale, config.MODEL)
        self.decom_conv = ComponentDecConv(config.MODEL.KERNEL_PATH, self.k_size)
        
    
    def forward(self, x):
        B, C, H, W = x.size()
        
        bic = F.interpolate(x, scale_factor=self.s, mode='bicubic', align_corners=False)
        pad = self.k_size // 2
        x_pad = F.pad(bic, pad=(pad, pad, pad, pad), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = self.decom_conv(x_pad).view(B, 3, -1, self.s * H, self.s * W)  # B, 3, N_K, Hs, Ws
        
        weight = self.w_conv(x)
        weight = weight.view(B, 1, -1, self.s * H, self.s * W)  # B, 1, N_K, Hs, Ws
        
        out = torch.sum(weight * x_com, dim=2)
        
        return out
    
    '''
    def __init__(self, config):
        super(Network, self).__init__()
        
        self.k_size = config.MODEL.KERNEL_SIZE
        self.s = config.MODEL.SCALE
        
        self.w_conv = WeightNet(config.MODEL)
        self.decom_conv = ComponentDecConv(config.MODEL.KERNEL_PATH, self.k_size)
        
        self.criterion = nn.L1Loss(reduction='mean')
        
        
    def forward(self, x, gt=None):
        B, C, H, W = x.size()
        
        bic = F.interpolate(x, scale_factor=self.s, mode='bicubic', align_corners=False)
        pad = self.k_size // 2
        x_pad = F.pad(bic, pad=(pad, pad, pad, pad), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = self.decom_conv(x_pad).view(B, 3, -1, self.s * H, self.s * W)  # B, 3, N_K, Hs, Ws
        
        weight = self.w_conv(x)
        weight = weight.view(B, 1, -1, self.s * H, self.s * W)  # B, 1, N_K, Hs, Ws
        
        out = torch.sum(weight * x_com, dim=2)
        
        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out
    '''

#>>>=============================================================================================== network.py 


#<<<=============================================================================================== solver.py
# https://github.com/dvlab-research/Simple-SR/blob/master/utils/solver.py

# LR scheduler 고치기

class CosineAnnealingLR_warmup(_LRScheduler):
    def __init__(self
                ,optimizer          # ADAM
                ,warm_iter          # 2000                     -> warm-up iters. Every restart, warm_iter period applied.
                ,warm_factor        # 0.1                      -> warm-up muliply factor for init LR of warm-up phase.
                ,base_lr            # 4e-4                     -> maximum LR
                ,min_lr             # 1e-7                     -> minimum LR
                ,t_period           # [200000, 400000, 600000] -> last element is max iter. This case, 600k is max iter.
                ,last_epoch = -1
                ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        
        #self.w_iter = config.SOLVER.WARM_UP_ITER
        #self.w_fac = config.SOLVER.WARM_UP_FACTOR
        #self.T_period = config.SOLVER.T_PERIOD
        
        self.w_iter = warm_iter
        self.w_fac = warm_factor
        self.T_period = t_period
        self.last_restart = 0
        self.T_max = self.T_period[0]
        #assert config.SOLVER.MAX_ITER == self.T_period[-1], 'Illegal training period setting.'
        
        
        super(CosineAnnealingLR_warmup, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch - self.last_restart < self.w_iter:
            ratio = self.w_fac + (1 - self.w_fac) * (self.last_epoch - self.last_restart) / self.w_iter
            return [(self.base_lr - self.min_lr) * ratio + self.min_lr for group in self.optimizer.param_groups]
        elif self.last_epoch in self.T_period:
            self.last_restart = self.last_epoch
            if self.last_epoch != self.T_period[-1]:
                self.T_max = self.T_period[self.T_period.index(self.last_epoch) + 1]
            return [self.min_lr for group in self.optimizer.param_groups]
        else:
            ratio = 1 / 2 * (1 + math.cos(
                (self.last_epoch - self.last_restart - self.w_iter) / (self.T_max - self.last_restart - self.w_iter) * math.pi))
            return [(self.base_lr - self.min_lr) * ratio + self.min_lr for group in self.optimizer.param_groups]
#>>>=============================================================================================== solver.py




print("EOF: model_lapar_a.py")