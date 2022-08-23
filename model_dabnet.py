# model_dabnet.py
#############################################################
#   
#   
#
#   model: DABNet (from paper "DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation")
#
#   paper link: https://bmvc2019.org/wp-content/papers/0955.html
#               https://arxiv.org/abs/1907.11357
#
#   paper info: Gen Li, Joongkyu Kim
#               DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation
#               Proceedings of the British Machine Vision Conference (BMVC) 2019 September
# 
#   github link: https://github.com/Reagan1311/DABNet
#
#   license info: MIT license 
#
#
#   How to Use
#   *CamVid 11 label 기준 원본 이미지를 그대로 입력에 사용함을 확인 (360x480)
#   *논문에서 Loss에 대한 언급을 못찾음 -> 코드에서 CrossEntropyLoss 로 명명된 loss 사용 확인됨
#   *논문에서 입력 이미지 정규화 여부 못찾음 -> 원본 그대로 입력 시행 예정

#   < import model >
#   from model_dabnet import DABNet, PolyLR
#   
#
#   < init >
#   model = DABNet(classes=11)
#   criterion = torch.nn.CrossEntropyLoss()
#   
#
#   < train >
#   
#   
#############################################################





#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
# https://github.com/Reagan1311/DABNet/blob/b8d62fe7f14ae4909a9e9aad1dd6e0ade98431cd/model/DABNet.py
# DABNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import _LRScheduler
import types
import warnings

#__all__ = ["DABNet"]


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                             padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class DABNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 3)

        # DAB Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + 3)

        # DAB Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)

        self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))

    def forward(self, input):

        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))

        # DAB Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))

        out = self.classifier(output2_cat)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        return out

# DABNet.py
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
# Poly Scheduler

class PolyLR(_LRScheduler):
    # modified version of LambdaLR from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
    def __init__(self, optimizer, last_epoch=-1, verbose=False, power=0.9, max_epoch=5000):
        self.optimizer = optimizer
        lr_lambda = lambda x : (1 - x/max_epoch)**power
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(PolyLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict


    def load_state_dict(self, state_dict):
        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        out_lr_raw = [base_lr * lmbda(self.last_epoch) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
        out_lr = out_lr_raw[0]
        
        try:    # complex 클래스 발생 문제 해결목적
            if out_lr < 0.0:
                warnings.warn("(PolyLR) lr fixed to 0.0")
                out_lr = 0.0
        except:
            warnings.warn("(PolyLR) lr generated as complex value")
            warnings.warn("(PolyLR) lr fixed to 0.0")
            out_lr = 0.0
        
        return [out_lr]

#=== End of PolyLR

print("EoF: model_dabnet.py")