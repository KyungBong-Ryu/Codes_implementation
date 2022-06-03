# model_dsrl.py

############################################################################
# [intro]
# Dual Super-Resolution Learning for semantic segmentation -> model & loss
# Paper 
# https://ieeexplore.ieee.org/document/9157434
# https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.pdf
# 
# Original code has gone... so I found Re-implementation one below. (with MIT license)
# https://github.com/Dootmaan/DSRL
# 
# [how to use]
# 1. model @@@ 각 인자들 설명 추가 필요
#   model = DeepLab_DSRL(num_classes= 
#                       ,backbone= "xception"
#                       )
#
#   output,output_sr,fea_seg,fea_sr = model(input_img)
#   
# 2. loss
#   criterion = loss_for_dsrl()
#   
#   loss = criterion.calc(output,output_sr,fea_seg,fea_sr,ans_label,ans_sr)
#   
############################################################################


import sys

# [model] ****************************************************************************************************
# https://github.com/Dootmaan/DSRL/blob/1822d6469dd8cca4b61afc42daa6df98067b4f80/modeling/deeplab.py

import torch
import torch.nn as nn
import torch.nn.functional as F

#fold "if True" for readability!
#<<< from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
if True:
    # -*- coding: utf-8 -*-
    # File   : batchnorm.py
    # Author : Jiayuan Mao
    # Email  : maojiayuan@gmail.com
    # Date   : 27/01/2018
    #
    # This file is part of Synchronized-BatchNorm-PyTorch.
    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # Distributed under MIT License.

    import collections

    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

    #<<< from .comm import SyncMaster
    if True:
        # -*- coding: utf-8 -*-
        # File   : comm.py
        # Author : Jiayuan Mao
        # Email  : maojiayuan@gmail.com
        # Date   : 27/01/2018
        #
        # This file is part of Synchronized-BatchNorm-PyTorch.
        # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
        # Distributed under MIT License.

        import queue
        import collections
        import threading

        __all__ = ['FutureResult', 'SlavePipe', 'SyncMaster']


        class FutureResult(object):
            """A thread-safe future implementation. Used only as one-to-one pipe."""

            def __init__(self):
                self._result = None
                self._lock = threading.Lock()
                self._cond = threading.Condition(self._lock)

            def put(self, result):
                with self._lock:
                    assert self._result is None, 'Previous result has\'t been fetched.'
                    self._result = result
                    self._cond.notify()

            def get(self):
                with self._lock:
                    if self._result is None:
                        self._cond.wait()

                    res = self._result
                    self._result = None
                    return res


        _MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
        _SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


        class SlavePipe(_SlavePipeBase):
            """Pipe for master-slave communication."""

            def run_slave(self, msg):
                self.queue.put((self.identifier, msg))
                ret = self.result.get()
                self.queue.put(True)
                return ret


        class SyncMaster(object):
            """An abstract `SyncMaster` object.
            - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
            call `register(id)` and obtain an `SlavePipe` to communicate with the master.
            - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
            and passed to a registered callback.
            - After receiving the messages, the master device should gather the information and determine to message passed
            back to each slave devices.
            """

            def __init__(self, master_callback):
                """
                Args:
                    master_callback: a callback to be invoked after having collected messages from slave devices.
                """
                self._master_callback = master_callback
                self._queue = queue.Queue()
                self._registry = collections.OrderedDict()
                self._activated = False

            def __getstate__(self):
                return {'master_callback': self._master_callback}

            def __setstate__(self, state):
                self.__init__(state['master_callback'])

            def register_slave(self, identifier):
                """
                Register an slave device.
                Args:
                    identifier: an identifier, usually is the device id.
                Returns: a `SlavePipe` object which can be used to communicate with the master device.
                """
                if self._activated:
                    assert self._queue.empty(), 'Queue is not clean before next initialization.'
                    self._activated = False
                    self._registry.clear()
                future = FutureResult()
                self._registry[identifier] = _MasterRegistry(future)
                return SlavePipe(identifier, self._queue, future)

            def run_master(self, master_msg):
                """
                Main entry for the master device in each forward pass.
                The messages were first collected from each devices (including the master device), and then
                an callback will be invoked to compute the message to be sent back to each devices
                (including the master device).
                Args:
                    master_msg: the message that the master want to send to itself. This will be placed as the first
                    message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.
                Returns: the message to be sent back to the master device.
                """
                self._activated = True

                intermediates = [(0, master_msg)]
                for i in range(self.nr_slaves):
                    intermediates.append(self._queue.get())

                results = self._master_callback(intermediates)
                assert results[0][0] == 0, 'The first result should belongs to the master.'

                for i, res in results:
                    if i == 0:
                        continue
                    self._registry[i].result.put(res)

                for i in range(self.nr_slaves):
                    assert self._queue.get() is True

                return results[0][1]

            @property
            def nr_slaves(self):
                return len(self._registry)

    #>>> from .comm import SyncMaster

    __all__ = ['SynchronizedBatchNorm1d', 'SynchronizedBatchNorm2d', 'SynchronizedBatchNorm3d']


    def _sum_ft(tensor):
        """sum over the first and last dimention"""
        return tensor.sum(dim=0).sum(dim=-1)


    def _unsqueeze_ft(tensor):
        """add new dementions at the front and the tail"""
        return tensor.unsqueeze(0).unsqueeze(-1)


    _ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
    _MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


    class _SynchronizedBatchNorm(_BatchNorm):
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
            super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

            self._sync_master = SyncMaster(self._data_parallel_master)

            self._is_parallel = False
            self._parallel_id = None
            self._slave_pipe = None

        def forward(self, input):
            # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
            if not (self._is_parallel and self.training):
                return F.batch_norm(
                    input, self.running_mean, self.running_var, self.weight, self.bias,
                    self.training, self.momentum, self.eps)

            # Resize the input to (B, C, -1).
            input_shape = input.size()
            input = input.view(input.size(0), self.num_features, -1)

            # Compute the sum and square-sum.
            sum_size = input.size(0) * input.size(2)
            input_sum = _sum_ft(input)
            input_ssum = _sum_ft(input ** 2)

            # Reduce-and-broadcast the statistics.
            if self._parallel_id == 0:
                mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
            else:
                mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

            # Compute the output.
            if self.affine:
                # MJY:: Fuse the multiplication for speed.
                output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
            else:
                output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

            # Reshape it.
            return output.view(input_shape)

        def __data_parallel_replicate__(self, ctx, copy_id):
            self._is_parallel = True
            self._parallel_id = copy_id

            # parallel_id == 0 means master device.
            if self._parallel_id == 0:
                ctx.sync_master = self._sync_master
            else:
                self._slave_pipe = ctx.sync_master.register_slave(copy_id)

        def _data_parallel_master(self, intermediates):
            """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

            # Always using same "device order" makes the ReduceAdd operation faster.
            # Thanks to:: Tete Xiao (http://tetexiao.com/)
            intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

            to_reduce = [i[1][:2] for i in intermediates]
            to_reduce = [j for i in to_reduce for j in i]  # flatten
            target_gpus = [i[1].sum.get_device() for i in intermediates]

            sum_size = sum([i[1].sum_size for i in intermediates])
            sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
            mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

            broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

            outputs = []
            for i, rec in enumerate(intermediates):
                outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))

            return outputs

        def _compute_mean_std(self, sum_, ssum, size):
            """Compute the mean and standard-deviation with sum and square-sum. This method
            also maintains the moving average on the master device."""
            assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
            mean = sum_ / size
            sumvar = ssum - sum_ * mean
            unbias_var = sumvar / (size - 1)
            bias_var = sumvar / size

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

            return mean, bias_var.clamp(self.eps) ** -0.5


    class SynchronizedBatchNorm1d(_SynchronizedBatchNorm):
        r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
        mini-batch.
        .. math::
            y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
        This module differs from the built-in PyTorch BatchNorm1d as the mean and
        standard-deviation are reduced across all devices during training.
        For example, when one uses `nn.DataParallel` to wrap the network during
        training, PyTorch's implementation normalize the tensor on each device using
        the statistics only on that device, which accelerated the computation and
        is also easy to implement, but the statistics might be inaccurate.
        Instead, in this synchronized version, the statistics will be computed
        over all training samples distributed on multiple devices.
        Note that, for one-GPU or CPU-only case, this module behaves exactly same
        as the built-in PyTorch implementation.
        The mean and standard-deviation are calculated per-dimension over
        the mini-batches and gamma and beta are learnable parameter vectors
        of size C (where C is the input size).
        During training, this layer keeps a running estimate of its computed mean
        and variance. The running sum is kept with a default momentum of 0.1.
        During evaluation, this running mean/variance is used for normalization.
        Because the BatchNorm is done over the `C` dimension, computing statistics
        on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm
        Args:
            num_features: num_features from an expected input of size
                `batch_size x num_features [x width]`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Default: 0.1
            affine: a boolean value that when set to ``True``, gives the layer learnable
                affine parameters. Default: ``True``
        Shape:
            - Input: :math:`(N, C)` or :math:`(N, C, L)`
            - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
        Examples:
            >>> # With Learnable Parameters
            >>> m = SynchronizedBatchNorm1d(100)
            >>> # Without Learnable Parameters
            >>> m = SynchronizedBatchNorm1d(100, affine=False)
            >>> input = torch.autograd.Variable(torch.randn(20, 100))
            >>> output = m(input)
        """

        def _check_input_dim(self, input):
            if input.dim() != 2 and input.dim() != 3:
                raise ValueError('expected 2D or 3D input (got {}D input)'
                                 .format(input.dim()))
            super(SynchronizedBatchNorm1d, self)._check_input_dim(input)


    class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
        r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
        of 3d inputs
        .. math::
            y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
        This module differs from the built-in PyTorch BatchNorm2d as the mean and
        standard-deviation are reduced across all devices during training.
        For example, when one uses `nn.DataParallel` to wrap the network during
        training, PyTorch's implementation normalize the tensor on each device using
        the statistics only on that device, which accelerated the computation and
        is also easy to implement, but the statistics might be inaccurate.
        Instead, in this synchronized version, the statistics will be computed
        over all training samples distributed on multiple devices.
        Note that, for one-GPU or CPU-only case, this module behaves exactly same
        as the built-in PyTorch implementation.
        The mean and standard-deviation are calculated per-dimension over
        the mini-batches and gamma and beta are learnable parameter vectors
        of size C (where C is the input size).
        During training, this layer keeps a running estimate of its computed mean
        and variance. The running sum is kept with a default momentum of 0.1.
        During evaluation, this running mean/variance is used for normalization.
        Because the BatchNorm is done over the `C` dimension, computing statistics
        on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm
        Args:
            num_features: num_features from an expected input of
                size batch_size x num_features x height x width
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Default: 0.1
            affine: a boolean value that when set to ``True``, gives the layer learnable
                affine parameters. Default: ``True``
        Shape:
            - Input: :math:`(N, C, H, W)`
            - Output: :math:`(N, C, H, W)` (same shape as input)
        Examples:
            >>> # With Learnable Parameters
            >>> m = SynchronizedBatchNorm2d(100)
            >>> # Without Learnable Parameters
            >>> m = SynchronizedBatchNorm2d(100, affine=False)
            >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
            >>> output = m(input)
        """

        def _check_input_dim(self, input):
            if input.dim() != 4:
                raise ValueError('expected 4D input (got {}D input)'
                                 .format(input.dim()))
            super(SynchronizedBatchNorm2d, self)._check_input_dim(input)


    class SynchronizedBatchNorm3d(_SynchronizedBatchNorm):
        r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
        of 4d inputs
        .. math::
            y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
        This module differs from the built-in PyTorch BatchNorm3d as the mean and
        standard-deviation are reduced across all devices during training.
        For example, when one uses `nn.DataParallel` to wrap the network during
        training, PyTorch's implementation normalize the tensor on each device using
        the statistics only on that device, which accelerated the computation and
        is also easy to implement, but the statistics might be inaccurate.
        Instead, in this synchronized version, the statistics will be computed
        over all training samples distributed on multiple devices.
        Note that, for one-GPU or CPU-only case, this module behaves exactly same
        as the built-in PyTorch implementation.
        The mean and standard-deviation are calculated per-dimension over
        the mini-batches and gamma and beta are learnable parameter vectors
        of size C (where C is the input size).
        During training, this layer keeps a running estimate of its computed mean
        and variance. The running sum is kept with a default momentum of 0.1.
        During evaluation, this running mean/variance is used for normalization.
        Because the BatchNorm is done over the `C` dimension, computing statistics
        on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
        or Spatio-temporal BatchNorm
        Args:
            num_features: num_features from an expected input of
                size batch_size x num_features x depth x height x width
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Default: 0.1
            affine: a boolean value that when set to ``True``, gives the layer learnable
                affine parameters. Default: ``True``
        Shape:
            - Input: :math:`(N, C, D, H, W)`
            - Output: :math:`(N, C, D, H, W)` (same shape as input)
        Examples:
            >>> # With Learnable Parameters
            >>> m = SynchronizedBatchNorm3d(100)
            >>> # Without Learnable Parameters
            >>> m = SynchronizedBatchNorm3d(100, affine=False)
            >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
            >>> output = m(input)
        """

        def _check_input_dim(self, input):
            if input.dim() != 5:
                raise ValueError('expected 5D input (got {}D input)'
                                 .format(input.dim()))
            super(SynchronizedBatchNorm3d, self)._check_input_dim(input)
#>>> from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

#<<< from modeling.aspp import build_aspp
if True:
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _ASPPModule(nn.Module):
        def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
            super(_ASPPModule, self).__init__()
            self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                                stride=1, padding=padding, dilation=dilation, bias=False)
            self.bn = BatchNorm(planes)
            self.relu = nn.ReLU()

            self._init_weight()

        def forward(self, x):
            x = self.atrous_conv(x)
            x = self.bn(x)

            return self.relu(x)

        def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    class ASPP(nn.Module):
        def __init__(self, backbone, output_stride, BatchNorm):
            super(ASPP, self).__init__()
            if backbone == 'drn':
                inplanes = 512
            elif backbone == 'mobilenet':
                inplanes = 320
            else:
                inplanes = 2048
            if output_stride == 16:
                dilations = [1, 6, 12, 18]
            elif output_stride == 8:
                dilations = [1, 12, 24, 36]
            else:
                raise NotImplementedError

            self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
            self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
            self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
            self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                                 BatchNorm(256),
                                                 nn.ReLU())
            self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
            self.bn1 = BatchNorm(256)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self._init_weight()

        def forward(self, x):
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x5 = self.global_avg_pool(x)
            x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            return self.dropout(x)

        def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    # m.weight.data.normal_(0, math.sqrt(2. / n))
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


    def build_aspp(backbone, output_stride, BatchNorm):
        return ASPP(backbone, output_stride, BatchNorm)
#>>> from modeling.aspp import build_aspp

#<<< from modeling.decoder import build_decoder
if True:
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Decoder(nn.Module):
        def __init__(self, num_classes, backbone, BatchNorm):
            super(Decoder, self).__init__()
            if backbone == 'resnet' or backbone == 'drn':
                low_level_inplanes = 256
            elif backbone == 'xception':
                low_level_inplanes = 128
            elif backbone == 'mobilenet':
                low_level_inplanes = 24
            else:
                raise NotImplementedError

            self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
            self.bn1 = BatchNorm(48)
            self.relu = nn.ReLU()
            self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           BatchNorm(256),
                                           nn.ReLU(),
                                           nn.Dropout(0.5),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           BatchNorm(256),
                                           nn.ReLU(),
                                           nn.Dropout(0.1),
                                           nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            self._init_weight()


        def forward(self, x, low_level_feat):
            low_level_feat = self.conv1(low_level_feat)
            low_level_feat = self.bn1(low_level_feat)
            low_level_feat = self.relu(low_level_feat)

            x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, low_level_feat), dim=1)
            x = self.last_conv(x)

            return x

        def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def build_decoder(num_classes, backbone, BatchNorm):
        return Decoder(num_classes, backbone, BatchNorm)
#>>> from modeling.decoder import build_decoder

#<<< from modeling.backbone import build_backbone
if True:
    import torch
    import torch.nn as nn
    import torch.utils.model_zoo as model_zoo

    #<<< === drn
    if True:
        webroot = 'http://dl.yf.io/drn/'

        model_urls = {
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
            'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
            'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
            'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
            'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
            'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
            'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
        }


        def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=padding, bias=False, dilation=dilation)


        class BasicBlock(nn.Module):
            expansion = 1

            def __init__(self, inplanes, planes, stride=1, downsample=None,
                         dilation=(1, 1), residual=True, BatchNorm=None):
                super(BasicBlock, self).__init__()
                self.conv1 = conv3x3(inplanes, planes, stride,
                                     padding=dilation[0], dilation=dilation[0])
                self.bn1 = BatchNorm(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes,
                                     padding=dilation[1], dilation=dilation[1])
                self.bn2 = BatchNorm(planes)
                self.downsample = downsample
                self.stride = stride
                self.residual = residual

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    residual = self.downsample(x)
                if self.residual:
                    out += residual
                out = self.relu(out)

                return out


        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, inplanes, planes, stride=1, downsample=None,
                         dilation=(1, 1), residual=True, BatchNorm=None):
                super(Bottleneck, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
                self.bn1 = BatchNorm(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                       padding=dilation[1], bias=False,
                                       dilation=dilation[1])
                self.bn2 = BatchNorm(planes)
                self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
                self.bn3 = BatchNorm(planes * 4)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual
                out = self.relu(out)

                return out


        class DRN(nn.Module):

            def __init__(self, block, layers, arch='D',
                         channels=(16, 32, 64, 128, 256, 512, 512, 512),
                         BatchNorm=None):
                super(DRN, self).__init__()
                self.inplanes = channels[0]
                self.out_dim = channels[-1]
                self.arch = arch

                if arch == 'C':
                    self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                                           padding=3, bias=False)
                    self.bn1 = BatchNorm(channels[0])
                    self.relu = nn.ReLU(inplace=True)

                    self.layer1 = self._make_layer(
                        BasicBlock, channels[0], layers[0], stride=1, BatchNorm=BatchNorm)
                    self.layer2 = self._make_layer(
                        BasicBlock, channels[1], layers[1], stride=2, BatchNorm=BatchNorm)

                elif arch == 'D':
                    self.layer0 = nn.Sequential(
                        nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3,
                                  bias=False),
                        BatchNorm(channels[0]),
                        nn.ReLU(inplace=True)
                    )

                    self.layer1 = self._make_conv_layers(
                        channels[0], layers[0], stride=1, BatchNorm=BatchNorm)
                    self.layer2 = self._make_conv_layers(
                        channels[1], layers[1], stride=2, BatchNorm=BatchNorm)

                self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, BatchNorm=BatchNorm)
                self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, BatchNorm=BatchNorm)
                self.layer5 = self._make_layer(block, channels[4], layers[4],
                                               dilation=2, new_level=False, BatchNorm=BatchNorm)
                self.layer6 = None if layers[5] == 0 else \
                    self._make_layer(block, channels[5], layers[5], dilation=4,
                                     new_level=False, BatchNorm=BatchNorm)

                if arch == 'C':
                    self.layer7 = None if layers[6] == 0 else \
                        self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
                                         new_level=False, residual=False, BatchNorm=BatchNorm)
                    self.layer8 = None if layers[7] == 0 else \
                        self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
                                         new_level=False, residual=False, BatchNorm=BatchNorm)
                elif arch == 'D':
                    self.layer7 = None if layers[6] == 0 else \
                        self._make_conv_layers(channels[6], layers[6], dilation=2, BatchNorm=BatchNorm)
                    self.layer8 = None if layers[7] == 0 else \
                        self._make_conv_layers(channels[7], layers[7], dilation=1, BatchNorm=BatchNorm)

                self._init_weight()

            def _init_weight(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, SynchronizedBatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()


            def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                            new_level=True, residual=True, BatchNorm=None):
                assert dilation == 1 or dilation % 2 == 0
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        BatchNorm(planes * block.expansion),
                    )

                layers = list()
                layers.append(block(
                    self.inplanes, planes, stride, downsample,
                    dilation=(1, 1) if dilation == 1 else (
                        dilation // 2 if new_level else dilation, dilation),
                    residual=residual, BatchNorm=BatchNorm))
                self.inplanes = planes * block.expansion
                for i in range(1, blocks):
                    layers.append(block(self.inplanes, planes, residual=residual,
                                        dilation=(dilation, dilation), BatchNorm=BatchNorm))

                return nn.Sequential(*layers)

            def _make_conv_layers(self, channels, convs, stride=1, dilation=1, BatchNorm=None):
                modules = []
                for i in range(convs):
                    modules.extend([
                        nn.Conv2d(self.inplanes, channels, kernel_size=3,
                                  stride=stride if i == 0 else 1,
                                  padding=dilation, bias=False, dilation=dilation),
                        BatchNorm(channels),
                        nn.ReLU(inplace=True)])
                    self.inplanes = channels
                return nn.Sequential(*modules)

            def forward(self, x):
                if self.arch == 'C':
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                elif self.arch == 'D':
                    x = self.layer0(x)

                x = self.layer1(x)
                x = self.layer2(x)

                x = self.layer3(x)
                low_level_feat = x

                x = self.layer4(x)
                x = self.layer5(x)

                if self.layer6 is not None:
                    x = self.layer6(x)

                if self.layer7 is not None:
                    x = self.layer7(x)

                if self.layer8 is not None:
                    x = self.layer8(x)

                return x, low_level_feat


        class DRN_A(nn.Module):

            def __init__(self, block, layers, BatchNorm=None):
                self.inplanes = 64
                super(DRN_A, self).__init__()
                self.out_dim = 512 * block.expansion
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False)
                self.bn1 = BatchNorm(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm=BatchNorm)
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2, BatchNorm=BatchNorm)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                               dilation=2, BatchNorm=BatchNorm)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, BatchNorm=BatchNorm)

                self._init_weight()

            def _init_weight(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, SynchronizedBatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        BatchNorm(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample, BatchNorm=BatchNorm))
                self.inplanes = planes * block.expansion
                for i in range(1, blocks):
                    layers.append(block(self.inplanes, planes,
                                        dilation=(dilation, dilation, ), BatchNorm=BatchNorm))

                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                return x

        def drn_a_50(BatchNorm, pretrained=True):
            model = DRN_A(Bottleneck, [3, 4, 6, 3], BatchNorm=BatchNorm)
            if pretrained:
                model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
            return model


        def drn_c_26(BatchNorm, pretrained=True):
            model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', BatchNorm=BatchNorm)
            if pretrained:
                pretrained = model_zoo.load_url(model_urls['drn-c-26'])
                del pretrained['fc.weight']
                del pretrained['fc.bias']
                model.load_state_dict(pretrained)
            return model


        def drn_c_42(BatchNorm, pretrained=True):
            model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', BatchNorm=BatchNorm)
            if pretrained:
                pretrained = model_zoo.load_url(model_urls['drn-c-42'])
                del pretrained['fc.weight']
                del pretrained['fc.bias']
                model.load_state_dict(pretrained)
            return model


        def drn_c_58(BatchNorm, pretrained=True):
            model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', BatchNorm=BatchNorm)
            if pretrained:
                pretrained = model_zoo.load_url(model_urls['drn-c-58'])
                del pretrained['fc.weight']
                del pretrained['fc.bias']
                model.load_state_dict(pretrained)
            return model


        def drn_d_22(BatchNorm, pretrained=True):
            model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', BatchNorm=BatchNorm)
            if pretrained:
                pretrained = model_zoo.load_url(model_urls['drn-d-22'])
                del pretrained['fc.weight']
                del pretrained['fc.bias']
                model.load_state_dict(pretrained)
            return model


        def drn_d_24(BatchNorm, pretrained=True):
            model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', BatchNorm=BatchNorm)
            if pretrained:
                pretrained = model_zoo.load_url(model_urls['drn-d-24'])
                del pretrained['fc.weight']
                del pretrained['fc.bias']
                model.load_state_dict(pretrained)
            return model


        def drn_d_38(BatchNorm, pretrained=True):
            model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', BatchNorm=BatchNorm)
            if pretrained:
                pretrained = model_zoo.load_url(model_urls['drn-d-38'])
                del pretrained['fc.weight']
                del pretrained['fc.bias']
                model.load_state_dict(pretrained)
            return model


        def drn_d_40(BatchNorm, pretrained=True):
            model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', BatchNorm=BatchNorm)
            if pretrained:
                pretrained = model_zoo.load_url(model_urls['drn-d-40'])
                del pretrained['fc.weight']
                del pretrained['fc.bias']
                model.load_state_dict(pretrained)
            return model


        def drn_d_54(BatchNorm, pretrained=True):
            model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', BatchNorm=BatchNorm)
            if pretrained:
                pretrained = model_zoo.load_url(model_urls['drn-d-54'])
                del pretrained['fc.weight']
                del pretrained['fc.bias']
                model.load_state_dict(pretrained)
            return model


        def drn_d_105(BatchNorm, pretrained=True):
            model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', BatchNorm=BatchNorm)
            if pretrained:
                pretrained = model_zoo.load_url(model_urls['drn-d-105'])
                del pretrained['fc.weight']
                del pretrained['fc.bias']
                model.load_state_dict(pretrained)
            return model
        
        '''
        if __name__ == "__main__":
            import torch
            model = drn_a_50(BatchNorm=nn.BatchNorm2d, pretrained=True)
            input = torch.rand(1, 3, 512, 512)
            output, low_level_feat = model(input)
            print(output.size())
            print(low_level_feat.size())
        '''
    #>>> === drn

    #<<< === mobilenet
    if True:
        def conv_bn(inp, oup, stride, BatchNorm):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                BatchNorm(oup),
                nn.ReLU6(inplace=True)
            )


        def fixed_padding(inputs, kernel_size, dilation):
            kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
            return padded_inputs


        class InvertedResidual(nn.Module):
            def __init__(self, inp, oup, stride, dilation, expand_ratio, BatchNorm):
                super(InvertedResidual, self).__init__()
                self.stride = stride
                assert stride in [1, 2]

                hidden_dim = round(inp * expand_ratio)
                self.use_res_connect = self.stride == 1 and inp == oup
                self.kernel_size = 3
                self.dilation = dilation

                if expand_ratio == 1:
                    self.conv = nn.Sequential(
                        # dw
                        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                        BatchNorm(hidden_dim),
                        nn.ReLU6(inplace=True),
                        # pw-linear
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, 1, bias=False),
                        BatchNorm(oup),
                    )
                else:
                    self.conv = nn.Sequential(
                        # pw
                        nn.Conv2d(inp, hidden_dim, 1, 1, 0, 1, bias=False),
                        BatchNorm(hidden_dim),
                        nn.ReLU6(inplace=True),
                        # dw
                        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                        BatchNorm(hidden_dim),
                        nn.ReLU6(inplace=True),
                        # pw-linear
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0, 1, bias=False),
                        BatchNorm(oup),
                    )

            def forward(self, x):
                x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
                if self.use_res_connect:
                    x = x + self.conv(x_pad)
                else:
                    x = self.conv(x_pad)
                return x


        class MobileNetV2(nn.Module):
            def __init__(self, output_stride=8, BatchNorm=None, width_mult=1., pretrained=True):
                super(MobileNetV2, self).__init__()
                block = InvertedResidual
                input_channel = 32
                current_stride = 1
                rate = 1
                interverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                ]

                # building first layer
                input_channel = int(input_channel * width_mult)
                self.features = [conv_bn(3, input_channel, 2, BatchNorm)]
                current_stride *= 2
                # building inverted residual blocks
                for t, c, n, s in interverted_residual_setting:
                    if current_stride == output_stride:
                        stride = 1
                        dilation = rate
                        rate *= s
                    else:
                        stride = s
                        dilation = 1
                        current_stride *= s
                    output_channel = int(c * width_mult)
                    for i in range(n):
                        if i == 0:
                            self.features.append(block(input_channel, output_channel, stride, dilation, t, BatchNorm))
                        else:
                            self.features.append(block(input_channel, output_channel, 1, dilation, t, BatchNorm))
                        input_channel = output_channel
                self.features = nn.Sequential(*self.features)
                self._initialize_weights()

                if pretrained:
                    self._load_pretrained_model()

                self.low_level_features = self.features[0:4]
                self.high_level_features = self.features[4:]

            def forward(self, x):
                low_level_feat = self.low_level_features(x)
                x = self.high_level_features(low_level_feat)
                return x, low_level_feat

            def _load_pretrained_model(self):
                pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
                model_dict = {}
                state_dict = self.state_dict()
                for k, v in pretrain_dict.items():
                    if k in state_dict:
                        model_dict[k] = v
                state_dict.update(model_dict)
                self.load_state_dict(state_dict)

            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        # m.weight.data.normal_(0, math.sqrt(2. / n))
                        torch.nn.init.kaiming_normal_(m.weight)
                    elif isinstance(m, SynchronizedBatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
        
        '''
        if __name__ == "__main__":
            input = torch.rand(1, 3, 512, 512)
            model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d)
            output, low_level_feat = model(input)
            print(output.size())
            print(low_level_feat.size())
        '''
    #>>> === mobilenet

    #<<< === resnet
    if True:
        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
                super(Bottleneck, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
                self.bn1 = BatchNorm(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                       dilation=dilation, padding=dilation, bias=False)
                self.bn2 = BatchNorm(planes)
                self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
                self.bn3 = BatchNorm(planes * 4)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride
                self.dilation = dilation

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual
                out = self.relu(out)

                return out

        class ResNet(nn.Module):

            def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
                self.inplanes = 64
                super(ResNet, self).__init__()
                blocks = [1, 2, 4]
                if output_stride == 16:
                    strides = [1, 2, 2, 1]
                    dilations = [1, 1, 1, 2]
                elif output_stride == 8:
                    strides = [1, 2, 1, 1]
                    dilations = [1, 1, 2, 4]
                else:
                    raise NotImplementedError

                # Modules
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
                self.bn1 = BatchNorm(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

                self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
                self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
                self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
                # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
                self._init_weight()

                if pretrained:
                    self._load_pretrained_model()

            def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        BatchNorm(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
                self.inplanes = planes * block.expansion
                for i in range(1, blocks):
                    layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

                return nn.Sequential(*layers)

            def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        BatchNorm(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                                    downsample=downsample, BatchNorm=BatchNorm))
                self.inplanes = planes * block.expansion
                for i in range(1, len(blocks)):
                    layers.append(block(self.inplanes, planes, stride=1,
                                        dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

                return nn.Sequential(*layers)

            def forward(self, input):
                x = self.conv1(input)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                low_level_feat = x
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                return x, low_level_feat

            def _init_weight(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, SynchronizedBatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def _load_pretrained_model(self):
                pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
                model_dict = {}
                state_dict = self.state_dict()
                for k, v in pretrain_dict.items():
                    if k in state_dict:
                        model_dict[k] = v
                state_dict.update(model_dict)
                self.load_state_dict(state_dict)

        def ResNet101(output_stride, BatchNorm, pretrained=True):
            """Constructs a ResNet-101 model.
            Args:
                pretrained (bool): If True, returns a model pre-trained on ImageNet
            """
            model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
            return model
        
        '''
        if __name__ == "__main__":
            import torch
            model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
            input = torch.rand(1, 3, 512, 512)
            output, low_level_feat = model(input)
            print(output.size())
            print(low_level_feat.size())
        '''
    #>>> === resnet

    #<<< === xception
    if True:
        def fixed_padding(inputs, kernel_size, dilation):
            kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
            return padded_inputs


        class SeparableConv2d(nn.Module):
            def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
                super(SeparableConv2d, self).__init__()

                self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                                       groups=inplanes, bias=bias)
                self.bn = BatchNorm(inplanes)
                self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

            def forward(self, x):
                x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
                x = self.conv1(x)
                x = self.bn(x)
                x = self.pointwise(x)
                return x


        class Block(nn.Module):
            def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                         start_with_relu=True, grow_first=True, is_last=False):
                super(Block, self).__init__()

                if planes != inplanes or stride != 1:
                    self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
                    self.skipbn = BatchNorm(planes)
                else:
                    self.skip = None

                self.relu = nn.ReLU(inplace=True)
                rep = []

                filters = inplanes
                if grow_first:
                    rep.append(self.relu)
                    rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
                    rep.append(BatchNorm(planes))
                    filters = planes

                for i in range(reps - 1):
                    rep.append(self.relu)
                    rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
                    rep.append(BatchNorm(filters))

                if not grow_first:
                    rep.append(self.relu)
                    rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
                    rep.append(BatchNorm(planes))

                if stride != 1:
                    rep.append(self.relu)
                    rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
                    rep.append(BatchNorm(planes))

                if stride == 1 and is_last:
                    rep.append(self.relu)
                    rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
                    rep.append(BatchNorm(planes))

                if not start_with_relu:
                    rep = rep[1:]

                self.rep = nn.Sequential(*rep)

            def forward(self, inp):
                x = self.rep(inp)

                if self.skip is not None:
                    skip = self.skip(inp)
                    skip = self.skipbn(skip)
                else:
                    skip = inp

                x = x + skip

                return x


        class AlignedXception(nn.Module):
            """
            Modified Alighed Xception
            """
            def __init__(self, output_stride, BatchNorm,
                         pretrained=True):
                super(AlignedXception, self).__init__()

                if output_stride == 16:
                    entry_block3_stride = 2
                    middle_block_dilation = 1
                    exit_block_dilations = (1, 2)
                elif output_stride == 8:
                    entry_block3_stride = 1
                    middle_block_dilation = 2
                    exit_block_dilations = (2, 4)
                else:
                    raise NotImplementedError


                # Entry flow
                self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
                self.bn1 = BatchNorm(32)
                self.relu = nn.ReLU(inplace=True)

                self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
                self.bn2 = BatchNorm(64)

                self.block1 = Block(64, 128, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
                self.block2 = Block(128, 256, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False,
                                    grow_first=True)
                self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm,
                                    start_with_relu=True, grow_first=True, is_last=True)

                # Middle flow
                self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
                self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)

                # Exit flow
                self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                                     BatchNorm=BatchNorm, start_with_relu=True, grow_first=False, is_last=True)

                self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
                self.bn3 = BatchNorm(1536)

                self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
                self.bn4 = BatchNorm(1536)

                self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
                self.bn5 = BatchNorm(2048)

                # Init weights
                self._init_weight()

                # Load pretrained model
                if pretrained:
                    self._load_pretrained_model()

            def forward(self, x):
                # Entry flow
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)

                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)

                x = self.block1(x)
                # add relu here
                x = self.relu(x)
                low_level_feat = x
                x = self.block2(x)
                x = self.block3(x)

                # Middle flow
                x = self.block4(x)
                x = self.block5(x)
                x = self.block6(x)
                x = self.block7(x)
                x = self.block8(x)
                x = self.block9(x)
                x = self.block10(x)
                x = self.block11(x)
                x = self.block12(x)
                x = self.block13(x)
                x = self.block14(x)
                x = self.block15(x)
                x = self.block16(x)
                x = self.block17(x)
                x = self.block18(x)
                x = self.block19(x)

                # Exit flow
                x = self.block20(x)
                x = self.relu(x)
                x = self.conv3(x)
                x = self.bn3(x)
                x = self.relu(x)

                x = self.conv4(x)
                x = self.bn4(x)
                x = self.relu(x)

                x = self.conv5(x)
                x = self.bn5(x)
                x = self.relu(x)

                return x, low_level_feat

            def _init_weight(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, SynchronizedBatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()


            def _load_pretrained_model(self):
                pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
                model_dict = {}
                state_dict = self.state_dict()

                for k, v in pretrain_dict.items():
                    if k in state_dict:
                        if 'pointwise' in k:
                            v = v.unsqueeze(-1).unsqueeze(-1)
                        if k.startswith('block11'):
                            model_dict[k] = v
                            model_dict[k.replace('block11', 'block12')] = v
                            model_dict[k.replace('block11', 'block13')] = v
                            model_dict[k.replace('block11', 'block14')] = v
                            model_dict[k.replace('block11', 'block15')] = v
                            model_dict[k.replace('block11', 'block16')] = v
                            model_dict[k.replace('block11', 'block17')] = v
                            model_dict[k.replace('block11', 'block18')] = v
                            model_dict[k.replace('block11', 'block19')] = v
                        elif k.startswith('block12'):
                            model_dict[k.replace('block12', 'block20')] = v
                        elif k.startswith('bn3'):
                            model_dict[k] = v
                            model_dict[k.replace('bn3', 'bn4')] = v
                        elif k.startswith('conv4'):
                            model_dict[k.replace('conv4', 'conv5')] = v
                        elif k.startswith('bn4'):
                            model_dict[k.replace('bn4', 'bn5')] = v
                        else:
                            model_dict[k] = v
                state_dict.update(model_dict)
                self.load_state_dict(state_dict)
        
        
        '''
        if __name__ == "__main__":
            import torch
            model = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)
            input = torch.rand(1, 3, 512, 512)
            output, low_level_feat = model(input)
            print(output.size())
            print(low_level_feat.size())
        '''
    #>>> === xception

    def build_backbone(backbone, output_stride, BatchNorm, pretrained = True):
        if backbone == 'resnet':
            return ResNet101(output_stride, BatchNorm, pretrained)
        elif backbone == 'xception':
            return AlignedXception(output_stride, BatchNorm, pretrained)
        elif backbone == 'drn':
            return drn_d_54(BatchNorm, pretrained)
        elif backbone == 'mobilenet':
            return MobileNetV2(output_stride, BatchNorm, pretrained)
        else:
            raise NotImplementedError
#>>> from modeling.backbone import build_backbone

#<<< from modeling.sr_decoder import build_sr_decoder
if True:
    class Decoder(nn.Module):
        def __init__(self, num_classes, backbone, BatchNorm):
            super(Decoder, self).__init__()
            if backbone == 'resnet' or backbone == 'drn':
                low_level_inplanes = 256
            elif backbone == 'xception':
                low_level_inplanes = 128
            elif backbone == 'mobilenet':
                low_level_inplanes = 24
            else:
                raise NotImplementedError

            self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
            self.bn1 = BatchNorm(48)
            self.relu = nn.ReLU()
            self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           BatchNorm(256),
                                           nn.ReLU(),
                                           nn.Dropout(0.5),
                                           nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                           BatchNorm(128),
                                           nn.ReLU(),
                                           nn.Dropout(0.1),
                                           nn.Conv2d(128, 64, kernel_size=1, stride=1))
            self._init_weight()


        def forward(self, x, low_level_feat):
            low_level_feat = self.conv1(low_level_feat)
            low_level_feat = self.bn1(low_level_feat)
            low_level_feat = self.relu(low_level_feat)

            x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, low_level_feat), dim=1)
            x = self.last_conv(x)

            return x

        def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def build_sr_decoder(num_classes, backbone, BatchNorm):
        return Decoder(num_classes, backbone, BatchNorm)
#>>> from modeling.sr_decoder import build_sr_decoder

class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )

        # self.relu=torch.nn.ReLU(inplace=True)

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)


class DeepLab_DSRL(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, pretrained_backbone = False):
        super(DeepLab_DSRL, self).__init__()
        
        print("\n model num_classes:", num_classes)
        
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, pretrained_backbone)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.sr_decoder = build_sr_decoder(num_classes,backbone,BatchNorm)
        self.pointwise = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes,3,1),
            torch.nn.BatchNorm2d(3),  #添加了BN层 -> translation: BN layer has been added
            torch.nn.ReLU(inplace=True)
        )

        self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2) 
        self.up_edsr_1 = EDSRConv(64,64)
        self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2) 
        self.up_edsr_2 = EDSRConv(32,32)
        self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2) 
        self.up_edsr_3 = EDSRConv(16,16)
        self.up_conv_last = nn.Conv2d(16,3,1)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x_seg = self.decoder(x, low_level_feat)
        x_sr= self.sr_decoder(x, low_level_feat)
        x_seg_up = F.interpolate(x_seg, size=input.size()[2:], mode='bilinear', align_corners=True)
        x_seg_up = F.interpolate(x_seg_up,size=[2*i for i in input.size()[2:]], mode='bilinear', align_corners=True)

        x_sr_up = self.up_sr_1(x_sr)
        x_sr_up=self.up_edsr_1(x_sr_up)

        x_sr_up = self.up_sr_2(x_sr_up)
        x_sr_up=self.up_edsr_2(x_sr_up)

        x_sr_up = self.up_sr_3(x_sr_up)
        x_sr_up=self.up_edsr_3(x_sr_up)
        x_sr_up=self.up_conv_last(x_sr_up)

        return x_seg_up,x_sr_up,self.pointwise(x_seg_up),x_sr_up

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

'''
if __name__ == "__main__":
    model = DeepLab_DSRL(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
'''


# [loss] *****************************************************************************************************
# code from... 
#               https://github.com/Dootmaan/DSRL/blob/1822d6469dd8cca4b61afc42daa6df98067b4f80/utils/loss.py
#               https://github.com/Dootmaan/DSRL/blob/1822d6469dd8cca4b61afc42daa6df98067b4f80/utils/fa_loss.py
#
# as mentioned in paper (https://ieeexplore.ieee.org/document/9157434) 3.3.Optimization, Loss is composed with 
# (1) Cross Entropy loss:       for Semantic segmentation,          weight = 1.0
#       -> torch.nn.CrossEntropyLoss
# (2) Mean Squared Error loss:  for Single Image Super Resolution,  weight = 0.1
#       -> torch.nn.MSELoss
# (3) Feature Affinity loss:    for Structured Relation Term,       weight = 1.0
#       -> use class code below


''' # origianl code

import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())

'''

'''
#have potential bug that occurs when batch_size>1 <-oriiginal code comment
class FALoss(torch.nn.Module):
    def __init__(self,subscale=0.0625):
        super(FALoss,self).__init__()
        self.subscale=int(1/subscale)

    def forward(self,feature1,feature2):
        feature1=torch.nn.AvgPool2d(self.subscale)(feature1)
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)
        
        m_batchsize, C, height, width = feature1.size()
        feature1 = feature1.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[N,1,W*H]
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)  #haven't implemented in torch 0.4.1, so i use repeat instead
        # feature1=torch.div(feature1,L2norm)
        mat1 = torch.bmm(feature1.permute(0,2,1),feature1) #[N,W*H,W*H]

        m_batchsize, C, height, width = feature2.size()
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[N,C,W*H]
        # L2norm=torch.norm(feature2,2,1,keepdim=True).repeat(1,C,1)
        # # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        # feature2=torch.div(feature2,L2norm)
        mat2 = torch.bmm(feature2.permute(0,2,1),feature2) #[N,W*H,W*H]

        L1norm=torch.norm(mat2-mat1,1)

        return L1norm/((height*width)**2) 
'''


# converted version for easy use
class loss_for_dsrl():
    def __init__(self, weight_loss_ce = 1.0, weight_loss_mse = 0.1, weight_loss_fa = 1.0):
        print("\ninit class loss_for_dsrl")
        
        self.weight_loss_ce = weight_loss_ce
        self.loss_ce = nn.CrossEntropyLoss()
        print("weight_loss_ce:", self.weight_loss_ce)
        
        self.weight_loss_mse = weight_loss_mse
        self.loss_mse = nn.MSELoss()
        print("weight_loss_mse:", self.weight_loss_mse)
        
        self.weight_loss_fa = weight_loss_fa
        
        print("weight_loss_fa:", self.weight_loss_fa)
        
        print("use .calc to calculate loss")
        
    
    
    def FALoss(self, in_ft_1, in_ft_2):
        in_b_1, in_c_1, in_h_1, in_w_1 = in_ft_1.size()
        in_b_2, in_c_2, in_h_2, in_w_2 = in_ft_2.size()
        if in_c_1 != 1 or in_c_2 != 1:
            print("(exc) feature map should be [b, 1, h, w] shape")
            print("feature map 1 shape:", in_b_1, in_c_1, in_h_1, in_w_1)
            print("feature map 2 shape:", in_b_2, in_c_2, in_h_2, in_w_2)
            sys.exit(9)
        
        # SubSample
        sub_sample_rate = 8
        ft_1 = torch.nn.AvgPool2d(sub_sample_rate)(in_ft_1)
        ft_2 = torch.nn.AvgPool2d(sub_sample_rate)(in_ft_2)
        
        # L2 norm
        ft_1_norm = torch.linalg.norm(ft_1, dim = (2,3), ord = 2, keepdims = True)
        ft_2_norm = torch.linalg.norm(ft_2, dim = (2,3), ord = 2, keepdims = True)
        
        ft_1_div = torch.div(ft_1, ft_1_norm)
        ft_2_div = torch.div(ft_2, ft_2_norm)
        
        # transposed ([b,c,h,w] -> [b,c,w,h])
        ft_1_tran = torch.transpose(ft_1_div, dim0 = 2, dim1 = 3)
        ft_2_tran = torch.transpose(ft_2_div, dim0 = 2, dim1 = 3)
        
        # Similarity Matrix (3D shape [b*c,h,w], sm_1 & sm_2)
        # 4D -> 3D for bmm ([b,c,h,w] -> [b*c,h,w])
        in_b_1, in_c_1, in_h_1, in_w_1 = ft_1_tran.size()
        in_b_2, in_c_2, in_h_2, in_w_2 = ft_1_div.size()
        sm_1 = torch.bmm(ft_1_tran.view(-1, in_h_1, in_w_1), ft_1_div.view(-1, in_h_2, in_w_2))
        
        in_b_1, in_c_1, in_h_1, in_w_1 = ft_2_tran.size()
        in_b_2, in_c_2, in_h_2, in_w_2 = ft_2_div.size()
        sm_2 = torch.bmm(ft_2_tran.view(-1, in_h_1, in_w_1), ft_2_div.view(-1, in_h_2, in_w_2))
        
        in_bc, in_h, in_w = sm_1.size()
        
        loss_l1 = torch.nn.L1Loss()
        
        return loss_l1(sm_1, sm_2)
        
    
    def calc(self, hypo_label, hypo_sr, feature_label, feature_sr, ans_label, ans_sr):
        
        return (self.loss_ce(hypo_label, ans_label)    * self.weight_loss_ce
               +self.loss_mse(hypo_sr, ans_sr)         * self.weight_loss_mse
               +self.FALoss(feature_label, feature_sr) * self.weight_loss_fa
               )







print("End of model_dsrl.py")
