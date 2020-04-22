# @Author   : panjianning
# @Email    : 2393661347@qq.com
# @FileName : cbnet.py
# @DateTime : 2020/3/30 22:03


import torch.nn as nn

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.ops import (ContextBlock, GeneralizedAttention, build_conv_layer,
                       build_norm_layer)
from mmdet.utils import get_root_logger
from .res2net import Res2Net
from .resnet import ResNet
from .resnet_vd import ResNet_VD
from ..registry import BACKBONES


@BACKBONES.register_module
class CBNet(nn.Module):
    arch_cfg = {
        'Res2Net': Res2Net,
        'ResNet': ResNet,
        'ResNet_VD': ResNet_VD
    }

    def __init__(self,
                 num_repeat,
                 connect_norm_eval,
                 use_act=True,
                 backbone_type='ResNet',
                 **backbone_args):
        super(CBNet, self).__init__()

        assert backbone_type in self.arch_cfg
        self.backbone_type = backbone_type
        self.num_repeat = num_repeat
        self.connect_norm_eval = connect_norm_eval
        self.backbone_names = []
        self.connect_op_names = []
        self.use_act = use_act
        assert num_repeat >= 2
        for i in range(1, num_repeat + 1):
            backbone = self.arch_cfg[backbone_type](**backbone_args)
            backbone_name = 'cb{}'.format(i)
            self.add_module(backbone_name, backbone)
            self.backbone_names.append(backbone_name)

        left_out_channels = [256, 512, 1024, 2048]
        right_in_channels = [64, 256, 512, 1024]
        for i, _ in enumerate(left_out_channels):
            conv = build_conv_layer(
                backbone.conv_cfg,
                left_out_channels[i],
                right_in_channels[i],
                kernel_size=1,
                padding=0,
                stride=1,
                bias=False)
            # constant_init(conv, 0)
            _, norm = build_norm_layer(backbone.norm_cfg, right_in_channels[i])
            # BatchNorm
            # default value of running_mean is 0
            # default value of running_var is 1
            # default weight is 1
            # default bias is 0
            # default ((x - 0)/1)* weight + bias = x,
            # in eval mode, running_mean and running_var will not be updated.
            # if requires_grad, weight and bias will be updated.
            connect_op = nn.Sequential(conv, norm)
            connect_op_name = 'connect_op{}'.format(i + 1)
            self.add_module(connect_op_name, connect_op)
            self.connect_op_names.append(connect_op_name)

    def forward(self, x):
        backbone = getattr(self, self.backbone_names[0])

        if self.backbone_type == 'ResNet_VD':
            res = backbone.conv1(x)
            res = backbone.norm1(res)
            res = backbone.relu1(res)
            res = backbone.conv2(res)
            res = backbone.norm2(res)
            res = backbone.relu2(res)
            res = backbone.conv3(res)
            res = backbone.norm3(res)
            res = backbone.relu3(res)
            res = backbone.maxpool(res)
        else:
            res = backbone.conv1(x)
            res = backbone.norm1(res)
            res = backbone.relu(res)
            res = backbone.maxpool(res)

        res_endpoints = []
        for i, layer_name in enumerate(backbone.res_layers):
            res_layer = getattr(backbone, layer_name)
            res = res_layer(res)
            res_endpoints.append(res)

        for backbone_name in self.backbone_names[1:]:
            backbone = getattr(self, backbone_name)

            if self.backbone_type == 'ResNet_VD':
                res = backbone.conv1(x)
                res = backbone.norm1(res)
                res = backbone.relu1(res)
                res = backbone.conv2(res)
                res = backbone.norm2(res)
                res = backbone.relu2(res)
                res = backbone.conv3(res)
                res = backbone.norm3(res)
                res = backbone.relu3(res)
                res = backbone.maxpool(res)
            else:
                res = backbone.conv1(x)
                res = backbone.norm1(res)
                res = backbone.relu(res)
                res = backbone.maxpool(res)

            for i in range(len(res_endpoints)):
                # k-1 backbone's i-th output connect to k backbone i-th input
                connect_op = getattr(self, self.connect_op_names[i])
                connect_out = connect_op(res_endpoints[i])
                if self.use_act:
                    connect_out = nn.ReLU(inplace=True)(connect_out)
                connect_out = nn.UpsamplingNearest2d(res.size()[2:])(connect_out)
                res = connect_out + res
                res_layer = getattr(backbone, backbone.res_layers[i])
                res = res_layer(res)
                res_endpoints[i] = res
        outs = []
        for i, res_endpoint in enumerate(res_endpoints):
            if i in backbone.out_indices:
                outs.append(res_endpoint)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger,
                            map_location=lambda storage, loc: storage)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
            for m in self.modules():
                if isinstance(m, Res2Net) or isinstance(m, ResNet) or isinstance(m, ResNet_VD):
                    m.init_weights(pretrained)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        super(CBNet, self).train(mode)
        if mode and self.connect_norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
