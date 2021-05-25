#!/usr/bin/env python
# coding=utf-8
"""
该脚本将中间层特征提取模块和FPN模块组装起来，形成一个从输入到
金字塔特征输出的完整模块
"""
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models import densenet

from .layergetter import IntermediateLayerGetter, DenseNetLayerGetter
from .fpn import FeaturePyramidNetwork, MaxpoolOnP5, Bottom_up_path
from .fpn import AttFeaturePyramidNetwork
from .fpn import PANModule
from .misc import FrozenBatchNorm2d


class BackboneWithFPN(nn.Module):
    """
    作用:该模块将组装一个完整的backbone+FPN结构
    参数:
        backbone: 特征提取网络
        return_layers: 要提取的中间层
        in_channels_list: C2,C3,C4,C5的通道数
        out_channel: P2,P3,P4,P5的通道数，它们都一样
    """

    def __init__(self, backbone, return_layers,
                 in_channels_list, out_channel):
        super(BackboneWithFPN, self).__init__()
        # 中间层特征提取模块
        self.body = IntermediateLayerGetter(backbone, return_layers)
        # FPN模块
        # self.fpn = FeaturePyramidNetwork(in_channels_list,
        #                                  out_channel,
        #                                  extra_block=MaxpoolOnP5())
        self.afpn = AttFeaturePyramidNetwork(in_channels_list,
                                             out_channel,
                                             extra_block=MaxpoolOnP5())
        # 实验用pan代替FPN
        # self.pan = PANModule(in_channels_list,
        #                      out_channel)
        # self.bottom_up = Bottom_up_path([256,256,256,256],
        #                                 out_channel,
        #                                 extra_block=MaxpoolOnP5())
        
        self.out_channels = out_channel

    def forward(self, x):
        x = self.body(x)
        x = self.afpn(x)
        # x = self.fpn(x)
        # x = self.bottom_up(x)
        return x


class BackboneWithFPNForDensenet(nn.Module):
    """
    针对Densenet的BackboneWithFPN
    """
    def __init__(self, backbone, in_channels_list, out_channel):
        super(BackboneWithFPNForDensenet, self).__init__()
        self.body = DenseNetLayerGetter(backbone)
        self.fpn = FeaturePyramidNetwork(in_channels_list,
                                         out_channel,
                                         extra_block=MaxpoolOnP5())
        # self.afpn = AttFeaturePyramidNetwork(in_channels_list,
        #                                      out_channel,
        #                                      extra_block=MaxpoolOnP5())
        # 实验用pan代替FPN
        # self.pan = PANModule(in_channels_list,
        #                      out_channel)
        # self.bottom_up = Bottom_up_path([256,256,256,256],
        #                                 out_channel,
        #                                 extra_block=MaxpoolOnP5())
        self.out_channels = out_channel

    def forward(self, x):
        x = self.body(x)
        # x = self.afpn(x)
        x = self.fpn(x)
        # x = self.bottom_up(x)
        return x


# def resnet_fpn_backbone(backbone_name, pretrained,
#                         norm_layer=FrozenBatchNorm2d):
#     """
#     作用: 建立resnet为backbone的BackboneWithFPN模块
#     参数:
#         backbone_name: 要使用的resnet的名称
#         pretrained: 是否为预训练网络
#         norm_layer: 把bn层固定住
#     """
#     backbone = resnet.__dict__[backbone_name](
#         pretrained=pretrained,
#         norm_layer=norm_layer
#     )
#     # 把C2,C3,C4,C5以外的层的参数固定住
#     for name, param in backbone.named_parameters():
#         if "layer2" not in name and "layer3" not in name and "layer4" not in name:
#             param.requires_grad_(False)

#     return_layers = {"layer1": "0", "layer2": "1",
#                      "layer3": "2", "layer4": "3"}

#     # C2的通道数
#     in_channels_stage2 = backbone.inplanes // 8
#     # C2, C3, C4, C5的通道数列表
#     in_channels_list = [
#         in_channels_stage2,
#         in_channels_stage2 * 2,
#         in_channels_stage2 * 4,
#         in_channels_stage2 * 8,
#     ]
#     # P2,P3,P4,P5的通道数
#     out_channel = 256
#     return BackboneWithFPN(backbone, return_layers,
#                           in_channels_list, out_channel)
def resnet_fpn_backbone(backbone_name, pretrained):
    """
    作用: 建立resnet为backbone的BackboneWithFPN模块
    参数:
        backbone_name: 要使用的resnet的名称
        pretrained: 是否为预训练网络
    """
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained
    )
    # 把C2,C3,C4,C5以外的层的参数固定住
    for name, param in backbone.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name:
            param.requires_grad_(False)

    return_layers = {"layer1": "0", "layer2": "1",
                     "layer3": "2", "layer4": "3"}

    # C2的通道数
    in_channels_stage2 = backbone.inplanes // 8
    # C2, C3, C4, C5的通道数列表
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    # P2,P3,P4,P5的通道数
    out_channel = 256
    return BackboneWithFPN(backbone, return_layers,
                          in_channels_list, out_channel)


def densenet_fpn_backbone(backbone_name, pretrained):
    """
    以densenet为backbone的fpn骨架网络
    """
    backbone = densenet.__dict__[backbone_name](
        pretrained=pretrained
    )
    for name, param in backbone.features.named_parameters():
        if "denseblock" not in name and "transition" not in name:
            param.requires_grad_(False)

    in_channels_list = [128, 256, 512, 1024] # densenet121
    # in_channels_list = [192, 384, 1056, 2208] # densenet161
    # in_channels_list = [128, 256, 640, 1664]  # densenet169
    # in_channels_list = [128, 256, 896, 1920]
    out_channel = 256
    return BackboneWithFPNForDensenet(backbone,
                                      in_channels_list,
                                      out_channel)


if __name__ == "__main__":
    import torch

    x = torch.randn(1, 3, 224, 224)
    net = resnet_fpn_backbone("resnet50", True)
    # net = densenet_fpn_backbone("densenet161", True)
    out = net(x)
    import ipdb;ipdb.set_trace()

