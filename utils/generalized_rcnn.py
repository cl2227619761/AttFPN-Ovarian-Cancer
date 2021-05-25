#!/usr/bin/env python
# coding=utf-8
"""
本脚本是关于Faster R-CNN模块基类的组装
"""
import torch
import torch.nn as nn
from torch.jit.annotations import Dict, List, Tuple, Optional
from torch import Tensor

from collections import OrderedDict
import warnings


class GeneralizedRCNN(nn.Module):
    """
    作用:
        Faster R-CNN模型基类的组装
    参数:
        transform: 原始图片和真实框的预处理模块
        backbone: BackboneWithFPN，特征提取模块
        rpn: RegionProposalNetwork模块，生成候选框
        roi_heads: RoI模块，即Fast R-CNN部分
    """

    def __init__(self, transform, backbone, rpn, roi_heads):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        作用:
            模块的前向传播
        """
        if self.training and targets is None:
            raise ValueError("Targets should be passed during training time")
        # 这个尺寸的作用是把预测的坐标对应到原始图像
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            img_size = img.shape[-2:]
            assert len(img_size) == 2
            original_image_sizes.append((img_size[0], img_size[1]))
        # 原始图像预处理
        images, targets = self.transform(images, targets)
        # backbone提取特征
        features = self.backbone(images.tensors)
        if isinstance(features, Tensor):
            # 如果只有一个水平的特征
            features = OrderedDict([("0", features)])
        # 候选框生成网络
        proposals, proposal_losses = self.rpn(images, features, targets)
        # Roi部分
        detections, detector_losses = self.roi_heads(features, proposals,
                                                     images.image_sizes,
                                                     targets)
        detections = self.transform.postprocess(detections,
                                                images.image_sizes,
                                                original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (loss, detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)

