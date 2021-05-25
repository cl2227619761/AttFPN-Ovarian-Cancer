#!/usr/bin/env python
# coding=utf-8
"""
本脚本的作用是将预处理后的4D图像张量和尺寸列表捆绑在一起
"""
import torch
from torch.jit.annotations import List, Tuple


class ImageList(object):
    """
    作用:
        将图像和尺寸放在该结构中，便于调用
    """

    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]])
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

