#!/usr/bin/env python
# coding=utf-8
"""
本脚本的作用是将原始图像和标注信息进行预处理
"""
import torch
from torch import Tensor
import torch.nn as nn
from torch.jit.annotations import List, Tuple, Dict, Optional
import math

from .image_list import ImageList


def resize_image(image, self_min_size, self_max_size, target):
    # type: (Tensor, float, float, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
    """
    作用:
        对图像进行resize操作的函数
    """
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))  # 图像的短边
    max_size = float(torch.max(im_shape))  # 图像的长边
    scale_factor = self_min_size / min_size  # 缩放因子
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear",
        align_corners=False
    )[0]
    if target is None:
        return image, target
    return image, target


def resize_boxes(boxes, original_size, new_size):
    """
    作用:
        对真实框进行相应的resize操作
    参数:
        boxes: 该图片上的真实框
        original_size: 该图片resize前的尺寸
        new_size: 该图片resize后的尺寸
    返回:
        经过resize之后的真实框坐标
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_ori, dtype=torch.float32, device=boxes.device)
        for s, s_ori in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(dim=1)
    
    xmin = xmin * ratio_w
    xmax = xmax * ratio_w
    ymin = ymin * ratio_h
    ymax = ymax * ratio_h

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class GeneralizedRCNNTransform(nn.Module):
    """
    作用:
        对原始图像做预处理操作
    参数:
        min_size: 图像缩放后的短边尺寸
        max_size: 图像缩放后的长边尺寸
        image_mean: 图像归一化时的均值
        image_std: 图像归一化时的标准差
    返回:
        image_list: 预处理后的图像
        targets: 预处理后的标注信息
    """
    
    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        作用:
            具体的预处理过程
        参数:
            images: 原始图像张量的列表
            targets: 原始图像的标注信息
        """
        images = [img for img in images]
        for i in range(len(images)):
            # 逐图片进行预处理
            image = images[i]
            target_index = targets[i] if targets is not None else None
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors")
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
            
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)  # 将图像列表变为4D张量
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))
        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        """
        作用:
            对图像进行归一化
        返回:
            归一化后的图像
        """
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image-mean[:, None, None])/std[:, None, None]

    def torch_choice(self, l):
        # type: (List[int])
        index = int(torch.empty(1).uniform_(0., float(len(l))).item())
        return l[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]])
        """
        作用:
            对图像和真实框进行resize
        返回:
            resize以后的图像和真实框
        """
        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])

        image, target = resize_image(image, size,
                                     float(self.max_size), target)
        if target is None:
            return image, target

        # 如果有真实框，要对真实框做对应的resize
        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        """
        作用:
            为了将图像列表变为4D张量，我们需要同一图像的尺寸
            按照最大的h和最大的w进行统一
        参数:
            the_list: 图像的尺寸列表[[C, H1, W1],[C,H2,W2]...]
        返回:
            最终统一的最大的尺寸[C, H_MAX, W_MAX]
        """
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int)
        """
        作用:
            将图像列表变为4D张量
        """
        max_size = self.max_by_axis([list(img.shape)
                                     for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1])/stride)*stride)
        max_size[2] = int(math.ceil(float(max_size[2])/stride)*stride)

        # 4D张量的形状
        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        # type: (List[Dict[str, Tensor]], List[Tuple[int, int]], List[Tuple[int, int]])
        """
        作用:
            在测试阶段的时候将预测框的坐标映射回原始图像
            在训练阶段直接返回预测结果
        """
        if self.training:
            return result
        for i, (pred, im_s, o_img_s) in enumerate(zip(
            result, image_shapes, original_image_sizes
        )):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_img_s)
            result[i]["boxes"] = boxes
        return result

