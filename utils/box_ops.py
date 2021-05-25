#!/usr/bin/env python
# coding=utf-8
"""
该脚本的作用是提供一些关于框的运算操作，比如nms等运算
"""
import torch
import torchvision
from torchvision.ops import nms
from torch.jit.annotations import Tuple
from torch import Tensor


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float)
    """
    作用:
        在每个水平或者每个类别中独立地做NMS
    参数:
        boxes: 一张图像上的预测框张量
        scores: 这些框对应的logit值张量
        idxs: 这些框对应的类别或者水平张量
        iou_threshold: NMS时设置的iou阈值
    返回:
        keep: 要保留的框的索引张量
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    max_coordination = boxes.max()
    offsets = idxs.to(boxes) * (max_coordination + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int])
    """
    作用:
        保证一张图像上的候选框都在图像的内部，没有超出边界
    参数:
        boxes: 一张图像的预测框坐标张量
        size: 一张图像的尺寸
    返回:
        经过限制处理以后的候选框坐标张量
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))

    else:
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float)
    """
    作用:
        去除尺寸小于指定最小尺寸的预测框
    参数:
        boxes: 预测框的坐标张量
        min_size: 指定的最小尺寸
    返回:
        要保留的预测框的索引
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = keep.nonzero().squeeze(1)
    return keep


def box_area(boxes):
    """
    作用:
        计算一张图片上所有框的面积
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    作用:
        计算boxes1中的每个框和boxes2中的每个框的iou
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

