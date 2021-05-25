#!/usr/bin/env python
# coding=utf-8
"""
本脚本中主要包括一些和框选择，框坐标和偏移量之间的转换的类和函数
"""
import torch
from torch.jit.annotations import List, Tuple

import math


def zeros_like(tensor, dtype):
    # type: (Tensor, int) -> Tensor
    """
    单独写出来，是因为使用原装的zeros_like函数有bug
    """
    return torch.zeros_like(tensor, dtype=dtype, layout=tensor.layout,
                            device=tensor.device,
                            pin_memory=tensor.is_pinned())


@torch.jit.script
class BalancedPositiveNegativeSampler(object):
    """
    作用:
        从每张图片中随机抽取一定数量的预测框，其中有一定比例的
        前景框和背景框
    参数:
        batch_size_per_img: 每张图片中要抽取的预测框的数量
        postive_fraction: 其中前景框所占的比例
    返回:
        前景框的索引掩膜
        背景框的索引掩膜
    """
    def __init__(self, batch_size_per_img, positive_fraction):
        # type: (int, float)
        self.batch_size_per_img = batch_size_per_img
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor])
        """
        参数:
            matched_idxs: 锚点框所匹配的标签
        """
        pos_idx = []  # 存储每张图片的选取的前景框掩膜结果
        neg_idx = []  # 存储每张图片的选取的背景框掩膜结果

        for matched_idxs_per_img in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_img >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_img == 0).squeeze(1)

            num_pos = int(self.batch_size_per_img * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_img - num_pos
            num_neg = min(negative.numel(), num_neg)
            
            perm1 = torch.randperm(
                positive.numel(), device=positive.device
            )[:num_pos]
            perm2 = torch.randperm(
                negative.numel(), device=negative.device
            )[:num_neg]

            pos_idx_per_img = positive[perm1]
            neg_idx_per_img = negative[perm2]

            # 创建前景框掩膜
            pos_idx_per_img_mask = zeros_like(
                matched_idxs_per_img, dtype=torch.uint8
            )
            # 创建背景框掩膜
            neg_idx_per_img_mask = zeros_like(
                matched_idxs_per_img, dtype=torch.uint8
            )

            pos_idx_per_img_mask[pos_idx_per_img] = torch.tensor(
                1, dtype=torch.uint8
            )
            neg_idx_per_img_mask[neg_idx_per_img] = torch.tensor(
                1, dtype=torch.uint8
            )

            pos_idx.append(pos_idx_per_img_mask)
            neg_idx.append(neg_idx_per_img_mask)

        return pos_idx, neg_idx


@torch.jit.script
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    作用:
        将匹配到的真实框坐标转变为相对于锚点框坐标的偏移量
    参数:
        reference_boxes: 锚点框匹配到的真实框的坐标
        proposals: 锚点框坐标
        weights: 权重参数，同下
    返回:
        偏移量张量
    """
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    anchor_widths = proposals_x2 - proposals_x1
    anchor_heights = proposals_y2 - proposals_y1
    anchor_ctr_x = proposals_x1 + 0.5 * anchor_widths
    anchor_ctr_y = proposals_y1 + 0.5 * anchor_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    dx = wx * (gt_ctr_x - anchor_ctr_x) / anchor_widths
    dy = wy * (gt_ctr_y - anchor_ctr_y) / anchor_heights
    dw = ww * torch.log(gt_widths / anchor_widths)
    dh = wh * torch.log(gt_heights / anchor_heights)

    targets = torch.cat((dx, dy, dw, dh), dim=1)
    return targets


@torch.jit.script
class BoxCoder(object):
    """
    作用:
        提供框的偏移量和坐标之间的转换方法
    参数:
        weights: 偏移量所除以的权重
        bbox_xform_clip: 为防止exp出现溢出
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float)
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor])
        """
        作用:
            使用encode_single函数进行坐标和偏移量的转换
        """
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, dim=0)

    def encode_single(self, reference_boxes, proposals):
        """
        作用:
            使用encode_boxes函数进行坐标和偏移量的转换
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor])
        """
        作用:
            利用下面的decode_single函数将偏移量换算为预测框坐标
        参数:
            rel_codes: 预测框的偏移量张量
            boxes: 锚点框的坐标张量列表
        """
        # 确认数据是否符合要求
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)

        # 每张图片的锚点框数量
        boxes_per_image = [b.shape[0] for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        pred_boxes = self.decode_single(
            rel_codes.reshape(box_sum, -1),
            concat_boxes
        )
        return pred_boxes.reshape(box_sum, -1, 4)

    def decode_single(self, rel_codes, boxes):
        """
        作用:
            将预测框相对于锚点框的偏移量换算为预测框的坐标
        参数:
            rel_codes:预测框的偏移量
            bboxes: 锚点框的坐标
        返回:
            所有图片所有预测框的坐标张量[num_pred_boxes, 4]
        """
        boxes = boxes.to(rel_codes.dtype)
        
        # 锚点框的宽度张量，形状[num_boxes]
        widths = boxes[:, 2] - boxes[:, 0]
        # 锚点框的高度张量，形状[num_boxes]
        heights = boxes[:, 3] - boxes[:, 1]
        # 锚点框的中心点横坐标，形状[num_boxes]
        ctr_x = boxes[:, 0] + 0.5 * widths
        # 锚点框的中心点纵坐标，形状[num_boxes]
        ctr_y = boxes[:, 1] + 0.5 * heights

        # 偏移量所除以的权重
        wx, wy, ww, wh = self.weights
        
        dx = rel_codes[:, 0::4] / wx  # 偏移量tx,形状[num_boxes, 1]
        dy = rel_codes[:, 1::4] / wy  # 偏移量ty,形状[num_boxes, 1]
        dw = rel_codes[:, 2::4] / ww  # 偏移量tw,形状[num_boxes, 1]
        dh = rel_codes[:, 3::4] / wh  # 偏移量th,形状[num_boxes, 1]

        # 由于反推坐标的时候计算涉及到exp(dw),为了防止溢出
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # 利用公式反推预测框的中心点坐标和预测框的宽高
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # 利用求得的预测框的中心点坐标和宽高得出预测框坐标
        pred_x1 = pred_ctr_x - torch.tensor(
            0.5, dtype=pred_ctr_x.dtype, device=pred_w.device
        ) * pred_w
        pred_y1 = pred_ctr_y - torch.tensor(
            0.5, dtype=pred_ctr_y.dtype, device=pred_h.device
        ) * pred_h
        pred_x2 = pred_ctr_x + torch.tensor(
            0.5, dtype=pred_ctr_x.dtype, device=pred_w.device
        ) * pred_w
        pred_y2 = pred_ctr_y + torch.tensor(
            0.5, dtype=pred_ctr_y.dtype, device=pred_h.device
        ) * pred_h

        pred_boxes = torch.stack(
            (pred_x1, pred_y1, pred_x2, pred_y2), dim=2
        ).flatten(1)
        return pred_boxes  # 形状[num_boxes, 4]


@torch.jit.script
class Matcher(object):
    """
    作用:
        根据iou张量对锚点框和真实框进行匹配
    参数:
        high_threshold: 前景阈值
        low_threshold: 背景阈值
        allow_low_quality_matches: 是否加入匹配质量差的框
    """
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2
    
    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, high_threshold, low_threshold,
                 allow_low_quality_matches=False):
        # type: (float, float, bool)
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2

    def __call__(self, match_quality_matrix):
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = torch.tensor(self.BELOW_LOW_THRESHOLD)
        matches[between_thresholds] = torch.tensor(self.BETWEEN_THRESHOLDS)

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(
                matches, all_matches, match_quality_matrix
            )

        return matches


    def set_low_quality_matches_(self, matches, all_matches,
                                 match_quality_matrix):
        """
        作用:
            匹配质量稍差的框
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


if __name__ == "__main__":
    import torch


    box_coder = BoxCoder(weights=[1.,1.,1.,1.])
    rel_boxes = torch.arange(0, 16, dtype=torch.float32).reshape(2, -1)
    boxes = [torch.arange(1, 5, dtype=torch.float32).reshape(1, -1),
             torch.arange(6, 10, dtype=torch.float32).reshape(1, -1)]

    pred_boxes = box_coder.decode(rel_boxes, boxes)
    import ipdb;ipdb.set_trace()
