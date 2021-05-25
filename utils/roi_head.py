#!/usr/bin/env python
# coding=utf-8
"""
本脚本是继RPNHead之后的RoIHead模块
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.jit.annotations import List, Dict, Optional, Tuple

from . import box_ops
from . import box_util
from .focalloss import CEFocalLoss


cefocalloss = CEFocalLoss(class_nums=2)


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor])
    """
    作用:
        计算RoIHeads的损失，分类损失和回归损失
    参数:
        class_logits: 分类头的预测结果
        box_regression: 回归头的预测结果
        labels: 候选框匹配到的真实标签
        regression_targets: 候选框匹配到的真实坐标的偏移量
    返回:
        classification_loss: 分类损失
        box_loss: 回归损失
    """
    # 由于labels是列表，故拼接起来
    labels = torch.cat(labels, dim=0)
    # 由于regression targets也是列表，故拼接起来
    regression_targets = torch.cat(regression_targets, dim=0)
    # 分类损失
    classification_loss = F.cross_entropy(class_logits, labels)
    # classification_loss = cefocalloss(class_logits, labels)
    # 回归损失的计算需要稍微进行一下变形，因为只用非背景框进行计算回归损失
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    # 由于box_regression的形状是[N, 类别数x4]，因此需要进行变形
    box_regression = box_regression.reshape(N, -1, 4)
    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum"
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


class RoIHeads(nn.Module):
    """
    作用:
        进行RPN模块之后的一系列操作
    参数:

    """
    __annotations__ = {
        "proposal_matcher": box_util.Matcher,
        "fg_bg_sampler": box_util.BalancedPositiveNegativeSampler,
        "box_coder": box_util.BoxCoder,
    }
    def __init__(self,
                 # RoIAlign
                 box_roi_pool,
                 # RoIAlign后面的两层全连接神经网络
                 box_head,
                 # 最终的分类头和回归头
                 box_predictor,
                 # 匹配候选框和真实框时的前景阈值和背景阈值
                 fg_iou_thresh, bg_iou_thresh,
                 # 每张图像中随机选取的候选框的数量
                 batch_size_per_img,
                 # 其中前景框所占比例
                 positive_fraction,
                 # 预测时规定的概率得分阈值
                 score_thresh,
                 # 预测时进行NMS的iou阈值
                 nms_thresh,
                 # 预测时经过NMS之后要保留的前若干个预测框
                 detections_per_img,
                 # 候选框的匹配到的坐标变为偏移量时的权重
                 bbox_reg_weights=None):
        super(RoIHeads, self).__init__()

        # 为候选框匹配真实框标签
        self.proposal_matcher = box_util.Matcher(
            fg_iou_thresh, bg_iou_thresh,
            allow_low_quality_matches=False
        )
        # 从匹配了标签的候选框中随机选取一部分
        self.fg_bg_sampler = box_util.BalancedPositiveNegativeSampler(
            batch_size_per_img, positive_fraction
        )
        
        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = box_util.BoxCoder(bbox_reg_weights)
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def subsample(self, labels):
        # type: (List[Tensor])
        """
        作用:
            从已经被赋予标签的候选框中随机选取一部分框
        参数:
            labels: 已经被赋予的标签
        """
        sampled_pos_masks, sampled_neg_masks = self.fg_bg_sampler(labels)
        sampled_inds = []  # 用来存储每张图片选出的框的索引
        for img_idx, (pos_mask, neg_mask) in enumerate(
            zip(sampled_pos_masks, sampled_neg_masks)
        ):
            img_sampled_inds = torch.nonzero(pos_mask | neg_mask).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor])
        """
        作用:
            为候选框匹配真实框索引
        参数:
            proposals: 候选框
            gt_boxes: 真实框
            gt_labels: 真实标签
        返回:
            候选框匹配到的真实框的索引
            为候选框所赋予的0, -1, >0的标签
        """
        matched_idxs = []  # 用来存放匹配到的真实框的索引
        labels = []  # 用来存放所赋予的标签
        for proposals_in_img, gt_boxes_in_img, gt_labels_in_img in zip(
            proposals, gt_boxes, gt_labels
        ):
            if gt_boxes_in_img.numel() == 0:
                # 对于背景图片
                device = proposals_in_img.device
                clamped_matched_idxs_in_img = torch.zeros(
                    (proposals_in_img.shape[0],), dtype=torch.int64,
                    device=device
                )
                labels_in_img = torch.zeros(
                    (proposals_in_img.shape[0],), dtype=torch.int64,
                    device=device
                )
            else:
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_img, proposals_in_img
                )
                matched_idxs_in_img = self.proposal_matcher(match_quality_matrix)
                clamped_matched_idxs_in_img = matched_idxs_in_img.clamp(
                    min=0
                )
                labels_in_img = gt_labels_in_img[clamped_matched_idxs_in_img]
                labels_in_img = labels_in_img.to(dtype=torch.int64)
                # 背景标签变为0
                bg_indices = matched_idxs_in_img == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_img[bg_indices] = torch.tensor(0)
                # 要舍弃的标签变为-1
                ignore_indices = matched_idxs_in_img == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_img[ignore_indices] = torch.tensor(-1)

            matched_idxs.append(clamped_matched_idxs_in_img)
            labels.append(labels_in_img)
        return matched_idxs, labels

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor])
        """
        作用:
            将候选框和真实框合并起来
        参数:
            proposals: RPNHead产生的候选框
            gt_boxes: 真实框列表
        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def select_training_samples(self, proposals, targets):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        作用:
            在候选框中随机选择一部分用于后续的训练
        参数:
            proposals: RPNHead生成的boxes
            targets: 真实的标注信息
        返回:
            proposals: 被选中的那些候选框
            matched_idxs: 被选中的那些候选框匹配到的真实框的索引
            labels: 选中的候选框匹配到的真实框的标签
            regression_targets: 真实框相对于选中的候选框的偏移量
        """
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        # 得到每张图片的真实框坐标
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        # 得到每张图片的真实框的标签
        gt_labels = [t["labels"] for t in targets]

        # 将候选框和真实框合并，一起作为候选框
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        # 为候选框匹配真实框标签
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        # 从匹配好标签的候选框中随机选出一部分
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            gt_boxes_in_img = gt_boxes[img_id]
            if gt_boxes_in_img.numel() == 0:
                gt_boxes_in_img = torch.zeros((1, 4), dtype=dtype,
                                              device=device)
            matched_gt_boxes.append(gt_boxes_in_img[matched_idxs[img_id]])
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self, class_logits, box_regression,
                               proposals, image_shapes):
        # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])
        """
        作用:
            RoIHeads预测阶段的后处理过程
        参数:
            class_logits: 分类头的预测结果
            box_regression: 回归头的预测结果
            proposals: 候选框的坐标
            image_shapes: 图像的尺寸
        返回:
            all_boxes: 预测框的坐标列表
            all_scores: 预测框的概率列表
            all_labels: 预测框的类别列表
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_img.shape[0]
                           for boxes_in_img in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # 把背景类抛掉
            boxes = boxes[:, 1:]
            labels = labels[:, 1:]
            scores = scores[:, 1:]

            # 进行变形
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # 去除概率小于指定阈值的预测框
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # 移除面积小于min_sizexmin_size的预测框
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # 进行NMS处理，在各类别间独立进行
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # 选出排名靠前的预测框
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels
            
    def forward(self, features, proposals, image_shapes, targets=None):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]], Optional[List[Dict[str, Tensor]]])
        """
        RoIHeads的前向传播过程
        """
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        # 使用RoIAlign获取RoI特征
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # RoI特征进入一个两层全连接神经网络
        box_features = self.box_head(box_features)
        # 该特征进入最终的分类头和回归头
        class_logits, box_regression = self.box_predictor(box_features)

        # 用于存放预测结果
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}  # 用于存放损失
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
        return result, losses

