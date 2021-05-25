#!/usr/bin/env python
# coding=utf-8
"""
本脚本是Faster R-CNN模块的组装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from utils.pooler import NewMultiScaleRoIAlign
from utils.generalized_rcnn import GeneralizedRCNN
from utils.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from utils.roi_head import RoIHeads
from utils.transform import GeneralizedRCNNTransform



class FasterRCNN(GeneralizedRCNN):
    """
    作用:
        组装最终的FasterRCNN模块
    """

    def __init__(self, backbone, num_classes=None,
                 # transform参数
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN参数
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000,
                 rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000,
                 rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_img=256,
                 rpn_positive_fraction=0.5,
                 # RoIHead参数
                 box_roi_pool=None,
                 box_head=None,
                 box_predictor=None,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_img=512,
                 box_positive_fraction=0.25,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 box_detections_per_img=100,
                 # box_detections_per_img=20,
                 bbox_reg_weights=None):
        
        if not hasattr(backbone, "out_channels"):
            # backbone要有out_channels属性，后面要用
            raise ValueError("backbone should have the out_channels attr")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        # assert isinstance(box_roi_pool, (NewMultiScaleRoIAlign, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            #用的这个
            anchor_sizes = ((32,),(64,),(128,),(256,),(512,))
            # anchor_sizes = ((32,64,128),(32,64,128),(64,128,256),(64,128,256),(128,256,512))
            #用的这个
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            # anchor_sizes = ((64,),(128,),(192,),(256,),(320,))
            # aspect_ratios = ((0.5, 1.0, 4.0),(0.5, 1.0, 3.5),(0.5, 1.0, 5.0),(0.5, 1.0, 3.0),(0.5, 1.0, 4.0))
            # 单一水平实验
            # anchor_sizes = ((128, 256, 512),)
            # aspect_ratios = ((0.5, 1.0, 2.0),)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes,
                                                   aspect_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(out_channels,
                               rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train,
            testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train,
            testing=rpn_post_nms_top_n_test
        )
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n, rpn_post_nms_top_n,
            rpn_nms_thresh, rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_img, rpn_positive_fraction
        )
        
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"],
                output_size=7,
                sampling_ratio=2
            )
        # if box_roi_pool is None:
        #     box_roi_pool = MultiScaleRoIAlign(
        #         featmap_names=["3"],
        #         output_size=7,
        #         sampling_ratio=2
        #     )
        # if box_roi_pool is None:
        #     box_roi_pool = MultiScaleRoIAlign(
        #         featmap_names=["3"],
        #         output_size=7,
        #         sampling_ratio=2
        #     )
            # box_roi_pool = NewMultiScaleRoIAlign(
            #     featmap_names=["0", "1", "2", "3"],
            #     output_size=7,
            #     sampling_ratio=2
            # )
        if box_head is None:
            # roi特征的尺寸
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLHead(
                out_channels * resolution ** 2,
                representation_size
            )
            # box_head = NewTwoMLHead(
            #     out_channels,
            #     resolution,
            #     representation_size
            # )

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes=num_classes
            )

        roi_heads = RoIHeads(
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_img, box_positive_fraction,
            box_score_thresh, box_nms_thresh,
            box_detections_per_img,
            bbox_reg_weights
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size,
                                             image_mean, image_std)
        super(FasterRCNN, self).__init__(transform, backbone, rpn, roi_heads)


class TwoMLHead(nn.Module):
    """
    作用:
        RoIAlign以后得到的box_feature需要经过该模块
        这是一个两层全连接神经网络
    参数:
        in_channels: 输入的神经元数
        representation_size
    """
    
    def __init__(self, in_channels, representation_size):
        super(TwoMLHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class NewTwoMLHead(nn.Module):

    def __init__(self, in_channels, roi_size, representation_size):
        super(NewTwoMLHead, self).__init__()
        self.fc6 = nn.ModuleList()
        num_levels = 4
        for i in range(num_levels):
            self.fc6.append(
                nn.Sequential(
                    nn.Linear(in_channels*roi_size**2, representation_size),
                    nn.GroupNorm(32, representation_size, 1e-5),
                    nn.ReLU(inplace=True)
                )
            )

        self.fc7 = nn.Sequential(
            nn.Linear(representation_size, representation_size),
            nn.GroupNorm(32, representation_size, 1e-5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size = x[0].shape[0]
        for i in range(len(x)):
            x[i] = self.fc6[i](x[i].view(batch_size, -1))
        for i in range(1, len(x)):
            x[0] = torch.max(x[0], x[i])
        x = x[0]
        x = self.fc7(x)
        return x


class FastRCNNPredictor(nn.Module):
    """
    作用:
        最终的分类头和回归头
    返回:
        scores: 类别logits值
        bbox_deltas: 预测框偏移量
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes*4)
        # 新加的初始化方式，配合focalloss
        # nn.init.normal_(self.cls_score.weight, 0., 0.01)
        # nn.init.constant_(self.cls_score.bias, -2.0)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


if __name__ == "__main__":
    from utils.backbone_utils import resnet_fpn_backbone

    backbone = resnet_fpn_backbone("resnet50", True)
    model = FasterRCNN(backbone, num_classes=2)
    print(model)
    import ipdb;ipdb.set_trace()

