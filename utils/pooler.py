#!/usr/bin/env python
# coding=utf-8
"""
将所有的roi在每个水平的特征上都进行一次roialign操作，参考论文
Path Aggregation Network
"""
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import roi_align


class NewMultiScaleRoIAlign(MultiScaleRoIAlign):
    """
    继承基类，对其进行改造，因为基类是参考FPN论文进行编写的，而我们
    并不需要按照proposal的尺寸将其映射到对应的水平，我们要做的是在
    每个水平上都进行roialign
    """

    def __init__(self, featmap_names, output_size, sampling_ratio):
        super(NewMultiScaleRoIAlign, self).__init__(
            featmap_names, output_size, sampling_ratio
        )

    def forward(self, x, boxes, image_shapes):
        """
        修改基类的forward方法
        参数:
            x: Dict[str, Tensor]
            boxes: List[Tensor]
            image_shapes: List[Tuple[int, int]]
        其中x是所有水平的特征构成的字典，boxes是proposals
        """
        x_filtered = []
        for k, v in x.items():
            if k in self.featmap_names:
                x_filtered.append(v)
        rois = self.convert_to_roi_format(boxes)
        if self.scales is None:
            self.setup_scales(x_filtered, image_shapes)

        scales = self.scales
        assert scales is not None
        result = []
        for level, (per_level_feature, scale) in enumerate(
            zip(x_filtered, scales)
        ):
            this_level_out = roi_align(
                per_level_feature, rois,
                output_size=self.output_size,
                spatial_scale=scale, sampling_ratio=self.sampling_ratio
            )
            result.append(this_level_out)
        return result


