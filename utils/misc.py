#!/usr/bin/env python
# coding=utf-8
"""
该脚本是提供将backbone网络的batchnorm层进行固定的
"""
import torch.nn as nn
import torch


class FrozenBatchNorm2d(nn.Module):
    """
    作用:固定gamma, beta, rm, rv版的BatchNorm2d
    参数:
        n: 输入的张量的通道数
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # 将gamma, beta, rm, rv注册为buffer
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """
        作用:删除以num_batch_tracked为后缀的键
        """
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        gamma = self.weight.reshape(1, -1, 1, 1)
        beta = self.bias.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        scale = gamma * rv.rsqrt()
        bias = beta - rm * scale
        return x * scale + bias

