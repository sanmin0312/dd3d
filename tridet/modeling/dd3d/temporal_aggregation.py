import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from typing import Optional, Any, Union, Callable
import torch.utils.checkpoint as checkpoint
from tridet.layers.normalization import ModuleListDial, Scale
from detectron2.layers import Conv2d, batched_nms, cat, get_norm
import copy
from detectron2.layers import Conv2d, cat, get_norm


class TemporalWeight(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.use_per_level_predictors = cfg.DD3D.FCOS3D.PER_LEVEL_PREDICTORS

        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        num_levels = self.num_levels if cfg.DD3D.FCOS3D.PER_LEVEL_PREDICTORS else 1

        self.temporal_weight = nn.ModuleList([
            Conv2d(in_channels*2, 1, kernel_size=3, stride=1, padding=1, bias=True)
            for _ in range(num_levels)
        ])

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):

        for l in self.temporal_weight.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(l.weight, a=1)
                if l.bias is not None:  # depth head may not have bias.
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, cur, prev):
        out, weights = [], []

        for l, (feat_cur, feat_prev) in enumerate(zip(cur, prev)):
            _l = l if self.use_per_level_predictors else 0

            temp_weight = self.temporal_weight[_l](torch.cat((feat_cur,feat_prev), dim=1))
            temp_weight = self.sigmoid(temp_weight)

            out.append(feat_cur*temp_weight+feat_prev*(1-temp_weight))
            weights.append(temp_weight)

        return out, weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))