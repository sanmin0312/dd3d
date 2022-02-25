import logging

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy

from tridet.layers import smooth_l1_loss
from detectron2.config import configurable

LOG = logging.getLogger(__name__)


class PoseHead(nn.Module):
    #TODO pose haed activation 고민해 볼 것
    #TODO PoseHead 위치 적절한지
    def __init__(self, cfg, num_features):
        super().__init__()
        posehead = nn.Sequential(nn.Linear(cfg.ATTENTION.EMBED_DIM, 7), #yaw, x, y, z
                                 )

        self.posehead = _get_clones(posehead,num_features)

    def forward(self, x):

        for i, (x_level, posenet) in enumerate(zip(x, self.posehead)):
            x[i] = posenet(x_level).type(torch.float32)

        return x


class PoseLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.beta = cfg.DD3D.FCOS3D.LOSS.SMOOTH_L1_BETA
        self.loss_weight = cfg.DD3D.FCOS3D.POSE_HEAD.LOSS_WEIGHT

    def forward(self, pose_gt, pose_pred):

        loss = smooth_l1_loss(pose_pred, pose_gt, beta=self.beta, reduction='mean')

        return {"loss_ego_pose": self.loss_weight * loss}


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))