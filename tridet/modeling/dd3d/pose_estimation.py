import logging

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy

from tridet.layers import smooth_l1_loss
from detectron2.config import configurable

from tridet.layers.normalization import ModuleListDial
from detectron2.layers import Conv2d, get_norm


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


class PoseHead_single(nn.Module):
    #TODO pose haed activation 고민해 볼 것
    #TODO PoseHead 위치 적절한지
    def __init__(self, cfg, num_features):
        super().__init__()
        self.posehead = nn.Sequential(nn.Linear(cfg.ATTENTION.EMBED_DIM, 7), #yaw, x, y, z
                                 )


    def forward(self, x):

        x = self.posehead(x).type(torch.float32)

        return x


class FCOSPoseHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.num_classes = cfg.DD3D.NUM_CLASSES
        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)

        self._version = cfg.DD3D.FCOS2D._VERSION

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        num_pose_convs = cfg.DD3D.FCOS2D.NUM_CLS_CONVS
        use_deformable = cfg.DD3D.FCOS2D.USE_DEFORMABLE
        norm = cfg.DD3D.FCOS2D.NORM

        if use_deformable:
            raise ValueError("Not supported yet.")

        head_configs = {'pose': num_pose_convs}

        for head_name, num_convs in head_configs.items():
            tower = []
            if self._version == "v1":
                for _ in range(num_convs):
                    conv_func = nn.Conv2d
                    tower.append(conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True))
                    if norm == "GN":
                        raise NotImplementedError()
                    elif norm == "NaiveGN":
                        raise NotImplementedError()
                    elif norm == "BN":
                        tower.append(ModuleListDial([nn.BatchNorm2d(in_channels) for _ in range(self.num_levels)]))
                    elif norm == "SyncBN":
                        raise NotImplementedError()
                    tower.append(nn.ReLU())
            elif self._version == "v2":
                for _ in range(num_convs):
                    if norm in ("BN", "FrozenBN"):
                        # Each FPN level has its own batchnorm layer.
                        # "BN" is converted to "SyncBN" in distributed training (see train.py)
                        norm_layer = ModuleListDial([get_norm(norm, in_channels) for _ in range(self.num_levels)])
                    else:
                        norm_layer = get_norm(norm, in_channels)
                    tower.append(
                        Conv2d(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=norm_layer is None,
                            norm=norm_layer,
                            activation=F.relu
                        )
                    )
            else:
                raise ValueError(f"Invalid FCOS2D version: {self._version}")
            self.add_module(f'{head_name}_tower', nn.Sequential(*tower))

        self.pose = nn.Conv2d(in_channels, 7, kernel_size=3, stride=1, padding=1)


        self.init_weights()

    def init_weights(self):

        for tower in [self.pose_tower]:
            for l in tower.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        predictors = [self.pose]

        for modules in predictors:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        pose = []

        for l, feature in enumerate(x):
            pose_tower_out = self.pose_tower(feature)
            pose.append(self.pose(pose_tower_out).mean(dim=(2, 3)))

        return pose


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