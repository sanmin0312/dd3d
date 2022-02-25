# Copyright 2021 Toyota Research Institute.  All rights reserved.
import torch
from torch import nn
import numpy as np

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess as resize_instances
from detectron2.structures import Instances

from tridet.modeling.dd3d.fcos2d import FCOS2DHead, FCOS2DInference, FCOS2DLoss
from tridet.modeling.dd3d.fcos3d import FCOS3DHead, FCOS3DInference, FCOS3DLoss
from tridet.modeling.dd3d.postprocessing import nuscenes_sample_aggregate
from tridet.modeling.dd3d.prepare_targets import DD3DTargetPreparer
from tridet.modeling.feature_extractor import build_feature_extractor
from tridet.structures.image_list import ImageList
from tridet.utils.tensor2d import compute_features_locations as compute_locations_per_level

from tridet.modeling.dd3d.attention_encoder import AttEncoder
from tridet.modeling.dd3d.pose_estimation import PoseHead, PoseLoss
from tridet.modeling.dd3d.dense_depth import DD3DVIDEODenseDepth
from tridet.modeling.dd3d.dense_depth_loss import build_dense_depth_loss
from tridet.structures.pose import Pose


@META_ARCH_REGISTRY.register()
class DD3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_feature_extractor(cfg)

        backbone_output_shape = self.backbone.output_shape()
        self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        self.backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.fcos2d_head = FCOS2DHead(cfg, self.backbone_output_shape)
        self.fcos2d_loss = FCOS2DLoss(cfg)
        self.fcos2d_inference = FCOS2DInference(cfg)

        if cfg.MODEL.BOX3D_ON:
            self.fcos3d_head = FCOS3DHead(cfg, self.backbone_output_shape)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        self.prepare_targets = DD3DTargetPreparer(cfg, self.backbone_output_shape)

        self.postprocess_in_inference = cfg.DD3D.INFERENCE.DO_POSTPROCESS

        self.do_nms = cfg.DD3D.INFERENCE.DO_NMS
        self.do_bev_nms = cfg.DD3D.INFERENCE.DO_BEV_NMS
        self.bev_nms_iou_thresh = cfg.DD3D.INFERENCE.BEV_NMS_IOU_THRESH

        # nuScenes inference aggregates detections over all 6 cameras.
        self.nusc_sample_aggregate_in_inference = cfg.DD3D.INFERENCE.NUSC_SAMPLE_AGGREGATE
        self.num_classes = cfg.DD3D.NUM_CLASSES

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        if 'intrinsics' in batched_inputs[0]:
            intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None
        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)

        gt_dense_depth = None
        if 'depth' in batched_inputs[0]:
            gt_dense_depth = [x["depth"].to(self.device) for x in batched_inputs]
            gt_dense_depth = ImageList.from_tensors(
                gt_dense_depth, self.backbone.size_divisibility, intrinsics=intrinsics
            )

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth = self.fcos3d_head(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        if self.training:
            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            if gt_dense_depth is not None:
                training_targets.update({"dense_depth": gt_dense_depth})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)
            return losses
        else:
            pred_instances, fcos2d_info = self.fcos2d_inference(
                logits, box2d_reg, centerness, locations, images.image_sizes
            )
            if not self.only_box2d:
                # This adds 'pred_boxes3d' and 'scores_3d' to Instances in 'pred_instances' in place.
                self.fcos3d_inference(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, inv_intrinsics, pred_instances,
                    fcos2d_info
                )

                # 3D score == 2D score x confidence.
                score_key = "scores_3d"
            else:
                score_key = "scores"

            # Transpose to "image-first", i.e. (B, L)
            pred_instances = list(zip(*pred_instances))
            pred_instances = [Instances.cat(instances) for instances in pred_instances]

            # 2D NMS and pick top-K.
            if self.do_nms:
                pred_instances = self.fcos2d_inference.nms_and_top_k(pred_instances, score_key)

            if not self.only_box2d and self.do_bev_nms:
                # Bird-eye-view NMS.
                dummy_group_idxs = {i: [i] for i, _ in enumerate(pred_instances)}
                if 'pose' in batched_inputs[0]:
                    poses = [x['pose'] for x in batched_inputs]
                else:
                    poses = [x['extrinsics'] for x in batched_inputs]
                pred_instances = nuscenes_sample_aggregate(
                    pred_instances,
                    dummy_group_idxs,
                    self.num_classes,
                    poses,
                    iou_threshold=self.bev_nms_iou_thresh,
                    include_boxes3d_global=False
                )

            if self.postprocess_in_inference:
                processed_results = []
                for results_per_image, input_per_image, image_size in \
                        zip(pred_instances, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = resize_instances(results_per_image, height, width)
                    processed_results.append({"instances": r})
            else:
                processed_results = [{"instances": x} for x in pred_instances]

            return processed_results

    def compute_locations(self, features):
        locations = []
        in_strides = [x.stride for x in self.backbone_output_shape]
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations_per_level(
                h, w, in_strides[level], feature.dtype, feature.device, offset=self.feature_locations_offset
            )
            locations.append(locations_per_level)
        return locations

@META_ARCH_REGISTRY.register()
class DD3D_VIDEO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_feature_extractor(cfg)

        backbone_output_shape = self.backbone.output_shape()
        self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        self.backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.fcos2d_head = FCOS2DHead(cfg, self.backbone_output_shape)
        self.fcos2d_loss = FCOS2DLoss(cfg)
        self.fcos2d_inference = FCOS2DInference(cfg)

        if cfg.MODEL.BOX3D_ON:
            self.fcos3d_head = FCOS3DHead(cfg, self.backbone_output_shape)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        if cfg.MODEL.VIDEO_ON:
            self.attencoder = AttEncoder(cfg, len(self.backbone_output_shape))
            self.pose_head = PoseHead(cfg, len(self.backbone_output_shape))
            self.pose_loss = PoseLoss(cfg)

        if cfg.MODEL.DEPTH_ON:
            self.dense_depth_head = DD3DVIDEODenseDepth(cfg, self.backbone_output_shape)
            self.depth_loss = build_dense_depth_loss(cfg)

        self.prepare_targets = DD3DTargetPreparer(cfg, self.backbone_output_shape)

        self.postprocess_in_inference = cfg.DD3D.INFERENCE.DO_POSTPROCESS

        self.do_nms = cfg.DD3D.INFERENCE.DO_NMS
        self.do_bev_nms = cfg.DD3D.INFERENCE.DO_BEV_NMS
        self.bev_nms_iou_thresh = cfg.DD3D.INFERENCE.BEV_NMS_IOU_THRESH

        # nuScenes inference aggregates detections over all 6 cameras.
        self.nusc_sample_aggregate_in_inference = cfg.DD3D.INFERENCE.NUSC_SAMPLE_AGGREGATE
        self.num_classes = cfg.DD3D.NUM_CLASSES

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        # import previous images
        # if 'image_prev' in batched_inputs[0]

        images_prev = [x["image_prev"].to(self.device) for x in batched_inputs]
        images_prev = [self.preprocess_image(x) for x in images_prev]

        if 'intrinsics' in batched_inputs[0]:
            intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None

        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)
        images_prev = ImageList.from_tensors(images_prev, self.backbone.size_divisibility, intrinsics=intrinsics)

        gt_dense_depth = None
        if 'depth' in batched_inputs[0]:
            gt_dense_depth = [x["depth"].to(self.device) for x in batched_inputs]
            gt_dense_depth = ImageList.from_tensors(
                gt_dense_depth, self.backbone.size_divisibility, intrinsics=intrinsics
            )

        gt_ego_pose = None
        if 'ego_pose' in batched_inputs[0]:
            gt_ego_pose = [x["ego_pose"] for x in batched_inputs]
            gt_ego_pose = torch.tensor(gt_ego_pose, dtype=torch.float32).to(self.device)

        features_ = self.backbone(torch.cat((images.tensor,images_prev.tensor)))

        #self-attention & temporal cross-attention
        feat_last_size = [features_[f].size(-1) for f in features_]
        features = [features_[f][:len(batched_inputs)].flatten(2) for f in self.in_features]
        features_prev = [features_[f][len(batched_inputs):].flatten(2) for f in self.in_features]
        features, feat_egopose = self.attencoder(features, features_prev)
        features = [f.transpose(1, 2).reshape(f.size(0), f.size(2), -1, feat_last_size[i]) for i, f in enumerate(features)]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth = self.fcos3d_head(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        #dense depth estimation (sparse depth)
        if self.dense_depth_head != None:
            dense_depth = self.dense_depth_head(features, images.intrinsics)

        if self.pose_head != None:
            ego_pose = self.pose_head(feat_egopose)

        if self.training:
            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            if gt_dense_depth is not None:
                training_targets.update({"dense_depth": gt_dense_depth})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)

            # depth loss
            if gt_dense_depth is not None:
                for lvl, x in enumerate(dense_depth):
                    loss_lvl = self.depth_loss(x, gt_dense_depth.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** lvl)  # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

            #ego pose loss
            if gt_ego_pose is not None:
                for lvl, x in enumerate(ego_pose):
                    loss_ego_lvl = self.pose_loss(gt_ego_pose, x)["loss_ego_pose"]
                    loss_ego_lvl = loss_ego_lvl / (np.sqrt(2) ** lvl)  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_lvl_{lvl}": loss_ego_lvl})

            return losses
        else:
            pred_instances, fcos2d_info = self.fcos2d_inference(
                logits, box2d_reg, centerness, locations, images.image_sizes
            )
            if not self.only_box2d:
                # This adds 'pred_boxes3d' and 'scores_3d' to Instances in 'pred_instances' in place.
                self.fcos3d_inference(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, inv_intrinsics, pred_instances,
                    fcos2d_info
                )

                # 3D score == 2D score x confidence.
                score_key = "scores_3d"
            else:
                score_key = "scores"

            # Transpose to "image-first", i.e. (B, L)
            pred_instances = list(zip(*pred_instances))
            pred_instances = [Instances.cat(instances) for instances in pred_instances]

            # 2D NMS and pick top-K.
            if self.do_nms:
                pred_instances = self.fcos2d_inference.nms_and_top_k(pred_instances, score_key)

            if not self.only_box2d and self.do_bev_nms:
                # Bird-eye-view NMS.
                dummy_group_idxs = {i: [i] for i, _ in enumerate(pred_instances)}
                if 'pose' in batched_inputs[0]:
                    poses = [x['pose'] for x in batched_inputs]
                else:
                    poses = [x['extrinsics'] for x in batched_inputs]
                pred_instances = nuscenes_sample_aggregate(
                    pred_instances,
                    dummy_group_idxs,
                    self.num_classes,
                    poses,
                    iou_threshold=self.bev_nms_iou_thresh,
                    include_boxes3d_global=False
                )

            if self.postprocess_in_inference:
                processed_results = []
                for results_per_image, input_per_image, image_size in \
                        zip(pred_instances, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = resize_instances(results_per_image, height, width)
                    processed_results.append({"instances": r})
            else:
                processed_results = [{"instances": x} for x in pred_instances]

            return processed_results

    def compute_locations(self, features):
        locations = []
        in_strides = [x.stride for x in self.backbone_output_shape]
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations_per_level(
                h, w, in_strides[level], feature.dtype, feature.device, offset=self.feature_locations_offset
            )
            locations.append(locations_per_level)
        return locations
