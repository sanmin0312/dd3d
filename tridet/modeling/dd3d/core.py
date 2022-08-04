# Copyright 2021 Toyota Research Institute.  All rights reserved.
import fvcore.nn
import torch
from torch import nn
import numpy as np
import os

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess as resize_instances
from detectron2.structures import Instances

from tridet.modeling.dd3d.fcos2d import FCOS2DHead, FCOS2DInference, FCOS2DLoss
from tridet.modeling.dd3d.fcos3d import FCOS3DHead, FCOS3DHead_prediction, FCOS3DInference, FCOS3DLoss, UncertaintyLoss
# from tridet.modeling.dd3d.fcos3d_origin import FCOS3DHead, FCOS3DInference, FCOS3DLoss

from tridet.modeling.dd3d.postprocessing import nuscenes_sample_aggregate
from tridet.modeling.dd3d.prepare_targets import DD3DTargetPreparer
from tridet.modeling.feature_extractor import build_feature_extractor
from tridet.structures.image_list import ImageList
from tridet.utils.tensor2d import compute_features_locations as compute_locations_per_level
from tridet.modeling.dd3d.pose_estimation import PoseLoss
from tridet.modeling.dd3d.temporal_aggregation import TemporalWeight
from tridet.modeling.dd3d.dense_depth_loss import build_dense_depth_loss

from DeformableDETR.models.deformable_transformer import DeformableTransformer
from DeformableDETR.models.position_encoding import PositionEmbeddingLearned


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
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, _ = self.fcos3d_head(features)
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


"no empty query"
"transformer"
"object depth + dense depth"
@META_ARCH_REGISTRY.register()
class DD3D_VIDEO_PREDICTION2_reserve(nn.Module):
    #TODO Positional Encoding for DeformAtt
    #TODO Temporal Feature Aggregation for current and past

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_feature_extractor(cfg)

        # for para in self.backbone.parameters():
        #     para.requires_grad = False

        backbone_output_shape = self.backbone.output_shape()
        self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        self.backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.fcos2d_head = FCOS2DHead(cfg, self.backbone_output_shape)
        self.fcos2d_loss = FCOS2DLoss(cfg)
        self.fcos2d_inference = FCOS2DInference(cfg)

        self.prediction_on = cfg.MODEL.PREDICTION
        self.video_on = cfg.MODEL.VIDEO_ON
        self.depth_on = cfg.MODEL.DEPTH_ON

        if cfg.MODEL.BOX3D_ON:
            self.fcos3d_head = FCOS3DHead(cfg,
                                          self.backbone_output_shape,
                                          depth=self.depth_on,
                                          video=self.video_on,
                                          prediction=False)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        if cfg.MODEL.PREDICTION:
            self.fcos2d_head_prediction = FCOS2DHead(cfg, self.backbone_output_shape)
            self.fcos2d_loss_prediction = FCOS2DLoss(cfg, prediction=True)

            if cfg.MODEL.BOX3D_ON:
                self.fcos3d_head_prediction = FCOS3DHead(cfg,
                                                         self.backbone_output_shape,
                                                         depth=self.depth_on,
                                                         video=self.video_on,
                                                         prediction=self.prediction_on)
                self.fcos3d_inference_prediction = FCOS3DInference(cfg)
                self.fcos3d_loss_prediction = FCOS3DLoss(cfg, prediction=self.prediction_on)

        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3))


        if cfg.MODEL.DEPTH_ON:
            self.depth_loss = build_dense_depth_loss(cfg)
            self.depth_loss_pred = build_dense_depth_loss(cfg)


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
        batch_size = len(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        if 'intrinsics' in batched_inputs[0]:
            intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None

        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)

        # import previous images
        images_t1, images_t2 = None, None
        if self.prediction_on:
            images_t1 = [x["image_t1"].to(self.device) for x in batched_inputs]
            images_t1 = [self.preprocess_image(x) for x in images_t1]

            images_t2 = [x["image_t2"].to(self.device) for x in batched_inputs]
            images_t2 = [self.preprocess_image(x) for x in images_t2]

            images_t1 = ImageList.from_tensors(images_t1, self.backbone.size_divisibility, intrinsics=intrinsics)
            images_t2 = ImageList.from_tensors(images_t2, self.backbone.size_divisibility, intrinsics=intrinsics)


        gt_ego_pose = None
        if self.video_on and 'ego_pose' in batched_inputs[0]:
            gt_ego_pose = [x["ego_pose"] for x in batched_inputs]
            gt_ego_pose = torch.tensor(gt_ego_pose, dtype=torch.float32).to(self.device)

        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev, features_cur = [], []

        for f in self.in_features:
            feat_cur = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[0] + self.temporal_embedding[0, None, None, None, None]
            feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:] + self.temporal_embedding[1:, None, None, None, None]

            features_cur.append(feat_cur)
            features_prev.append(feat_prev.transpose(0,1))

        # temporal aggregation

        features_prev = self.temporal_head(features_prev, masks=None, pos_embeds=None)
        features = [feat_cur + feat_prev for (feat_cur, feat_prev) in zip(features_cur, features_prev)]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, _ = self.fcos3d_head(features, inv_intrinsics)

        if self.training:

            gt_dense_depth_c, gt_dense_depth_f = None, None
            if dense_depth is not None:
                gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
                gt_dense_depth_c = ImageList.from_tensors(
                    gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
                )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            if gt_dense_depth_c is not None:
                training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)

            if self.prediction_on:
                # prediction from previous images
                logits_prev, box2d_reg_prev, centerness_prev, _ = self.fcos2d_head_prediction(features_prev)
                if not self.only_box2d:
                    box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, dense_depth_prev, ego_pose = self.fcos3d_head_prediction(
                        features_prev, inv_intrinsics)

                # detection loss from previous images
                fcos2d_prev_loss, fcos2d_prev_info = self.fcos2d_loss_prediction(logits_prev, box2d_reg_prev, centerness_prev, training_targets)
                losses.update(fcos2d_prev_loss)

                if not self.only_box2d:
                    fcos3d_prev_loss = self.fcos3d_loss_prediction(
                        box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, _, inv_intrinsics,
                        fcos2d_prev_info, training_targets
                    )
                    losses.update(fcos3d_prev_loss)

            if dense_depth is not None:
                for lvl, x in enumerate(dense_depth):
                    loss_lvl = self.depth_loss(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 1)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

            if dense_depth_prev is not None:
                for lvl, x in enumerate(dense_depth_prev):
                    loss_lvl = self.depth_loss_pred(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_pred_lvl_{lvl}": loss_lvl})

                # for lvl, x in enumerate(dense_depth_prev):
                #     loss_lvl = self.depth_loss(x[:, 1], gt_dense_depth_f.tensor)["loss_dense_depth"]
                #     loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 1))  # Is sqrt(2) good?
                #     losses.update({f"loss_dense_depth_f_lvl_{lvl}": loss_lvl})

            #ego pose loss
            if gt_ego_pose is not None and ego_pose is not None:
                for lvl, x in enumerate(ego_pose):
                    loss_lvl = self.pose_loss(x, gt_ego_pose)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_lvl_{lvl}": loss_lvl})

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


"no empty query"
"transformer"
"object depth + dense depth"
@META_ARCH_REGISTRY.register()
class DD3D_VIDEO_PREDICTION2(nn.Module):
    #TODO Positional Encoding for DeformAtt
    #TODO Temporal Feature Aggregation for current and past

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_feature_extractor(cfg)

        # for para in self.backbone.parameters():
        #     para.requires_grad = False

        backbone_output_shape = self.backbone.output_shape()
        self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        self.backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.fcos2d_head = FCOS2DHead(cfg, self.backbone_output_shape)
        self.fcos2d_loss = FCOS2DLoss(cfg)
        self.fcos2d_inference = FCOS2DInference(cfg)

        self.prediction_on = cfg.MODEL.PREDICTION
        self.video_on = cfg.MODEL.VIDEO_ON
        self.depth_on = cfg.MODEL.DEPTH_ON

        if cfg.MODEL.BOX3D_ON:
            self.fcos3d_head = FCOS3DHead(cfg,
                                          self.backbone_output_shape,
                                          depth=self.depth_on,
                                          video=self.video_on,
                                          prediction=False)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        if cfg.MODEL.PREDICTION:
            self.fcos2d_head_prediction = FCOS2DHead(cfg, self.backbone_output_shape)
            self.fcos2d_loss_prediction = FCOS2DLoss(cfg, prediction=True)

            if cfg.MODEL.BOX3D_ON:
                self.fcos3d_head_prediction = FCOS3DHead(cfg,
                                                         self.backbone_output_shape,
                                                         depth=self.depth_on,
                                                         video=self.video_on,
                                                         prediction=self.prediction_on)
                self.fcos3d_inference_prediction = FCOS3DInference(cfg)
                self.fcos3d_loss_prediction = FCOS3DLoss(cfg, prediction=self.prediction_on)

        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3))


        if cfg.MODEL.DEPTH_ON:
            self.depth_loss = build_dense_depth_loss(cfg)
            self.depth_loss_pred = build_dense_depth_loss(cfg)


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
        batch_size = len(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        if 'intrinsics' in batched_inputs[0]:
            intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None

        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)

        # import previous images
        images_t1, images_t2 = None, None
        if self.prediction_on:
            images_t1 = [x["image_t1"].to(self.device) for x in batched_inputs]
            images_t1 = [self.preprocess_image(x) for x in images_t1]

            images_t2 = [x["image_t2"].to(self.device) for x in batched_inputs]
            images_t2 = [self.preprocess_image(x) for x in images_t2]

            images_t1 = ImageList.from_tensors(images_t1, self.backbone.size_divisibility, intrinsics=intrinsics)
            images_t2 = ImageList.from_tensors(images_t2, self.backbone.size_divisibility, intrinsics=intrinsics)


        gt_ego_pose = None
        if self.video_on and 'ego_pose' in batched_inputs[0]:
            gt_ego_pose = [x["ego_pose"] for x in batched_inputs]
            gt_ego_pose = torch.tensor(gt_ego_pose, dtype=torch.float32).to(self.device)

        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev, features_cur = [], []

        for f in self.in_features:
            feat_cur = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[0] + self.temporal_embedding[0, None, None, None, None]
            feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:] + self.temporal_embedding[1:, None, None, None, None]

            features_cur.append(feat_cur)
            features_prev.append(feat_prev.transpose(0,1))

        # temporal aggregation

        features_prev = self.temporal_head(features_prev, masks=None, pos_embeds=None)
        features = [feat_cur + feat_prev for (feat_cur, feat_prev) in zip(features_cur, features_prev)]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        logits, box2d_reg, centerness, _ = self.fcos2d_head_prediction(features_prev)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, _ = self.fcos3d_head_prediction(features_prev, inv_intrinsics)

        if self.training:

            gt_dense_depth_c, gt_dense_depth_f = None, None
            if dense_depth is not None:
                gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
                gt_dense_depth_c = ImageList.from_tensors(
                    gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
                )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            if gt_dense_depth_c is not None:
                training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)

            if self.prediction_on:
                # prediction from previous images
                logits_prev, box2d_reg_prev, centerness_prev, _ = self.fcos2d_head_prediction(features_prev)
                if not self.only_box2d:
                    box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, dense_depth_prev, ego_pose = self.fcos3d_head_prediction(
                        features_prev, inv_intrinsics)

                # detection loss from previous images
                fcos2d_prev_loss, fcos2d_prev_info = self.fcos2d_loss_prediction(logits_prev, box2d_reg_prev, centerness_prev, training_targets)
                losses.update(fcos2d_prev_loss)

                if not self.only_box2d:
                    fcos3d_prev_loss = self.fcos3d_loss_prediction(
                        box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, _, inv_intrinsics,
                        fcos2d_prev_info, training_targets
                    )
                    losses.update(fcos3d_prev_loss)

            if dense_depth is not None:
                for lvl, x in enumerate(dense_depth):
                    loss_lvl = self.depth_loss(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 1)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

            if dense_depth_prev is not None:
                for lvl, x in enumerate(dense_depth_prev):
                    loss_lvl = self.depth_loss_pred(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_pred_lvl_{lvl}": loss_lvl})

                # for lvl, x in enumerate(dense_depth_prev):
                #     loss_lvl = self.depth_loss(x[:, 1], gt_dense_depth_f.tensor)["loss_dense_depth"]
                #     loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 1))  # Is sqrt(2) good?
                #     losses.update({f"loss_dense_depth_f_lvl_{lvl}": loss_lvl})

            #ego pose loss
            if gt_ego_pose is not None and ego_pose is not None:
                for lvl, x in enumerate(ego_pose):
                    loss_lvl = self.pose_loss(x, gt_ego_pose)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_lvl_{lvl}": loss_lvl})

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


"no empty query"
"transformer"
"object depth + dense depth"
"without prediction head"
@META_ARCH_REGISTRY.register()
class DD3D_VIDEO_PREDICTION2_woPred(nn.Module):
    #TODO Positional Encoding for DeformAtt
    #TODO Temporal Feature Aggregation for current and past

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_feature_extractor(cfg)

        # for para in self.backbone.parameters():
        #     para.requires_grad = False

        backbone_output_shape = self.backbone.output_shape()
        self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        self.backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.fcos2d_head = FCOS2DHead(cfg, self.backbone_output_shape)
        self.fcos2d_loss = FCOS2DLoss(cfg)
        self.fcos2d_inference = FCOS2DInference(cfg)

        self.prediction_on = cfg.MODEL.PREDICTION
        self.video_on = cfg.MODEL.VIDEO_ON
        self.depth_on = cfg.MODEL.DEPTH_ON

        if cfg.MODEL.BOX3D_ON:
            self.fcos3d_head = FCOS3DHead(cfg,
                                          self.backbone_output_shape,
                                          depth=self.depth_on,
                                          video=self.video_on,
                                          prediction=False)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3))


        if cfg.MODEL.DEPTH_ON:
            self.depth_loss = build_dense_depth_loss(cfg)
            self.depth_loss_pred = build_dense_depth_loss(cfg)


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
        batch_size = len(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        if 'intrinsics' in batched_inputs[0]:
            intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None

        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)

        # import previous images
        images_t1, images_t2 = None, None
        if self.prediction_on:
            images_t1 = [x["image_t1"].to(self.device) for x in batched_inputs]
            images_t1 = [self.preprocess_image(x) for x in images_t1]

            images_t2 = [x["image_t2"].to(self.device) for x in batched_inputs]
            images_t2 = [self.preprocess_image(x) for x in images_t2]

            images_t1 = ImageList.from_tensors(images_t1, self.backbone.size_divisibility, intrinsics=intrinsics)
            images_t2 = ImageList.from_tensors(images_t2, self.backbone.size_divisibility, intrinsics=intrinsics)


        gt_ego_pose = None


        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev, features_cur = [], []

        for f in self.in_features:
            feat_cur = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[0] + self.temporal_embedding[0, None, None, None, None]
            feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:] + self.temporal_embedding[1:, None, None, None, None]

            features_cur.append(feat_cur)
            features_prev.append(feat_prev.transpose(0,1))

        # temporal aggregation

        features_prev = self.temporal_head(features_prev, masks=None, pos_embeds=None)
        features = [feat_cur + feat_prev for (feat_cur, feat_prev) in zip(features_cur, features_prev)]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, _ = self.fcos3d_head(features, inv_intrinsics)

        if self.training:

            gt_dense_depth_c, gt_dense_depth_f = None, None
            if dense_depth is not None:
                gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
                gt_dense_depth_c = ImageList.from_tensors(
                    gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
                )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            if gt_dense_depth_c is not None:
                training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)


            if dense_depth is not None:
                for lvl, x in enumerate(dense_depth):
                    loss_lvl = self.depth_loss(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 1)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

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

"no empty query"
"transformer"
"object depth + dense depth"
@META_ARCH_REGISTRY.register()
class DD3D_VIDEO_PREDICTION2_onlyPred(nn.Module):
    #TODO Positional Encoding for DeformAtt
    #TODO Temporal Feature Aggregation for current and past

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_feature_extractor(cfg)

        # for para in self.backbone.parameters():
        #     para.requires_grad = False

        backbone_output_shape = self.backbone.output_shape()
        self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        self.backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.fcos2d_head = FCOS2DHead(cfg, self.backbone_output_shape)
        self.fcos2d_loss = FCOS2DLoss(cfg)
        self.fcos2d_inference = FCOS2DInference(cfg)

        self.prediction_on = cfg.MODEL.PREDICTION
        self.video_on = cfg.MODEL.VIDEO_ON
        self.depth_on = cfg.MODEL.DEPTH_ON

        if cfg.MODEL.BOX3D_ON:
            self.fcos3d_head = FCOS3DHead(cfg,
                                          self.backbone_output_shape,
                                          depth=self.depth_on,
                                          video=self.video_on,
                                          prediction=False)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3))


        if cfg.MODEL.DEPTH_ON:
            self.depth_loss = build_dense_depth_loss(cfg)
            self.depth_loss_pred = build_dense_depth_loss(cfg)


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
        batch_size = len(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        if 'intrinsics' in batched_inputs[0]:
            intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None

        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)

        # import previous images
        images_t1, images_t2 = None, None
        if self.prediction_on:
            images_t1 = [x["image_t1"].to(self.device) for x in batched_inputs]
            images_t1 = [self.preprocess_image(x) for x in images_t1]

            images_t2 = [x["image_t2"].to(self.device) for x in batched_inputs]
            images_t2 = [self.preprocess_image(x) for x in images_t2]

            images_t1 = ImageList.from_tensors(images_t1, self.backbone.size_divisibility, intrinsics=intrinsics)
            images_t2 = ImageList.from_tensors(images_t2, self.backbone.size_divisibility, intrinsics=intrinsics)


        gt_ego_pose = None


        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev, features_cur = [], []

        for f in self.in_features:
            feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:] + self.temporal_embedding[1:, None, None, None, None]

            features_prev.append(feat_prev.transpose(0,1))

        # temporal aggregation

        features = self.temporal_head(features_prev, masks=None, pos_embeds=None)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, _ = self.fcos3d_head(features, inv_intrinsics)

        if self.training:

            gt_dense_depth_c, gt_dense_depth_f = None, None
            if dense_depth is not None:
                gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
                gt_dense_depth_c = ImageList.from_tensors(
                    gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
                )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            if gt_dense_depth_c is not None:
                training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)


            if dense_depth is not None:
                for lvl, x in enumerate(dense_depth):
                    loss_lvl = self.depth_loss(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 1)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

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


"no empty query"
"transformer"
"positional+temporal"
@META_ARCH_REGISTRY.register()
class DD3D_VIDEO_PREDICTION3(nn.Module):
    #TODO Positional Encoding for DeformAtt
    #TODO Temporal Feature Aggregation for current and past

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_feature_extractor(cfg)

        # for para in self.backbone.parameters():
        #     para.requires_grad = False

        backbone_output_shape = self.backbone.output_shape()
        self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        self.backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.fcos2d_head = FCOS2DHead(cfg, self.backbone_output_shape)
        self.fcos2d_loss = FCOS2DLoss(cfg)
        self.fcos2d_inference = FCOS2DInference(cfg)

        self.prediction_on = cfg.MODEL.PREDICTION
        self.video_on = cfg.MODEL.VIDEO_ON
        self.depth_on = cfg.MODEL.DEPTH_ON

        if cfg.MODEL.BOX3D_ON:
            self.fcos3d_head = FCOS3DHead(cfg,
                                          self.backbone_output_shape,
                                          depth=self.depth_on,
                                          video=self.video_on,
                                          prediction=False)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        if cfg.MODEL.PREDICTION:
            self.fcos2d_head_prediction = FCOS2DHead(cfg, self.backbone_output_shape)
            self.fcos2d_loss_prediction = FCOS2DLoss(cfg, prediction=True)

            if cfg.MODEL.BOX3D_ON:
                self.fcos3d_head_prediction = FCOS3DHead(cfg,
                                                         self.backbone_output_shape,
                                                         depth=self.depth_on,
                                                         video=self.video_on,
                                                         prediction=self.prediction_on)
                self.fcos3d_inference_prediction = FCOS3DInference(cfg)
                self.fcos3d_loss_prediction = FCOS3DLoss(cfg, prediction=self.prediction_on)

        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3))
            self.positional_encoder = PositionEmbeddingLearned(int(cfg.FE.FPN.OUT_CHANNELS))


        if cfg.MODEL.DEPTH_ON:
            self.depth_loss = build_dense_depth_loss(cfg)
            self.depth_loss_pred = build_dense_depth_loss(cfg)

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
        batch_size = len(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        if 'intrinsics' in batched_inputs[0]:
            intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None

        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)

        # import previous images
        images_t1, images_t2 = None, None
        if self.prediction_on:
            images_t1 = [x["image_t1"].to(self.device) for x in batched_inputs]
            images_t1 = [self.preprocess_image(x) for x in images_t1]

            images_t2 = [x["image_t2"].to(self.device) for x in batched_inputs]
            images_t2 = [self.preprocess_image(x) for x in images_t2]

            images_t1 = ImageList.from_tensors(images_t1, self.backbone.size_divisibility, intrinsics=intrinsics)
            images_t2 = ImageList.from_tensors(images_t2, self.backbone.size_divisibility, intrinsics=intrinsics)


        gt_ego_pose = None
        if self.video_on and 'ego_pose' in batched_inputs[0]:
            gt_ego_pose = [x["ego_pose"] for x in batched_inputs]
            gt_ego_pose = torch.tensor(gt_ego_pose, dtype=torch.float32).to(self.device)

        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev, features_cur = [], []

        for i,f in enumerate(self.in_features):

            feat_cur = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[0]
            feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:]

            pos_embed = self.positional_encoder(feat_cur)

            embed_cur = pos_embed + self.temporal_embedding[0]
            embed_prev = pos_embed[None, :] + self.temporal_embedding[1:][:, None, None, None, None]

            features_cur.append(feat_cur+embed_cur)
            features_prev.append((feat_prev+embed_prev).transpose(0,1))

            # temporal aggregation

        features_prev = self.temporal_head(features_prev, masks=None, pos_embeds=None)
        features = [feat_cur + feat_prev for (feat_cur, feat_prev) in zip(features_cur, features_prev)]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, _ = self.fcos3d_head(features, inv_intrinsics)

        if self.training:

            gt_dense_depth_c, gt_dense_depth_f = None, None
            if dense_depth is not None:
                gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
                gt_dense_depth_c = ImageList.from_tensors(
                    gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
                )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            if gt_dense_depth_c is not None:
                training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)

            if self.prediction_on:
                # prediction from previous images
                logits_prev, box2d_reg_prev, centerness_prev, _ = self.fcos2d_head_prediction(features_prev)
                if not self.only_box2d:
                    box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, dense_depth_prev, ego_pose = self.fcos3d_head_prediction(
                        features_prev, inv_intrinsics)

                # detection loss from previous images
                fcos2d_prev_loss, fcos2d_prev_info = self.fcos2d_loss_prediction(logits_prev, box2d_reg_prev, centerness_prev, training_targets)
                losses.update(fcos2d_prev_loss)

                if not self.only_box2d:
                    fcos3d_prev_loss = self.fcos3d_loss_prediction(
                        box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, _, inv_intrinsics,
                        fcos2d_prev_info, training_targets
                    )
                    losses.update(fcos3d_prev_loss)

            if dense_depth is not None:
                for lvl, x in enumerate(dense_depth):
                    loss_lvl = self.depth_loss(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

            if dense_depth_prev is not None:
                for lvl, x in enumerate(dense_depth_prev):
                    loss_lvl = self.depth_loss_pred(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_pred_lvl_{lvl}": loss_lvl})

                # for lvl, x in enumerate(dense_depth_prev):
                #     loss_lvl = self.depth_loss(x[:, 1], gt_dense_depth_f.tensor)["loss_dense_depth"]
                #     loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 1))  # Is sqrt(2) good?
                #     losses.update({f"loss_dense_depth_f_lvl_{lvl}": loss_lvl})

            #ego pose loss
            if gt_ego_pose is not None and ego_pose is not None:
                for lvl, x in enumerate(ego_pose):
                    loss_lvl = self.pose_loss(x, gt_ego_pose)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_lvl_{lvl}": loss_lvl})

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

"no empty query"
"transformer"
"positional+temporal+level embedding"
@META_ARCH_REGISTRY.register()
class DD3D_VIDEO_PREDICTION4(nn.Module):
    #TODO Positional Encoding for DeformAtt
    #TODO Temporal Feature Aggregation for current and past

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_feature_extractor(cfg)

        # for para in self.backbone.parameters():
        #     para.requires_grad = False

        backbone_output_shape = self.backbone.output_shape()
        self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        self.backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.fcos2d_head = FCOS2DHead(cfg, self.backbone_output_shape)
        self.fcos2d_loss = FCOS2DLoss(cfg)
        self.fcos2d_inference = FCOS2DInference(cfg)

        self.prediction_on = cfg.MODEL.PREDICTION
        self.video_on = cfg.MODEL.VIDEO_ON
        self.depth_on = cfg.MODEL.DEPTH_ON

        if cfg.MODEL.BOX3D_ON:
            self.fcos3d_head = FCOS3DHead(cfg,
                                          self.backbone_output_shape,
                                          depth=self.depth_on,
                                          video=self.video_on,
                                          prediction=False)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        if cfg.MODEL.PREDICTION:
            self.fcos2d_head_prediction = FCOS2DHead(cfg, self.backbone_output_shape)
            self.fcos2d_loss_prediction = FCOS2DLoss(cfg, prediction=True)

            if cfg.MODEL.BOX3D_ON:
                self.fcos3d_head_prediction = FCOS3DHead(cfg,
                                                         self.backbone_output_shape,
                                                         depth=self.depth_on,
                                                         video=self.video_on,
                                                         prediction=self.prediction_on)
                self.fcos3d_inference_prediction = FCOS3DInference(cfg)
                self.fcos3d_loss_prediction = FCOS3DLoss(cfg, prediction=self.prediction_on)

        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3, int(cfg.FE.FPN.OUT_CHANNELS)))
            self.level_embedding = nn.Parameter(torch.Tensor(5, int(cfg.FE.FPN.OUT_CHANNELS)))
            self.positional_encoder = PositionEmbeddingLearned(int(cfg.FE.FPN.OUT_CHANNELS))

            nn.init.uniform_(self.temporal_embedding)
            nn.init.uniform_(self.level_embedding)

        if cfg.MODEL.DEPTH_ON:
            self.depth_loss = build_dense_depth_loss(cfg)
            self.depth_loss_pred = build_dense_depth_loss(cfg)


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
        batch_size = len(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        if 'intrinsics' in batched_inputs[0]:
            intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None

        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)

        # import previous images
        images_t1, images_t2 = None, None
        if self.prediction_on:
            images_t1 = [x["image_t1"].to(self.device) for x in batched_inputs]
            images_t1 = [self.preprocess_image(x) for x in images_t1]

            images_t2 = [x["image_t2"].to(self.device) for x in batched_inputs]
            images_t2 = [self.preprocess_image(x) for x in images_t2]

            images_t1 = ImageList.from_tensors(images_t1, self.backbone.size_divisibility, intrinsics=intrinsics)
            images_t2 = ImageList.from_tensors(images_t2, self.backbone.size_divisibility, intrinsics=intrinsics)


        gt_ego_pose = None
        if self.video_on and 'ego_pose' in batched_inputs[0]:
            gt_ego_pose = [x["ego_pose"] for x in batched_inputs]
            gt_ego_pose = torch.tensor(gt_ego_pose, dtype=torch.float32).to(self.device)

        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev, features_cur = [], []

        for i,f in enumerate(self.in_features):

            feat_cur = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[0]
            feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:]

            pos_embed = self.positional_encoder(feat_cur)

            embed_cur = pos_embed + self.temporal_embedding[0][None, :, None, None] + self.level_embedding[i][None, :, None, None]
            embed_prev = pos_embed[None, :] + self.temporal_embedding[1:][:, None, :, None, None] + self.level_embedding[i][None, None, :, None, None]

            features_cur.append(feat_cur+embed_cur)
            features_prev.append((feat_prev+embed_prev).transpose(0,1))

            # temporal aggregation

        features_prev = self.temporal_head(features_prev, masks=None, pos_embeds=None)
        features = [feat_cur + feat_prev for (feat_cur, feat_prev) in zip(features_cur, features_prev)]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, _ = self.fcos3d_head(features, inv_intrinsics)

        if self.training:

            gt_dense_depth_c, gt_dense_depth_f = None, None
            if dense_depth is not None:
                gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
                gt_dense_depth_c = ImageList.from_tensors(
                    gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
                )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            if gt_dense_depth_c is not None:
                training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)

            if self.prediction_on:
                # prediction from previous images
                logits_prev, box2d_reg_prev, centerness_prev, _ = self.fcos2d_head_prediction(features_prev)
                if not self.only_box2d:
                    box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, dense_depth_prev, ego_pose = self.fcos3d_head_prediction(
                        features_prev, inv_intrinsics)

                # detection loss from previous images
                fcos2d_prev_loss, fcos2d_prev_info = self.fcos2d_loss_prediction(logits_prev, box2d_reg_prev, centerness_prev, training_targets)
                losses.update(fcos2d_prev_loss)

                if not self.only_box2d:
                    fcos3d_prev_loss = self.fcos3d_loss_prediction(
                        box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, _, inv_intrinsics,
                        fcos2d_prev_info, training_targets
                    )
                    losses.update(fcos3d_prev_loss)

            if dense_depth is not None:
                for lvl, x in enumerate(dense_depth):
                    loss_lvl = self.depth_loss(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

            if dense_depth_prev is not None:
                for lvl, x in enumerate(dense_depth_prev):
                    loss_lvl = self.depth_loss_pred(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_pred_lvl_{lvl}": loss_lvl})

                # for lvl, x in enumerate(dense_depth_prev):
                #     loss_lvl = self.depth_loss(x[:, 1], gt_dense_depth_f.tensor)["loss_dense_depth"]
                #     loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 1))  # Is sqrt(2) good?
                #     losses.update({f"loss_dense_depth_f_lvl_{lvl}": loss_lvl})

            #ego pose loss
            if gt_ego_pose is not None and ego_pose is not None:
                for lvl, x in enumerate(ego_pose):
                    loss_lvl = self.pose_loss(x, gt_ego_pose)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_lvl_{lvl}": loss_lvl})

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


"temporal depth uncertainty"
@META_ARCH_REGISTRY.register()
class DD3D_VIDEO_PREDICTION8(nn.Module):
    #TODO Positional Encoding for DeformAtt
    #TODO Temporal Feature Aggregation for current and past

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_feature_extractor(cfg)

        # for para in self.backbone.parameters():
        #     para.requires_grad = False

        backbone_output_shape = self.backbone.output_shape()
        self.in_features = cfg.DD3D.IN_FEATURES or list(backbone_output_shape.keys())
        self.backbone_output_shape = [backbone_output_shape[f] for f in self.in_features]

        self.feature_locations_offset = cfg.DD3D.FEATURE_LOCATIONS_OFFSET

        self.fcos2d_head = FCOS2DHead(cfg, self.backbone_output_shape)
        self.fcos2d_loss = FCOS2DLoss(cfg)
        self.fcos2d_inference = FCOS2DInference(cfg)

        self.prediction_on = cfg.MODEL.PREDICTION
        self.video_on = cfg.MODEL.VIDEO_ON
        self.depth_on = cfg.MODEL.DEPTH_ON

        if cfg.MODEL.BOX3D_ON:
            self.fcos3d_head = FCOS3DHead(cfg,
                                          self.backbone_output_shape,
                                          depth=self.depth_on)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True

        if cfg.MODEL.PREDICTION:
            self.fcos2d_head_prediction = FCOS2DHead(cfg, self.backbone_output_shape)
            self.fcos2d_loss_prediction = FCOS2DLoss(cfg, prediction=True)

            if cfg.MODEL.BOX3D_ON:
                self.fcos3d_head_prediction = FCOS3DHead_prediction(cfg,
                                                         self.backbone_output_shape,
                                                         depth=self.depth_on,
                                                         video=self.video_on)
                self.fcos3d_inference_prediction = FCOS3DInference(cfg)
                self.fcos3d_loss_prediction = FCOS3DLoss(cfg, prediction=self.prediction_on)

        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg)
            self.temporal_weight = TemporalWeight(cfg, self.backbone_output_shape)
            self.depth_loss_pred = build_dense_depth_loss(cfg)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3, int(cfg.FE.FPN.OUT_CHANNELS)))
            self.level_embedding = nn.Parameter(torch.Tensor(5, int(cfg.FE.FPN.OUT_CHANNELS)))
            # self.temporal_embedding = nn.Parameter(torch.Tensor(3))
            # self.level_embedding = nn.Parameter(torch.Tensor(5))
            self.positional_encoder = PositionEmbeddingLearned(int(cfg.FE.FPN.OUT_CHANNELS))

            nn.init.uniform_(self.temporal_embedding)
            nn.init.uniform_(self.level_embedding)

        if cfg.MODEL.DEPTH_ON:
            self.depth_loss = build_dense_depth_loss(cfg)
            self.uncertainty_loss = UncertaintyLoss(cfg)

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
        batch_size = len(batched_inputs)

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.preprocess_image(x) for x in images]

        if 'intrinsics' in batched_inputs[0]:
            intrinsics = [x['intrinsics'].to(self.device) for x in batched_inputs]
        else:
            intrinsics = None

        images = ImageList.from_tensors(images, self.backbone.size_divisibility, intrinsics=intrinsics)

        # import previous images
        images_t1, images_t2 = None, None
        if self.prediction_on:
            images_t1 = [x["image_t1"].to(self.device) for x in batched_inputs]
            images_t1 = [self.preprocess_image(x) for x in images_t1]

            images_t2 = [x["image_t2"].to(self.device) for x in batched_inputs]
            images_t2 = [self.preprocess_image(x) for x in images_t2]

            images_t1 = ImageList.from_tensors(images_t1, self.backbone.size_divisibility, intrinsics=intrinsics)
            images_t2 = ImageList.from_tensors(images_t2, self.backbone.size_divisibility, intrinsics=intrinsics)


        gt_ego_pose = None
        if self.video_on and 'ego_pose' in batched_inputs[0]:
            gt_ego_pose = [x["ego_pose"] for x in batched_inputs]
            gt_ego_pose = torch.tensor(gt_ego_pose, dtype=torch.float32).to(self.device)

        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev, features_cur = [], []
        query = []
        pos_embed = []

        for i,f in enumerate(self.in_features):
            feat_cur = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[0]
            feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:]

            pos_embed = self.positional_encoder(feat_cur)

            embed_cur = pos_embed + self.temporal_embedding[0][None, :, None, None] + self.level_embedding[i][None, :, None, None]
            embed_prev = pos_embed[None, :] + self.temporal_embedding[1:][:, None, :, None, None] + self.level_embedding[i][None, None, :, None, None]
            # embed_cur = pos_embed + self.temporal_embedding[0] + self.level_embedding[i]
            # embed_prev = pos_embed[None, :] + self.temporal_embedding[1:][:, None, None, None, None] + self.level_embedding[i]

            # embed_cur = pos_embed
            # embed_prev = pos_embed[None, :]

            features_cur.append(feat_cur+embed_cur)
            features_prev.append((feat_prev+embed_prev).transpose(0,1))


        # temporal aggregation
        features_prev = self.temporal_head(features_prev, masks=None, pos_embeds=None)
        # features = self.temporal_weight(features_cur, features_prev)
        features = [feat_cur + feat_prev for (feat_cur, feat_prev) in zip(features_cur, features_prev)]


        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, dense_depth_low, depth_un = self.fcos3d_head(features, inv_intrinsics)

        if self.training:

            gt_dense_depth_c, gt_dense_depth_f = None, None
            if dense_depth is not None:
                gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
                gt_dense_depth_c = ImageList.from_tensors(
                    gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
                )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            if gt_dense_depth_c is not None:
                training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)

            if self.prediction_on:
                # prediction from previous images
                logits_prev, box2d_reg_prev, centerness_prev, _ = self.fcos2d_head_prediction(features_prev)
                if not self.only_box2d:
                    box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, dense_depth_prev, dense_depth_low_prev, ego_pose = self.fcos3d_head_prediction(
                        features_prev, inv_intrinsics)

                # detection loss from previous images
                fcos2d_prev_loss, fcos2d_prev_info = self.fcos2d_loss_prediction(logits_prev, box2d_reg_prev, centerness_prev, training_targets)
                losses.update(fcos2d_prev_loss)

                if not self.only_box2d:
                    fcos3d_prev_loss = self.fcos3d_loss_prediction(
                        box3d_quat_prev, box3d_ctr_prev, box3d_depth_prev, box3d_size_prev, box3d_conf_prev, _, inv_intrinsics,
                        fcos2d_prev_info, training_targets
                    )
                    losses.update(fcos3d_prev_loss)

            if dense_depth is not None and dense_depth_prev is not None:
                for lvl, (c,p) in enumerate(zip(dense_depth, dense_depth_prev)):
                    loss_lvl = self.depth_loss(c, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 1)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

                    loss_lvl = self.depth_loss_pred(p, gt_dense_depth_c.tensor)["loss_dense_depth"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
                    losses.update({f"loss_dense_depth_pred_lvl_{lvl}": loss_lvl})

            if depth_un is not None:
                for lvl, (c, p, un) in enumerate(zip(dense_depth_low, dense_depth_low_prev, depth_un)):
                    loss_lvl = ((torch.abs(c-p) / (torch.exp(un)+1e-7)) + un) / np.sqrt(4)
                    loss_lvl = self.uncertainty_loss(loss_lvl)["loss_uncertainty"]

                    losses.update({f"loss_uncertainty_lvl_{lvl}": loss_lvl})

            #ego pose loss
            if gt_ego_pose is not None and ego_pose is not None:
                for lvl, x in enumerate(ego_pose):
                    loss_lvl = self.pose_loss(x, gt_ego_pose)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(4) ** (3-lvl))  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_lvl_{lvl}": loss_lvl})

            return losses

        else:
            pred_instances, fcos2d_info = self.fcos2d_inference(
                logits, box2d_reg, centerness, locations, images.image_sizes
            )
            if not self.only_box2d:
                # This adds 'pred_boxes3d' and 'scores_3d' to Instances in 'pred_instances' in place.
                self.fcos3d_inference(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, depth_un, inv_intrinsics, pred_instances,
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

