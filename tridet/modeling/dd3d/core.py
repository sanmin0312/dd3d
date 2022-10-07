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
# from tridet.modeling.dd3d.fcos3d_rev import FCOS3DHead, FCOS3DHead_prediction, FCOS3DInference, FCOS3DLoss, UncertaintyLoss
from tridet.modeling.dd3d.fcos3d import FCOS3DHead, FCOS3DInference, FCOS3DLoss

from tridet.modeling.dd3d.postprocessing import nuscenes_sample_aggregate
from tridet.modeling.dd3d.prepare_targets import DD3DTargetPreparer
from tridet.modeling.feature_extractor import build_feature_extractor
from tridet.structures.image_list import ImageList
from tridet.utils.tensor2d import compute_features_locations as compute_locations_per_level
from tridet.modeling.dd3d.pose_estimation import PoseLoss
from tridet.modeling.dd3d.temporal_aggregation import TemporalWeight
from tridet.modeling.dd3d.dense_depth_loss import build_dense_depth_loss, build_dense_depth_uncertainty_loss
from tridet.modeling.dd3d.pose_estimation import FCOSPoseHead
from tridet.modeling.dd3d.dense_depth import DD3DDenseDepthHead

from torchmetrics.functional import structural_similarity_index_measure as SSIM

from DeformableDETR.models.deformable_transformer import DeformableTransformer, DeformableTransformer_warping

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


"""
1. Temporal Deformable Cross Attention
2. 
"""
@META_ARCH_REGISTRY.register()
class P2D(nn.Module):
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
            self.temporal_head = DeformableTransformer(cfg, )

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3,256))
            torch.nn.init.kaiming_uniform_(self.temporal_embedding, a=1)

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
            feat_cur = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[0] + self.temporal_embedding[0][None, :, None, None]
            feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:] + self.temporal_embedding[1:, None, :, None, None]
            # feat_cur = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[0]
            # feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:]

            features_cur.append(feat_cur)
            features_prev.append(feat_prev.transpose(0,1))

        # temporal aggregation
        features_prev = self.temporal_head(features_prev, masks=None, pos_embeds=None)


        # features = [torch.stack((f_c, f_p), dim=1) for f_p, f_c in zip(features_prev, features_cur)]
        # features = self.temporal_head(features, masks=None, pos_embeds=None)
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
        # if True:
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

    # def pos_embeddings(self, feature_shape):
    #
    #     "feature_shape: [image_num, batch_size, channel_size, H, W]"
    #
    #     h=feature_shape[-2]
    #     w=feature_shape[-1]
    #
    #     pos_h = torch.arange(0, 1, 1./h)
    #     pos_w = torch.arange(0, 1, 1./w)


@META_ARCH_REGISTRY.register()
class P2D_FS(nn.Module):
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
                                          video=self.video_on)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True


        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg, )
            self.attention_warping = DeformableTransformer_warping(cfg, )
            self.temporal_weight = TemporalWeight(cfg, self.backbone_output_shape)

            #ego pose estimation network
            self.pose_haed=FCOSPoseHead(cfg, self.backbone_output_shape)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3,256))
            torch.nn.init.kaiming_uniform_(self.temporal_embedding, a=1)

            self.positional_embedding = PositionEmbeddingLearned()

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


        gt_ego_pose, gt_ego_pose_prev = None, None
        if self.video_on and 'ego_pose' in batched_inputs[0]:
            gt_ego_pose = [x["ego_pose"] for x in batched_inputs]
            gt_ego_pose = torch.tensor(gt_ego_pose, dtype=torch.float32).to(self.device)

        if self.video_on and 'ego_pose_prev' in batched_inputs[0]:
            gt_ego_pose_prev = [x["ego_pose_prev"] for x in batched_inputs]
            gt_ego_pose_prev = torch.tensor(gt_ego_pose_prev, dtype=torch.float32).to(self.device)

        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev, features_cur = [], []

        for f in self.in_features:
            pos_embedding = self.positional_embedding(features[f][:1])
            feat = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1) + self.temporal_embedding[:, None, :, None, None] + pos_embedding.unsqueeze(dim=0)

            features_cur.append(feat[0])
            features_prev.append(feat[1:].transpose(0,1))

        # temporal aggregation
        features_prev = self.temporal_head(features_prev, masks=None, pos_embeds=None) #features_prev = [B, T, C, H, W]

        # past pose estimation
        ego_pose_prev = self.pose_haed(features_prev)

        features_prev = self.attention_warping([torch.stack((feat_cur, feat_prev), dim=1) for (feat_cur, feat_prev) in zip(features_cur, features_prev)], masks=None, pos_embeds=None)
        # features = [torch.stack((f_c, f_p), dim=1) for f_p, f_c in zip(features_prev, features_cur)]
        # features = self.temporal_head(features, masks=None, pos_embeds=None)

        features, _ = self.temporal_weight(features_cur, features_prev)
        # features = [feat_cur + feat_prev for (feat_cur, feat_prev) in zip(features_cur, features_prev)]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        locations = self.compute_locations(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        logits, box2d_reg, centerness, _ = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, ego_pose = self.fcos3d_head(features, inv_intrinsics)

        if self.training:
        # if True:
            # gt_dense_depth_c, gt_dense_depth_f = None, None
            # if dense_depth is not None:
            #     gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
            #     gt_dense_depth_c = ImageList.from_tensors(
            #         gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
            #     )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            # if gt_dense_depth_c is not None:
            #     training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)

            #
            # if dense_depth is not None:
            #     for lvl, x in enumerate(dense_depth):
            #         loss_lvl = self.depth_loss(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
            #         loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
            #         losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

            if True:
                for lvl, (cur, prev) in enumerate(zip(features_cur, features_prev)):
                    if lvl > 2:
                        continue

                    mask = 3-lvl

                    # loss_lvl = 1-SSIM(prev[:,:,mask:-1*mask, mask:-1*mask], cur[:,:,mask:-1*mask, mask:-1*mask].detach(), gaussian_kernel=False, kernel_size=5)
                    loss_lvl = 1 - SSIM(prev,
                                        cur.detach(),
                                        gaussian_kernel=False,
                                        kernel_size=5
                                        )
                    losses.update({f"loss_ssim_lvl_{lvl}": loss_lvl})


            # if dense_depth_prev is not None:
            #     for lvl, x in enumerate(dense_depth_prev):
            #         loss_lvl = self.depth_loss_pred(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
            #         loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
            #         losses.update({f"loss_dense_depth_pred_lvl_{lvl}": loss_lvl})

            #ego pose loss
            if gt_ego_pose is not None and ego_pose is not None:
                for lvl, x in enumerate(ego_pose):
                    loss_lvl = self.pose_loss(x, gt_ego_pose)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_lvl_{lvl}": loss_lvl})

            if gt_ego_pose_prev is not None and ego_pose_prev is not None:
                for lvl, x in enumerate(ego_pose_prev):
                    loss_lvl = self.pose_loss(x, gt_ego_pose_prev)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_prev_lvl_{lvl}": loss_lvl})

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

    # def pos_embeddings(self, feature_shape):
    #
    #     "feature_shape: [image_num, batch_size, channel_size, H, W]"
    #
    #     h=feature_shape[-2]
    #     w=feature_shape[-1]
    #
    #     pos_h = torch.arange(0, 1, 1./h)
    #     pos_w = torch.arange(0, 1, 1./w)


@META_ARCH_REGISTRY.register()
class P2D_FS2(nn.Module):
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
                                          video=self.video_on)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True


        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg, )
            self.attention_warping = DeformableTransformer_warping(cfg, )
            self.temporal_weight = TemporalWeight(cfg, self.backbone_output_shape)

            #ego pose estimation network
            self.pose_haed=FCOSPoseHead(cfg, self.backbone_output_shape)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3,256))
            torch.nn.init.kaiming_uniform_(self.temporal_embedding, a=1)

            self.positional_embedding = PositionEmbeddingLearned()

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


        gt_ego_pose, gt_ego_pose_prev = None, None
        if self.video_on and 'ego_pose' in batched_inputs[0]:
            gt_ego_pose = [x["ego_pose"] for x in batched_inputs]
            gt_ego_pose = torch.tensor(gt_ego_pose, dtype=torch.float32).to(self.device)

        if self.video_on and 'ego_pose_prev' in batched_inputs[0]:
            gt_ego_pose_prev = [x["ego_pose_prev"] for x in batched_inputs]
            gt_ego_pose_prev = torch.tensor(gt_ego_pose_prev, dtype=torch.float32).to(self.device)

        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev1, features_prev2, features_cur = [], [], []

        for f in self.in_features:
            pos_embedding = self.positional_embedding(features[f][:1])
            feat = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1) + self.temporal_embedding[:, None, :, None, None] + pos_embedding.unsqueeze(dim=0)

            features_cur.append(feat[0])
            features_prev1.append(feat[1])
            features_prev2.append(feat[2])

        # past pose estimation
        ego_pose_prev, pose_prev_embed = self.pose_haed([torch.cat((x1, x2), dim=1) for x1, x2 in zip(features_prev1, features_prev2)])
        ego_pose, pose_embed = self.pose_haed([torch.cat((x1, x2), dim=1) for x1, x2 in zip(features_cur, features_prev1)])

        # temporal aggregation
        features_prev = self.temporal_head([torch.stack((x1+embed[:, :, None, None] , x2), dim=1) for x1, x2, embed in zip(features_prev1, features_prev2, pose_prev_embed)], masks=None, pos_embeds=None) #features_prev = [B, T, C, H, W]

        features_prev = self.attention_warping([torch.stack((x1+embed[:, :, None, None] , x1+embed[:, :, None, None]), dim=1) for x1, embed in zip(features_prev, pose_embed)], masks=None, pos_embeds=None)
        # features = [torch.stack((f_c, f_p), dim=1) for f_p, f_c in zip(features_prev, features_cur)]
        # features = self.temporal_head(features, masks=None, pos_embeds=None)

        # features, _ = self.temporal_weight(features_cur, features_prev)
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
        # if True:
            # gt_dense_depth_c, gt_dense_depth_f = None, None
            # if dense_depth is not None:
            #     gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
            #     gt_dense_depth_c = ImageList.from_tensors(
            #         gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
            #     )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            # if gt_dense_depth_c is not None:
            #     training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)

            #
            # if dense_depth is not None:
            #     for lvl, x in enumerate(dense_depth):
            #         loss_lvl = self.depth_loss(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
            #         loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
            #         losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

            if True:
                for lvl, (cur, prev) in enumerate(zip(features_cur, features_prev)):
                    if lvl > 2:
                        continue

                                        # loss_lvl = 1-SSIM(prev[:,:,mask:-1*mask, mask:-1*mask], cur[:,:,mask:-1*mask, mask:-1*mask].detach(), gaussian_kernel=False, kernel_size=5)
                    loss_lvl = 1 - SSIM(prev,
                                        cur.detach(),
                                        gaussian_kernel=False,
                                        kernel_size=3
                                        )

                    loss_lvl = loss_lvl * ((np.sqrt(2) ** (3-lvl)))  # Is sqrt(2) good?

                    losses.update({f"loss_ssim_lvl_{lvl}": loss_lvl})


            # if dense_depth_prev is not None:
            #     for lvl, x in enumerate(dense_depth_prev):
            #         loss_lvl = self.depth_loss_pred(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
            #         loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
            #         losses.update({f"loss_dense_depth_pred_lvl_{lvl}": loss_lvl})

            #ego pose loss
            if gt_ego_pose is not None and ego_pose is not None:
                for lvl, x in enumerate(ego_pose):
                    loss_lvl = self.pose_loss(x, gt_ego_pose)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_lvl_{lvl}": loss_lvl})

            if gt_ego_pose_prev is not None and ego_pose_prev is not None:
                for lvl, x in enumerate(ego_pose_prev):
                    loss_lvl = self.pose_loss(x, gt_ego_pose_prev)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_prev_lvl_{lvl}": loss_lvl})

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

    # def pos_embeddings(self, feature_shape):
    #
    #     "feature_shape: [image_num, batch_size, channel_size, H, W]"
    #
    #     h=feature_shape[-2]
    #     w=feature_shape[-1]
    #
    #     pos_h = torch.arange(0, 1, 1./h)
    #     pos_w = torch.arange(0, 1, 1./w)



@META_ARCH_REGISTRY.register()
class P2D_FS2_1006(nn.Module):
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
                                          video=self.video_on)
            self.fcos3d_loss = FCOS3DLoss(cfg)
            self.fcos3d_inference = FCOS3DInference(cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True


        if cfg.MODEL.VIDEO_ON:
            self.pose_loss = PoseLoss(cfg)
            self.temporal_head = DeformableTransformer(cfg, )
            self.attention_warping = DeformableTransformer_warping(cfg, )
            self.temporal_weight = TemporalWeight(cfg, self.backbone_output_shape)

            #ego pose estimation network
            self.pose_haed=FCOSPoseHead(cfg, self.backbone_output_shape)

            #temporal embedding
            self.temporal_embedding = nn.Parameter(torch.Tensor(3,256))
            torch.nn.init.kaiming_uniform_(self.temporal_embedding, a=1)

            self.positional_embedding = PositionEmbeddingLearned()

            #feature similarity loss
            self.mseloss = nn.MSELoss(reduction='mean')

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


        gt_ego_pose, gt_ego_pose_prev = None, None
        if self.video_on and 'ego_pose' in batched_inputs[0]:
            gt_ego_pose = [x["ego_pose"] for x in batched_inputs]
            gt_ego_pose = torch.tensor(gt_ego_pose, dtype=torch.float32).to(self.device)

        if self.video_on and 'ego_pose_prev' in batched_inputs[0]:
            gt_ego_pose_prev = [x["ego_pose_prev"] for x in batched_inputs]
            gt_ego_pose_prev = torch.tensor(gt_ego_pose_prev, dtype=torch.float32).to(self.device)

        feature_cur = self.backbone(images.tensor)

        with torch.no_grad():
            feature_prev = self.backbone(torch.cat((images_t1.tensor, images_t2.tensor)))

        features_prev1, features_prev2, features_cur = [], [], []

        for f in self.in_features:
            pos_embedding = self.positional_embedding(feature_prev[f][:1])
            feat_prev = feature_prev[f].reshape(2, batch_size, feature_prev[f].size(1), feature_prev[f].size(2), -1) + self.temporal_embedding[1:, None, :, None, None] + pos_embedding.unsqueeze(dim=0)
            feat_cur = feature_cur[f].reshape(1, batch_size, feature_prev[f].size(1), feature_prev[f].size(2), -1) + self.temporal_embedding[:1, None, :, None, None] + pos_embedding.unsqueeze(dim=0)

            features_cur.append(feat_cur[0])
            features_prev1.append(feat_prev[0])
            features_prev2.append(feat_prev[1])

        # past pose estimation
        ego_pose_prev, pose_prev_embed = self.pose_haed([torch.cat((x1, x2), dim=1) for x1, x2 in zip(features_prev1, features_prev2)])
        ego_pose, pose_embed = self.pose_haed([torch.cat((x1, x2), dim=1) for x1, x2 in zip(features_cur, features_prev1)])

        # temporal aggregation
        features_prev = self.temporal_head([torch.stack((x1+embed[:, :, None, None] , x2), dim=1) for x1, x2, embed in zip(features_prev1, features_prev2, pose_prev_embed)], masks=None, pos_embeds=None) #features_prev = [B, T, C, H, W]

        features_prev = self.attention_warping([torch.stack((x1+embed[:, :, None, None] , x1), dim=1) for x1, embed in zip(features_prev, pose_embed)], masks=None, pos_embeds=None)
        # features = [torch.stack((f_c, f_p), dim=1) for f_p, f_c in zip(features_prev, features_cur)]
        # features = self.temporal_head(features, masks=None, pos_embeds=None)

        features, _ = self.temporal_weight(features_cur, [x.clone().detach() for x in features_prev])
        # features = [feat_cur + feat_prev for (feat_cur, feat_prev) in zip(features_cur, features_prev)]

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
        # if True:
            # gt_dense_depth_c, gt_dense_depth_f = None, None
            # if dense_depth is not None:
            #     gt_dense_depth_c = [x["depth_c"].to(self.device) for x in batched_inputs]
            #     gt_dense_depth_c = ImageList.from_tensors(
            #         gt_dense_depth_c, self.backbone.size_divisibility, intrinsics=intrinsics
            #     )

            assert gt_instances is not None
            feature_shapes = [x.shape[-2:] for x in features]
            training_targets = self.prepare_targets(locations, gt_instances, feature_shapes)
            # if gt_dense_depth_c is not None:
            #     training_targets.update({"dense_depth_c": gt_dense_depth_c})

            losses = {}
            fcos2d_loss, fcos2d_info = self.fcos2d_loss(logits, box2d_reg, centerness, training_targets)
            losses.update(fcos2d_loss)

            if not self.only_box2d:
                fcos3d_loss = self.fcos3d_loss(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, _, inv_intrinsics,
                    fcos2d_info, training_targets
                )
                losses.update(fcos3d_loss)

            #
            # if dense_depth is not None:
            #     for lvl, x in enumerate(dense_depth):
            #         loss_lvl = self.depth_loss(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
            #         loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
            #         losses.update({f"loss_dense_depth_lvl_{lvl}": loss_lvl})

            if True:
                for lvl, (cur, prev) in enumerate(zip(features_cur, features_prev)):
                    if lvl > 2:
                        continue


                    mask = 5-lvl**2


                    cur = cur[:, :, mask:-1 * mask, mask:-1 * mask].detach()
                    prev = prev[:, :, mask:-1 * mask, mask:-1 * mask]

                    loss_lvl = self.mseloss(prev, cur)


                    # loss_lvl = 1 - SSIM(prev,
                    #                     cur.detach(),
                    #                     gaussian_kernel=False,
                    #                     kernel_size=3
                    #                     )

                    # loss_lvl = loss_lvl * ((np.sqrt(2) ** (3-lvl)))  # Is sqrt(2) good?

                    losses.update({f"loss_ssim_lvl_{lvl}": loss_lvl})


            # if dense_depth_prev is not None:
            #     for lvl, x in enumerate(dense_depth_prev):
            #         loss_lvl = self.depth_loss_pred(x, gt_dense_depth_c.tensor)["loss_dense_depth"]
            #         loss_lvl = loss_lvl / (np.sqrt(4) ** (lvl + 2)) # Is sqrt(2) good?
            #         losses.update({f"loss_dense_depth_pred_lvl_{lvl}": loss_lvl})

            #ego pose loss
            if gt_ego_pose is not None and ego_pose is not None:
                for lvl, x in enumerate(ego_pose):
                    loss_lvl = self.pose_loss(x, gt_ego_pose)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_lvl_{lvl}": loss_lvl})

            if gt_ego_pose_prev is not None and ego_pose_prev is not None:
                for lvl, x in enumerate(ego_pose_prev):
                    loss_lvl = self.pose_loss(x, gt_ego_pose_prev)["loss_ego_pose"]
                    loss_lvl = loss_lvl / (np.sqrt(2) ** (3-lvl)) *2  # Is sqrt(2) good?
                    losses.update({f"loss_ego_pose_prev_lvl_{lvl}": loss_lvl})

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