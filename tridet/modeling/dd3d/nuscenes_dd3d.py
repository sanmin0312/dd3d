# Copyright 2021 Toyota Research Institute.  All rights reserved.
import torch
import torch.nn.functional as F
from fvcore.nn.smooth_l1_loss import smooth_l1_loss
from torch import nn

from detectron2.layers import Conv2d, cat
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess as resize_instances
from detectron2.structures import Instances
from detectron2.utils import comm as d2_comm

from tridet.data.datasets.nuscenes.build import MAX_NUM_ATTRIBUTES
from tridet.modeling.dd3d.core import DD3D, DD3D_VIDEO_PREDICTION4
from tridet.modeling.dd3d.postprocessing import get_group_idxs, nuscenes_sample_aggregate
from tridet.modeling.dd3d.prepare_targets import DD3DTargetPreparer
from tridet.structures.boxes3d import Boxes3D
from tridet.structures.image_list import ImageList
from tridet.utils.comm import reduce_sum

INF = 100000000.


class NuscenesDD3DTargetPreparer(DD3DTargetPreparer):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        assert self.dd3d_enabled, f"{type(self).__name__} requires dd3d_enabled = True"

    def __call__(self, locations, gt_instances, feature_shapes):
        num_loc_list = [len(loc) for loc in locations]

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(loc_to_size_range_per_level[None].expand(num_loc_list[l], -1))

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(locations, dim=0)

        training_targets = self.compute_targets_for_locations(locations, gt_instances, loc_to_size_range, num_loc_list)

        training_targets["locations"] = [locations.clone() for _ in range(len(gt_instances))]
        training_targets["im_inds"] = [
            locations.new_ones(locations.size(0), dtype=torch.long) * i for i in range(len(gt_instances))
        ]

        box2d = training_targets.pop("box2d", None)

        # transpose im first training_targets to level first ones
        training_targets = {k: self._transpose(v, num_loc_list) for k, v in training_targets.items() if k != "box2d"}

        training_targets["fpn_levels"] = [
            loc.new_ones(len(loc), dtype=torch.long) * level for level, loc in enumerate(training_targets["locations"])
        ]

        # Flatten targets: (L x B x H x W, TARGET_SIZE)
        labels = cat([x.reshape(-1) for x in training_targets["labels"]])
        box2d_reg_targets = cat([x.reshape(-1, 4) for x in training_targets["box2d_reg"]])

        target_inds = cat([x.reshape(-1) for x in training_targets["target_inds"]])
        locations = cat([x.reshape(-1, 2) for x in training_targets["locations"]])
        im_inds = cat([x.reshape(-1) for x in training_targets["im_inds"]])
        fpn_levels = cat([x.reshape(-1) for x in training_targets["fpn_levels"]])

        pos_inds = torch.nonzero(labels != self.num_classes).squeeze(1)

        targets = {
            "labels": labels,
            "box2d_reg_targets": box2d_reg_targets,
            "locations": locations,
            "target_inds": target_inds,
            "im_inds": im_inds,
            "fpn_levels": fpn_levels,
            "pos_inds": pos_inds
        }

        if self.dd3d_enabled:
            box3d_targets = Boxes3D.cat(training_targets["box3d"])
            targets.update({"box3d_targets": box3d_targets})

            if box2d is not None:
                # Original format is B x L x (H x W, 4)
                # Need to be in L x (B, 4, H, W).
                batched_box2d = []
                for lvl, per_lvl_box2d in enumerate(zip(*box2d)):
                    # B x (H x W, 4)
                    h, w = feature_shapes[lvl]
                    batched_box2d_lvl = torch.stack([x.T.reshape(4, h, w) for x in per_lvl_box2d], dim=0)
                    batched_box2d.append(batched_box2d_lvl)
                targets.update({"batched_box2d": batched_box2d})

        # Nuscenes targets -- attribute / speed
        attributes = cat([x.reshape(-1) for x in training_targets["attributes"]])
        speeds = cat([x.reshape(-1) for x in training_targets["speeds"]])

        targets.update({'attributes': attributes, 'speeds': speeds})

        return targets

    def compute_targets_for_locations(self, locations, targets, size_ranges, num_loc_list):
        labels = []
        box2d_reg = []

        if self.dd3d_enabled:
            box3d = []

        target_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        # NuScenes targets  -- attribute / speed
        attributes, speeds = [], []

        num_targets = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                # reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                box2d_reg.append(locations.new_zeros((locations.size(0), 4)))
                target_inds.append(labels_per_im.new_zeros(locations.size(0)) - 1)

                if self.dd3d_enabled:
                    box3d.append(
                        Boxes3D(
                            locations.new_zeros(locations.size(0), 4),
                            locations.new_zeros(locations.size(0), 2),
                            locations.new_zeros(locations.size(0), 1),
                            locations.new_zeros(locations.size(0), 3),
                            locations.new_zeros(locations.size(0), 3, 3),
                        ).to(torch.float32)
                    )
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            # reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            box2d_reg_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(bboxes, num_loc_list, xs, ys)
            else:
                is_in_boxes = box2d_reg_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = box2d_reg_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            box2d_reg_per_im = box2d_reg_per_im[range(len(locations)), locations_to_gt_inds]
            target_inds_per_im = locations_to_gt_inds + num_targets
            num_targets += len(targets_per_im)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            box2d_reg.append(box2d_reg_per_im)
            target_inds.append(target_inds_per_im)

            if self.dd3d_enabled:
                # 3D box targets
                box3d_per_im = targets_per_im.gt_boxes3d[locations_to_gt_inds]
                box3d.append(box3d_per_im)

            # NuScenes targets  -- attribute / speed
            attributes_per_im = targets_per_im.gt_attributes[locations_to_gt_inds]
            speeds_per_im = targets_per_im.gt_speeds[locations_to_gt_inds]
            attributes.append(attributes_per_im)
            speeds.append(speeds_per_im)

        ret = {"labels": labels, "box2d_reg": box2d_reg, "target_inds": target_inds}
        if self.dd3d_enabled:
            ret.update({"box3d": box3d})

        # NuScenes targets  -- attribute / speed
        ret.update({"attributes": attributes, "speeds": speeds})

        return ret


class NuscenesLoss():
    def __init__(self, cfg):
        self.attr_loss_weight = cfg.DD3D.NUSC.LOSS.WEIGHT_ATTR
        self.speed_loss_weight = cfg.DD3D.NUSC.LOSS.WEIGHT_SPEED

    def __call__(self, attr_logits, speeds, fcos2d_info, targets):
        # Flatten predictions
        attr_logits = cat([x.permute(0, 2, 3, 1).reshape(-1, MAX_NUM_ATTRIBUTES) for x in attr_logits])
        speeds = cat([x.permute(0, 2, 3, 1).reshape(-1) for x in speeds])

        pos_inds = targets['pos_inds']

        if pos_inds.numel() == 0:
            losses = {"loss_attr": attr_logits.sum() * 0., "loss_speed": speeds.sum() * 0.}
            # NOTE: This is probably un-reachable, because the training filter images with empty annotations.
            # NOTE: If not, attr_weights can be unavailable in the reduce_sum below().
            return losses

        losses = {}

        # 1. Attributes
        attr_logits = attr_logits[pos_inds]
        target_attr = targets['attributes'][pos_inds]
        valid_attr_mask = target_attr != MAX_NUM_ATTRIBUTES  # No attrs associated with class, or just attr missing.

        attr_weights = fcos2d_info['centerness_targets'][valid_attr_mask]
        # Denominator for all foreground losses -- re-computed for features with valid attributes.
        # attr_loss_denom = max(reduce_sum(attr_weights.sum()).item() / d2_comm.get_world_size(), 1e-6)
        # NOTE: compute attr_weights_sum, and then feed it to reduce_sum() works, but not above.
        attr_weights_sum = attr_weights.sum()
        attr_loss_denom = max(reduce_sum(attr_weights_sum).item() / d2_comm.get_world_size(), 1e-6)

        if valid_attr_mask.sum() == 0:
            losses.update({"loss_attr": attr_logits.sum() * 0.})
        else:
            attr_logits = attr_logits[valid_attr_mask]
            target_attr = target_attr[valid_attr_mask]

            xent = F.cross_entropy(attr_logits, target_attr)
            loss_attr = (xent * attr_weights).sum() / attr_loss_denom

            losses.update({"loss_attr": self.attr_loss_weight * loss_attr})

        # 2. Speed
        speeds = speeds[pos_inds]
        target_speeds = targets['speeds'][pos_inds]
        # NOTE: some GT speeds are NaN.
        valid_gt_mask = torch.logical_not(torch.isnan(target_speeds))

        speed_weights = fcos2d_info['centerness_targets'][valid_gt_mask]
        # Denominator for all foreground losses -- re-computed for features with valid speeds.
        # speed_loss_denom = max(reduce_sum(speed_weights.sum()).item() / d2_comm.get_world_size(), 1e-6)
        speed_weights_sum = speed_weights.sum()
        speed_loss_denom = max(reduce_sum(speed_weights_sum).item() / d2_comm.get_world_size(), 1e-6)

        if valid_gt_mask.sum() == 0:
            losses.update({"loss_speed": speeds.sum() * 0.})
            # return losses
        else:
            speeds = speeds[valid_gt_mask]
            target_speeds = target_speeds[valid_gt_mask]

            l1_error = smooth_l1_loss(speeds, target_speeds, beta=0.05)
            loss_speed = (l1_error * speed_weights).sum() / speed_loss_denom
            losses.update({"loss_speed": self.speed_loss_weight * loss_speed})

        return losses


class NuscenesInference():
    def __init__(self, cfg):
        pass

    def __call__(self, attr_logits, speeds, pred_instances, fcos2d_info):
        """Add 'pred_attribute', 'pred_speed' to Instances in 'pred_instances'."""
        N = attr_logits[0].shape[0]
        for lvl, (attr_logits_lvl, speed_lvl, info_lvl, instances_lvl) in \
            enumerate(zip(attr_logits, speeds, fcos2d_info, pred_instances)):

            attr_logits_lvl = attr_logits_lvl.permute(0, 2, 3, 1).reshape(N, -1, MAX_NUM_ATTRIBUTES)
            speed_lvl = speed_lvl.permute(0, 2, 3, 1).reshape(N, -1)
            for i in range(N):
                fg_inds_per_im = info_lvl['fg_inds_per_im'][i]
                topk_indices = info_lvl['topk_indices'][i]

                attr_logits_per_im = attr_logits_lvl[i][fg_inds_per_im]
                speed_per_im = speed_lvl[i][fg_inds_per_im]

                if topk_indices is not None:
                    attr_logits_per_im = attr_logits_per_im[topk_indices]
                    speed_per_im = speed_per_im[topk_indices]

                if len(attr_logits_per_im) == 0:
                    instances_lvl[i].pred_attributes = instances_lvl[i].pred_classes.new_tensor([])
                    instances_lvl[i].pred_speeds = instances_lvl[i].scores.new_tensor([])
                else:
                    instances_lvl[i].pred_attributes = attr_logits_per_im.argmax(dim=1)
                    instances_lvl[i].pred_speeds = speed_per_im


@META_ARCH_REGISTRY.register()
class NuscenesDD3D(DD3D):
    def __init__(self, cfg):
        super().__init__(cfg)

        backbone_output_shape = self.backbone_output_shape
        in_channels = backbone_output_shape[0].channels

        # --------------------------------------------------------------------------
        # NuScenes predictions -- attribute / speed, computed from cls_tower output.
        # --------------------------------------------------------------------------
        self.attr_logits = Conv2d(in_channels, MAX_NUM_ATTRIBUTES, kernel_size=3, stride=1, padding=1, bias=True)
        self.speed = Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=True, activation=F.relu)

        # init weights
        for modules in [self.attr_logits, self.speed]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)

        # Re-define target preparer
        del self.prepare_targets
        self.prepare_targets = NuscenesDD3DTargetPreparer(cfg, backbone_output_shape)

        self.nuscenes_loss = NuscenesLoss(cfg)
        self.nuscenes_inference = NuscenesInference(cfg)

        # self.num_images_per_sample = cfg.MODEL.FCOS3D.NUSC_NUM_IMAGES_PER_SAMPLE
        self.num_images_per_sample = cfg.DD3D.NUSC.INFERENCE.NUM_IMAGES_PER_SAMPLE

        assert self.num_images_per_sample == 6
        assert cfg.DATALOADER.TEST.NUM_IMAGES_PER_GROUP == 6

        # NOTE: NuScenes evaluator allows max. 500 detections per sample.
        self.max_num_dets_per_sample = cfg.DD3D.NUSC.INFERENCE.MAX_NUM_DETS_PER_SAMPLE

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
        logits, box2d_reg, centerness, fcos2d_extra_output = self.fcos2d_head(features)
        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth = self.fcos3d_head(features)
        inv_intrinsics = images.intrinsics.inverse() if images.intrinsics is not None else None

        # --------------------------------------------------------------------------
        # NuScenes predictions -- attribute / speed, computed from cls_tower output.
        # --------------------------------------------------------------------------
        attr_logits, speeds = [], []
        for x in fcos2d_extra_output['cls_tower_out']:
            attr_logits.append(self.attr_logits(x))
            speeds.append(self.speed(x))

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

            # Nuscenes loss -- attribute / speed
            nuscenes_loss = self.nuscenes_loss(attr_logits, speeds, fcos2d_info, training_targets)
            losses.update(nuscenes_loss)
            return losses
        else:
            pred_instances, fcos2d_info = self.fcos2d_inference(
                logits, box2d_reg, centerness, locations, images.image_sizes
            )
            if not self.only_box2d:
                # This adds 'pred_boxes3d' and 'scores_3d' to Instances in 'pred_instances'.
                self.fcos3d_inference(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, inv_intrinsics, pred_instances,
                    fcos2d_info
                )
                score_key = "scores_3d"
            else:
                score_key = "scores"

            # This adds 'pred_attributes', 'pred_speed' to Instances in 'pred_instances'.
            self.nuscenes_inference(attr_logits, speeds, pred_instances, fcos2d_info)

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

                # ----------------------------------------------------------
                # NuScenes specific: cross-image (i.e. sample-level) BEV NMS.
                # ----------------------------------------------------------
                sample_tokens = [x['sample_token'] for x in batched_inputs]
                group_idxs = get_group_idxs(sample_tokens, self.num_images_per_sample)

                instances = [x['instances'] for x in processed_results]
                global_poses = [x['pose'] for x in batched_inputs]

                filtered_instances = nuscenes_sample_aggregate(
                    instances,
                    group_idxs,
                    self.num_classes,
                    global_poses,
                    self.bev_nms_iou_thresh,
                    max_num_dets_per_sample=self.max_num_dets_per_sample
                )
                processed_results = [{"instances": x} for x in filtered_instances]
            else:
                processed_results = [{"instances": x} for x in pred_instances]

            return processed_results


@META_ARCH_REGISTRY.register()
class NuscenesDD3D_PREDICTION(DD3D_VIDEO_PREDICTION4):
    def __init__(self, cfg):
        super().__init__(cfg)

        backbone_output_shape = self.backbone_output_shape
        in_channels = backbone_output_shape[0].channels

        # --------------------------------------------------------------------------
        # NuScenes predictions -- attribute / speed, computed from cls_tower output.
        # --------------------------------------------------------------------------
        self.attr_logits = Conv2d(in_channels, MAX_NUM_ATTRIBUTES, kernel_size=3, stride=1, padding=1, bias=True)
        self.speed = Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=True, activation=F.relu)

        # init weights
        for modules in [self.attr_logits, self.speed]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)

        # Re-define target preparer
        del self.prepare_targets
        self.prepare_targets = NuscenesDD3DTargetPreparer(cfg, backbone_output_shape)

        self.nuscenes_loss = NuscenesLoss(cfg)
        self.nuscenes_inference = NuscenesInference(cfg)

        # self.num_images_per_sample = cfg.MODEL.FCOS3D.NUSC_NUM_IMAGES_PER_SAMPLE
        self.num_images_per_sample = cfg.DD3D.NUSC.INFERENCE.NUM_IMAGES_PER_SAMPLE

        assert self.num_images_per_sample == 6
        assert cfg.DATALOADER.TEST.NUM_IMAGES_PER_GROUP == 6

        # NOTE: NuScenes evaluator allows max. 500 detections per sample.
        self.max_num_dets_per_sample = cfg.DD3D.NUSC.INFERENCE.MAX_NUM_DETS_PER_SAMPLE

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

        gt_dense_depth = None
        if 'depth' in batched_inputs[0]:
            gt_dense_depth = [x["depth"].to(self.device) for x in batched_inputs]
            gt_dense_depth = ImageList.from_tensors(
                gt_dense_depth, self.backbone.size_divisibility, intrinsics=intrinsics
            )

        features = self.backbone(torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))
        # features = checkpoint.checkpoint(self.backbone, torch.cat((images.tensor, images_t1.tensor, images_t2.tensor)))

        features_prev, features_cur = [], []

        for f in self.in_features:
            feat_cur = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[0]
            feat_prev = features[f].reshape(3, batch_size, features[f].size(1), features[f].size(2), -1)[1:]

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

        logits, box2d_reg, centerness, fcos2d_extra_output = self.fcos2d_head(features)

        if not self.only_box2d:
            box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, dense_depth, _ = self.fcos3d_head(features, inv_intrinsics)

        # --------------------------------------------------------------------------
        # NuScenes predictions -- attribute / speed, computed from cls_tower output.
        # --------------------------------------------------------------------------
        attr_logits, speeds = [], []
        for x in fcos2d_extra_output['cls_tower_out']:
            attr_logits.append(self.attr_logits(x))
            speeds.append(self.speed(x))

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

            # Nuscenes loss -- attribute / speed
            nuscenes_loss = self.nuscenes_loss(attr_logits, speeds, fcos2d_info, training_targets)
            losses.update(nuscenes_loss)
            return losses
        else:
            pred_instances, fcos2d_info = self.fcos2d_inference(
                logits, box2d_reg, centerness, locations, images.image_sizes
            )
            if not self.only_box2d:
                # This adds 'pred_boxes3d' and 'scores_3d' to Instances in 'pred_instances'.
                self.fcos3d_inference(
                    box3d_quat, box3d_ctr, box3d_depth, box3d_size, box3d_conf, inv_intrinsics, pred_instances,
                    fcos2d_info
                )
                score_key = "scores_3d"
            else:
                score_key = "scores"

            # This adds 'pred_attributes', 'pred_speed' to Instances in 'pred_instances'.
            self.nuscenes_inference(attr_logits, speeds, pred_instances, fcos2d_info)

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

                # ----------------------------------------------------------
                # NuScenes specific: cross-image (i.e. sample-level) BEV NMS.
                # ----------------------------------------------------------
                sample_tokens = [x['sample_token'] for x in batched_inputs]
                group_idxs = get_group_idxs(sample_tokens, self.num_images_per_sample)

                instances = [x['instances'] for x in processed_results]
                global_poses = [x['pose'] for x in batched_inputs]

                filtered_instances = nuscenes_sample_aggregate(
                    instances,
                    group_idxs,
                    self.num_classes,
                    global_poses,
                    self.bev_nms_iou_thresh,
                    max_num_dets_per_sample=self.max_num_dets_per_sample
                )
                processed_results = [{"instances": x} for x in filtered_instances]
            else:
                processed_results = [{"instances": x} for x in pred_instances]

            return processed_results