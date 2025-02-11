# Copyright 2021 Toyota Research Institute.  All rights reserved.
import tridet.modeling.dd3d
from tridet.modeling import feature_extractor
from tridet.modeling.dd3d import DD3DWithTTA, DD3D_VIDEOWithTTA, NuscenesDD3DWithTTA

TTA_MODELS = {
    "DD3D": DD3DWithTTA,
    "NuscenesDD3D": NuscenesDD3DWithTTA,
    "DD3D_VIDEO_PREDICTION7": DD3D_VIDEOWithTTA,
    "DD3D_DEPFORMATT_MULTIDEPTH": DD3D_VIDEOWithTTA,

}


def build_tta_model(cfg, model):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    assert meta_arch in TTA_MODELS, f"Test-time augmentation model is not available: {meta_arch}"
    return TTA_MODELS[meta_arch](cfg, model)
