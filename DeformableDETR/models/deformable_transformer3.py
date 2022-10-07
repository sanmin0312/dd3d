# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from DeformableDETR.util.misc import inverse_sigmoid
from DeformableDETR.models.ops.modules import MSDeformAttn

import torch.utils.checkpoint as checkpoint


class DeformableTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d_model = cfg.DEFORMABLETRASFORMER.D_MODEL
        nhead = cfg.DEFORMABLETRASFORMER.N_HEAD
        num_encoder_layers = cfg.DEFORMABLETRASFORMER.NUM_ENC_LAYERS
        dim_feedforward = cfg.DEFORMABLETRASFORMER.DIM_FFW
        dropout = cfg.DEFORMABLETRASFORMER.DROPOUT
        activation = cfg.DEFORMABLETRASFORMER.ACTIVATION
        num_feature_levels = cfg.DEFORMABLETRASFORMER.NUM_FEAT_LEVELS
        enc_n_points = cfg.DEFORMABLETRASFORMER.NUM_ENC_POINTS

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, qs, kvs, pos_embeds=None):
        # assert self.two_stage or query_embed is not None
        masks = []

        for f in kvs:
            masks.append(torch.zeros_like(f[:, 0, 0], dtype=torch.bool))

        if pos_embeds is None:
            pos_embeds = []
            for f in kvs:
                pos_embeds.append(torch.zeros_like(f[:, 0]))

        # prepare input for encoder
        qs_flatten, kvt1s_flatten, kvt2s_flatten = [], [], []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (q, kv, pos_embed) in enumerate(zip(qs, kvs, pos_embeds)):
            kvt1 = kv[:, 0]
            kvt2 = kv[:, 1]

            bs, c, h, w = q.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            q = q.flatten(2).transpose(1, 2)
            kvt1 = kvt1.flatten(2).transpose(1, 2)
            kvt2 = kvt2.flatten(2).transpose(1, 2)

            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

            qs_flatten.append(q)
            kvt1s_flatten.append(kvt1)
            kvt2s_flatten.append(kvt2)

        qs_flatten = torch.cat(qs_flatten, 1)
        kvt1s_flatten = torch.cat(kvt1s_flatten, 1)
        kvt2s_flatten = torch.cat(kvt2s_flatten, 1)

        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=qs_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        prev_memory, memory = self.encoder(qs_flatten, kvt1s_flatten, kvt2s_flatten, spatial_shapes, level_start_index, valid_ratios,
                              lvl_pos_embed_flatten, padding_mask=None)

        prev_output, output = [], []
        for (start_idx, spatial_shape) in zip(level_start_index, spatial_shapes):
            lvl_output = memory[:, start_idx:start_idx + spatial_shape.prod()]
            lvl_output = lvl_output.reshape(memory.shape[0], spatial_shape[0], spatial_shape[1], -1)
            lvl_output = lvl_output.permute(0, 3, 1, 2)
            output.append(lvl_output)

            lvl_output = prev_memory[:, start_idx:start_idx + spatial_shape.prod()]
            lvl_output = lvl_output.reshape(prev_memory.shape[0], spatial_shape[0], spatial_shape[1], -1)
            lvl_output = lvl_output.permute(0, 3, 1, 2)
            prev_output.append(lvl_output)

        return prev_output, output


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = nn.LayerNorm(d_model)

        # t-1 cross attention
        self.t1_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # t-2 cross attention
        self.t2_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout4 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout5 = nn.Dropout(dropout)
        self.norm5 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout4(self.activation(self.linear1(src))))
        src = src + self.dropout5(src2)
        src = self.norm5(src)
        return src

    def forward(self, q, kvt1, kvt2, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        #query self attention
        src1 = self.self_attn(q, reference_points, q, spatial_shapes, level_start_index, padding_mask)
        q = q + self.dropout0(src1)
        q = self.norm0(q)

        # t-1 cross attention
        src2 = self.t1_cross_attn(q, reference_points, self.with_pos_embed(kvt1, pos), spatial_shapes, level_start_index, padding_mask)
        # src2 = checkpoint.checkpoint(self.t1_cross_attn, q, reference_points, kvt1, spatial_shapes, level_start_index, padding_mask)

        src2 = self.dropout1(src2)
        src2 = self.norm1(src2)

        # t-2 cross attention
        src3 = self.t2_cross_attn(q, reference_points, self.with_pos_embed(kvt2, pos), spatial_shapes, level_start_index, padding_mask)
        src3 = src2 + self.dropout2(src3)
        src3 = self.norm2(src3)

        # ffn
        q = self.forward_ffn(q+src3)

        return src3, q


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, q, kvt1, kvt2, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = q
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=q.device)
        for _, layer in enumerate(self.layers):
            prev_output, output = layer(output, kvt1, kvt2, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            # output = checkpoint.checkpoint(layer, output, kvt1, kvt2, pos, reference_points, spatial_shapes, level_start_index, padding_mask)


        return prev_output, output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)


