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


class TemporalConv(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.use_per_level_predictors = cfg.DD3D.FCOS3D.PER_LEVEL_PREDICTORS

        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        num_levels = self.num_levels if cfg.DD3D.FCOS3D.PER_LEVEL_PREDICTORS else 1

        self.temporal_conv = nn.ModuleList([
            Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
            for _ in range(num_levels)
        ])

        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):

        for l in self.temporal_conv.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(l.weight, a=1)
                if l.bias is not None:  # depth head may not have bias.
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, cur, prev):
        out = []

        for l, (feat_cur, feat_prev) in enumerate(zip(cur, prev)):
            _l = l if self.use_per_level_predictors else 0

            out.append(self.temporal_conv[_l](torch.cat((feat_cur,feat_prev), dim=1)))

        return out


# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model:int, max_len: int = 300, dropout = 0.0):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         pe = pe.transpose(0,1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         pe = self.pe[:, :x.size(1)]
#         x = x + pe.broadcast_to(x.shape)
#         # return x
#         #
#         return self.dropout(x)
# class CoordinateEncoding(nn.Module):
#     def __init__(self, embed_dim):
#         super().__init__()
#
#         self.ce = nn.Sequential(nn.Linear(2, embed_dim),
#                            nn.ReLU())
#
#     def forward(self, embedding: Tensor, feat_last_size) -> Tensor:
#
#         dim1 = torch.arange(1, int((embedding.size(1)) / feat_last_size) + 1, 1)
#         dim2 = torch.arange(1, feat_last_size + 1, 1)
#
#         coord = torch.meshgrid(dim1, dim2)
#         coord = torch.stack(coord, dim=2)
#         coord = coord.view(-1, 2)
#         coord = coord.expand(embedding.size(0), -1, -1).to(embedding.device)
#         coord = coord.type(torch.float32)
#         coord = self.ce(coord)
#
#         return embedding+coord
# class CrossAttEncoder(nn.Module):
#     """
#     Only CrossAtt
#     """
#     __constants__ = ['norm']
#
#     def __init__(self, cfg, num_features):
#         super().__init__()
#
#         num_crossatt = cfg.ATTENTION.NUM_CROSSATT
#         self.relu = nn.ReLU(inplace=True)
#
#         embed_dim = cfg.ATTENTION.EMBED_DIM
#         num_heads = cfg.ATTENTION.NUM_HEADS
#         norm = cfg.ATTENTION.NORM
#
#         crossatt = CrossAttentionEncoderLayer(embed_dim, num_heads, batch_first=True)
#
#         self.crossatt = _get_clones(crossatt, num_crossatt)
#         # self.PoseEnc = PositionalEncoding(embed_dim)
#         self.ce = CoordinateEncoding(embed_dim)
#
#     def forward(self, feat, feat_prev, feat_last_size):
#         """
#         Args:
#             feat: List of Tensors from each level of feature map, [batch_size, feature_size (W*H), embedding_size]
#             feat_prev: feat: List of Tensors from each level of feature map, [batch_size, feature_size (W*H), embedding_size]
#         """
#
#         feat = feat.transpose(1, 2)
#         feat = self.ce(feat, feat_last_size)
#
#         feat_prev = feat_prev.transpose(1, 2)
#         feat_prev = self.ce(feat_prev, feat_last_size)
#
#         for i, mod_cross in enumerate(self.crossatt):
#             if i == 0:
#                 output = torch.cat((feat, torch.zeros(feat.size(0), 1, feat.size(2)).to(feat.device)), dim=1)
#             # output = self.PoseEnc(output)
#             # output = mod_cross(output, feat_prev, feat_prev)
#             output = checkpoint.checkpoint(mod_cross, output, feat_prev, feat_prev)
#
#         feat_egopose = output[:, -1]
#         feat = output[:, :-1]
#
#         return feat, feat_egopose
# class AttEncoder(nn.Module):
#     """
#     SelfAtt + CrossAtt
#     """
#     __constants__ = ['norm']
#
#     def __init__(self, cfg, num_features):
#         super().__init__()
#
#         num_seflatt = cfg.ATTENTION.NUM_SELFATT
#         num_crossatt = cfg.ATTENTION.NUM_CROSSATT
#         self.relu = nn.ReLU(inplace=True)
#
#         embed_dim = cfg.ATTENTION.EMBED_DIM
#         num_heads = cfg.ATTENTION.NUM_HEADS
#         norm = cfg.ATTENTION.NORM
#
#         selfatt  = nn.ModuleList(
#             [SelfAttentionEncoderLayer(
#                 embed_dim, num_heads, batch_first=True
#             ) for _ in range(num_features)]
#         )
#
#         crossatt = nn.ModuleList(
#             [CrossAttentionEncoderLayer(
#                 embed_dim, num_heads, batch_first=True
#             ) for _ in range(num_features)]
#         )
#
#         self.selfatt = _get_clones(selfatt, num_seflatt)
#         self.crossatt = _get_clones(crossatt, num_crossatt)
#         self.PoseEnc = PositionalEncoding(embed_dim)
#
#     def forward(self, feat, feat_prev):
#         """
#         Args:
#             feat: List of Tensors from each level of feature map, [batch_size, feature_size (W*H), embedding_size]
#             feat_prev: feat: List of Tensors from each level of feature map, [batch_size, feature_size (W*H), embedding_size]
#         """
#
#         feat = [f.transpose(1, 2) for f in feat]
#         feat_prev = [f.transpose(1, 2) for f in feat_prev]
#
#         for i, (mod_self, mod_cross) in enumerate(zip(self.selfatt, self.crossatt)):
#             for j, (mod_self_level, mod_cross_level, output, output_prev) in enumerate(zip(mod_self, mod_cross, feat, feat_prev)):
#                 if i == 0:
#                     output = torch.cat((output, torch.zeros(output.size(0), 1, output.size(2)).to(output.device)), dim=1)
#
#                 output = self.PoseEnc(output)
#                 # output = mod_self_level(output)
#                 # output = mod_cross_level(output, output_prev, output_prev)
#                 output = checkpoint.checkpoint(mod_self_level,output)
#                 output = checkpoint.checkpoint(mod_cross_level,output, output_prev, output_prev)
#
#                 feat[j] = output
#
#         feat_egopose = [f[:, -1] for f in feat]
#         feat = [f[:, :-1] for f in feat]
#
#         return feat, feat_egopose
# class AttEncoder_patch(nn.Module):
#     """
#     CrossAtt
#     query: each pixel in feature maps
#     key, value: surrounding patch from feature maps
#     """
#
#     __constants__ = ['norm']
#
#     def __init__(self, cfg, num_features):
#         super().__init__()
#
#         num_crossatt = cfg.ATTENTION.NUM_CROSSATT
#         self.relu = nn.ReLU(inplace=True)
#
#         embed_dim = cfg.ATTENTION.EMBED_DIM
#         num_heads = cfg.ATTENTION.NUM_HEADS
#         norm = cfg.ATTENTION.NORM
#
#         crossatt = nn.ModuleList(
#             [CrossAttentionEncoderLayer(
#                 embed_dim, num_heads, batch_first=False
#             ) for _ in range(num_features)]
#         )
#
#         self.crossatt = _get_clones(crossatt, num_crossatt)
#
#         self.PoseEnc = CoordinateEncoding(embed_dim)
#
#     def forward(self, features):
#         feat = []
#         """
#         Args:
#             feat: List of Tensors from each level of feature map, [batch_size, feature_size (W*H), embedding_size]
#             feat_prev: feat: List of Tensors from each level of feature map, [batch_size, feature_size (W*H), embedding_size]
#         """
#         c_features = [f[:,0] for f in features]
#         p_features = [f[:,1] for f in features]
#
#         kvs = self.kv_generation(p_features)
#         for i, mod_cross in enumerate(self.crossatt):
#
#             for j, (mod_cross_level, output, kv) in enumerate(zip(mod_cross, c_features, kvs)):
#                 # if i == 0:
#                 #     output = torch.cat((output, torch.zeros(output.size(0), 1, output.size(2)).to(output.device)), dim=1)
#
#                 out_shape = output.shape
#
#                 output = output.permute(0, 2, 3, 1)
#                 output = output.reshape(1, -1, output.size(-1))
#
#                 kv = kv.permute(4, 5, 0, 2, 3, 1)
#                 kv = kv.reshape(-1, output.size(1), output.size(-1))
#                 # output = self.PoseEnc(output)
#
#                 output = checkpoint.checkpoint(mod_cross_level,output, kv, kv)
#                 output = output.reshape(out_shape[0], out_shape[2], out_shape[3], -1)
#
#                 c_features[j] = output.permute(0,3,1,2)
#
#         return c_features
#
#
#     def kv_generation (self, features):
#
#         kvs = []
#
#         for i, feature in enumerate(features):
#
#             window_size = int(16/2**(i+2)+1)
#             if i == 4:
#                 window_size = 0
#             pad = nn.ZeroPad2d(window_size)
#
#             feature = pad(feature)
#
#             kvs.append(feature.unfold(2,window_size*2+1,1).unfold(3,window_size*2+1,1))
#
#         return kvs
# class SelfAttentionEncoderLayer(nn.Module):
#     """Copied from torch.nn.modules.transformer"""
#
#     #TODO 수정사항 정리하기
#
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     This standard encoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.
#
#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of the intermediate layer, can be a string
#             ("relu" or "gelu") or a unary callable. Default: relu
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False``.
#         norm_first: if ``True``, layer norm is done prior to attention and feedforward
#             operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
#
#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = encoder_layer(src)
#
#     Alternatively, when ``batch_first`` is ``True``:
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
#         >>> src = torch.rand(32, 10, 512)
#         >>> out = encoder_layer(src)
#     """
#     __constants__ = ['batch_first', 'norm_first']
#
#     def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0.0, activation=F.relu,
#                  layer_norm_eps=1e-5, batch_first=True, norm_first=False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(SelfAttentionEncoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout, inplace=True)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
#
#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = nn.Dropout(dropout, inplace=True)
#         self.dropout2 = nn.Dropout(dropout, inplace=True)
#
#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation
#
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(SelfAttentionEncoderLayer, self).__setstate__(state)
#
#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layer.
#
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#
#         Shape:
#             see the docs in Transformer class.
#         """
#
#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
#
#         x = src
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
#             x = x + self._ff_block(self.norm2(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
#             x = self.norm2(x + self._ff_block(x))
#
#         return x
#
#
#     # self-attention block
#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)
#
#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout2(x)
# class CrossAttentionEncoderLayer(nn.Module):
#     """Copied from torch.nn.modules.transformer"""
#
#     #TODO 수정사항 정리하기
#
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     This standard encoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.
#
#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of the intermediate layer, can be a string
#             ("relu" or "gelu") or a unary callable. Default: relu
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False``.
#         norm_first: if ``True``, layer norm is done prior to attention and feedforward
#             operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
#
#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = encoder_layer(src)
#
#     Alternatively, when ``batch_first`` is ``True``:
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
#         >>> src = torch.rand(32, 10, 512)
#         >>> out = encoder_layer(src)
#     """
#     __constants__ = ['batch_first', 'norm_first']
#
#     def __init__(self, d_model, nhead, dim_feedforward=64, dropout=0.0, activation=F.relu,
#                  layer_norm_eps=1e-5, batch_first=True, norm_first=False,
#                  device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(CrossAttentionEncoderLayer, self).__init__()
#         self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
#                                             **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout, inplace=True)
#         self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
#
#         self.norm_first = norm_first
#         self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
#         self.dropout1 = nn.Dropout(dropout, inplace=True)
#         self.dropout2 = nn.Dropout(dropout, inplace=True)
#
#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation
#
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(CrossAttentionEncoderLayer, self).__setstate__(state)
#
#     def forward(self, q: Tensor, k: Tensor, v: Tensor,
#                 src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layer.
#
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#
#         Shape:
#             see the docs in Transformer class.
#         """
#
#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
#
#         if self.norm_first:
#             q = q + self._ca_block(self.norm1(q), self.norm1(k), self.norm1(v), src_mask, src_key_padding_mask)
#             q = q + self._ff_block(self.norm2(q))
#         else:
#             q = self.norm1(q + self._ca_block(q, k, v, src_mask, src_key_padding_mask))
#             q = self.norm2(q + self._ff_block(q))
#
#         return q
#
#
#     # cross-attention block
#     def _ca_block(self, q: Tensor, k:Tensor, v:Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x = self.cross_attn(q, k, v,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)
#
#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout2(x)
# class TemporalAgg(nn.Module):
#     def __init__(self, cfg, input_shape):
#         super().__init__()
#
#         self.num_levels = len(input_shape)
#
#         in_channels = [s.channels for s in input_shape]
#         assert len(set(in_channels)) == 1, "Each level must have the same channel!"
#         in_channels = in_channels[0]
#
#         num_convs = 2
#         norm = cfg.DD3D.FCOS3D.NORM
#
#         for _ in range(num_convs):
#             tower = []
#
#             if norm in ("BN", "FrozenBN"):
#                 # Each FPN level has its own batchnorm layer.
#                 # "BN" is converted to "SyncBN" in distributed training (see train.py)
#                 norm_layer = ModuleListDial([get_norm(norm, in_channels) for _ in range(self.num_levels)])
#             else:
#                 norm_layer = get_norm(norm, in_channels)
#             tower.append(
#                 Conv2d(
#                     in_channels*2,
#                     in_channels,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                     bias=norm_layer is None,
#                     norm=norm_layer,
#                     activation=F.relu
#                 )
#             )
#         self.add_module(f'tempagg_tower', nn.Sequential(*tower))
#
#         self.init_weights()
#
#     def init_weights(self):
#
#         for l in self.tempagg_tower.modules():
#             if isinstance(l, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='relu')
#                 if l.bias is not None:
#                     torch.nn.init.constant_(l.bias, 0)
#
#         pass
#
#
#     def forward(self, x):
#
#         out = []
#         for l, feature in enumerate(x):
#             feature = feature.reshape(feature.size(0), -1, feature.size(3), feature.size(4))
#             feature = self.tempagg_tower(feature)
#             out.append(feature)
#
#         return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))