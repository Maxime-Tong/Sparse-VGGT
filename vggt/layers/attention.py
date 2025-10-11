# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from typing import Dict

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        layer=0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.layer = layer

    def _hierarchy_dot_product(self, q: Tensor, k: Tensor, v: Tensor, hierarchy_data: Dict, topk_ratio: float = 0.1) -> Tensor:
        B, H, N, D = q.shape
        device = q.device
        
        S = hierarchy_data["num_frames"]
        keyframes = hierarchy_data["keyframes"]
        clusters = hierarchy_data["clusters"]
        K = len(clusters)
        tokens_per_frame = N // S
        
        # ---- Step 1. Coarse-level attention among keyframes ----
        # Select all query tokens belonging to keyframes
        key_q_indices = []
        for f_idx in keyframes:
            start = f_idx * tokens_per_frame
            end = (f_idx + 1) * tokens_per_frame
            key_q_indices.append(torch.arange(start, end, device=device))
        key_q_indices = torch.cat(key_q_indices, dim=0)  # [K * P_k]

        q_proto = q[:, :, key_q_indices, :]  # (B, H, K*P_k, D)
        k_proto = k[:, :, key_q_indices, :]

        # All keys are used at coarse level
        attn_scores = torch.matmul(q_proto * self.scale, k.transpose(-2, -1))  # (B, H, K*P_k, K*P_k)
        # attn_weights = F.softmax(attn_scores, dim=-1)

        # ---- Step 2. Select top-k keys for each keyframe ----
        topk_tokens = []
        topk_num = max(1, int(tokens_per_frame * K * topk_ratio))  # e.g. 10%
        topk_vals, topk_idx = torch.topk(attn_scores, k=topk_num, dim=-1, sorted=False)

        # ---- Step 3. Fine-level attention within each cluster ----
        x_out = torch.zeros_like(q)

        for cluster_idx, frame_indices in enumerate(clusters):
            # Collect all q indices belonging to this cluster
            q_indices = []
            for f_idx in frame_indices:
                start = f_idx * tokens_per_frame
                end = (f_idx + 1) * tokens_per_frame
                q_indices.append(torch.arange(start, end, device=device))
            q_indices = torch.cat(q_indices, dim=0)

            # The selected key tokens (topk) for the corresponding keyframe
            cluster_key_idx = topk_tokens[cluster_idx]
            q_cluster = q[:, :, q_indices, :]       # (B, H, Qc, D)
            k_cluster = k[:, :, cluster_key_idx, :] # (B, H, Kc, D)
            v_cluster = v[:, :, cluster_key_idx, :] # (B, H, Kc, D)

            attn = torch.matmul(q_cluster * self.scale, k_cluster.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            x_cluster = torch.matmul(attn, v_cluster).type(q.dtype)

            # write back the results
            x_out[:, :, q_indices, :] = x_cluster

        return x_out.view(B, H, N, D)

    def forward(self, x: Tensor, pos=None, hierarchy_data: Dict = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if hierarchy_data is not None:
            x = self._hierarchy_dot_product(q, k, v, hierarchy_data)
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, hierarchy_data=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, hierarchy_data=hierarchy_data)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
