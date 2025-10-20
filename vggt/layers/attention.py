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

from vggt.utils.reduce import TokenReducer

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
        layer=0,
        mode=''
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
        
        self.mode = mode
        self.reducer = TokenReducer(n_hashes=3, n_buckets=64)

    @torch.amp.autocast('cuda')
    def _hierarchy_dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hierarchy_data: Dict, top_ratio: float = 0.2) -> torch.Tensor:
        """
        Optimized Hierarchical Attention:
        - Coarse: cluster-level mean over frames (patch preserved for fine)
        - Fine: intra-cluster + neighbor clusters sparse attention
        """

        B, H, N, D = q.shape
        S = hierarchy_data["num_frames"]
        clusters = hierarchy_data["clusters"]
        ref_cluster = hierarchy_data["ref_cluster"]
        keyframes = hierarchy_data["keyframes"]
        K = len(clusters)
        P = N // S
        device = q.device

        # ---- Step 1. reshape ----
        q = q.view(B, H, S, -1, D)
        k = k.view(B, H, S, -1, D)
        v = v.view(B, H, S, -1, D)

        # ---- Step 2. cluster-level prototypes (mean over frames, preserve patch dimension) ----
        # (B,H,K,P,D)
        q_proto = q[:, :, keyframes]
        k_proto = k[:, :, keyframes]
        
        top_patch_ratio = 0.2
        top_k_patch = max(1, int(P * top_patch_ratio))
        attn_score = torch.matmul(q_proto, k_proto.transpose(-2, -1)) * self.scale # [B, H, K, P, P]
        attn_score = attn_score.softmax(-1).mean(3).mean(1)[0] # [K, P]
        patch_values, patch_indices = torch.topk(attn_score, k=top_k_patch)
        del attn_score, patch_values

        # ---- Step 3. Coarse attention (frame-level mean only, no full patch flatten)
        # reduce patch dimension before computing cluster-to-cluster similarity
        patch_gather_indices = patch_indices[None, None, :, :, None].expand(B, H, -1, -1, D)
        q_selected = torch.gather(
            q_proto, # [B, H, K, P, D]
            index=patch_gather_indices,
            dim=3
        ).view(B, H, -1, D)
        k_selected = torch.gather(
            k_proto,
            index=patch_gather_indices,
            dim=3
        ).view(B, H, -1, D)
        
        attn_scores = torch.matmul(q_selected, k_selected.transpose(-2, -1)) * self.scale  # (B, H, K*top_k, K*top_k)
        attn_scores = attn_scores.softmax(-1)
        
        # 平均head，找每个cluster最相关的topK
        top_k = max(1, int(K * top_ratio))
        _, cluster_topk = torch.topk(attn_scores.view(B, H, K, top_k_patch, K, top_k_patch).mean(dim=(1, 3, 5)), k=top_k, dim=-1)  # (B, K, K)
        del q_selected, k_selected, attn_scores, q_proto, k_proto

        # ---- Step 4. Fine attention ----
        out_frames = torch.zeros(B, H, S, P, D, device=device, dtype=q.dtype)

        for ci, frame_ids in enumerate(clusters):
            q_local = q[:, :, frame_ids, :, :].reshape(B, H, -1, D)
            k_local = k[:, :, clusters[ref_cluster], :, :].reshape(B, H, -1, D)
            v_local = v[:, :, clusters[ref_cluster], :, :].reshape(B, H, -1, D)

            # 添加 coarse 阶段最相关 cluster（只添加一次，使用缓存）
            neighbor_clusters = cluster_topk[0, ci].tolist() # [top_k]
            for nc in neighbor_clusters:
                if nc != ref_cluster: 
                    k_selected = k[:, :, clusters[nc], :, :]
                    v_selected = v[:, :, clusters[nc], :, :]
                    
                    if 'P' in self.mode:
                        k_selected = k_selected[:, :, :, patch_indices[nc]]
                        v_selected = v_selected[:, :, :, patch_indices[nc]]
                        
                    k_local = torch.cat([k_local, k_selected.reshape(B, H, -1, D)], dim=2)
                    v_local = torch.cat([v_local, v_selected.reshape(B, H, -1, D)], dim=2)

            # 稀疏 attention (coarse 约束 + chunked)
            out_local = F.scaled_dot_product_attention(q_local, k_local, v_local, dropout_p=0.0, is_causal=False).type(q.dtype)

            out_frames[:, :, frame_ids, :, :] = out_local.view(B, H, len(frame_ids), P, D)

            # 显式清理，防止 block 级内存占用累积
            del q_local, k_local, v_local, out_local
        return out_frames.view(B, H, N, D)


    def forward(self, x: Tensor, pos=None, hierarchy_data: Dict = None) -> Tensor:
        B, N, C = x.shape
        # Compute QKV and split (keep as minimal temp tensors)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        del qkv  # Free temp QKV tensor immediately
        
        q, k = self.q_norm(q), self.k_norm(k)
        
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if hierarchy_data is not None:
            H, D = self.num_heads, self.head_dim
            S = hierarchy_data['num_frames']
            P = N // S

            # Reduce tokens and merge Q/K/V with memory-efficient steps
            self.reducer.compute_maps(x.view(S, P, C))  # (S, P, C) -> reduction maps
            
            # Merge Q with immediate view conversion and temp cleanup
            # q_merged = self.reducer.merge(q.view(H, S, P, D)).view(B, H, -1, D)
            # del q  # Free original Q after merging
            
            # Merge K with immediate view conversion and temp cleanup
            k_merged = self.reducer.merge(k.view(H, S, P, D)).view(B, H, -1, D)
            del k  # Free original K after merging
            
            # Merge V with immediate view conversion and temp cleanup
            v_merged = self.reducer.merge(v.view(H, S, P, D)).view(B, H, -1, D)
            del v  # Free original V after merging
            
            # Update to merged versions
            k, v = k_merged, v_merged
            # print("q:", q.shape, k.shape, v.shape)

        # Attention computation (uses merged Q/K/V if hierarchy is active)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.attn_drop.p if self.training else 0.0
            )     
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        if hierarchy_data is not None:
            # Unmerge and restore original shape
            # x = self.reducer.unmerge(x.view(H, -1, D), (H, S, P, D)).view(B, H, N, D)
            self.reducer.clean()  # Release reducer GPU memory immediately
            del q, k, v  # Free merged Q/K/V after unmerge

        # Final projection
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
