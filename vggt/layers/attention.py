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

from flash_attn import flash_attn_func
from vggt.utils.reduce import TokenReducer, PatchAttention

XFORMERS_AVAILABLE = False

def get_memory_info():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB"

def adaptive_sparsity(layer_idx, total_layers=24, min_sparsity=0.3, max_sparsity=0.7):
    import numpy as np
    x = layer_idx * np.pi / (total_layers - 1)
    weight = (1 - np.cos(x)) / 2
    sparsity_ratio = min_sparsity + (max_sparsity - min_sparsity) * weight
    return float(sparsity_ratio)

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
        # self.reducer = TokenReducer(scale=self.scale, n_hashes=3, n_buckets=32)
        self.reducer = PatchAttention(scale=self.scale)

    @torch.amp.autocast('cuda')
    def _hierarchy_dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hierarchy_data: Dict) -> torch.Tensor:
        """
        Optimized Hierarchical Attention:
        - Coarse: cluster-level mean over frames (patch preserved for fine)
        - Fine: intra-cluster + neighbor clusters sparse attention
        """
        S = hierarchy_data["num_frames"]
        clusters = hierarchy_data["clusters"]   # list of lists of frame indices, len K
        keyframes = hierarchy_data["keyframes"]
        K = len(clusters)

        B, H, N, D = q.shape
        P = N // S
        device = q.device
        dtype = torch.bfloat16

        # reshape to (B, H, S, P, D)
        q = q.view(B, H, S, P, D).type(dtype)
        k = k.view(B, H, S, P, D).type(dtype)
        v = v.view(B, H, S, P, D).type(dtype)

        # prototypes from keyframes: (B, H, K, P, D)
        q_proto = q[:, :, keyframes]
        k_proto = k[:, :, keyframes]

        # Select patch indices
        top_ratio = adaptive_sparsity(self.layer)
        top_k_patch = max(1, int(P * top_ratio))
        attn_score = torch.max(torch.sum(k_proto * q_proto, dim=-1), dim=1)[0].mean(0)
        _, patch_indices = torch.topk(attn_score, k=top_k_patch, dim=-1)  # (K, top_k_patch)
        del attn_score
        
        # calculate sparse attention by cluster
        out_frames = torch.zeros(B, H, S, P, D, device=device, dtype=dtype)
        
        for ci, frame_ids in enumerate(clusters):
            Lq = len(frame_ids) * P
            q_local = q[:, :, frame_ids, :, :].reshape(B, H, Lq, D)

            k_parts = []
            v_parts = []
            for nc in range(K):
                frames_nc = clusters[nc]  # python list of frame indices
                if len(frames_nc) == 0: 
                    continue
                
                if nc in [hierarchy_data['ref_cluster'], ci]:
                    sampled_frames_nc = frames_nc
                else:
                    frame_subsample_ratio = 0.7
                    num_frames_to_sample = max(1, int(len(frames_nc) * frame_subsample_ratio))
                    frame_indices = torch.randperm(len(frames_nc), device=device)[:num_frames_to_sample]
                    sampled_frames_nc = frames_nc[frame_indices]
                
                pidx = patch_indices[nc]  # (top_k_patch,)
                k_sel_nc = k[:, :, sampled_frames_nc, :, :][:, :, :, pidx, :].reshape(B, H, -1, D)
                v_sel_nc = v[:, :, sampled_frames_nc, :, :][:, :, :, pidx, :].reshape(B, H, -1, D)
                k_parts.append(k_sel_nc)
                v_parts.append(v_sel_nc)

            if len(k_parts) == 0:
                continue

            # concatenate along source-length dim
            k_local = torch.cat(k_parts, dim=2)   # (B, H, Lk, D)
            v_local = torch.cat(v_parts, dim=2)   # (B, H, Lk, D)

            # compute attention: queries q_local attend to keys k_local -> outputs (B,H,Lq,D)
            q_local = q_local.transpose(1, 2).contiguous()
            k_local = k_local.transpose(1, 2).contiguous()
            v_local = v_local.transpose(1, 2).contiguous()
            
            out_local = flash_attn_func(q_local, k_local, v_local)
            
            out_local = out_local.transpose(1, 2).contiguous().type(dtype)
            out_frames[:, :, frame_ids, :, :] += out_local.view(B, H, len(frame_ids), P, D)
            
        del q_local, k_local, v_local, out_local, k_parts, v_parts
        torch.cuda.empty_cache()
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
            
        if hierarchy_data is not None and 'P' in self.mode:
            # x = self.reducer.cluster_attention(q, k, v, hierarchy_data)
            print("Before execute hierarchy attention", get_memory_info())
            x = self._hierarchy_dot_product(q, k, v, hierarchy_data)
            print("After execute hierarchy attention", get_memory_info())
        elif self.fused_attn:
            if hierarchy_data is not None:
                print("Before scaled_dot_product_attention", get_memory_info())
                
            x = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.attn_drop.p if self.training else 0.0
            )     
            
            if hierarchy_data is not None:
                print("After scaled_dot_product_attention", get_memory_info())
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

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
