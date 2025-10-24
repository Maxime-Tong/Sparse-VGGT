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
        self.reducer = TokenReducer(scale=self.scale, n_hashes=3, n_buckets=32)

    @torch.amp.autocast('cuda')
    def _hierarchy_dot_product(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hierarchy_data: Dict, top_ratio: float = 0.5) -> torch.Tensor:
        """
        Optimized Hierarchical Attention:
        - Coarse: cluster-level mean over frames (patch preserved for fine)
        - Fine: intra-cluster + neighbor clusters sparse attention
        """

        B, H, N, D = q.shape
        S = hierarchy_data["num_frames"]
        clusters = hierarchy_data["clusters"]   # list of lists of frame indices, len K
        K = len(clusters)
        keyframes = hierarchy_data["keyframes"]
        P = N // S
        device = q.device
        dtype = q.dtype

        # reshape to (B, H, S, P, D)
        q = q.view(B, H, S, P, D)
        k = k.view(B, H, S, P, D)
        v = v.view(B, H, S, P, D)

        # prototypes from keyframes: (B, H, K, P, D)
        q_proto = q[:, :, keyframes]   # keyframes length == K
        k_proto = k[:, :, keyframes]

        # select top-k patches per cluster using averaged coarse scores
        top_k_patch = max(1, int(P * top_ratio))
        # compute per-cluster per-patch score, average over B,H
        attn_score = torch.sum(q_proto * k_proto, dim=-1) * self.scale   # (B,H,K,P)
        attn_score = attn_score.softmax(-1).mean(dim=(0, 1))              # (K, P)
        _, patch_indices = torch.topk(attn_score, k=top_k_patch, dim=-1)  # (K, top_k_patch)
        del attn_score
        
        # gather selected patches from prototypes
        # prepare index for gather on dim=3 (patch dim)
        patch_idx_exp = patch_indices[None, None, :, :, None].expand(B, H, -1, -1, D)  # (B,H,K,top_k_patch,D)
        q_sel = torch.gather(q_proto, index=patch_idx_exp, dim=3).view(B, H, -1, D)    # (B, H, K*top_k_patch, D)
        k_sel = torch.gather(k_proto, index=patch_idx_exp, dim=3).view(B, H, -1, D)
        
        # compute inter-patch attention aggregated into cluster-to-cluster scores
        attn_mat = torch.matmul(q_sel, k_sel.transpose(-2, -1)) * self.scale   # (B,H,K*tk,K*tk)
        # reshape to (B,H,K,tk,K,tk) then max over batch/head and patch dims to get (K,K)
        tk = top_k_patch
        attn_cluster = attn_mat.softmax(-1).view(B, H, K, tk, K, tk)
        cluster_scores = torch.amax(attn_cluster, dim=(0, 1, 3, 5))  # (K, K)
        # keep as tensor on device to avoid cpu transfer
        del q_sel, k_sel, attn_mat, attn_cluster, q_proto, k_proto
        torch.cuda.empty_cache()

        # prepare output container
        out_frames = torch.zeros(B, H, S, P, D, device=device, dtype=dtype)
        # For each cluster, gather neighbor clusters (vectorized selection per neighbor)
        for ci, frame_ids in enumerate(clusters):
            # q_local: queries for frames in this cluster, shape (B, H, Lq, D) where Lq = len(frame_ids)*P
            Lq = len(frame_ids) * P
            q_local = q[:, :, frame_ids, :, :].reshape(B, H, Lq, D)

            # neighbor mask: select clusters with significant cross-score
            scores_row = cluster_scores[ci]                                  # (K,)
            thr = 0.1 * torch.mean(scores_row)
            neighbor_mask = scores_row >= thr
            # always include self if numerical issues
            neighbor_mask[ci] = True

            selected_nc = torch.nonzero(neighbor_mask, as_tuple=False).squeeze(-1)
            if selected_nc.numel() == 0:
                # no neighbors (shouldn't happen because self included), skip
                continue

            # Build k_local and v_local by concatenating selected clusters' frames and their top patches.
            # This inner loop iterates only over selected neighbor clusters (usually small).
            k_parts = []
            v_parts = []
            for nc in selected_nc.tolist():
                # frames in cluster nc
                frames_nc = clusters[nc]  # python list of frame indices
                if len(frames_nc) == 0:
                    continue
                # select patches for this cluster: index into patch dim
                pidx = patch_indices[nc]  # (top_k_patch,)
                # gather keys/vals: shape (B, H, frames_nc, top_k_patch, D)
                k_sel_nc = k[:, :, frames_nc, :, :][:, :, :, pidx, :].reshape(B, H, -1, D)
                v_sel_nc = v[:, :, frames_nc, :, :][:, :, :, pidx, :].reshape(B, H, -1, D)
                k_parts.append(k_sel_nc)
                v_parts.append(v_sel_nc)

            if len(k_parts) == 0:
                # no neighbor content, skip
                continue

            # concatenate along source-length dim
            k_local = torch.cat(k_parts, dim=2)   # (B, H, Lk, D)
            v_local = torch.cat(v_parts, dim=2)   # (B, H, Lk, D)

            # compute attention: queries q_local attend to keys k_local -> outputs (B,H,Lq,D)
            # use PyTorch scaled_dot_product_attention for best performance
            out_local = F.scaled_dot_product_attention(q_local, k_local, v_local, dropout_p=0.0, is_causal=False)
            # out_local = out_local.type(dtype)

            # write back to output frames (reshape Lq -> (len(frame_ids), P))
            out_frames[:, :, frame_ids, :, :] += out_local.view(B, H, len(frame_ids), P, D)

            # free temporaries to reduce peak memory
            del q_local, k_local, v_local, out_local, k_parts, v_parts

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
            x = self._hierarchy_dot_product(q, k, v, hierarchy_data)
        elif self.fused_attn:
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
