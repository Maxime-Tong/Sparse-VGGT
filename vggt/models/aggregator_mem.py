# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import Tensor
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

import numpy as np
from sklearn.decomposition import PCA
import kmedoids
from sklearn.metrics.pairwise import euclidean_distances

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

import csv
import time

def _print_mem(stage: str, log_to_file=False, file_path="vggt_aggregator_mem_log.csv"):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    msg = f"[MEM] {stage:<30} alloc={alloc:8.1f}MB | reserved={reserved:8.1f}MB | peak={max_alloc:8.1f}MB"
    print(msg)
    if log_to_file:
        with open(file_path, "a") as f:
            csv.writer(f).writerow([stage, alloc, reserved, max_alloc])


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        mode=''
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    layer=_,
                    mode=mode
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False
        
        self.mode = mode
        self.enable_mem_log = True
        self.log_to_file = True

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)
                
    def _cluster_frames(self, tokens: torch.Tensor, pca_dim: int = 256):
        assert tokens.dim() == 3, "tokens must have shape (B*S, P, C)"
        num_frames = tokens.shape[0]
        device = tokens.device

        X = tokens.reshape(num_frames, -1).cpu().numpy()  # [N, P*C]

        # ---- Optional PCA reduction ----
        if X.shape[0] > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=42)
            X = pca.fit_transform(X)
        import time
        start_time = time.time()
        
        n_clusters = max(5, int(np.sqrt(num_frames)))  # empirical rule K≈√N
        km = kmedoids.KMedoids(n_clusters=n_clusters, method='fasterpam', random_state=42)
        diss = euclidean_distances(X)
        result = km.fit(diss)
        labels = result.labels_
        medoid_indices = result.medoid_indices_
        
        end_time = time.time()
        print("PAM took: %.2f s" % ((end_time - start_time)))
        
        clusters, keyframes = [], []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            clusters.append(torch.tensor(cluster_indices, dtype=torch.long, device=device))
            keyframes.append(int(medoid_indices[i]))

        result = {
            "num_frames": num_frames,
            "num_clusters": n_clusters,
            "clusters": clusters,
            "keyframes": torch.tensor(keyframes, dtype=torch.long, device=device)
        }
        return result

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape
        if self.enable_mem_log:
            _print_mem("Start Aggregator", self.log_to_file)

        # Normalize
        images = (images - self._resnet_mean) / self._resnet_std
        if self.enable_mem_log:
            _print_mem("After normalization", self.log_to_file)

        # Patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        if self.enable_mem_log:
            _print_mem("After patch_embed", self.log_to_file)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        # optional frame clustering
        hierarchy_data = None
        if not self.training and self.mode != '':
            t0 = time.time()
            hierarchy_data = self._cluster_frames(patch_tokens)
            if self.enable_mem_log:
                print(f"[INFO] Frame clustering took {time.time()-t0:.2f}s")
                _print_mem("After _cluster_frames", self.log_to_file)

        # token preparation
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        if self.enable_mem_log:
            _print_mem("After token concatenation", self.log_to_file)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
            
        if self.enable_mem_log:
            _print_mem("After pos embedding", self.log_to_file)

        # alternating attention blocks
        _, P, C = tokens.shape
        frame_idx = 0
        global_idx = 0
        output_list = []

        for layer_id in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    if self.enable_mem_log:
                        _print_mem(f"Before frame block {frame_idx}", self.log_to_file)
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                    if self.enable_mem_log:
                        _print_mem(f"After frame block {frame_idx}", self.log_to_file)

                elif attn_type == "global":
                    if self.enable_mem_log:
                        _print_mem(f"Before global block {global_idx}", self.log_to_file)
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos, hierarchy_data=hierarchy_data
                        )
                        _print_mem(f"After global block {global_idx}", self.log_to_file)

            intermediates_frames = [4, 11, 17, 23]
            if 'L' not in self.mode or layer_id in intermediates_frames:
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
                    
        if self.enable_mem_log:
            _print_mem("End of Aggregator", self.log_to_file)
        del concat_inter
        del frame_intermediates
        del global_intermediates

        return output_list, self.patch_start_idx


    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, hierarchy_data=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, hierarchy_data=hierarchy_data)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
