import torch
import torch.nn.functional as F

class TokenReducer:
    """
    LSH-based TokenReducer
    ---------------------------
    Input:
        tokens: (S, P, C)
    Output:
        merged_indices: List[Tensor]
            Each element is (num_buckets_s, variable-length index list)
            Representing patch indices under each bucket in frame s
    """

    def __init__(self, n_hashes=4, n_buckets=4, seed=42, n_special_tokens=5, device=None):
        self.n_hashes = n_hashes
        self.n_buckets = n_buckets
        self.seed = seed
        self.n_special_tokens = n_special_tokens
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.proj = None
        self.reduction_maps = None  # Stores {bucket_id -> patch_indices} for each frame
        self.global_bucket_map = None  # Stores [(frame_idx, local_bucket_idx)]

    def _init_proj(self, C):
        """Initialize random projection matrix"""
        self.proj = torch.randn(self.n_hashes, C, self.n_buckets, device=self.device)

    @torch.no_grad()
    def compute_maps(self, tokens: torch.Tensor):
        """
        Perform LSH clustering on input tokens, return bucket→patch mappings
        Args:
            tokens: (S, P, C)
        Returns:
            reduction_maps: List[Dict[int, Tensor]]
                Mapping from hash bucket index to patch ids for each frame
        """
        assert tokens.dim() == 3, "tokens must have shape (S, P, C)"
        S, P, C = tokens.shape
        n_special = self.n_special_tokens
        p_normal = P - n_special
        if self.proj is None:
            self._init_proj(C)
        tokens = F.normalize(tokens, dim=-1)
        reduction_maps = []

        for s in range(S):
            feats = tokens[s]  # (P, C)
            
            frame_map = {}
            for i in range(n_special):
                frame_map[-i-1] = torch.tensor([i], device=self.device)
            
            # Compute LSH hash with temporary tensors
            normal_feats = feats[n_special:]
            hash_logits = torch.einsum("pc,hcb->hpb", normal_feats, self.proj)
            hash_codes = torch.argmax(hash_logits, dim=-1)  # (n_hashes, P)
            del hash_logits  # Free immediate temporary
            
            # Combine into global bucket code
            combined_code = torch.zeros(p_normal, dtype=torch.long, device=self.device)
            base = 1
            for i in range(self.n_hashes):
                combined_code += hash_codes[i] * base
                base *= self.n_buckets
            del hash_codes  # Free after use
            
            # Build bucket→patch indices mapping
            unique_codes, inverse = torch.unique(combined_code, return_inverse=True)
            del combined_code  # Free after unique computation
            
            for b_id in range(len(unique_codes)):
                mask = inverse == b_id
                patch_ids = torch.nonzero(mask, as_tuple=True)[0] + n_special
                frame_map[b_id] = patch_ids
            del inverse  # Free per-frame temporary
            reduction_maps.append(frame_map)

        self.reduction_maps = reduction_maps
        return reduction_maps

    @torch.no_grad()
    def merge(self, feats: torch.Tensor):
        """
        Merge features by averaging based on reduction_maps, concatenate all frames
        Args:
            feats: (H, S, P, C)
        Returns:
            merged_feats: (H, total_bucket_num, C)
        """
        assert self.reduction_maps is not None, "Call reduce() before merge()."
        H, S, P, C = feats.shape
        merged_feats = []
        self.global_bucket_map = []  # [(frame_idx, local_bucket_idx)]

        for s in range(S):
            frame_map = self.reduction_maps[s]
            bucket_feats = []
            for b_id, patch_ids in frame_map.items():
                selected = feats[:, s, patch_ids, :]  # (H, n_patches_in_bucket, C)
                mean_feat = selected.mean(dim=1, keepdim=True)  # (H, 1, C)
                bucket_feats.append(mean_feat)
                self.global_bucket_map.append((s, b_id))
                del selected  # Free per-selection temporary
            merged_frame = torch.cat(bucket_feats, dim=1)  # (H, num_bucket_s, C)
            merged_feats.append(merged_frame)
            del bucket_feats  # Free per-frame list

        merged_feats = torch.cat(merged_feats, dim=1)  # (H, total_bucket_num, C)
        return merged_feats

    @torch.no_grad()
    def unmerge(self, merged_feats: torch.Tensor, full_shape):
        """
        Broadcast merged features back to original patch positions
        Args:
            merged_feats: (H, total_bucket_num, C)
            full_shape: (H, S, P, C)
        Returns:
            unmerged_feats: (H, S, P, C)
        """
        H, S, P, C = full_shape
        assert self.reduction_maps is not None
        assert self.global_bucket_map is not None

        device = merged_feats.device
        unmerged = torch.zeros(full_shape, device=device, dtype=merged_feats.dtype)

        for global_b_id, (frame_idx, local_b_id) in enumerate(self.global_bucket_map):
            patch_ids = self.reduction_maps[frame_idx][local_b_id]
            unmerged[:, frame_idx, patch_ids, :] = merged_feats[:, global_b_id, :].unsqueeze(1)

        return unmerged

    def clean(self):
        """Release all GPU memory occupied by the TokenReducer instance"""
        # Clear projection matrix
        if self.proj is not None:
            del self.proj
            self.proj = None
        
        # Clear reduction maps and their contained tensors
        if self.reduction_maps is not None:
            for frame_map in self.reduction_maps:
                for key in list(frame_map.keys()):
                    del frame_map[key]
            del self.reduction_maps
            self.reduction_maps = None
        
        # Clear global bucket map
        if self.global_bucket_map is not None:
            del self.global_bucket_map
            self.global_bucket_map = None
        
        # Explicitly trigger CUDA memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()