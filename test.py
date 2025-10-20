import torch
import torch.nn.functional as F

class TokenReducer:
    """
    LSH-based TokenReducer
    ---------------------------
    输入:
        tokens: (S, P, C)
    输出:
        merged_indices: List[Tensor]
            每个元素为 (num_buckets_s, variable-length index list)
            表示第 s 帧中每个 bucket 下对应的 patch indices
    """

    def __init__(self, n_hashes=4, n_buckets=4, seed=42, device=None):
        self.n_hashes = n_hashes
        self.n_buckets = n_buckets
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.proj = None
        self.reduction_maps = None  # 存储每帧的 {bucket_id -> patch_indices}

    def _init_proj(self, C):
        """初始化随机投影矩阵"""
        self.proj = torch.randn(self.n_hashes, C, self.n_buckets, device=self.device)

    @torch.no_grad()
    def reduce(self, tokens: torch.Tensor):
        """
        对输入tokens执行LSH聚类，只返回bucket→patch映射
        参数:
            tokens: (S, P, C)
        返回:
            reduction_maps: List[Dict[int, Tensor]]
                每帧的hash bucket索引到patch id的映射
        """
        assert tokens.dim() == 3, "tokens must have shape (S, P, C)"
        S, P, C = tokens.shape
        if self.proj is None:
            self._init_proj(C)

        tokens = F.normalize(tokens, dim=-1)
        reduction_maps = []

        for s in range(S):
            feats = tokens[s]  # (P, C)
            # 计算 LSH hash
            hash_logits = torch.einsum("pc,hcb->hpb", feats, self.proj)
            hash_codes = torch.argmax(hash_logits, dim=-1)  # (n_hashes, P)

            # 组合为全局bucket编码
            combined_code = torch.zeros(P, dtype=torch.long, device=self.device)
            base = 1
            for i in range(self.n_hashes):
                combined_code += hash_codes[i] * base
                base *= self.n_buckets

            # 构建 bucket → patch indices 的映射
            unique_codes, inverse = torch.unique(combined_code, return_inverse=True)
            frame_map = {}
            for b_id in range(len(unique_codes)):
                mask = inverse == b_id
                patch_ids = torch.nonzero(mask, as_tuple=True)[0]
                frame_map[b_id] = patch_ids
            reduction_maps.append(frame_map)

        self.reduction_maps = reduction_maps
        return reduction_maps

    @torch.no_grad()
    def merge(self, feats: torch.Tensor):
        """
        根据 reduction_maps 对特征进行均值融合，并拼接所有帧
        参数:
            feats: (H, S, P, C)
        返回:
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
            merged_feats.append(torch.cat(bucket_feats, dim=1))  # (H, num_bucket_s, C)

        merged_feats = torch.cat(merged_feats, dim=1)  # (H, total_bucket_num, C)
        return merged_feats


    @torch.no_grad()
    def unmerge(self, merged_feats: torch.Tensor, full_shape):
        """
        将merge后的特征广播回原patch位置
        参数:
            merged_feats: (H, total_bucket_num, C)
            full_shape: (H, S, P, C)
        返回:
            unmerged_feats: (H, S, P, C)
        """
        H, S, P, C = full_shape
        assert self.reduction_maps is not None
        assert self.global_bucket_map is not None

        device = merged_feats.device
        unmerged = torch.zeros(full_shape, device=device)

        for global_b_id, (frame_idx, local_b_id) in enumerate(self.global_bucket_map):
            frame_map = self.reduction_maps[frame_idx]
            patch_ids = frame_map[local_b_id]
            unmerged[:, frame_idx, patch_ids, :] = merged_feats[:, global_b_id, :].unsqueeze(1)

        return unmerged



# Example usage
if __name__ == "__main__":
    S, P, C, H = 2, 1024, 64, 4
    tokens = torch.randn(S, P, C, device="cuda")
    qkv = torch.randn(H, S, P, C, device="cuda")

    reducer = TokenReducer(n_hashes=4, n_buckets=16)
    reducer.reduce(tokens)

    merged_feats = reducer.merge(qkv)
    restored_feats = reducer.unmerge(merged_feats, qkv.shape)

    print(qkv.shape, merged_feats.shape, restored_feats.shape)
    print((restored_feats - qkv).abs().mean())