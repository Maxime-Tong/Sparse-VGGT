import torch
import torch.nn.functional as F
from typing import Dict

class TokenReducer:
    def __init__(self, scale, n_hashes=4, n_buckets=4, seed=42, device=None):
        self.scale = scale
        self.n_hashes = n_hashes
        self.n_buckets = n_buckets
        self.topk_ratio = 0.5
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.proj = None

    def _init_proj(self, C):
        self.proj = torch.randn(self.n_hashes, C, self.n_buckets, device=self.device)
        self.proj = F.normalize(self.proj, dim=1)

    def compute_maps(self, tokens: torch.Tensor, hierarchy_data: Dict):
        B, H, N, C = tokens.shape
        assert B == 1
        
        S = hierarchy_data['num_frames']
        P = N // S
        D = H * C
        tokens = tokens.transpose(1, 2).reshape(S, P, D)
        
        if self.proj is None:
            self._init_proj(D)
        tokens = F.normalize(tokens, dim=-1)
        
        clusters = hierarchy_data['clusters']
        
        n_global_cluster = 0
        global_cluster_map = []
        for cluster in clusters:
            cluster_feats = tokens[cluster]
            Lp = len(cluster) * P
            feats = cluster_feats.view(Lp, D)
            
            hash_logits = torch.einsum("pc,hcb->hpb", feats, self.proj)
            hash_codes = torch.argmax(hash_logits, dim=-1)
            del hash_logits
            
            combined_code = torch.zeros(Lp, dtype=torch.long, device=self.device)
            base = 1
            for i in range(self.n_hashes):
                combined_code += hash_codes[i] * base
                base *= self.n_buckets
            del hash_codes
            
            unique_codes, inverse = torch.unique(combined_code, return_inverse=True)
            del combined_code
            
            global_cluster_map += [inverse + n_global_cluster]
            n_global_cluster += len(unique_codes)
            
        global_cluster_map = torch.cat(global_cluster_map, dim=-1)
        return global_cluster_map, n_global_cluster
    
    def reorganize_tokens(self, feats, cluster_map):
        B, H, N, C = feats.shape
        device = feats.device

        sorted_indices = torch.argsort(cluster_map)
        feats_agg = feats[:, :, sorted_indices]
        sorted_cluster_ids = cluster_map[sorted_indices]
        
        cluster_boundaries = torch.cat([
            torch.tensor([0], device=device),
            torch.where(sorted_cluster_ids[1:] != sorted_cluster_ids[:-1])[0] + 1,
            torch.tensor([N], device=device)
        ])
        
        return feats_agg, sorted_indices, cluster_boundaries
    

    def compute_centroids(self, feats, ranges, n_cluster):
        centroids = []
        for i in range(n_cluster):
            feats_local = feats[:, :, ranges[i]:ranges[i+1]]
            centroid = feats_local.mean(dim=2)
            centroids.append(centroid)
        centroids = torch.stack(centroids, dim=2)
        return centroids            
    
    def cluster_attention(self, q, k, v, hierarchy_data):
        B, H, N, C = q.shape
        device = q.device
        
        cluster_map_q, Kq = self.compute_maps(q, hierarchy_data)
        cluster_map_k, Kk = self.compute_maps(k, hierarchy_data)
        
        q_agg, inverse_q, cluster_q_ranges = self.reorganize_tokens(q, cluster_map_q)
        k_agg, inverse_k, cluster_k_ranges = self.reorganize_tokens(k, cluster_map_k)
        v_agg = v[:, :, inverse_k]
        
        proto_q = self.compute_centroids(q_agg, cluster_q_ranges, Kq)
        proto_k = self.compute_centroids(k_agg, cluster_k_ranges, Kk)
        
        attn = torch.matmul(proto_q, proto_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1).view(B, H, Kq, Kk).mean(dim=(0, 1))
        
        topk = max(1, int(Kk * self.topk_ratio))
        _, topk_clusters = torch.topk(attn, k=topk, dim=-1)
        del attn
        
        chunk_size = 64
        out = torch.zeros_like(q)
        
        cluster_q_ranges_list = cluster_q_ranges.tolist()
        
        start_cluster = 0
        while start_cluster < Kq:
            end_cluster = min(start_cluster + chunk_size, Kq)
            
            # Create attention mask for current chunk
            chunk_size_total = cluster_q_ranges_list[end_cluster] - cluster_q_ranges_list[start_cluster]
            
            # Vectorized attention mask creation
            attn_mask = torch.zeros(chunk_size_total, N, device=device, dtype=torch.bool)
            
            # Set attention for top-k clusters
            for i, cluster_idx in enumerate(range(start_cluster, end_cluster)):
                cluster_start = cluster_q_ranges_list[cluster_idx] - cluster_q_ranges_list[start_cluster]
                cluster_end = cluster_q_ranges_list[cluster_idx + 1] - cluster_q_ranges_list[start_cluster]
                attn_mask[cluster_start:cluster_end, topk_clusters[cluster_idx]] = True
            
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, chunk_size_total, N)
            
            # Extract current chunk
            start_idx = cluster_q_ranges_list[start_cluster]
            end_idx = cluster_q_ranges_list[end_cluster]
            
            # Use sparse attention for efficiency
            q_chunk = q_agg[:, :, start_idx:end_idx]
            
            # Apply masked attention
            out_chunk = F.scaled_dot_product_attention(
                q_chunk, k_agg, v_agg,
                attn_mask=attn_mask,
                dropout_p=0.0
            )
            
            out[:, :, start_idx:end_idx] = out_chunk
            start_cluster = end_cluster
        
        # Restore original token order
        out = out[:, :, torch.argsort(inverse_q)]
        return out

    def clean(self):
        if self.proj is not None:
            del self.proj
            self.proj = None
        torch.cuda.empty_cache()