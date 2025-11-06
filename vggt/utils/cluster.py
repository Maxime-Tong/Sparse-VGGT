import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple, Dict, List

class ClusterAttention:
    def __init__(self, n_hashes=4, n_buckets=64, salient_ratio=0.5, device='cuda', seed=42):
        self.n_hashes = n_hashes
        self.n_buckets = n_buckets
        self.salient_ratio = salient_ratio
        
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        
        self.proj = None
        
    def _init_proj(self, dim):
        self.proj = torch.randn(self.n_hashes, dim, self.n_buckets, device=self.device)
        self.proj = F.normalize(self.proj, dim=1)
        
    def _hash_tokens(self, tokens: Tensor):
        B, H, N, D = tokens.shape
        assert B == 1
        tokens = tokens.permute(0, 2, 1, 3).reshape(N, H * D)
        
        if self.proj is None:
            self._init_proj(H * D)
        
        tokens = F.normalize(tokens, dim=1)
        hash_logits = torch.einsum("nd,hdb->hnb", tokens, self.proj).abs()
        hash_codes = torch.argmax(hash_logits, dim=-1)
        del hash_logits
        
        combined = torch.zeros(N, dtype=torch.long, device=self.device)
        base = 1
        for t in range(self.n_hashes):
            combined += hash_codes[t] * base
            base *= self.n_buckets
            
        unique_codes, inverse = torch.unique(combined, return_inverse=True)
        del combined
        
        return unique_codes, inverse
    
    def _agg_trivial(self, tokens: Tensor, buckets: Tensor, labels: Tensor):
        B, H, N, D = tokens.shape
        dtype = tokens.dtype
        
        n_labels = len(buckets)
        tokens_agg = torch.zeros((B, H, n_labels, D), dtype=dtype, device=self.device)
        index = labels.view(1, 1, N, 1).expand(B, H, N, D)
        tokens_agg.scatter_reduce_(dim=2, index=index, src=tokens, reduce='mean', include_self=False)
        
        return tokens_agg
    
    def _select_salient(self, qs: Tensor, ks: Tensor, hierarchy_data: Dict, n_special_tokens=5):
        B, H, S, P, D = qs.shape
        keyframes = hierarchy_data["keyframes"]
        K = len(keyframes)
        
        q_proto = qs[:, :, keyframes]
        k_proto = ks[:, :, keyframes]
        
        top_ratio = self.salient_ratio
        top_k_patch = int(P * top_ratio)
        attn_score = torch.amax(torch.sum(k_proto * q_proto, dim=-1), dim=(0, 1))
        _, topk_indices = torch.topk(attn_score, k=top_k_patch, dim=-1)  # (K, top_k_patch)
        del attn_score
        
        special_indices = torch.arange(n_special_tokens, device=self.device, dtype=torch.long).unsqueeze(0).expand(K, n_special_tokens)
        combined = torch.cat([special_indices, topk_indices], dim=-1)
        
        salient_indices = torch.full((K, top_k_patch), -1, device=self.device, dtype=torch.long)
        for i in range(K):
            unique_vals = torch.unique(combined[i])
            salient_indices[i] = unique_vals[:top_k_patch]
        return salient_indices
    
    def attention(self, q: Tensor, k: Tensor, v: Tensor, hierarchy_data: Dict):
        S = hierarchy_data["num_frames"]
        clusters = hierarchy_data["clusters"]
        
        B, H, N, D = q.shape
        P = N // S
        
        qs = q.view(B, H, S, P, D)
        ks = k.view(B, H, S, P, D)
        vs = v.view(B, H, S, P, D)
        
        salient_pacthes = self._select_salient(qs, ks, hierarchy_data) # (K, topk_patch)
        n_salient = salient_pacthes.shape[1]
        
        meta = []
        q_combined, k_combined, v_combined = [], [], []
        for ci, frame_ids in enumerate(clusters):
            n_frames = len(frame_ids)
            sal_pid = salient_pacthes[ci]
            
            sal_mask = torch.zeros(P, dtype=torch.bool, device=self.device)
            sal_mask[sal_pid] = True
            non_sal_mask = ~sal_mask
            
            qc = qs[:, :, frame_ids]
            
            frame_subsample_ratio = 0.5
            num_frames_to_sample = max(1, int(len(frame_ids) * frame_subsample_ratio))
            frame_indices = torch.randperm(len(frame_ids), device=self.device)[:num_frames_to_sample]
            sampled_frames_nc = frame_ids[frame_indices]
            
            kc = ks[:, :, sampled_frames_nc]
            vc = vs[:, :, sampled_frames_nc]
            
            qc_rem = qc[:, :, :, non_sal_mask].reshape(B, H, -1, D)
            kc_rem = kc[:, :, :, non_sal_mask].reshape(B, H, -1, D)
            vc_rem = vc[:, :, :, non_sal_mask].reshape(B, H, -1, D)
            
            qc_buckets, qc_labels = self._hash_tokens(qc_rem)
            kc_buckets, kc_labels = self._hash_tokens(kc_rem)
            
            qc_trival = self._agg_trivial(qc_rem, qc_buckets, qc_labels)
            kc_trival = self._agg_trivial(kc_rem, kc_buckets, kc_labels)
            vc_trival = self._agg_trivial(vc_rem, kc_buckets, kc_labels)
            
            qc_sal = qc[:, :, :, sal_mask].reshape(B, H, -1, D)
            kc_sal = kc[:, :, :, sal_mask].reshape(B, H, -1, D)
            vc_sal = vc[:, :, :, sal_mask].reshape(B, H, -1, D)
            
            q_combined.append(torch.cat([qc_sal, qc_trival], dim=2))
            k_combined.append(torch.cat([kc_sal, kc_trival], dim=2))
            v_combined.append(torch.cat([vc_sal, vc_trival], dim=2))
            
            n_sal = n_frames * n_salient
            n_trival = len(qc_buckets)
            _, trival_inv = torch.unique(qc_labels, return_inverse=True)
            
            meta.append((ci, (n_sal, n_trival), sal_mask, trival_inv, n_frames))
            
        q_combined = torch.cat(q_combined, dim=2)
        k_combined = torch.cat(k_combined, dim=2)
        v_combined = torch.cat(v_combined, dim=2)
        # print(q_combined.shape, q.shape)
        
        del q, k, v
        
        out = F.scaled_dot_product_attention(q_combined, k_combined, v_combined)
        
        out_full = torch.zeros((B, H, S, P, D), dtype=out.dtype, device=self.device)
        
        start_idx = 0
        for ci, (n_sal, n_trival), sal_mask, trival_inv, n_frames in meta:
            out_local = torch.zeros((B, H, n_frames, P, D), dtype=out.dtype, device=self.device)
            
            sal_start = start_idx
            sal_end = start_idx + n_sal
            out_local[:, :, :, sal_mask] = out[:, :, sal_start:sal_end, :].reshape(B, H, n_frames, n_salient, D)
            
            trival_start = sal_end
            trival_end = trival_start + n_trival
            out_trival = out[:, :, trival_start:trival_end]
            
            non_sal_mask = ~sal_mask
            out_local[:, :, :, non_sal_mask] = out_trival[:, :, trival_inv, :].reshape(B, H, n_frames, P-n_salient, D)            
            out_full[:, :, clusters[ci]] = out_local
            
            start_idx = trival_end
            
        return out_full.reshape(B, H, N, D)
    
    def cleanup(self):
        if self.proj is not None:
            del self.proj
            self.proj = None
        torch.cuda.empty_cache()
            
            
            
        
        