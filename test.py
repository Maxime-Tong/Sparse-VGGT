import torch
import torch.nn.functional as F
import numpy as np

def adaptive_sparsity(layer_idx, total_layers=24, min_sparsity=0.3, max_sparsity=0.7):
    import numpy as np
    x = layer_idx * np.pi / (total_layers - 1)
    weight = (1 - np.cos(x)) / 2
    sparsity_ratio = min_sparsity + (max_sparsity - min_sparsity) * weight
    return float(sparsity_ratio)

sparsity = [adaptive_sparsity(l) for l in range(24)]
print(sparsity)
print(sum(sparsity) / len(sparsity))