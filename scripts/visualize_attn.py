import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attn_dict: dict, num_images: int, figsize: tuple = (12, 10), cmap: str = "viridis"):
    # 对层号进行排序，确保从小到大显示
    sorted_layers = sorted(attn_dict.keys())  # 关键修改：对层号排序
    num_layers = len(sorted_layers)
    
    # 计算子图布局（尽量保持正方形）
    n_rows = int(np.ceil(np.sqrt(num_layers)))
    n_cols = int(np.ceil(num_layers / n_rows))
    
    # 创建画布
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # 展平轴数组，便于迭代
    
    # 预处理所有注意力图
    processed_maps = {}
    for layer in sorted_layers:  # 使用排序后的层列表
        attn = attn_dict[layer]  # [num_heads, num_images*p, num_images*p]
        n_p = attn.shape[1]
        p = n_p // num_images
        # 重塑并使用max合并注意力头和patch维度
        attn_reshaped = attn.reshape(attn.shape[0], num_images, p, num_images, p)
        print(attn_reshaped.shape)
        attn_map = attn_reshaped[-1].max(1).max(2)
        processed_maps[layer] = attn_map
    
    # 绘制所有热力图（按排序后的层顺序）
    for i, layer in enumerate(sorted_layers):  # 按排序后的顺序绘图
        ax = axes[i]
        attn_map = processed_maps[layer]
        
        # 绘制热力图（使用各层自身的颜色标尺）
        sns.heatmap(attn_map, ax=ax, cmap=cmap, cbar=True,
                   xticklabels=5, yticklabels=5)     # 每隔5个patch显示刻度
        
        # 设置标题和标签，使用层号作为标识
        title = f"Layer {layer}"
        title += f" (image size: {num_images}x{num_images})"
        ax.set_title(title)
        ax.set_xlabel("Key Patch Index")
        ax.set_ylabel("Query Patch Index")
    
    # 隐藏多余的子图
    for i in range(num_layers, n_rows * n_cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

attn_dict = {}  # 用字典暂存每个层的注意力权重8
layers = [0, 4, 8, 12, 16, 20]  # 目标层（这里即使顺序混乱，最终也会按层号排序）
layers = [8, 9, 10, 11, 12, 13, 14, 15]  # 目标层（这里即使顺序混乱，最终也会按层号排序）

import torch
import torch.nn.functional as F
patch_tokens = torch.load('/data/xthuang/code/vggt/output/tmp/tokens_0_100_10_xyz.pt')['x_norm_patchtokens']
N, P, D = patch_tokens.shape
print(N, P, D)

patch_tokens = patch_tokens.reshape(N * P, D)
patch_tokens = F.normalize(patch_tokens, dim=-1)
token_sim = (patch_tokens @ patch_tokens.transpose(-1, -2)).unsqueeze(0)
# print(patch_tokens.shape) # [100, 1369, 1024]
tokens_np = patch_tokens.detach().cpu().numpy()
token_sim = (tokens_np @ tokens_np.T)[np.newaxis, :, :]
attn_dict[-1] = token_sim

for file in os.listdir( 'output/tmp/attn_weights_0_100_10_xyz'):
    try:
        layer = int(file.split('_')[1].split('.')[0])  # 提取层号
    except (IndexError, ValueError):
        continue  # 跳过格式不符合的文件
    
    if layer in layers:
        # 加载注意力权重
        file_path = os.path.join( 'output/tmp/attn_weights_0_100_10_xyz', file)
        attn_weight = np.load(file_path).squeeze(0)  # 保留[num_heads, seq_len, seq_len]
        print(f"Loaded {file}, shape: {attn_weight.shape}")
        attn_dict[layer] = attn_weight

# 检查是否所有目标层都已加载
missing_layers = [layer for layer in layers if layer not in attn_dict]
if missing_layers:
    print(f"Warning: Missing layers {missing_layers}")

# 可视化并保存
fig = visualize_attention(attn_dict, 10)  # 第二个参数为图像数量
plt.savefig("output/tmp/attn_map.png")
plt.close()  # 释放内存