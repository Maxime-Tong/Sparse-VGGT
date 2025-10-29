import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import math

import torch
from torch.nn.functional import cosine_similarity

from typing import Dict, List, Optional, Tuple

def plot_qk_similarity_heatmap(q: torch.Tensor, 
                               k: torch.Tensor, 
                               batch_idx: int = 0, 
                               head_idx: int = 0,
                               P : int = 1400,
                               is_query: bool = True, 
                               title: str = None, 
                               figsize: tuple = (10, 8), 
                               cmap: str = "viridis"):
    """
    绘制query或key向量两两之间的余弦相似度热力图。
    
    参数:
        q: 模型输出的query张量，形状为 (B, num_heads, N, head_dim)
        k: 模型输出的key张量，形状为 (B, num_heads, N, head_dim)
        batch_idx: 选择批次中的第几个样本（默认0）
        head_idx: 选择第几个注意力头（默认0）
        is_query: 若为True，可视化query的相似性；否则可视化key的相似性
        title: 热力图标题，若为None则自动生成
        figsize: 图像大小
        cmap: 颜色映射（如"viridis", "RdBu_r"等）
    """
    # 校验输入张量形状
    assert q.dim() == 4 and k.dim() == 4, "q和k必须是4维张量 (B, num_heads, N, head_dim)"
    B, num_heads, N, head_dim = q.shape
    assert batch_idx < B, f"batch_idx {batch_idx} 超出范围（总批次 {B}）"
    assert head_idx < num_heads, f"head_idx {head_idx} 超出范围（总头数 {num_heads}）"
    
    # 提取单批次、单头的向量（形状：(N, head_dim)）
    if is_query:
        vectors = q[batch_idx, head_idx, :P]  # 取query向量
        vec_type = "Query"
    else:
        vectors = k[batch_idx, head_idx, :P]  # 取key向量
        vec_type = "Key"
    
    # 计算两两余弦相似度矩阵（形状：(N, N)）
    # 方法：通过广播计算所有向量对 (i,j) 的相似度
    sim_matrix = cosine_similarity(
        vectors.unsqueeze(1),  # 扩展为 (N, 1, head_dim)
        vectors.unsqueeze(0),  # 扩展为 (1, N, head_dim)
        dim=-1  # 沿特征维度计算余弦相似度
    )
    
    # 绘制热力图
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        sim_matrix.cpu().detach().numpy(),  # 转为numpy数组
        cmap=cmap,
        vmin=-1.0,  # 余弦相似度范围固定为[-1, 1]
        vmax=1.0,
        annot=False,  # 若N较小（如<10）可设为True显示数值
        fmt=".2f",
        cbar=True
    )
    
    # 设置标题和标签
    if title is None:
        title = f"{vec_type} Similarity Heatmap (Batch {batch_idx}, Head {head_idx})"
    plt.title(title, fontsize=12)
    plt.xlabel(f"{vec_type} Index", fontsize=10)
    plt.ylabel(f"{vec_type} Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"output/tmp/q_k_sim/{title}.png")
    return ax  # 返回轴对象，方便后续处理

def visualize_patches_on_image(data, ax, title='Attention Patches', alpha=0.3):
    """
    在原图上以黄色渐变高亮选中的patch（按patch_indices顺序由亮到暗）
    
    Args:
        data: dict 包含
            - image: RGB图像 (H, W, 3)
            - selected_patches: 被选中patch索引list
            - patch_size: 每个patch的大小
        ax: matplotlib Axes
        title: 图标题
        alpha: 透明度（控制总体视觉强度）
    """
    image = data['image']
    patch_indices = data['selected_patches']
    patch_size = data['patch_size']

    H, W, _ = image.shape

    def _compute_patch_coordinates(patch_id):
        patches_per_row = math.ceil(W / patch_size)
        row = patch_id // patches_per_row
        col = patch_id % patches_per_row
        y_start = row * patch_size
        x_start = col * patch_size
        return x_start, y_start, patch_size, patch_size

    # 显示原图
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')

    # 创建RGBA图层
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    base_color = np.array([1.0, 1.0, 0.0])  # 黄色 (R,G,B)

    # 亮度渐变：前→亮，后→暗
    n = len(patch_indices)
    brightness_levels = np.linspace(1.0, 0.4, n)  # 可调范围 [亮, 暗]

    for i, patch_id in enumerate(patch_indices):
        x, y, w, h = _compute_patch_coordinates(patch_id)
        brightness = brightness_levels[i]
        color = base_color * brightness  # 按亮度缩放黄色
        overlay[y:y+h, x:x+w, :3] = color
        overlay[y:y+h, x:x+w, 3] = alpha  # alpha固定控制透明度

    # 叠加高亮层
    ax.imshow(overlay)

    return ax

def visualize_multiple_patches(patches_dict, images_dict, patch_size = 14, figsize: tuple = (12, 10)):
    num_layers = len(patches_dict)
    sorted_layers = sorted(patches_dict.keys())
    
    n_rows = int(np.ceil(np.sqrt(num_layers)))
    n_cols = int(np.ceil(num_layers / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, layer_name in enumerate(sorted_layers):
        title = f'Layer name: {layer_name}'
        data = {
            'image': images_dict[layer_name],
            'selected_patches': patches_dict[layer_name],
            'patch_size': patch_size
        }
        visualize_patches_on_image(data, axes[i], title, alpha=0.6)
    
    for j in range(num_layers, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig