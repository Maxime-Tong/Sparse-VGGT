import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention(attn_dict: dict, num_images: int, figsize: tuple = (12, 10), cmap: str = "tab10", sample_ratio: float = 0.05):
    """
    可视化所有 images 的所有 query patch 对 image0 的每个 key patch 的注意力。
    不对 query 求平均，每个散点对应一个注意力值。
    
    参数:
      - attn_dict: {layer: np.ndarray[num_heads, num_images*p, num_images*p]}
      - num_images: 图像数量
      - sample_ratio: 若 query patch 过多，可随机采样显示，避免过密
    """
    sorted_layers = sorted(attn_dict.keys())
    num_layers = len(sorted_layers)

    n_rows = int(np.ceil(np.sqrt(num_layers)))
    n_cols = int(np.ceil(num_layers / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, layer in enumerate(sorted_layers):
        ax = axes[i]
        attn = attn_dict[layer]  # [num_heads, num_images*p, num_images*p]
        num_heads, n_p, _ = attn.shape
        p = n_p // num_images

        # 取所有 query 对 image0 key patch 的注意力
        attn = attn.transpose(0, 2, 1)
        attn_to_image0 = attn[:, :, :p]  # [num_heads, num_images*p, p]
        colors = plt.get_cmap(cmap, num_heads)

        for h in range(num_heads):
            attn_h = attn_to_image0[h]  # [num_images*p, p]

            # 随机采样部分 query，防止绘制过多点（否则太密）
            total_queries = attn_h.shape[0]
            sample_size = max(1, int(total_queries * sample_ratio))
            sampled_idx = np.random.choice(total_queries, sample_size, replace=False)
            attn_sampled = attn_h[sampled_idx]  # [sample_size, p]

            # 构造散点坐标
            x = np.tile(np.arange(p), sample_size)
            y = attn_sampled.flatten()

            ax.scatter(
                x, y,
                color=colors(h),
                s=4,
                alpha=0.4,
                label=f"Head {h}" if i == 0 else None  # legend只在第一个subplot显示
            )

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Key Patch Index (image0)")
        ax.set_ylabel("Attention Weight")
        # ax.set_ylim(0, 1)

    # 全局图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=8)

    # 隐藏多余子图
    for j in range(num_layers, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig

attn_dict = {}
# layers = [8, 12]
layers = [8, 9, 10, 11, 12, 13, 14, 15]
attn_path = 'output/tmp/attn_weights_0_100_10_xyz'

for file in os.listdir(attn_path):
    try:
        layer = int(file.split('_')[1].split('.')[0])
    except (IndexError, ValueError):
        continue
    if layer in layers:
        attn_weight = np.load(os.path.join(attn_path, file)).squeeze(0)
        print(f"Loaded {file}, shape: {attn_weight.shape}")
        attn_dict[layer] = attn_weight

missing_layers = [layer for layer in layers if layer not in attn_dict]
if missing_layers:
    print(f"Warning: Missing layers {missing_layers}")

# 绘图
fig = visualize_attention(attn_dict, num_images=10, sample_ratio=0.05)
plt.savefig("output/tmp/attn_scatter_full.png", dpi=300)
plt.close()
