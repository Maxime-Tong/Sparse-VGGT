import os
import pickle
import argparse
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

from vggt.utils.visual_test import visualize_multiple_patches
from utils.config_utils import load_config
from utils.dataset import load_dataset

special_tokens = 5
base_dir = "output/tmp/patches_on_images/"
data_dir = base_dir + "data/"
selected_cluster = 0
selected_layers = [4, 11, 17, 23]

def parse_args():
    parser = argparse.ArgumentParser(description="ATE TEST")
    parser.add_argument("--config", type=str, required=True, help="Directory containing the scene configs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

args = parse_args()
config = load_config(args.config)
dataset = load_dataset(args, '', config)

patches = {}
images = {}
for layer_file in os.listdir(data_dir):
    data_path = data_dir + layer_file
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    layer_name = data['layer']
    if layer_name not in selected_layers:
        continue
    patches_indices = data['patch_indices'][selected_cluster]
    patches_indices = patches_indices[patches_indices >= special_tokens]
    patches[layer_name] = patches_indices - special_tokens
    
    frame_id = data['keyframes'][selected_cluster]
    color_path = dataset.color_paths[frame_id]
    img = Image.open(color_path)
    img = img.convert("RGB")
    width, height = img.size
    target_szie = 518
    new_width = target_szie
    img = img.resize((target_szie, target_szie), Image.Resampling.BICUBIC)
    img = np.array(img)
    images[layer_name] = img
    
fig = visualize_multiple_patches(patches, images, patch_size=14)
plt.savefig("output/tmp/patches_on_images/p_k_heatmap.png", dpi=300)
plt.close()