import numpy as np
import pycolmap
import argparse

import os
import matplotlib.pyplot as plt

import evo
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS

from utils.config_utils import load_config
from utils.dataset import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="ATE TEST")
    parser.add_argument("--config", type=str, required=True, help="Directory containing the scene configs")
    parser.add_argument("--output", type=str, required=True, help="Directory for saving output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def read_colmap_poses(colmap_path, camera_type="SIMPLE_PINHOLE"):
    reconstruction = pycolmap.Reconstruction(colmap_path)
    num_images = len(reconstruction.images)
    timestamps = []
    poses = []
    
    # 按图像ID顺序遍历（与示例函数中for i in range(num_images)逻辑一致）
    for i in range(num_images):
        # 示例函数中使用 i+1 作为图像ID（COLMAP图像ID从1开始）
        img_id = i + 1
        pyimg = reconstruction.images[img_id]
        
        # 1. 解析时间戳（从图像文件名提取，假设文件名含时间戳，如TUM格式）
        # 例如图像名为"1305031102.175304.jpg"，提取"1305031102.175304"转为float
        ts_str = pyimg.name.split('.jpg')[0].split('.png')[0]  # 兼容.jpg和.png
        try:
            timestamp = float(ts_str)
        except ValueError:
            raise ValueError(f"无法从图像名 {pyimg.name} 解析时间戳，请检查文件名格式")
        timestamps.append(timestamp)
        
        # 2. 提取外参矩阵（与示例函数一致：pyimg.cam_from_world.matrix()）
        # 外参定义：相机坐标系到世界坐标系的变换矩阵（cam_from_world）
        # 矩阵格式为4x4: [[R, t], [0, 1]]，其中R为旋转矩阵，t为平移向量
        extrinsics_matrix = pyimg.cam_from_world.matrix()  # 直接使用示例函数中的外参提取方式
        
        w2c = np.eye(4)
        w2c[:3, :4] = extrinsics_matrix
        pose = np.linalg.inv(w2c)
        poses.append(pose)
    
    # 转换为numpy数组
    timestamps = np.array(timestamps, dtype=np.float64)
    poses = np.stack(poses, axis=0)  # 形状为(N, 4, 4)
    
    # 按时间戳排序（确保轨迹时间顺序正确）
    sorted_indices = np.argsort(timestamps)
    timestamps = timestamps[sorted_indices]
    poses = poses[sorted_indices]
    
    return timestamps, poses

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    # traj_ref = PosePath3D(poses_se3=poses_gt)
    # traj_est = PosePath3D(poses_se3=poses_est)
    # traj_est_aligned = trajectory.align_trajectory(
    #     traj_est, traj_ref, correct_scale=monocular
    # )
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est_aligned = PosePath3D(poses_se3=poses_est)
    traj_est_aligned.align(
        traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    print(f"RMSE ATE {ape_stat}[m]")

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stat

timestamps, est_poses = read_colmap_poses('/data/xthuang/code/vggt/output/tum/desk1/sparse')

args = parse_args()
config = load_config(args.config)
dataset = load_dataset(args, '', config)

N_images = len(timestamps)
gt_poses = []
for frame_idx in range(N_images):
    gt_color, gt_depth, gt_w2c = dataset[frame_idx]
    gt_pose = np.linalg.inv(gt_w2c.cpu().numpy())
    gt_poses.append(gt_pose)
gt_poses = np.stack(gt_poses)

evaluate_evo(gt_poses, est_poses, 'output/tmp', 'vggt', monocular=True)

