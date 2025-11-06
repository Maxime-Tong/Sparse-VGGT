# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import os
import os.path as osp

import open3d as o3d

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from collections import defaultdict
from tqdm import tqdm


from eval_utils.utils import accuracy, completion
from eval_utils.criterion import Regr3D_t_ScaleShiftInv, L21
from eval_utils.timer import CudaTimer

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from datasets.seven_scenes.data import SevenScenes

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
# from vggt.utils.load_fn import load_and_preprocess_images_square
# from vggt.utils.geometry import unproject_depth_map_to_point_map
# from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
# from vggt.dependency.track_predict import predict_tracks
# from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Evaluation")
    parser.add_argument("--kf", type=int, default=2, help="Key frame")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output")
    parser.add_argument("--use_proj", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--mode", type=str, default='', help="Run VGGT mode")
    return parser.parse_args()

def run_VGGT(model, images, dtype, resolution=None):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    if resolution is not None:
        if isinstance(resolution, int):
            images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
        else:  
            images = F.interpolate(images, size=resolution, mode="bilinear", align_corners=False)

    timer = CudaTimer("Inference time")
    timer.start()
    
    predictions = {}
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images)

            # Predict Cameras
            pose_enc_list = model.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]
            predictions["pose_enc_list"] = pose_enc_list
            
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc_list[-1], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            
            # Predict Depth Maps
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            predictions["depth"] = depth_map
            predictions["depth_conf"] = depth_conf
            
            pts3d, pts3d_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf
    
    elapsed_time_ms = timer.stop()
    print(f"Inference time: {elapsed_time_ms:.2f}ms")
    return predictions, elapsed_time_ms
    

def move_data_to_device(batch, device, ignore_keys=None):
    """Move batch data to device"""
    if ignore_keys is None:
        ignore_keys = set(["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"])
    
    for view in batch:
        for name in view:
            if name in ignore_keys:
                continue
            if isinstance(view[name], (tuple, list)):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)

def preprocess_images(batch):
    """Normalize and prepare images for inference"""
    if isinstance(batch, dict) and "img" in batch:
        batch["img"] = (batch["img"] + 1.0) / 2.0
        imgs_tensor = batch["img"]
    elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
        for view in batch:
            view["img"] = (view["img"] + 1.0) / 2.0
        imgs_tensor = torch.cat([v["img"] for v in batch], dim=0)
    return imgs_tensor

def organize_predictions(predictions, views):
    """Organize model predictions into per-view format"""
    if "pose_enc" in predictions:
        B, S = predictions["pose_enc"].shape[:2]
    elif "world_points" in predictions:
        B, S = predictions["world_points"].shape[:2]
    else:
        raise KeyError("predictions is missing a key to infer sequence length")

    ress = []
    for s in range(S):
        res = {
            "pts3d_in_other_view": predictions["world_points"][:, s],
            "conf": predictions["world_points_conf"][:, s],
            "depth": predictions["depth"][:, s],
            "depth_conf": predictions["depth_conf"][:, s],
            "camera_pose": predictions["pose_enc"][:, s, :],
        }
        if isinstance(views, list) and s < len(views) and "valid_mask" in views[s]:
            res["valid_mask"] = views[s]["valid_mask"]
        if "track" in predictions:
            res.update({
                "track": predictions["track"][:, s],
                "vis": predictions.get("vis", None)[:, s] if "vis" in predictions else None,
                "track_conf": predictions.get("conf", None)[:, s] if "conf" in predictions else None,
            })
        ress.append(res)
        
    return ress

def extract_and_crop_data(batch, preds, gt_pts, pred_pts):
    """Extract and crop point cloud data and images"""
    in_camera1 = None
    pts_all, pts_gt_all, images_all, masks_all, conf_all = [], [], [], [], []

    for j, view in enumerate(batch):
        if in_camera1 is None:
            in_camera1 = view["camera_pose"][0].cpu()

        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
        mask = view["valid_mask"].cpu().numpy()[0]
        pts = pred_pts[j].cpu().numpy()[0]
        conf = preds[j]["conf"].cpu().data.numpy()[0]
        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

        # Center crop
        H, W = image.shape[:2]
        cx, cy = W // 2, H // 2
        l, t, r, b = cx - 112, cy - 112, cx + 112, cy + 112
        image = image[t:b, l:r]
        mask = mask[t:b, l:r]
        pts = pts[t:b, l:r]
        pts_gt = pts_gt[t:b, l:r]

        images_all.append(image[None, ...])
        pts_all.append(pts[None, ...])
        pts_gt_all.append(pts_gt[None, ...])
        masks_all.append(mask[None, ...])
        conf_all.append(conf[None, ...])

    return (np.concatenate(images_all, axis=0),
            np.concatenate(pts_all, axis=0),
            np.concatenate(pts_gt_all, axis=0),
            np.concatenate(masks_all, axis=0))

def prepare_point_clouds(pts_all, pts_gt_all, images_all, masks_all, use_proj=False):
    """Prepare point clouds for evaluation with optional alignment"""
    pts_all_masked = pts_all[masks_all > 0]
    pts_gt_all_masked = pts_gt_all[masks_all > 0]
    images_all_masked = images_all[masks_all > 0]

    # Remove invalid points
    valid_mask = np.isfinite(pts_all_masked)
    pts_all_masked = pts_all_masked[valid_mask]
    pts_gt_all_masked = pts_gt_all_masked[np.isfinite(pts_gt_all_masked)]
    
    # Reshape to point clouds
    pts_all_masked = pts_all_masked.reshape(-1, 3)
    pts_gt_all_masked = pts_gt_all_masked.reshape(-1, 3)
    images_all_masked = images_all_masked.reshape(-1, 3)

    # Sample if too many points
    max_points = 999999
    if pts_all_masked.shape[0] > max_points:
        sample_indices = np.random.choice(pts_all_masked.shape[0], max_points, replace=False)
        pts_all_masked = pts_all_masked[sample_indices]
        images_all_masked = images_all_masked[sample_indices]
    
    if pts_gt_all_masked.shape[0] > max_points:
        sample_indices_gt = np.random.choice(pts_gt_all_masked.shape[0], max_points, replace=False)
        pts_gt_all_masked = pts_gt_all_masked[sample_indices_gt]

    # Apply alignment if needed
    if use_proj:
        pts_all_masked = apply_umeyama_alignment(pts_all_masked, pts_gt_all_masked)

    # Create Open3D point clouds
    pcd = create_o3d_pointcloud(pts_all_masked, images_all_masked)
    pcd_gt = create_o3d_pointcloud(pts_gt_all_masked, images_all_masked)

    return pcd, pcd_gt

def apply_umeyama_alignment(src, dst, with_scale=True):
    """Apply Umeyama alignment between source and target point clouds"""
    assert src.shape == dst.shape
    N, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    Sigma = dst_c.T @ src_c / N
    U, D, Vt = np.linalg.svd(Sigma)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    s = (D * S.diagonal()).sum() / ((src_c**2).sum() / N) if with_scale else 1.0
    t = mu_dst - s * R @ mu_src

    return (s * (R @ src.T)).T + t

def create_o3d_pointcloud(points, colors=None):
    """Create Open3D point cloud from numpy arrays"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def register_point_clouds(pcd, pcd_gt, threshold=0.1):
    """Register point clouds using ICP"""
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, pcd_gt, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd = pcd.transform(reg_p2p.transformation)
    return pcd

def compute_metrics(pcd_gt, pcd):
    """Compute accuracy, completion and normal consistency metrics"""
    pcd.estimate_normals()
    pcd_gt.estimate_normals()

    gt_normal = np.asarray(pcd_gt.normals)
    pred_normal = np.asarray(pcd.normals)

    acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    
    return (acc, acc_med, comp, comp_med, nc1, nc1_med, nc2, nc2_med)

def log_metrics(scene_id, metrics, log_file):
    """Log metrics to file and console"""
    acc, acc_med, comp, comp_med, nc1, nc1_med, nc2, nc2_med = metrics
    log_str = f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
    print(log_str)
    print(log_str, file=open(log_file, "a"))
    return (acc, acc_med, comp, comp_med, nc1, nc1_med, nc2, nc2_med)

def update_accumulators(accum, metrics):
    """Update metric accumulators"""
    acc, acc_med, comp, comp_med, nc1, nc1_med, nc2, nc2_med = metrics
    accum['acc'] += acc
    accum['acc_med'] += acc_med
    accum['comp'] += comp
    accum['comp_med'] += comp_med
    accum['nc1'] += nc1
    accum['nc1_med'] += nc1_med
    accum['nc2'] += nc2
    accum['nc2_med'] += nc2_med
    
def save_final_results(save_path, scene_infer_times, metrics_accum, dataset_len):
    """Save final evaluation results"""
    # Read existing logs
    to_write = ""
    if osp.exists(osp.join(save_path, "logs.txt")):
        with open(osp.join(save_path, "logs.txt"), "r") as f_sub:
            to_write += f_sub.read()

    # Write final summary
    with open(osp.join(save_path, "logs_all.txt"), "w") as f:
        # Add inference time summary
        time_lines = []
        for sid, times in scene_infer_times.items():
            if len(times) > 0:
                time_lines.append(f"Idx: {sid}, Time_avg_ms: {np.mean(times):.2f}")
        time_block = "\n".join(time_lines) + ("\n" if time_lines else "")

        # Calculate mean metrics
        mean_metrics = {k: v / dataset_len for k, v in metrics_accum.items()}
        
        # Format output string
        print_str = "mean".ljust(20) + ": "
        for m_name, value in mean_metrics.items():
            print_str += f"{m_name}: {value:.3f} | "
        print_str += "\n"

        f.write(to_write + time_block + print_str)

def main(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT(mode=args.mode)
    _URL = "weights/model.pt"
    model.load_state_dict(torch.load(_URL, map_location="cpu"), strict=False)
    model.eval()
    model = model.to(dtype).to(device)
    print(f"Model loaded")

    # Load Dataset
    resolution = (518, 392)
    dataset = SevenScenes(
        split="test",
        ROOT="/data/xthuang/dataset/slam/7scenes/7scenes_source",
        resolution=resolution,
        num_seq=1,
        full_video=True,
        kf_every=args.kf,
    )
    
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = osp.join(args.output_dir, "logs.txt")
    metrics_accum = defaultdict(float)
    scene_infer_times = defaultdict(list)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
        pin_memory=True
    )
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        move_data_to_device(batch, device)
        images = preprocess_images(batch)

        predictions, infer_time_ms = run_VGGT(model, images, dtype)
        
        predictions = organize_predictions(predictions, batch)
        
        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = criterion.get_all_pts3d_t(batch, predictions)
        
        images_all, pts_all, pts_gt_all, masks_all = extract_and_crop_data(batch, predictions, gt_pts, pred_pts)

        scene_id = batch[0]["label"][0].rsplit("/", 1)[0]
        scene_infer_times[scene_id].append(infer_time_ms)
        
        pcd, pcd_gt = prepare_point_clouds(pts_all, pts_gt_all, images_all, masks_all, args.use_proj)
        
        pcd = register_point_clouds(pcd, pcd_gt)
        
        metrics = compute_metrics(pcd_gt, pcd)
        
        logged_metrics = log_metrics(scene_id, metrics, log_file)
        update_accumulators(metrics_accum, logged_metrics)
        
        torch.cuda.empty_cache()

    # Save final results
    save_final_results(args.output_dir, scene_infer_times, metrics_accum, len(dataset))
        
if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        main(args)