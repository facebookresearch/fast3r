#!/usr/bin/env python3
"""
fast3r_re10k_pose_eval_multi_gpu.py

1) Root set to /home/jianingy/research/fast3r/src via rootutils.
2) Splits RealEstate10K test folders between 2 GPUs in parallel (by default).
3) Loads & runs DUSt3R model on each GPU, aggregates final results.
4) Allows evaluating only a subset of scene folders (via --subset_file).
5) Correctly parses RealEstate10K extrinsics as c2w.
"""

import os
import glob
import random
import time
import copy
import math
import json
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
from numpy.linalg import inv
import torch
import hydra
import open3d as o3d
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

# For CLI
import argparse

# Attempt pretty console table with 'rich'
try:
    from rich.table import Table
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Rootutils: set project root => /home/jianingy/research/fast3r/src
import rootutils
rootutils.setup_root(
    "/home/jianingy/research/fast3r/src",
    indicator=".project-root",  # or remove if not needed
    pythonpath=True
)

# Project-specific imports
from src.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from src.dust3r.inference_multiview import inference
from src.dust3r.model import FlashDUSt3R
from src.dust3r.utils.image import rgb, imread_cv2
from src.dust3r.datasets.utils.transforms import ImgNorm
import src.dust3r.datasets.utils.cropping as cropping

from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


########################
# 1) Utility: fix old references in config
########################
def replace_dust3r_in_config(cfg):
    for key, value in cfg.items():
        if isinstance(value, dict):
            replace_dust3r_in_config(value)
        elif isinstance(value, str):
            if "dust3r." in value and "src.dust3r." not in value:
                cfg[key] = value.replace("dust3r.", "src.dust3r.")
    return cfg


########################
# 2) Crop & Resize
########################
def crop_resize_if_necessary(
    image,
    intrinsics_3x3,
    target_resolution=(512, 288),
    rng=None,
    info=None
):
    """Crop around principal point + downscale => (512×288) or (288×512)."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    W_org, H_org = image.size
    cx, cy = int(round(intrinsics_3x3[0, 2])), int(round(intrinsics_3x3[1, 2]))

    min_margin_x = min(cx, W_org - cx)
    min_margin_y = min(cy, H_org - cy)

    left = cx - min_margin_x
    top = cy - min_margin_y
    right = cx + min_margin_x
    bottom = cy + min_margin_y
    crop_bbox = (left, top, right, bottom)

    image_c, _, intrinsics_c = cropping.crop_image_depthmap(
        image,
        None,
        intrinsics_3x3,
        crop_bbox
    )

    # Check orientation
    W_c, H_c = image_c.size
    if H_c > W_c:
        # swap if portrait
        target_resolution = (target_resolution[1], target_resolution[0])

    # Downscale
    image_rs, _, intrinsics_rs = cropping.rescale_image_depthmap(
        image_c, None, intrinsics_c, np.array(target_resolution)
    )
    intrinsics2 = cropping.camera_matrix_of_crop(
        intrinsics_rs, image_rs.size, target_resolution, offset_factor=0.5
    )
    final_bbox = cropping.bbox_from_intrinsics_in_out(
        intrinsics_rs, intrinsics2, target_resolution
    )
    image_out, _, intrinsics_out = cropping.crop_image_depthmap(
        image_rs, None, intrinsics_rs, final_bbox
    )

    return image_out, intrinsics_out


########################
# 3) Worker function: processes a subset of folders on a given GPU
########################
def process_folders(args):
    """
    args: tuple(
       video_folders, device_id,
       re10k_video_root, re10k_txt_root,
       checkpoint_path, output_folder
    )
    Each worker loads the model on 'cuda:device_id' and runs inference.
    Returns a list of dicts with final metrics for each folder.
    """
    (video_folders, device_id,
     re10k_video_root, re10k_txt_root,
     checkpoint_path, output_folder) = args

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"[Process GPU {device_id}] => device = {device}")

    # 1) Load model
    print(f"[Process GPU {device_id}] Loading checkpoint: {checkpoint_path}")
    cfg_path = os.path.join(os.path.dirname(checkpoint_path), "../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg.model.net = replace_dust3r_in_config(cfg.model.net)

    if "encoder_args" in cfg.model.net:
        cfg.model.net.encoder_args.patch_embed_cls = "PatchEmbedDust3R"
        cfg.model.net.head_args.landscape_only = False
    else:
        cfg.model.net.patch_embed_cls = "PatchEmbedDust3R"
        cfg.model.net.landscape_only = False

    cfg.model.net.decoder_args.random_image_idx_embedding = True
    cfg.model.net.decoder_args.attn_bias_for_inference_enabled = False

    lit_module = hydra.utils.instantiate(
        cfg.model, train_criterion=None, validation_criterion=None
    )

    lit_module = MultiViewDUSt3RLitModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        net=lit_module.net,
        train_criterion=lit_module.train_criterion,
        validation_criterion=lit_module.validation_criterion
    )
    lit_module.eval()
    model = lit_module.net.to(device)
    # Optionally compile
    # model = torch.compile(model)

    results_list = []

    # 2) For each folder => sample frames => run inference => evaluate poses
    for vid_folder in tqdm(video_folders, desc=f"[GPU {device_id}] Evaluate"):
        folder_path = os.path.join(re10k_video_root, vid_folder)
        if not os.path.isdir(folder_path):
            continue

        txt_path = os.path.join(re10k_txt_root, vid_folder + ".txt")
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r") as f:
            txt_lines = f.read().strip().split("\n")
        if len(txt_lines) <= 1:
            continue
        txt_lines = txt_lines[1:]  # skip first line (URL)

        lines_map = {}
        for line in txt_lines:
            parts = line.strip().split()
            # Expect at least 19 columns
            if len(parts) < 19:
                continue
            frame_id = parts[0]
            lines_map[frame_id] = parts

        frame_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        if len(frame_files) < 2:
            continue

        # Sample up to 10 frames per folder
        n_to_sample = min(10, len(frame_files))
        sampled_frames = random.sample(frame_files, n_to_sample)
        sampled_frames.sort()

        selected_views = []
        for frame_path in sampled_frames:
            basename = os.path.splitext(os.path.basename(frame_path))[0]
            if basename not in lines_map:
                continue

            columns = lines_map[basename]
            # parse fx, fy, cx, cy
            fx = float(columns[1])
            fy = float(columns[2])
            cx = float(columns[3])
            cy = float(columns[4])

            # parse extrinsic row-major 3×4 => build 4×4, then invert to get c2w
            extrinsic_val = [float(v) for v in columns[7:19]]  # 12 floats
            extrinsic = np.array(extrinsic_val, dtype=np.float64).reshape(3, 4)
            # Build 4x4
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :3] = extrinsic[:3, :3]
            pose_4x4[:3, 3]  = extrinsic[:3, 3]  # translation in last column
            poses_c2w_gt = inv(pose_4x4).astype(np.float32)

            # Load image
            img_rgb = imread_cv2(frame_path)
            if img_rgb is None:
                continue

            H_org, W_org = img_rgb.shape[:2]
            # Build K
            K_3x3 = np.array([
                [fx * W_org, 0.0,       cx * W_org],
                [0.0,        fy * H_org, cy * H_org],
                [0.0,        0.0,       1.0]
            ], dtype=np.float32)

            # Crop+resize => final_img_pil, final_intrinsics
            pil_img = Image.fromarray(img_rgb)
            final_img_pil, final_intrinsics_3x3 = crop_resize_if_necessary(
                pil_img, K_3x3, target_resolution=(512, 288)
            )
            tensor_chw = ImgNorm(final_img_pil)

            # Put data on GPU
            view_dict = {
                "img": tensor_chw.unsqueeze(0).to(device),              # (1,3,H,W)
                "camera_pose": torch.from_numpy(poses_c2w_gt).unsqueeze(0).to(device),  # (1,4,4)
                "camera_intrinsics": torch.from_numpy(final_intrinsics_3x3).unsqueeze(0).to(device),  # (1,3,3)
                "dataset": ["RealEstate10K"],
                "true_shape": torch.tensor([[final_img_pil.size[1], final_img_pil.size[0]]], device=device),
            }
            selected_views.append(view_dict)

        if len(selected_views) < 2:
            continue

        output = inference(
            selected_views,
            model=model,
            device=device,
            dtype=torch.float32,
            verbose=False,
            profiling=False
        )

        # Evaluate camera poses
        pose_results = lit_module.evaluate_camera_poses(
            views=output["views"],
            preds=output["preds"],
            niter_PnP=100,
            focal_length_estimation_method='first_view_from_global_head'
        )
        if len(pose_results) > 0:
            metrics_dict = pose_results[0]
            metrics_dict["video_name"] = vid_folder

            # Save result to file
            out_path = os.path.join(output_folder, f"{vid_folder}.txt")
            with open(out_path, "w") as ff:
                ff.write(str(metrics_dict))

            results_list.append(metrics_dict)

    return results_list


########################
# 4) The main: splits data for 2 GPUs, spawns processes, aggregates final metrics
########################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_file", type=str, default=None,
                        help="Optional path to a text file with subset folder names to evaluate")
    parser.add_argument("--gpu_count", type=int, default=2, help="Number of GPUs to use")
    args = parser.parse_args()

    # Setup paths
    re10k_video_root = "/data/jianingy/RealEstate10K/videos/test"
    re10k_txt_root   = "/data/jianingy/RealEstate10K/test"
    output_folder    = "/home/jianingy/research/fast3r/notebooks/RealEstate10K_eval"
    os.makedirs(output_folder, exist_ok=True)

    # Checkpoint
    checkpoint_dir = "/data/jianingy/dust3r_data/fast3r_checkpoints/super_long_training_5175604"
    possible_dir = os.path.join(checkpoint_dir, "checkpoints", "last.ckpt")
    if os.path.isdir(possible_dir):
        # Convert zero checkpoint
        ckpt_agg = os.path.join(checkpoint_dir, "checkpoints", "last_aggregated.ckpt")
        if not os.path.exists(ckpt_agg):
            convert_zero_checkpoint_to_fp32_state_dict(possible_dir, ckpt_agg, tag=None)
        CKPT_PATH = ckpt_agg
    else:
        CKPT_PATH = os.path.join(checkpoint_dir, "checkpoints", "last.ckpt")

    # All video folders
    all_folders = sorted(os.listdir(re10k_video_root))
    all_folders = [f for f in all_folders if os.path.isdir(os.path.join(re10k_video_root, f))]

    # If user specified a subset file, only keep those folders
    if args.subset_file and os.path.exists(args.subset_file):
        with open(args.subset_file, "r") as f:
            subset_scenes = set(line.strip() for line in f if line.strip())
        all_folders = [fd for fd in all_folders if fd in subset_scenes]

    # If no folders remain, just exit
    if not all_folders:
        print("No matching folders found. Exiting.")
        return

    # Split in 'args.gpu_count' parts. By default =2
    n_gpus = args.gpu_count
    chunk_size = math.ceil(len(all_folders) / n_gpus)
    subfolders_list = []
    for i in range(n_gpus):
        start_i = i * chunk_size
        end_i = start_i + chunk_size
        subfolders_list.append(all_folders[start_i:end_i])

    # We'll run n_gpus processes in parallel
    tasks = []
    for i in range(n_gpus):
        if subfolders_list[i]:
            tasks.append((subfolders_list[i], i, re10k_video_root, re10k_txt_root, CKPT_PATH, output_folder))

    ctx = get_context("spawn")  # or "fork" if on Linux
    pool = ctx.Pool(processes=len(tasks))
    async_results = []
    for arg_tuple in tasks:
        ar = pool.apply_async(process_folders, (arg_tuple,))
        async_results.append(ar)

    # Collect
    all_results = []
    for ar in async_results:
        subset_res = ar.get()  # each is a list of metrics
        all_results.extend(subset_res)

    pool.close()
    pool.join()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Aggregate
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    aggregator = defaultdict(list)
    # Typical keys from evaluate_camera_poses
    metric_keys = ["RRA_at_5","RRA_at_15","RRA_at_30","RTA_at_5","RTA_at_15","RTA_at_30","mAA_30"]

    for res in all_results:
        for k in metric_keys:
            if k in res:
                aggregator[k].append(float(res[k]))

    final_means = {}
    for k in metric_keys:
        vals = aggregator.get(k, [])
        if vals:
            final_means[k] = sum(vals)/len(vals)
        else:
            final_means[k] = float("nan")

    # Print summary
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title="RealEstate10K Pose Metrics Summary (Multi-GPU aggregated)")
        table.add_column("Metric", justify="left")
        table.add_column("Mean Value", justify="right")
        for k in metric_keys:
            val = final_means[k]
            table.add_row(k, f"{val:.4f}")
        console.print(table)
    else:
        print("Final Means for RealEstate10K Pose:")
        for k in metric_keys:
            print(f"{k}: {final_means[k]:.4f}")

    print(f"[Main] Done! Processed {len(all_results)} video folders total.")


if __name__ == "__main__":
    main()
