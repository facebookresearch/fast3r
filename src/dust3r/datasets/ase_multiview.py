import os
import os.path as osp
import time
import cv2
import math
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from PIL import Image

from src.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.dust3r.utils.image import imread_cv2
from src.dust3r.datasets.aria.camera_utils import undistort_fisheye_to_pinhole_rgbd, VignetteCorrector

##########################################
# Camera parameters and utilities
##########################################

FISHEYE_CAM_PARAMS = torch.tensor(
    [
        297.638,
        357.66,
        349.192,
        0.365089,
        -0.173808,
        -0.753495,
        2.43479,
        -2.57786,
        0.878848,
        0.00080052,
        -0.000294238,
        0,
        0,
        0,
        0,
    ],
    dtype=torch.float32,
)[None, None, :]

PINHOLE_CAM_PARAMS = torch.tensor(
    [297.638, 297.638, 357.66, 349.192], dtype=torch.float32
)[None, None, :]

ASE_INTRINSICS = torch.tensor(
    [
        [PINHOLE_CAM_PARAMS[0, 0, 0], 0, PINHOLE_CAM_PARAMS[0, 0, 2]],
        [0, PINHOLE_CAM_PARAMS[0, 0, 1], PINHOLE_CAM_PARAMS[0, 0, 3]],
        [0, 0, 1],
    ],
    dtype=torch.float32,
)

T_DEVICE_FROM_CAMERA = torch.tensor(
    [
        [0.99606003, -0.04388682, 0.07706079, -0.0075301],
        [0.08210934, 0.78468796, -0.61442889, -0.01090855],
        [-0.03350334, 0.61833547, 0.78519983, -0.00359806],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)

def imread_cv2(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Image at {path} not found.")
    return img

def read_trajectory_file(filepath):
    assert osp.exists(filepath), f"Could not find trajectory file: {filepath}"
    with open(filepath, "r") as f:
        header = f.readline()
        positions = []
        rotations = []
        transforms = []
        timestamps = []
        for line in f.readlines():
            line = line.strip().split(",")
            timestamp = int(line[1])
            translation = np.array([float(line[3]), float(line[4]), float(line[5])], dtype=np.float32)
            quat_xyzw = np.array([float(line[6]), float(line[7]), float(line[8]), float(line[9])], dtype=np.float32)
            rot_matrix = R.from_quat(quat_xyzw).as_matrix()
            transform = np.eye(4, dtype=np.float32)
            transform[:3,:3] = rot_matrix
            transform[:3,3] = translation
            positions.append(translation)
            rotations.append(rot_matrix)
            transforms.append(transform)
            timestamps.append(timestamp)
    return {
        "positions": np.stack(positions),
        "rotations": np.stack(rotations),
        "Ts_world_from_device": np.stack(transforms),
        "timestamps": np.array(timestamps),
    }


##########################################
# Rotation about camera Z-axis
##########################################
def get_rotation_matrix_z(k):
    # k times 90 degrees clockwise about Z
    # For k=1 (90 deg cw): Rz = [[0,1,0],[-1,0,0],[0,0,1]]
    # For k=2 (180 deg): Rz = [[-1,0,0],[0,-1,0],[0,0,1]]
    # For k=3 (270 deg cw): Rz = [[0,-1,0],[1,0,0],[0,0,1]]
    k = k % 4
    if k == 0:
        return torch.eye(4, dtype=torch.float32)
    elif k == 1:
        R = torch.tensor([[0,1,0],
                          [-1,0,0],
                          [0,0,1]], dtype=torch.float32)
    elif k == 2:
        R = torch.tensor([[-1,0,0],
                          [0,-1,0],
                          [0,0,1]], dtype=torch.float32)
    elif k == 3:
        R = torch.tensor([[0,-1,0],
                          [1,0,0],
                          [0,0,1]], dtype=torch.float32)
    Rt = torch.eye(4, dtype=torch.float32)
    Rt[:3,:3] = R
    return Rt

def rotate_image_90_clockwise(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def adjust_intrinsics_for_90_clockwise_rotation(K, original_width, original_height):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    new_width = original_height
    new_height = original_width

    # After a 90° clockwise rotation:
    # fx' = fy
    # fy' = fx
    # cx' = cy
    # cy' = (W - 1) - cx
    new_fx = fy
    new_fy = fx
    new_cx = cy
    new_cy = (original_width - 1) - cx

    K_new = np.array([
        [new_fx, 0, new_cx],
        [0, new_fy, new_cy],
        [0,    0,   1   ]
    ], dtype=np.float32)
    return K_new

##########################################
# ASE_Multiview dataset
##########################################
class ASE_Multiview(BaseStereoViewDataset):
    def __init__(
        self,
        ROOT,
        split="train",
        num_views=4,
        window_size=10,
        num_samples_per_window=10,
        data_scaling=1.0,
        ordered=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, split=split, **kwargs)
        self.ROOT = ROOT
        self.split = split
        self.num_views = num_views
        self.window_size = window_size
        self.num_samples_per_window = num_samples_per_window
        self.data_scaling = data_scaling
        self.ordered = ordered

        # Load scenes
        self.scenes = sorted([d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))])[:1]
        if self.data_scaling < 1.0:
            num_scenes = max(1, int(len(self.scenes)*self.data_scaling))
            self.scenes = self.scenes[:num_scenes]

        self.scene_to_indices = {}
        self.metadata = []
        # Load frames from each scene
        for scene_id, scene_name in enumerate(self.scenes):
            scene_path = osp.join(self.ROOT, scene_name)
            trajectory = read_trajectory_file(osp.join(scene_path, "trajectory.csv"))
            n_frames = len(trajectory["Ts_world_from_device"])
            for frame_idx in range(n_frames):
                self.metadata.append((scene_id, frame_idx, scene_name, trajectory))
        
        from collections import defaultdict
        group_by_scene = defaultdict(list)
        for i,(scene_id, frame_idx, scene_name, traj) in enumerate(self.metadata):
            group_by_scene[scene_id].append(i)

        # Sort by frame_idx within each scene
        for sid in group_by_scene:
            group_by_scene[sid].sort(key=lambda i: self.metadata[i][1])

        self.scene_to_indices = group_by_scene
        self._generate_combinations()

        self.vignette_corrector = VignetteCorrector()

    def _generate_combinations(self):
        self.combinations = []
        for scene_id, indices in self.scene_to_indices.items():
            if len(indices) < self.num_views:
                continue
            max_index_diff = self.window_size
            for i, idx_center in enumerate(indices):
                # form a window around idx_center
                window_start = max(0, i - max_index_diff // 2)
                window_end = min(len(indices), i + max_index_diff // 2)
                window_indices = indices[window_start:window_end]

                for _ in range(self.num_samples_per_window):
                    if len(window_indices) >= self.num_views:
                        combo = random.sample(window_indices, self.num_views)
                        if self.ordered:
                            combo = sorted(combo, key=lambda x: window_indices.index(x))
                        self.combinations.append(tuple(combo))

        self.combinations = sorted(set(self.combinations))

    def __len__(self):
        return len(self.combinations)

    def _get_views(self, idx, resolution, rng):
        start_time = time.time()
        image_indices = self.combinations[idx]
        views = []
        for view_idx in image_indices:
            scene_id, frame_idx, scene_name, trajectory = self.metadata[view_idx]
            scene_dir = osp.join(self.ROOT, scene_name)

            # Load pose
            pose = trajectory["Ts_world_from_device"][frame_idx].copy()
            # Apply device->camera transform
            pose = pose @ T_DEVICE_FROM_CAMERA.numpy()

            # Load image and depth
            rgb_path = osp.join(scene_dir, "rgb", f"vignette{frame_idx:07d}.jpg")
            depth_path = osp.join(scene_dir, "depth", f"depth{frame_idx:07d}.png")
            rgb = imread_cv2(rgb_path, cv2.IMREAD_COLOR).astype(np.float32)
            depth = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            # Vignette correct
            rgb = self.vignette_corrector.correct(rgb)

            # Undistort (fisheye to pinhole)
            rgb_undistorted, depth_undistorted = undistort_fisheye_to_pinhole_rgbd(
                rgb, depth, FISHEYE_CAM_PARAMS, PINHOLE_CAM_PARAMS
            )

            # Rotate image and depth 90 deg clockwise
            rgb_rotated = rotate_image_90_clockwise(rgb_undistorted)
            depth_rotated = rotate_image_90_clockwise(depth_undistorted)

            # Adjust intrinsics for rotation
            H, W, _ = rgb_undistorted.shape
            intrinsics = ASE_INTRINSICS.clone().numpy()
            intrinsics = adjust_intrinsics_for_90_clockwise_rotation(intrinsics, W, H)

            # Also rotate the pose about camera Z to match the image rotation
            # Rotation 90 deg clockwise about camera Z:
            Rz = get_rotation_matrix_z(k=1).numpy()
            pose = pose @ Rz

            # Convert depth to meters
            depthmap = depth_rotated / 1000.0

            # Convert image to uint8 and create PIL image
            rgb_rotated_uint8 = np.clip(rgb_rotated, 0, 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb_rotated_uint8, mode="RGB")

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ASE',
                label=scene_name + f"_{frame_idx:07d}",
                instance=f'{idx}_{view_idx}',
            ))

        print(f"Time taken for loading views: {time.time() - start_time:.2f}s")
        return views


##########################################
# The simplified dataset
##########################################
class ASE_Multiview_Simple(BaseStereoViewDataset):
    """
    A simplified version that:
      - Finds the first scene in ROOT
      - Loads its trajectory
      - Creates a *sliding window* of size num_views across frames
      - For each item (idx), returns that contiguous set of frames
      - No random sampling / no fancy combination generation
    """
    def __init__(
        self,
        ROOT,
        split="train",
        num_views=4,
        data_scaling=1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, split=split, **kwargs)
        self.ROOT = ROOT
        self.split = split
        self.num_views = num_views
        self.data_scaling = data_scaling

        # Just pick the first scene (or all if you like)
        self.scenes = sorted(
            d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))
        )
        if not self.scenes:
            raise RuntimeError(f"No scenes found in {self.ROOT}")

        # For simplicity, we'll just use scene[0].
        self.scene_name = self.scenes[0]
        scene_dir = osp.join(self.ROOT, self.scene_name)

        # Load trajectory
        trajectory = read_trajectory_file(osp.join(scene_dir, "trajectory.csv"))
        self.poses_world_from_device = trajectory["Ts_world_from_device"]
        n_frames = len(self.poses_world_from_device)

        # Possibly scale down number of frames
        if self.data_scaling < 1.0:
            n_keep = max(1, int(n_frames * self.data_scaling))
            n_frames = min(n_frames, n_keep)

        # We'll store (scene_idx=0, frame_idx) in metadata
        # Actually we only have one scene in this example
        self.metadata = list(range(n_frames))

        # Build a simple *sliding window* of length num_views
        # so each dataset item is a set of frames [i, i+1, ..., i+num_views-1].
        # self.combinations = []
        # # Stop when i+num_views > n_frames
        # for i in range(n_frames - num_views + 1):
        #     frames = list(range(i, i + num_views))
        #     self.combinations.append(frames)

        # If you only want EXACTLY ONE item with the first `num_views` frames:
        step = 3
        assert step * self.num_views <= n_frames
        self.combinations = [list(range(0, step * self.num_views, step))]

        self.vignette_corrector = VignetteCorrector()

    def __len__(self):
        return len(self.combinations)

    def _get_views(self, idx, resolution, rng):
        """Implements the base class method for obtaining a list of views."""
        frame_indices = self.combinations[idx]
        scene_dir = osp.join(self.ROOT, self.scene_name)

        views = []
        for f_idx in frame_indices:
            # Pose is [4x4], as a NumPy array
            pose = self.poses_world_from_device[f_idx].copy()
            # device->camera
            pose = pose @ T_DEVICE_FROM_CAMERA.numpy()

            # Load images
            rgb_path = osp.join(scene_dir, "rgb", f"vignette{f_idx:07d}.jpg")
            depth_path = osp.join(scene_dir, "depth", f"depth{f_idx:07d}.png")

            rgb = imread_cv2(rgb_path, cv2.IMREAD_COLOR).astype(np.float32)
            depth = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            # Vignette correction
            rgb = self.vignette_corrector.correct(rgb)

            # Undistort
            rgb_undist, depth_undist = undistort_fisheye_to_pinhole_rgbd(
                rgb, depth, FISHEYE_CAM_PARAMS, PINHOLE_CAM_PARAMS
            )

            # Rotate image/depth 90 deg clockwise
            rgb_rot = rotate_image_90_clockwise(rgb_undist)
            depth_rot = rotate_image_90_clockwise(depth_undist)

            # Adjust intrinsics
            H, W, _ = rgb_undist.shape
            intrinsics = ASE_INTRINSICS.clone().numpy()
            intrinsics = adjust_intrinsics_for_90_clockwise_rotation(intrinsics, W, H)

            # Also rotate the pose about Z by 90 deg cw
            Rz = get_rotation_matrix_z(k=1).numpy()  # NumPy
            pose = pose @ Rz  # still NumPy

            # Convert depth to meters
            depthmap = depth_rot / 1000.0

            # Convert to PIL
            rgb_rot8 = np.clip(rgb_rot, 0, 255).astype(np.uint8)
            rgb_pil = Image.fromarray(rgb_rot8, mode="RGB")

            # Optionally do your crop/resize
            # (We'll just do a direct resize if resolution != size)
            rgb_pil, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_pil, depthmap, intrinsics, resolution, rng
            )

            # Construct a dictionary for each view
            views.append(dict(
                img=rgb_pil,
                depthmap=depthmap.astype(np.float32),
                camera_pose=pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ASE',
                label=f"{self.scene_name}_{f_idx:07d}",
                instance=f"{idx}_{f_idx}",
            ))
        return views

    def __getitem__(self, index):
        """
        The base class calls _get_views(self, idx, resolution, rng).
        The base class might handle some extra logic for augmentations, etc.
        """
        # This typically calls self._get_views(...) inside.
        return super().__getitem__(index)
