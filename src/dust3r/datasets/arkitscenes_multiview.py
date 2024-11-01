import os.path as osp
import cv2
import numpy as np
import random

from src.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.dust3r.utils.image import imread_cv2
from tqdm import tqdm

class ARKitScenes_Multiview(BaseStereoViewDataset):
    def __init__(self, num_views=4, window_size=6, num_samples_per_window=10, ordered=False, *args, split, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.num_views = num_views
        self.num_samples_per_window = num_samples_per_window
        self.window_size = window_size
        self.ordered = ordered

        if split == "train":
            self.split = "Training"
        elif split == "test":
            self.split = "Test"
        else:
            raise ValueError("Invalid split option")

        self.loaded_data = self._load_data(self.split)
        self._generate_combinations()

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, split, 'all_metadata.npz')) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']
            self.intrinsics = data['intrinsics'].astype(np.float32)
            self.trajectories = data['trajectories'].astype(np.float32)

    def _generate_combinations(self):
        """
        Generate combinations of image indices for multiview.
        """
        self.combinations = []

        # Group image indices by scene
        scene_to_indices = {}
        for idx, scene_id in enumerate(self.sceneids):
            if scene_id not in scene_to_indices:
                scene_to_indices[scene_id] = []
            scene_to_indices[scene_id].append(idx)

        # Sort each scene's indices by temporal order based on image names
        for scene_id, indices in scene_to_indices.items():
            scene_to_indices[scene_id] = sorted(indices, key=lambda idx: self.images[idx])

        # Generate combinations of views within each scene
        for indices in scene_to_indices.values():
            if len(indices) >= self.num_views:
                max_index_diff = self.window_size
                for i in range(len(indices)):
                    window_start = max(0, i - max_index_diff // 2)
                    window_end = min(len(indices), i + max_index_diff // 2)
                    window_indices = indices[window_start:window_end]

                    for _ in range(self.num_samples_per_window):
                        if len(window_indices) >= self.num_views:
                            # Randomly sample a combination
                            combo = random.sample(window_indices, self.num_views)

                            # If ordered flag is set, sort based on the original order in window_indices
                            if self.ordered:
                                combo = sorted(combo, key=lambda x: window_indices.index(x))

                            self.combinations.append(tuple(combo))

        # Remove duplicates and sort the combinations
        self.combinations = sorted(set(self.combinations))

    def __len__(self):
        return len(self.combinations)

    def _get_views(self, idx, resolution, rng):
        image_indices = self.combinations[idx]

        views = []
        for view_idx in image_indices:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_dir, 'vga_wide', basename.replace('.png', '.jpg')))
            # Load depthmap
            depthmap = imread_cv2(osp.join(scene_dir, 'lowres_depth', basename), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='arkitscenes',
                label=self.scenes[scene_id] + '_' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
            ))

        return views
