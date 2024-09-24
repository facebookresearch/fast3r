import itertools
import json
import os.path as osp
from collections import deque
import random

import cv2
import numpy as np

from src.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.dust3r.utils.image import imread_cv2

class Co3d_Multiview(BaseStereoViewDataset):
    def __init__(self, num_views=4, window_degree_range=360, num_samples_per_window=100, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.num_views = num_views
        self.window_degree_range = window_degree_range
        self.num_samples_per_window = num_samples_per_window
        assert mask_bg in (True, False, "rand")
        self.mask_bg = mask_bg

        # Load all scenes
        with open(osp.join(self.ROOT, f"selected_seqs_{self.split}.json"), "r") as f:
            self.scenes = json.load(f)
            self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
            self.scenes = {
                (k, k2): v2 for k, v in self.scenes.items() for k2, v2 in v.items()
            }
        self.scene_list = list(self.scenes.keys())

        self._generate_combinations(num_images=100, degree_range=window_degree_range, num_samples_per_window=num_samples_per_window)

        self.invalidate = {scene: {} for scene in self.scene_list}

    def _generate_combinations(self, num_images, degree_range, num_samples_per_window):
        """
        Generate all combinations of views such that the difference between
        the max and min index in one combo doesn't exceed the degree range.

        Args:
            num_images (int): Total number of images (e.g., 100).
            degree_range (int): Maximum degree range covered by the posed of the views (e.g., 180).
            num_samples_per_window (int): Number of combinations to sample within each window.
        """
        self.combinations = []
        max_index_diff = degree_range * num_images // 360  # Maximum index difference for the given degree range

        for i in range(num_images):
            window_start = max(0, i - max_index_diff // 2)
            window_end = min(num_images, i + max_index_diff // 2)
            window_indices = list(range(window_start, window_end))
            for _ in range(num_samples_per_window):
                combo = random.sample(window_indices, self.num_views)
                self.combinations.append(tuple(combo))

        # Remove duplicates and sort the combinations
        self.combinations = sorted(set(self.combinations))

    def __len__(self):
        return len(self.scene_list) * len(self.combinations)

    def _get_views(self, idx, resolution, rng):
        # Choose a scene
        obj, instance = self.scene_list[idx // len(self.combinations)]
        image_pool = self.scenes[obj, instance]
        im_indices = self.combinations[idx % len(self.combinations)]

        if resolution not in self.invalidate[obj, instance]:  # Flag invalid images
            self.invalidate[obj, instance][resolution] = [
                False for _ in range(len(image_pool))
            ]

        # Decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == "rand" and rng.choice(2))

        # Add a bit of randomness
        last = len(image_pool) - 1
        views = []
        imgs_idxs = [
            max(0, min(im_idx + rng.integers(-4, 5), last))
            for im_idx in im_indices
        ]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # Some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            if self.invalidate[obj, instance][resolution][im_idx]:
                # Search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(
                        image_pool
                    )
                    if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break

            view_idx = image_pool[im_idx]

            impath = osp.join(
                self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.jpg"
            )

            # Load camera params
            input_metadata = np.load(impath.replace("jpg", "npz"))
            camera_pose = input_metadata["camera_pose"].astype(np.float32)
            intrinsics = input_metadata["camera_intrinsics"].astype(np.float32)

            # Load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(
                impath.replace("images", "depths") + ".geometric.png",
                cv2.IMREAD_UNCHANGED,
            )
            depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(
                input_metadata["maximum_depth"]
            )

            if mask_bg:
                # Load object mask
                maskpath = osp.join(
                    self.ROOT, obj, instance, "masks", f"frame{view_idx:06n}.png"
                )
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # Update the depthmap with mask
                depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
            )

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # Problem, invalidate image and retry
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="Co3d_v2",
                    label=osp.join(obj, instance),
                    instance=osp.split(impath)[1],
                )
            )
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.utils.image import rgb
    from dust3r.viz import SceneViz, auto_cam_size
    from IPython.display import display

    dataset = Co3d_Multiview(
        split="train", num_views=4, ROOT="data/co3d_subset_processed", resolution=224, aug_crop=16,
    )

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == dataset.num_views
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]["camera_pose"] for view_idx in range(dataset.num_views)]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in range(dataset.num_views):
            pts3d = views[view_idx]["pts3d"]
            valid_mask = views[view_idx]["valid_mask"]
            colors = rgb(views[view_idx]["img"])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(
                pose_c2w=views[view_idx]["camera_pose"],
                focal=views[view_idx]["camera_intrinsics"][0, 0],
                color=(view_idx * 255, (1 - view_idx) * 255, 0),
                image=colors,
                cam_size=cam_size,
            )
        display(viz.show(point_size=100, viewer="notebook"))
        break
