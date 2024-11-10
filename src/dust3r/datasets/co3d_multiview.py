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
        self.invalid_scene_tracker = set()  # Track scenes that have all images invalid

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

    def _fetch_views_for_pool(self, obj, instance, image_pool, resolution, rng):
        """Attempt to get valid views from a single image_pool, with oversampling if needed."""
        last = len(image_pool) - 1
        imgs_idxs = deque([max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in self.combinations[0]])

        # Collect views and track validity
        views = []
        valid_imgs = []

        while imgs_idxs:
            im_idx = imgs_idxs.pop()
            if self.invalidate[obj, instance][resolution][im_idx]:  # Skip invalid images
                continue

            # Attempt to load view data
            view_data = self._load_view_data(obj, instance, image_pool, im_idx, resolution, rng)
            if view_data:
                views.append(view_data)
                valid_imgs.append(im_idx)  # Track valid images
                if len(views) == self.num_views:
                    return views  # Return if we have enough valid views

        # If not enough views, oversample from valid images in the pool
        while len(views) < self.num_views and valid_imgs:
            im_idx = random.choice(valid_imgs)  # Randomly oversample from valid images
            view_data = self._load_view_data(obj, instance, image_pool, im_idx, resolution, rng)
            if view_data:
                views.append(view_data)

        # Return views if enough valid ones were found; otherwise, return None
        return views if len(views) == self.num_views else None

    def _get_views(self, idx, resolution, rng, max_scene_retries=5):
        """Attempt to get views, retrying with different scenes if necessary."""
        for attempt in range(max_scene_retries):
            # Select a different scene on each retry by applying an offset based on `attempt`
            scene_idx = (idx + attempt) % len(self.scene_list)
            obj, instance = self.scene_list[scene_idx]

            # Skip scenes that are known to be invalid
            if (obj, instance) in self.invalid_scene_tracker:
                continue

            image_pool = self.scenes[obj, instance]
            if resolution not in self.invalidate[obj, instance]:
                self.invalidate[obj, instance][resolution] = [False] * len(image_pool)

            views = self._fetch_views_for_pool(obj, instance, image_pool, resolution, rng)
            if views:
                return views  # Successfully found views

            # If no valid views, mark the scene as invalid and log a warning
            print(f"Warning: Scene {obj, instance} has all images invalid. Skipping.")
            self.invalid_scene_tracker.add((obj, instance))

        raise ValueError(f"Exceeded {max_scene_retries=}. No valid views found.")

    def _load_view_data(self, obj, instance, image_pool, im_idx, resolution, rng):
        """Load the data for a single view, including image, depth, and camera parameters."""
        try:
            view_idx = image_pool[im_idx]
            impath = osp.join(self.ROOT, obj, instance, "images", f"frame{view_idx:06n}.jpg")
            input_metadata = np.load(impath.replace("jpg", "npz"))
            camera_pose = input_metadata["camera_pose"].astype(np.float32)
            intrinsics = input_metadata["camera_intrinsics"].astype(np.float32)

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(impath.replace("images", "depths") + ".geometric.png", cv2.IMREAD_UNCHANGED)
            depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata["maximum_depth"])

            # Background masking logic
            if self.mask_bg:
                maskpath = osp.join(self.ROOT, obj, instance, "masks", f"frame{view_idx:06n}.png")
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1
                depthmap *= maskmap

            # Crop, resize, and validate
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
            )
            if (depthmap > 0.0).sum() == 0:
                # Mark as invalid and return None if no valid depth
                self.invalidate[obj, instance][resolution][im_idx] = True
                return None

            return dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset="Co3d_v2",
                label=osp.join(obj, instance),
                instance=osp.split(impath)[1],
            )
        except Exception as e:
            print(f"Error loading view data for image {impath}: {e}")
            return None

    def __len__(self):
        return len(self.scene_list) * len(self.combinations)


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
