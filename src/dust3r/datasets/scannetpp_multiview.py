import os.path as osp
import cv2
import numpy as np
import random

from src.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.dust3r.utils.image import imread_cv2

class ScanNetpp_Multiview(BaseStereoViewDataset):
    def __init__(self, num_views=4, window_size=60, num_samples_per_window=100, ordered=False, *args, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.num_views = num_views
        self.window_size = window_size
        self.num_samples_per_window = num_samples_per_window
        self.ordered = ordered
        assert self.split == 'train'
        self.loaded_data = self._load_data()
        self._generate_combinations()

    def _load_data(self):
        with np.load(osp.join(self.ROOT, 'all_metadata.npz')) as data:
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
        self.scene_to_indices = {}

        # Group and sort indices by scene, then segregate by iPhone/DSLR patterns
        for idx, scene_id in enumerate(self.sceneids):
            if scene_id not in self.scene_to_indices:
                self.scene_to_indices[scene_id] = {'iphone': [], 'dslr': []}

            # Sort by image pattern type
            if "frame_" in self.images[idx]:  # iPhone pattern
                self.scene_to_indices[scene_id]['iphone'].append(idx)
            else:  # DSLR pattern
                self.scene_to_indices[scene_id]['dslr'].append(idx)

        # Sort indices in temporal order within each scene's iPhone and DSLR list
        for scene_dict in self.scene_to_indices.values():
            scene_dict['iphone'].sort(key=lambda idx: self.images[idx])
            scene_dict['dslr'].sort(key=lambda idx: self.images[idx])

        # Generate combinations within each segregated list
        for scene_dict in self.scene_to_indices.values():
            for device_type in ['iphone', 'dslr']:
                indices = scene_dict[device_type]
                if len(indices) >= self.num_views:
                    max_index_diff = self.window_size
                    for i in range(len(indices)):
                        window_start = max(0, i - max_index_diff // 2)
                        window_end = min(len(indices), i + max_index_diff // 2)
                        window_indices = indices[window_start:window_end]

                        for _ in range(self.num_samples_per_window):
                            if len(window_indices) >= self.num_views:
                                combo = random.sample(window_indices, self.num_views)
                                
                                # If ordered, sort based on the order in window_indices
                                if self.ordered:
                                    combo = sorted(combo, key=lambda x: window_indices.index(x))
                                
                                self.combinations.append(tuple(combo))

        # Remove duplicates and sort the combinations
        self.combinations = sorted(set(self.combinations))

    def __len__(self):
        return len(self.combinations)

    def _get_views(self, idx, resolution, rng):
        image_indices = self.combinations[idx]

        # Determine device type based on the first index in image_indices
        scene_id = self.sceneids[image_indices[0]]
        device_type = 'iphone' if "frame_" in self.images[image_indices[0]] else 'dslr'
        valid_indices = self.scene_to_indices[scene_id][device_type]

        # Add a bit of randomness within the valid indices
        random_offsets = [rng.integers(-2, 3) for _ in image_indices]
        image_indices = [
            valid_indices[max(0, min(valid_indices.index(im_idx) + offset, len(valid_indices) - 1))]
            for im_idx, offset in zip(image_indices, random_offsets)
        ]

        views = []
        for view_idx in image_indices:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_dir, 'images', basename + '.jpg'))
            # Load depthmap
            depthmap = imread_cv2(osp.join(scene_dir, 'depth', basename + '.png'), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ScanNet++',
                label=self.scenes[scene_id] + '_' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    from IPython.display import display

    dataset = ScanNetpp_Multiview(
        split="train", num_views=4, window_size=60, num_samples_per_window=10, ROOT="data/scannetpp_processed", resolution=224, aug_crop=16
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
