import os.path as osp
import cv2
import numpy as np
import random

from src.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.dust3r.utils.image import imread_cv2

# notebook tqdm 
from tqdm import tqdm

class ARKitScenes_Multiview(BaseStereoViewDataset):
    def __init__(self, num_views=4, num_samples_per_window=10, *args, split, ROOT, with_pairs=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.num_views = num_views
        self.num_samples_per_window = num_samples_per_window

        if split == "train":
            self.split = "Training"
        elif split == "test":
            self.split = "Test"
        else:
            raise ValueError("Invalid split option")

        self.loaded_data = self._load_data(self.split)
        self.with_pairs = with_pairs
        if self.with_pairs:
            self._generate_combinations_with_pairs()
        else:
            self._generate_combinations_without_pairs()

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, split, 'all_metadata.npz')) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']
            self.intrinsics = data['intrinsics'].astype(np.float32)
            self.trajectories = data['trajectories'].astype(np.float32)
            self.pairs = data['pairs'][:, :2].astype(int)  # You can remove this later since pairs are not used anymore

    # without pre-computed pairs
    def _generate_combinations_without_pairs(self, window_size=6):
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

        # Generate combinations of views within each scene
        for indices in scene_to_indices.values():
            if len(indices) >= self.num_views:
                max_index_diff = window_size
                for i in range(len(indices)):
                    window_start = max(0, i - max_index_diff // 2)
                    window_end = min(len(indices), i + max_index_diff // 2)
                    window_indices = indices[window_start:window_end]
                    for _ in range(self.num_samples_per_window):
                        if len(window_indices) >= self.num_views:
                            combo = random.sample(window_indices, self.num_views)
                            self.combinations.append(tuple(combo))

        # Remove duplicates and sort the combinations
        self.combinations = sorted(set(self.combinations))

    # with pre-computed pairs
    def _generate_combinations_with_pairs(self):
        """
        Generate combinations of image indices for multiview by selecting num_views//2 pairs per scene.
        The number of combinations to generate is controlled by num_samples_per_window.
        """
        self.combinations = []

        # Group image indices by scene
        scene_to_indices = {}
        for idx, scene_id in enumerate(self.sceneids):
            if scene_id not in scene_to_indices:
                scene_to_indices[scene_id] = []
            scene_to_indices[scene_id].append(idx)

        # Group pairs by scene
        scene_to_pairs = {}
        for pair in self.pairs:
            image_idx1, image_idx2 = pair
            scene_id = self.sceneids[image_idx1]  # Assume both images in the pair belong to the same scene
            if scene_id not in scene_to_pairs:
                scene_to_pairs[scene_id] = []
            scene_to_pairs[scene_id].append((image_idx1, image_idx2))

        # Iterate over each scene and build combinations
        for scene_id, indices in tqdm(scene_to_indices.items(), desc="Generating combinations"):
            pairs = scene_to_pairs.get(scene_id, [])
            num_images = len(indices)  # Number of images in the scene
            num_combinations = num_images * self.num_samples_per_window

            for _ in range(num_combinations):
                # Randomly select num_views//2 pairs
                if len(pairs) >= self.num_views // 2:
                    selected_pairs = random.sample(pairs, self.num_views // 2)
                else:
                    # If there are not enough pairs, select as many as possible
                    selected_pairs = pairs

                # Flatten the selected pairs to create a list of image indices
                base_combination = [img_idx for pair in selected_pairs for img_idx in pair]

                # Ensure that we only select unique image indices
                base_combination = list(set(base_combination))

                # If the number of unique views is less than num_views, randomly select additional views
                while len(base_combination) < self.num_views:
                    # Repeat available images if needed
                    additional_views = random.choices(indices, k=self.num_views - len(base_combination))
                    base_combination.extend(additional_views)

                # Convert to tuple and store the combination
                self.combinations.append(tuple(base_combination[:self.num_views]))

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


if __name__ == "__main__":
    import rootutils
    rootutils.setup_root("/opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/jianingy/research/accel-cortex/dust3r/fast3r/src", indicator=".project-root", pythonpath=True)

    import numpy as np

    from src.dust3r.datasets.arkitscenes_multiview import ARKitScenes_Multiview

    from src.dust3r.datasets.base.base_stereo_view_dataset import view_name
    from src.dust3r.utils.image import rgb
    from src.dust3r.viz import SceneViz, auto_cam_size
    from IPython.display import display

    dataset = ARKitScenes_Multiview(
        split='train', num_views=4, window_size=6, num_samples_per_window=10, ROOT="/fsx-cortex/jianingy/dust3r_data/arkitscenes_processed", resolution=224, aug_crop=16
    )

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == dataset.num_views
        print(dataset.num_views)
        print([view_name(view) for view in views])
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in range(dataset.num_views)]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in range(dataset.num_views):
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                            focal=views[view_idx]['camera_intrinsics'][0, 0],
                            color=(view_idx * 255, (1 - view_idx) * 255, 0),
                            image=colors,
                            cam_size=cam_size)
        display(viz.show())
        break
