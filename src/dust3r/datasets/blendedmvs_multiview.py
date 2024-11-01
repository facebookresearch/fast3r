import os.path as osp
import numpy as np
import random

from src.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.dust3r.utils.image import imread_cv2

class BlendedMVS_Multiview(BaseStereoViewDataset):
    """Multi-view Dataset of BlendedMVS scenes."""

    def __init__(self, num_views=4, num_samples_per_window=10, window_size=6, ordered=False, *args, ROOT, split=None, **kwargs):
        self.ROOT = ROOT
        self.num_views = num_views
        self.num_samples_per_window = num_samples_per_window
        self.window_size = window_size
        self.ordered = ordered
        super().__init__(*args, **kwargs)
        self._load_data(split)
        self._generate_combinations()

    def _load_data(self, split):
        pairs = np.load(osp.join(self.ROOT, 'blendedmvs_pairs.npy'))
        if split is None:
            selection = slice(None)
        if split == 'train':
            # select 90% of all scenes
            selection = (pairs['seq_low'] % 10) > 0
        if split == 'val':
            # select 10% of all scenes
            selection = (pairs['seq_low'] % 10) == 0
        self.pairs = pairs[selection]

        # Group image indices by scene for multiview sampling
        self.scene_to_indices = {}
        for idx, (seqh, seql, img1, img2, score) in enumerate(self.pairs):
            scene_id = f"{seqh:08x}{seql:016x}"
            if scene_id not in self.scene_to_indices:
                self.scene_to_indices[scene_id] = []
            self.scene_to_indices[scene_id].extend([img1, img2])

        # Sort each scene's indices by temporal order based on file names
        for scene_id, indices in self.scene_to_indices.items():
            self.scene_to_indices[scene_id] = sorted(set(indices))

    def _generate_combinations(self):
        """
        Generate combinations of image indices for multi-view sampling.
        """
        self.combinations = []

        # Generate combinations of views within each scene
        for indices in self.scene_to_indices.values():
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

                            # Sort combination if ordered is True
                            if self.ordered:
                                combo = sorted(combo, key=lambda x: window_indices.index(x))

                            self.combinations.append((scene_id, tuple(combo)))

        # Remove duplicates and sort the combinations
        self.combinations = sorted(set(self.combinations))

    def __len__(self):
        return len(self.combinations)

    def _get_views(self, idx, resolution, rng):
        scene_id, image_indices = self.combinations[idx]
        seq_path = osp.join(self.ROOT, scene_id)

        views = []
        for view_index in image_indices:
            impath = f"{view_index:08n}"
            image = imread_cv2(osp.join(seq_path, impath + ".jpg"))
            depthmap = imread_cv2(osp.join(seq_path, impath + ".exr"))
            camera_params = np.load(osp.join(seq_path, impath + ".npz"))

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = camera_params['R_cam2world']
            camera_pose[:3, 3] = camera_params['t_cam2world']

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, impath))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='BlendedMVS',
                label=scene_id,
                instance=impath))

        return views


if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = BlendedMVS_Multiview(split='train', ROOT="data/blendedmvs_processed", resolution=224, num_views=4, window_size=6, num_samples_per_window=10, ordered=True, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == dataset.num_views
        print(idx, [view_name(view) for view in views])
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in range(dataset.num_views)]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in range(dataset.num_views):
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(
                pose_c2w=views[view_idx]['camera_pose'],
                focal=views[view_idx]['camera_intrinsics'][0, 0],
                color=(idx * 255, (1 - idx) * 255, 0),
                image=colors,
                cam_size=cam_size)
        viz.show()
        break
