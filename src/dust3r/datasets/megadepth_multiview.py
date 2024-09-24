import os.path as osp
import numpy as np
import random

from src.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from src.dust3r.utils.image import imread_cv2

class MegaDepth_Multiview(BaseStereoViewDataset):
    def __init__(self, num_views=4, window_size=60, num_samples_per_window=100, *args, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.num_views = num_views
        self.window_size = window_size
        self.num_samples_per_window = num_samples_per_window
        self._load_data()

        if self.split is None:
            pass
        elif self.split == 'train':
            self.select_scene(('0015', '0022'), opposite=True)
        elif self.split == 'val':
            self.select_scene(('0015', '0022'))
        else:
            raise ValueError(f'bad {self.split=}')

        self._generate_scene_to_images_mapping()
        self._generate_combinations()

    def select_scene(self, scene, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = [s.startswith(scenes) for s in self.scenes]
        assert any(scene_id), 'no scene found'

        valid = np.in1d(self.sceneids, np.nonzero(scene_id)[0])

        if opposite:
            valid = ~valid
        assert valid.any()
        self.sceneids = self.sceneids[valid]
        self.images = self.images[valid]

    def _load_data(self):
        with np.load(osp.join(self.ROOT, 'all_metadata_for_multiview.npz')) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']

    def _generate_scene_to_images_mapping(self):
        self.scene_to_images = {}
        self.image_to_scene = {}
        for img_idx, scene_idx in enumerate(self.sceneids):
            scene = self.scenes[scene_idx]
            if scene not in self.scene_to_images:
                self.scene_to_images[scene] = []
            self.scene_to_images[scene].append(img_idx)
            self.image_to_scene[img_idx] = scene

    def _generate_combinations(self):
        """
        Generate combinations of image indices for multiview.
        """
        self.combinations = []

        # Generate combinations within each scene
        for indices in self.scene_to_images.values():
            if len(indices) >= self.num_views:
                max_index_diff = self.window_size
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

    def __len__(self):
        return len(self.combinations)

    def _get_views(self, idx, resolution, rng):
        image_indices = self.combinations[idx]

        # Ensure the indices stay within the scene boundaries
        scene_name = self.image_to_scene[image_indices[0]]
        valid_indices = self.scene_to_images[scene_name]

        # Add a bit of randomness
        random_offsets = [rng.integers(-2, 3) for _ in image_indices]
        image_indices = [valid_indices[max(0, min(valid_indices.index(im_idx) + offset, len(valid_indices) - 1))] for im_idx, offset in zip(image_indices, random_offsets)]

        scene, subscene = scene_name.split('/')
        seq_path = osp.join(self.ROOT, scene, subscene)

        views = []

        for im_id in image_indices:
            img = self.images[im_id]
            try:
                image = imread_cv2(osp.join(seq_path, img + '.jpg'))
                depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
                camera_params = np.load(osp.join(seq_path, img + ".npz"))
            except Exception as e:
                raise OSError(f'cannot load {img}, got exception {e}')

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=osp.relpath(seq_path, self.ROOT),
                instance=img))

        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    from IPython.display import display

    dataset = MegaDepth_Multiview(
        split="train", num_views=4, window_size=60, num_samples_per_window=10, ROOT="data/megadepth_processed", resolution=224, aug_crop=16
    )

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == dataset.num_views
        print(idx, view_name(views[0]), view_name(views[1]))
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
