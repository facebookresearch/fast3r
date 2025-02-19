import os.path as osp
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # noqa
import cv2  # noqa
import numpy as np
from PIL import Image
import json
import random

from src.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset

class Habitat_Multiview(BaseStereoViewDataset):
    def __init__(self, size=1_000_000, num_views=4, data_scaling=1.0, *args, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.num_views = num_views
        self.data_scaling = data_scaling

        assert self.split is not None
        # Loading list of scenes
        with open(osp.join(self.ROOT, f'Habitat_{size}_scenes_{self.split}.txt')) as f:
            self.scenes = f.read().splitlines()

        # Apply data scaling to limit the number of scenes
        if self.data_scaling < 1.0:
            num_scenes = max(1, int(len(self.scenes) * self.data_scaling))
            self.scenes = sorted(self.scenes)[:num_scenes]

        self.instances = list(range(1, 5))  # Instance views other than view 0

    def filter_scene(self, label, instance=None):
        if instance:
            subscene, instance = instance.split('_')
            label += '/' + subscene
            self.instances = [int(instance) - 1]
        valid = np.bool_([scene.startswith(label) for scene in self.scenes])
        assert sum(valid), f'No scene was selected for {label=}, {instance=}'
        self.scenes = [scene for i, scene in enumerate(self.scenes) if valid[i]]

    def _get_views(self, idx, resolution, rng):
        scene = self.scenes[idx]
        data_path, key = osp.split(osp.join(self.ROOT, scene))

        views = []
        # Always include view 0 (anchor view)
        selected_views = [0]

        # Check if num_views exceeds 5
        if self.num_views > 5:
            # Oversample views to reach num_views
            additional_views = random.choices(self.instances, k=self.num_views - 1)
        else:
            additional_views = random.sample(self.instances, min(len(self.instances), self.num_views - 1))

        selected_views.extend(additional_views)

        for view_index in selected_views:
            # Load the view (and use the next one if this one's broken)
            for ii in range(view_index, view_index + 5):
                try:
                    image, depthmap, intrinsics, camera_pose = self._load_one_view(data_path, key, ii % 5, resolution, rng)
                except FileNotFoundError:
                    continue  # View does not exist, try the next one
                if np.isfinite(camera_pose).all():
                    break
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='Habitat',
                label=osp.relpath(data_path, self.ROOT),
                instance=f"{key}_{view_index}"
            ))

        return views

    def _load_one_view(self, data_path, key, view_index, resolution, rng):
        view_index += 1  # File indices start at 1
        impath = osp.join(data_path, f"{key}_{view_index}.jpeg")
        image = Image.open(impath)

        depthmap_filename = osp.join(data_path, f"{key}_{view_index}_depth.exr")
        depthmap = cv2.imread(depthmap_filename, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

        camera_params_filename = osp.join(data_path, f"{key}_{view_index}_camera_params.json")
        with open(camera_params_filename, 'r') as f:
            camera_params = json.load(f)

        intrinsics = np.float32(camera_params['camera_intrinsics'])
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = camera_params['R_cam2world']
        camera_pose[:3, 3] = camera_params['t_cam2world']

        image, depthmap, intrinsics = self._crop_resize_if_necessary(
            image, depthmap, intrinsics, resolution, rng, info=impath)
        return image, depthmap, intrinsics, camera_pose


if __name__ == "__main__":
    import rootutils
    rootutils.setup_root("/opt/hpcaas/.mounts/fs-0565f60d669b6a2d3/home/jianingy/research/accel-cortex/dust3r/fast3r/src", indicator=".project-root", pythonpath=True)

    import numpy as np

    from src.dust3r.datasets.habitat_multiview import Habitat_Multiview

    from src.dust3r.datasets.base.base_stereo_view_dataset import view_name
    from src.dust3r.utils.image import rgb
    from src.dust3r.viz import SceneViz, auto_cam_size
    from IPython.display import display

    dataset = Habitat_Multiview(1_000, num_views=6, split='train', ROOT="/fsx-cortex/jianingy/dust3r_data/habitat_processed",
                                resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == dataset.num_views
        print(len(views))
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
