import os
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque

from src.dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset

'''
Preprocessing code of scannetpp is from splatam
'''

class Scannetpp(BaseManyViewDataset):
    def __init__(self, num_seq=100, num_frames=5, 
                 min_thresh=10, max_thresh=100, 
                 test_id=None, full_video=False, 
                 kf_every=1, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every

         # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq
    
    def load_all_scenes(self, base_dir, num_seq=200):
        
        if self.test_id is None:
            meta_split = osp.join(base_dir, 'splits', f'nvs_sem_{self.split}.txt')
            
            if not osp.exists(meta_split):
                raise FileNotFoundError(f"Split file {meta_split} not found")
            
            with open(meta_split) as f:
                self.scene_list = f.read().splitlines()
                
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")
            
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
                
            print(f"Test_id: {self.test_id}")
    
    def _get_views(self, idx, resolution, rng, attempts=0):
        scene_id = self.scene_list[idx // self.num_seq]

        cams_metadata_path = osp.join(self.ROOT, 'data', scene_id, 'dslr/nerfstudio/transforms_undistorted.json')
        cams_meta_data = json.load(open(cams_metadata_path, "r"))
        fx, fy, cx, cy = cams_meta_data['fl_x'], cams_meta_data['fl_y'], cams_meta_data['cx'], cams_meta_data['cy']

        frame_meta_data = cams_meta_data['frames']
        train_info_path = osp.join(self.ROOT, 'data', scene_id, 'dslr/train_test_lists.json')
        train_info = json.load(open(train_info_path, "r"))

        imgs_idxs_ = sorted(train_info['train'])
        imgs_idxs = self.sample_frame_idx(imgs_idxs_, rng, full_video=self.full_video)
        imgs_idxs = deque(imgs_idxs)

        filepath_index_mapping = {frame["file_path"]: index for index, frame in enumerate(frame_meta_data)}

        views = []
        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            # Load image data
            impath = osp.join(self.ROOT, 'data', scene_id, 'dslr/undistorted_images', im_idx)
            depthpath = osp.join(self.ROOT, 'data', scene_id, 'dslr/undistorted_depths', im_idx.replace('.JPG', '.png'))

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0

            # Load camera params
            frame_metadata = frame_meta_data[filepath_index_mapping.get(im_idx)]
            intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            
            camera_pose = np.array(frame_metadata["transform_matrix"], dtype=np.float32)
            # gl to cv
            camera_pose[:, 1:3] *= -1.0

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found for {impath}")
                    continue
                else:
                    if attempts >= 5:
                        new_idx = rng.integers(0, self.__len__()-1)
                        return self._get_views(new_idx, resolution, rng)
                    return self._get_views(idx, resolution, rng, attempts+1)
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='scannetpp',
                label=osp.join(scene_id, im_idx),
                instance=osp.split(impath)[1],
            ))
        return views
