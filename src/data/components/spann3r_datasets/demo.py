import os
import cv2
import numpy as np
import os.path as osp
from collections import deque

from src.dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


class Demo(BaseManyViewDataset):
    def __init__(self, num_seq=1, num_frames=5, 
                 min_thresh=10, max_thresh=100,
                 full_video=True, kf_every=1, 
                 *args, ROOT, **kwargs):
        
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.full_video = full_video
        self.kf_every = kf_every
    
    def __len__(self):
        return self.num_seq
    
    def _get_views(self, idx, resolution, rng):
        
        img_idxs = sorted(os.listdir(self.ROOT))
        valid_extensions = {'.jpg', '.jpeg', '.png', '.heic'}
        img_idxs = [idx for idx in img_idxs 
                    if idx.lower().endswith(tuple(valid_extensions)) and 'depth' not in idx.lower()]

        img_idxs = self.sample_frame_idx(img_idxs, rng, full_video=self.full_video)

        # pseudo intrinsics
        fx, fy = 1.0, 1.0

        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(self.ROOT, im_idx)
            if not osp.exists(impath):
                raise FileNotFoundError(f"Image not found: {impath}")

            print(f'Loading image: {impath}')

            if 'heic' in impath.lower():
                from PIL import Image
                rgb_image = Image.open(impath)
                if rgb_image.mode != 'RGB':
                    rgb_image = rgb_image.convert('RGB')
                rgb_image = np.array(rgb_image)
            else:
                rgb_image = imread_cv2(impath)
            

            depth_path = impath.split('.')[0] + '_depth.png'
            meta_data_path = impath.split('.')[0] + '.npz'

            if osp.exists(meta_data_path):
                input_metadata = np.load(meta_data_path)
                camera_pose = input_metadata['camera_pose'].astype(np.float32)
                intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
            else:
                cx, cy = rgb_image.shape[1]//2, rgb_image.shape[0]//2
                intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

                # pseudo camera pose
                camera_pose = np.eye(4).astype(np.float32)
            
            if not osp.exists(depth_path):
                depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1])).astype(np.float32)
            else:
                depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
                depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])
            # resize rgb to the same size as depth
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='demo',
                label=osp.join(self.ROOT, im_idx),
                instance=osp.split(impath)[1],
            ))
        return views


