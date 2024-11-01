import os
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque

from src.dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


class DTU(BaseManyViewDataset):
    def __init__(self, num_seq=49, num_frames=5, 
                 min_thresh=10, max_thresh=30, 
                 test_id=None, full_video=False, 
                 sample_pairs=False, kf_every=1, 
                 *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.sample_pairs = sample_pairs
    
        # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        
        if self.test_id is None:
            self.scene_list = os.listdir(osp.join(base_dir))
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")
            
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
                
            print(f"Test_id: {self.test_id}")
    
    def load_cam_mvsnet(self, file, interval_scale=1):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 192
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0
        
        
        extrinsic = cam[0].astype(np.float32)
        intrinsic = cam[1].astype(np.float32)

        return intrinsic, extrinsic
    
    def _get_views(self, idx, resolution, rng): 
        scene_id = self.scene_list[idx // self.num_seq]
        seq_id = idx % self.num_seq

        print('Scene ID:', scene_id)
        
        image_path = osp.join(self.ROOT, scene_id, 'images')
        depth_path = osp.join(self.ROOT, scene_id, 'depths')
        mask_path = osp.join(self.ROOT, scene_id, 'binary_masks')
        cam_path = osp.join(self.ROOT, scene_id, 'cams')
        pairs_path = osp.join(self.ROOT, scene_id, 'pair.txt')

        

        if not self.full_video:
            img_idxs = self.sample_pairs(pairs_path, seq_id)
        else:
            img_idxs = sorted(os.listdir(image_path))
            img_idxs = self.sample_frame_idx(img_idxs, rng, full_video=self.full_video)
        
        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.pop()
            impath = osp.join(image_path, im_idx)
            depthpath = osp.join(depth_path, im_idx.replace('.jpg', '.npy'))
            campath = osp.join(cam_path, im_idx.replace('.jpg', '_cam.txt'))
            maskpath = osp.join(mask_path, im_idx.replace('.jpg', '.png'))

            rgb_image = imread_cv2(impath)
            depthmap = np.load(depthpath)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

            mask = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED)/255.0
            mask = mask.astype(np.float32)

            mask[mask>0.5] = 1.0
            mask[mask<0.5] = 0.0

            mask = cv2.resize(mask, (depthmap.shape[1], depthmap.shape[0]), interpolation=cv2.INTER_NEAREST)
            kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
            mask = cv2.erode(mask, kernel, iterations=1)
            depthmap = depthmap * mask
            
            cur_intrinsics, camera_pose = self.load_cam_mvsnet(open(campath, 'r'))
            intrinsics = cur_intrinsics[:3, :3]
            camera_pose = np.linalg.inv(camera_pose)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='dtu',
                label=osp.join(scene_id, im_idx),
                instance=osp.split(impath)[1],
            ))

        return views


    
    
