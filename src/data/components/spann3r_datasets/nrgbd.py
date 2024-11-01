import os
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque

from src.dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


class NRGBD(BaseManyViewDataset):
    def __init__(self, num_seq=1, num_frames=5, 
                 min_thresh=10, max_thresh=100, 
                 test_id=None, full_video=False, 
                 tuple_path=None, seq_id=None,
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
        self.seq_id = seq_id

         # load all scenes
        self.load_all_tuples(tuple_path)
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_path):
        if tuple_path is not None:
            with open(tuple_path) as f:
                self.tuple_list = f.read().splitlines()
        
        else:
            self.tuple_list = None
    
    def load_all_scenes(self, base_dir):
        
        scenes = os.listdir(base_dir)
        
        if self.test_id is not None:
            self.scene_list = [self.test_id]
        
        else:
            self.scene_list = scenes
        
        print(f"Found {len(self.scene_list)} sequences in split {self.split}")
    
    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if 'nan' in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]]
                poses.append(pose_floats)

        return np.array(poses, dtype=np.float32), valid


    
    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]
        
        else:
            scene_id = self.scene_list[idx // self.num_seq]

            num_files = len(os.listdir(os.path.join(self.ROOT, scene_id, 'images')))
            img_idxs = [f'{i}' for i in range(num_files)]
            img_idxs = self.sample_frame_idx(img_idxs, rng, full_video=self.full_video)


        fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        
        posepath = osp.join(self.ROOT, scene_id, f'poses.txt')
        camera_poses, valids = self.load_poses(posepath)

        imgs_idxs = deque(img_idxs)
        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(self.ROOT, scene_id, 'images', f'img{im_idx}.png')
            depthpath = osp.join(self.ROOT, scene_id, 'depth',f'depth{im_idx}.png')

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap>10] = 0
            depthmap[depthmap<1e-3] = 0

            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            camera_pose = camera_poses[int(im_idx)]
            # gl to cv
            camera_pose[:, 1:3] *= -1.0

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='nrgbd',
                label=osp.join(scene_id, im_idx),
                instance=osp.split(impath)[1],
            ))

        return views

            





        