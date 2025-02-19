import os
import cv2
import json
import itertools
import numpy as np
import os.path as osp
from collections import deque

from src.dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


class Co3d(BaseManyViewDataset):
    def __init__(self, mask_bg=True, use_comb=True,
                 scene_class=None, scene_id=None,
                 num_seq=100, num_frames=5, 
                 min_thresh=10, max_thresh=100,
                 full_video=False, lb=0, ub=30,
                 kf_every=1, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
    
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh

        self.full_video = full_video
        self.kf_every = kf_every
        self.use_comb = use_comb

        self.scenes, self.scene_list = self.load_scene(scene_class, scene_id)

        self.combinations, self.num_seq = self.get_combinations(use_comb, lb, ub)
        self.invalidate = {scene: {} for scene in self.scene_list}
    

    def get_combinations(self, use_comb, lb, ub):

        if use_comb and not self.full_video:
            print('Using combinations')
            combinations = list(itertools.combinations(range(100), self.num_frames))
            combinations = [combo for combo in combinations if all(lb < abs(x-y) <= ub and abs(x-y) % 5 == 0 for x, y in zip(combo, combo[1:]))]
            num_seq = len(combinations)
            print('Number of sequences:', num_seq)
        else:
            combinations = None
            num_seq = self.num_seq
        
        return combinations, num_seq

    

    def load_scene(self, scene_class=None, scene_id=None):
        print('Loading scenes')
        with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
            scenes = json.load(f)
            
            if scene_class is not None:
                scenes = {k: v for k, v in scenes.items() if k == scene_class}
            else:
                scenes = {k: v for k, v in scenes.items() if len(v) > 0} # k is class (apple), v is corresponding list
            
            if scene_id is not None:
                scenes = {(k, k2): v2 for k, v in scenes.items() for k2, v2 in v.items() if k2 == scene_id}
            else:
                scenes = {(k, k2): v2 for k, v in scenes.items() 
                            for k2, v2 in v.items()} # k is class (apple), k2 is instance (110_13051_23361), v2 is list of image idx
        scene_list = list(scenes.keys())
        
        return scenes, scene_list
    
    def __len__(self):

        return len(self.scene_list) * self.num_seq
    
    def _get_views(self, idx, resolution, rng, attempts=0):
        obj, instance = self.scene_list[idx // self.num_seq]
        image_pool = self.scenes[obj, instance]

        if self.use_comb and not self.full_video:
            frame_idx = self.combinations[idx % len(self.combinations)]

            last = len(image_pool)-1
            imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in frame_idx]

        else:
            img_idx = range(0, len(image_pool))
            imgs_idxs = self.sample_frames(img_idx, rng)
        

        if resolution not in self.invalidate[obj, instance]:  # flag invalid images
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]
        
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))
       
        imgs_idxs = deque(imgs_idxs)

        max_depth_min = 1e8
        max_depth_max = 0.      
        max_depth_first = None  
        

        views = []

        while len(imgs_idxs) > 0: 
            im_idx = imgs_idxs.popleft()

            if self.invalidate[obj, instance][resolution][im_idx]:
                # search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break

            view_idx = image_pool[im_idx]

            impath = osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06d}.jpg')

            # load camera params
            input_metadata = np.load(impath.replace('jpg', 'npz'))
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(impath.replace('images', 'depths') + '.geometric.png', cv2.IMREAD_UNCHANGED)
            depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])

            if mask_bg:
                # load object mask
                maskpath = osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06d}.png')
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap
                            
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.appendleft(im_idx)
                continue

            if input_metadata['maximum_depth'] > max_depth_max:
                max_depth_max = input_metadata['maximum_depth']
            
            if input_metadata['maximum_depth'] < max_depth_min:
                max_depth_min = input_metadata['maximum_depth']
            
            if max_depth_first is None:
                max_depth_first = input_metadata['maximum_depth']
            
            
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='Co3d_v2',
                label=osp.join(obj, instance),
                instance=osp.split(impath)[1]
            ))

        if max_depth_max / max_depth_min > 100. or max_depth_max / max_depth_first > 10.:
            new_idx = rng.integers(0, self.__len__()-1)
            return self._get_views(new_idx, resolution, rng)
        
        return views







    


