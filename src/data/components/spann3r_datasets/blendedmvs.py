import os
import cv2
import numpy as np
import os.path as osp
from collections import deque

from src.dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset


class BlendMVS(BaseManyViewDataset):
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
    
    # def sample_pairs(self, pairs_path, rng, max_trials=10):
        
    #     cluster_lines = open(pairs_path).read().splitlines()
    #     image_num = int(cluster_lines[0])
    #     trials = 0
    #     while trials < max_trials:
    #         trials += 1
            
    #         sample_idx = rng.choice(image_num)
    #         ref_idx = int(cluster_lines[2 * sample_idx + 1])
    #         cluster_info =  cluster_lines[2 * sample_idx + 2].split()
    #         total_view_num = int(cluster_info[0])
            
    #         if total_view_num > self.num_frames-1:
    #             list_idx = ['{:08d}.jpg'.format(ref_idx)]

    #             sample_cidx = rng.choice(total_view_num, self.num_frames-1, replace=False)
    #             for cidx in sample_cidx:
    #                 list_idx.append('{:08d}.jpg'.format(int(cluster_info[2 * cidx + 1])))
                
    #             if rng.choice([True, False]):
    #                 list_idx.reverse()
                    
                
                
                    
    #             return list_idx
        
    #     return None

    def sample_pairs(self, pairs_path, rng, max_trials=10):
        # Read the pair.txt file and build a graph
        cluster_lines = open(pairs_path).read().splitlines()
        image_num = int(cluster_lines[0])

        # Build a dictionary mapping each image to its top neighbors
        image_neighbors = {}
        for i in range(image_num):
            ref_idx = int(cluster_lines[2 * i + 1])
            cluster_info = cluster_lines[2 * i + 2].split()
            total_view_num = int(cluster_info[0])
            neighbors = []
            for j in range(total_view_num):
                neighbor_idx = int(cluster_info[2 * j + 1])
                score = float(cluster_info[2 * j + 2])
                neighbors.append((neighbor_idx, score))
            image_neighbors[ref_idx] = neighbors

        trials = 0
        while trials < max_trials:
            trials += 1

            sample_idx = rng.choice(list(image_neighbors.keys()))
            ref_idx = sample_idx
            visited = set([ref_idx])
            candidate_images = [ref_idx]

            # Shuffle and add the top neighbors of ref_idx to the queue
            neighbors = image_neighbors.get(ref_idx, [])
            rng.shuffle(neighbors)  # Randomize the starting order of neighbors
            queue = deque(neighbors)

            while len(candidate_images) < self.num_frames and queue:
                neighbor_idx, score = queue.popleft()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    candidate_images.append(neighbor_idx)
                    # Add the neighbors of this neighbor to the queue
                    neighbors = image_neighbors.get(neighbor_idx, [])
                    rng.shuffle(neighbors)  # Introduce randomness in neighbor selection
                    queue.extend(neighbors)

            if len(candidate_images) >= self.num_frames:
                if rng.choice([True, False]):
                    candidate_images.reverse()
                # Format image filenames
                list_idx = ['{:08d}.jpg'.format(idx) for idx in candidate_images[:self.num_frames]]
                return list_idx

        return None

    def load_all_scenes(self, base_dir):
        
        if self.test_id is None:
            meta_split = osp.join(base_dir, f'{self.split}_list.txt')
            
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
    
    def load_cam_mvsnet(self, f, interval_scale=1):
        """ read camera txt file """
        # f = open(file)
        RT = np.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32)
        assert RT.shape == (4, 4)
        # RT = np.linalg.inv(RT)  # world2cam to cam2world

        K = np.loadtxt(f, skiprows=2, max_rows=3, dtype=np.float32)
        assert K.shape == (3, 3)

        return K, RT
    

    def _get_views(self, idx, resolution, rng, attempts=0):
        scene_id = self.scene_list[idx // self.num_seq]

        image_path = osp.join(self.ROOT, scene_id, 'blended_images')
        depth_path = osp.join(self.ROOT, scene_id, 'rendered_depth_maps')
        cam_path = osp.join(self.ROOT, scene_id, 'cams')
        pairs_path = osp.join(self.ROOT, scene_id, 'cams', 'pair.txt')

        if not self.full_video:
            img_idxs = self.sample_pairs(pairs_path, rng)
        
        else:
            img_idxs = sorted(os.listdir(image_path))
            img_idxs = img_idxs[::self.kf_every]
        
        if img_idxs is None:
            new_idx = rng.integers(0, self.__len__()-1)
            return self._get_views(new_idx, resolution, rng)

        
        imgs_idxs = deque(img_idxs)

        views = []

        max_depth_min = 1e8
        max_depth_max = 0.0
        max_depth_first = None  

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(image_path, im_idx)
            depthpath = osp.join(depth_path, im_idx.replace('.jpg', '.pfm'))
            campath = osp.join(cam_path, im_idx.replace('.jpg', '_cam.txt'))

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

            cur_intrinsics, camera_pose = self.load_cam_mvsnet(open(campath, 'r'))
            
            intrinsics = cur_intrinsics[:3, :3]
            camera_pose = np.linalg.inv(camera_pose)

            H, W = rgb_image.shape[:2]
            cx, cy = intrinsics[:2, 2].round().astype(int)
            min_margin_x = min(cx, W-cx)
            min_margin_y = min(cy, H-cy)
            
            if min_margin_x <= W/5 or min_margin_y <= H/5:
                new_idx = rng.integers(0, self.__len__()-1)
                return self._get_views(new_idx, resolution, rng)    
                        
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            
            input_depth_max = depthmap.max()
            if input_depth_max> max_depth_max:
                max_depth_max = input_depth_max
            
            if input_depth_max < max_depth_min:
                max_depth_min = input_depth_max
            
            if max_depth_first is None:
                max_depth_first = input_depth_max

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
                dataset='blendmvs',
                label=osp.join(scene_id, im_idx),
                instance=osp.split(impath)[1],
            ))
            
        if max_depth_max / max_depth_min > 100. or max_depth_max / max_depth_first > 10.:
            print(f"Warning: Depthmap range too large: {max_depth_max} {max_depth_min} {max_depth_first}")
            new_idx = rng.integers(0, self.__len__()-1)
            return self._get_views(new_idx, resolution, rng)
        
        return views

            







