import os
import cv2
import json
import numpy as np
import os.path as osp
from collections import deque

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset

class ArkitScene(BaseManyViewDataset):
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
        self.active_thresh= min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every

         # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq
    
    def load_all_scenes(self, base_dir, num_seq=200):
        
        if self.test_id is None:
            
            if self.split == 'train':
                scene_path = osp.join(base_dir, 'raw', 'Training')
            elif self.split == 'val':
                scene_path = osp.join(base_dir, 'raw', 'Validation')
            
            self.scene_path = scene_path
            self.scene_list = os.listdir(scene_path)
            
                
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")
            
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
                
            print(f"Test_id: {self.test_id}")
    
    def get_intrinsic(self, intrinsics_dir, frame_id, video_id):
        '''
        Nerfstudio
        '''
        intrinsic_fn = osp.join(intrinsics_dir, f"{video_id}_{frame_id}.pincam")

        if not osp.exists(intrinsic_fn):
            intrinsic_fn = osp.join(intrinsics_dir, f"{video_id}_{float(frame_id) - 0.001:.3f}.pincam")

        if not osp.exists(intrinsic_fn):
            intrinsic_fn = osp.join(intrinsics_dir, f"{video_id}_{float(frame_id) + 0.001:.3f}.pincam")

        _, _, fx, fy, hw, hh = np.loadtxt(intrinsic_fn)
        intrinsic = np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])
        return intrinsic
    
    def get_pose(self, frame_id, poses_from_traj):
        frame_pose = None
        if str(frame_id) in poses_from_traj:
            frame_pose = np.array(poses_from_traj[str(frame_id)])
        else:
            for my_key in poses_from_traj:
                if abs(float(frame_id) - float(my_key)) < 0.1:
                    frame_pose = np.array(poses_from_traj[str(my_key)])
        
        if frame_pose is None:
            print(f"Warning: No pose found for frame {frame_id}")
            
            return None

        assert frame_pose is not None
        frame_pose[0:3, 1:3] *= -1
        frame_pose = frame_pose[np.array([1, 0, 2, 3]), :]
        frame_pose[2, :] *= -1
        return frame_pose
    
    def traj_string_to_matrix(self, traj_string):
        """convert traj_string into translation and rotation matrices
        Args:
            traj_string: A space-delimited file where each line represents a camera position at a particular timestamp.
            The file has seven columns:
            * Column 1: timestamp
            * Columns 2-4: rotation (axis-angle representation in radians)
            * Columns 5-7: translation (usually in meters)
        Returns:
            ts: translation matrix
            Rt: rotation matrix
        """
        tokens = traj_string.split()
        assert len(tokens) == 7
        ts = tokens[0]
        # Rotation in angle axis
        angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        r_w_to_p, _ = cv2.Rodrigues(np.asarray(angle_axis))  # type: ignore
        # Translation
        t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
        extrinsics = np.eye(4, 4)
        extrinsics[:3, :3] = r_w_to_p
        extrinsics[:3, -1] = t_w_to_p
        Rt = np.linalg.inv(extrinsics)
        return (ts, Rt)
    
    def _get_views(self, idx, resolution, rng, attempts=0): 
        scene_id = self.scene_list[idx // self.num_seq]

        image_path = osp.join(self.scene_path, scene_id, 'lowres_wide')
        depth_path = osp.join(self.scene_path, scene_id, 'lowres_depth')
        intrinsics_path = osp.join(self.scene_path, scene_id, 'lowres_wide_intrinsics')
        pose_path = osp.join(self.scene_path, scene_id, 'lowres_wide.traj')

        if not osp.exists(image_path) or not osp.exists(depth_path) or not osp.exists(intrinsics_path) or not osp.exists(pose_path):
            print(f"Warning: Scene not found: {scene_id}")
            new_idx = rng.integers(0, self.__len__()-1)
            return self._get_views(new_idx, resolution, rng)

        img_idxs_ = [x for x in sorted(os.listdir(depth_path))]
        img_idxs_ = [x.split(".png")[0].split("_")[1] for x in img_idxs_]

        if len(img_idxs_) < self.num_frames:
            print(f"Warning: Not enough frames in {scene_id}, {len(img_idxs_)} < {self.num_frames}")
            new_idx = rng.integers(0, self.__len__()-1)
            return self._get_views(new_idx, resolution, rng)
        
        img_idxs = self.sample_frame_idx(img_idxs_, rng, full_video=self.full_video)
        imgs_idxs = deque(img_idxs)

        # Load trajectory
        poses_from_traj = {}
        with open(pose_path, "r", encoding="utf-8") as f:
            traj = f.readlines()

        for line in traj:
            poses_from_traj[f"{round(float(line.split(' ')[0]), 3):.3f}"] = np.array(
                self.traj_string_to_matrix(line)[1].tolist()
            )

        


        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            impath = osp.join(image_path, f'{scene_id}_{im_idx}.png')
            depthpath = osp.join(depth_path, f'{scene_id}_{im_idx}.png')

            camera_pose = self.get_pose(im_idx, poses_from_traj)
            intrinsics_ = self.get_intrinsic(intrinsics_path, im_idx, scene_id).astype(np.float32)

            if not osp.exists(impath) or not osp.exists(depthpath) or camera_pose is None:
                print (f"Warning: Image/Depth/Pose not found for {impath}")
                new_idx = rng.integers(0, self.__len__()-1)
                return self._get_views(new_idx, resolution, rng)

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0

            camera_pose = camera_pose.astype(np.float32)
            # gl to cv
            camera_pose[:, 1:3] *= -1.0

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath)
            
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
                dataset='arkit',
                label=osp.join(scene_id, im_idx),
                instance=osp.split(impath)[1],
            ))
        return views

if __name__ == "__main__":

    num_frames=5
    print('loading dataset')

    dataset = ArkitScene(split='train', ROOT="./data/arkit_lowres", resolution=224, num_seq=100, max_thresh=100)
            

        

      