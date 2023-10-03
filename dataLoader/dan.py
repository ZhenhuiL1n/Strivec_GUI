from torch.utils.data import Dataset
from PIL import Image
import json
import os
from torchvision import transforms as T
import numpy as np
import torch
import cv2
from tqdm import tqdm
from .ray_utils import *

import math

def calculate_focal_length(diagonal, camera_angle):
    focal_length = diagonal / (2 * math.tan(camera_angle / 2))
    return focal_length

def calculate_diagonal(width, height):
    diagonal = math.sqrt(width**2 + height**2)
    return diagonal


class DanDataset(Dataset):
    
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, frame_i=0, rnd_ray=False, args=None):
        self.N_vis = N_vis
        self.root_dir = datadir 
        self.split = split
        self.is_stack = is_stack
        self.device = torch.device('cuda')     

        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()
        self.frame_i = frame_i

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0,6.0]      
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 3)
        self.downsample=downsample

        self.unbounded_inward = False
        self.unbounded_inner_r = 0.0
        self.flip_y = False
        self.flip_x = False
        self.inverse_y = False
        self.ndc = False
        self.near_clip = None
        self.irregular_shape = False
        self.args = args
        
        # get the number of frames from the args and generate the frame id from the camera numbers:
        self.frames_num = args.frames_num
        self.cameras_num = args.cameras_num
   
        # create the number of frame id according to the camera numbers:
        fid = np.array([list(range(0, self.frames_num )) for _ in range(self.cameras_num)]).T.reshape(-1, 1).squeeze()  
        print("fid:", fid)

    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        n_cameras = len(self.meta['frames'])
        n_frames = self.meta['n_frames']
        h = self.meta['h']
        w = self.meta['w']

        self.img_wh = (w, h)
        if 'camera_angle_x' in self.meta:
            self.focal = self.img_wh[0] * 0.5 / np.tan(0.5 * self.meta['camera_angle_x'])
        
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = [] 
        self.all_depth = []
        self.raw_poses = []
        self.downsample=1.0
        self.intrinsics_all = []
        # ray directions for all pixels, same for all images (same H, W, focal)
        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(self.meta["frames"])})'):
            frame = self.meta['frames'][i]
            fx, fy = frame.get('fl_x', self.focal), frame.get('fl_y', self.focal)
            cx, cy = frame.get('cx', w / 2), frame.get('cy', h / 2)

            self.directions, _ = get_ray_directions(h, w, [fx, fy], [cx, cy])  # (h, w, 3)
            self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
                    
            # check the code how it works haha
            self.intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
            self.intrinsics_all += [torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()]
            
            raw_pose = np.array(frame['transform_matrix'])
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.raw_poses += [torch.FloatTensor(raw_pose)]
            self.poses += [c2w]
            
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        # read images
        if self.split == 'train':
            for i_frame in tqdm(list(range(n_frames)), desc=f'Loading images {self.split} ({n_frames * n_cameras})'):
                for i_camera in range(n_cameras):
                    image_path = os.path.join(self.root_dir, self.meta['frames'][i_camera]['file_path'] + '.png')
                    self.image_paths += [image_path]
                    img = Image.open(image_path)
                    # if self.downsample!=1.0:
                    #     img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img)  # (4, h, w)
                    if img.shape[0] == 4:
                        img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                    else:
                        img = img.view(3, -1).permute(1, 0)
                    self.all_rgbs += [torch.tensor(img)]
    
            # self.all_rgbs = self.all_rgbs[self.frame_i]
        elif self.split == 'test':
            for i_frame in tqdm(list(range(n_frames)), desc=f'Loading images {self.split} ({n_frames * n_cameras})'):
                for i_camera in range(n_cameras):
                    image_path = os.path.join(self.root_dir, self.meta['frames'][i_camera]['file_path'] + '.png')
                    self.image_paths += [image_path]
                    img = Image.open(image_path)
                    # if self.downsample!=1.0:
                    #     img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img)
                    if img.shape[0] == 4:
                        img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                    else:
                        img = img.view(3, -1).permute(1, 0)
                    self.all_rgbs += [torch.tensor(img)]
        
        elif self.split =='path':
            
            
            w = self.meta['w']
            h = self.meta['h']
            camera_angle_x = self.meta['camera_angle_x']
            diagonal = calculate_diagonal(w, h)
            focal = calculate_focal_length(diagonal, camera_angle_x)
            fx, fy = focal, focal
            cx, cy = w / 2, h / 2
            self.directions, _ = get_ray_directions(h, w, [fx, fy], [cx, cy])  # (h, w, 3)
            
            for i_frame in tqdm(list(range(n_frames)), desc=f'Loading images {self.split} ({n_frames * n_cameras})'):
                for i_camera in range(n_cameras):
                    image_path = os.path.join(self.root_dir, self.meta['frames'][i_camera]['file_path'] + '.png')
                    self.image_paths += [image_path]
                    img = Image.open(image_path)
                    # if self.downsample!=1.0:
                    #     img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img)
                    if img.shape[0] == 4:
                        img = img.view(4, -1).permute(1, 0)
                        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                    else:
                        img = img.view(3, -1).permute(1, 0)
                    self.all_rgbs += [torch.tensor(img)]
        
        
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.poses = torch.stack(self.poses)
        self.pose_all = self.poses.to(self.device)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            # self.all_rgbs = torch.stack([torch.cat(rgbs, 0) for rgbs in self.all_rgbs]) if self.all_rgbs else None # (n_frames*n_cameras*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) if self.all_rgbs else None
            print("=====> training all_rgbs shape:", self.all_rgbs.shape)
            
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            # self.all_rgbs = torch.stack([
            #         torch.stack(rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)
            #         for rgbs in self.all_rgbs
            # ]) if self.all_rgbs else None # (n_frames,n_cameras,h,w,3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3) if self.all_rgbs else None
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)


    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}


        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        # print('ray:', sample['rays'][0], sample['rays'][-1])
            
        return sample