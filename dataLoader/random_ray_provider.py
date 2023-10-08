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
from scipy.spatial.transform import Rotation as R
import torch


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot


    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])

class Random_ray_Dataset(Dataset):

    def __init__(self, args=None):
        self.N_vis = args.N_vis
        self.root_dir = args.datadir 
        self.device = torch.device('cuda')    
        self.frames_num = args.frames_nums
        self.cameras_num = args.cameras_num
        
        # get the time_emb_list can be later used for the gui
        self.fid = np.array([list(range(0, self.frames_num )) for _ in range(self.cameras_num)]).T.reshape(-1, 1).squeeze()  
        self.time_emb_list = (self.fid / self.frames_num *2) - 0.95       
        
        self.white_bg = True
        self.near_far = [args.near, args.far]

        self.define_transforms()
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # read the meta data from the path to get the image width, image height, focal_v, radius we need
        
        
         
    def generate_random_poses(self):
        pass
    def generate_random_ray(self):
        pass
    
    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
    