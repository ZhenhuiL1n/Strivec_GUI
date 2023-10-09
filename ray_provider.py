import torch
import os
from tqdm.auto import tqdm
import json, random
import numpy as np
import sys
import time

## getting the arguments
from opt_hier import config_parser
args = config_parser()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

from utils import *
from models.init_net.run import get_density_pnts
# from models.init_geo.run import get_density_pnts
from pyhocon import ConfigFactory
from models.masked_adam import MaskedAdam

from torch.utils.data import Dataset
from renderer import *
from models.apparatus import *
from preprocessing.recon_prior_hier import gen_geo
import datetime
from models.core.Strivec4d import Space_vec
from dataLoader.dan_video import DanDataset
import time
import math
from dataLoader.ray_utils import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

n_test = 100
path = f'/home/zhenhui/Nerf-Projects/Strivec_GUI/debug'

class render_ray_provider:
    
    def __init__(self, args):
        self.N_vis = args.N_vis
        self.device = torch.device('cuda')
        self.frames_num = args.frames_num
        self.cameras_num = args.cameras_num
        self.fid = np.array([i for i in range(0, 50, 1)])
        self.time_emb_list = (self.fid / self.frames_num *2) - 0.95
        # set the parameters for the image
        self.white_bg = True
        self.near_far = [args.near, args.far]  
        
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 3)     
    
        self.unbounded_inward = False
        self.unbounded_inner_r = 0.0
        self.flip_y = False
        self.flip_x = False
        self.inverse_y = False
        self.ndc = False
        self.near_clip = None
        self.irregular_shape = False
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!
        self.read_meta(args=args)
        self.interpolated()
        

    def interpolated(self):

        rots = R.from_matrix(np.stack([self.pose0[:3, :3], self.pose1[:3, :3]]))
        slerp = Slerp([0, 1], rots)

        self.poses = []
        for i in range(n_test + 1):
            ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = slerp(ratio).as_matrix()
            pose[:3, 3] = (1 - ratio) * self.pose0[:3, 3] + ratio * self.pose1[:3, 3]
            
            self.poses.append(pose)
        # save the poses 
        
        c2ws = np.stack(self.poses)
        output = {
            'render': { 'cam2world': c2ws.tolist()              
          }
        }
        with open(os.path.join(path, 'render.json'), 'w') as f:
            json.dump(output, f, indent=4)
    
    def read_meta(self, args):
        path = args.datadir
        meta_path = os.path.join(path,'transforms_path.json')
        # read the json file
        with open(meta_path) as f:
            self.meta = json.load(f)
            self.camera_angle_x = self.meta['camera_angle_x']
            width = self.meta['width']
            height = self.meta['height']
            self.image_wh = [width, height]
            
            # what should be the fx and fy, cx and cy here
        self.focal_x = self.image_wh[0] * 0.5 / np.tan(0.5 * self.meta['camera_angle_x'])
        self.focal_y = self.image_wh[1] * 0.5 / np.tan(0.5 * self.meta['camera_angle_x'])
        fx, fy = self.focal_x, self.focal_y
        cx, cy = width/2,  height/2
        self.directions, _ = get_ray_directions(height, width, [fx, fy], [cx, cy])
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        #get the intrinsic
        self.intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
        self.pose0 = np.array(self.meta['cameras'][0]['transform_matrix'])
        self.pose1 = np.array(self.meta['cameras'][-1]['transform_matrix'])

    def get_rays_for_render(self, downscale=1):
        self.intrinsics = self.intrinsics * downscale
        
        all_ray_o = []
        all_ray_d = []
        
        for pose in self.poses:
            pose = pose @ self.blender2opencv
            pose = torch.tensor(pose).float()
            rays_o, rays_d = get_rays(self.directions, pose)
            
            all_ray_o += [rays_o]
            all_ray_d += [rays_d]

        all_ray_o = torch.stack(all_ray_o, dim=0)
        all_ray_d = torch.stack(all_ray_d, dim=0)
        
        print(all_ray_o.shape)
        print(all_ray_d.shape)

        data = {
            'rays_o': all_ray_o,
            'rays_d': all_ray_d,
            'H': self.image_wh[0],
            'W': self.image_wh[1],
        }
        
        return data
            
