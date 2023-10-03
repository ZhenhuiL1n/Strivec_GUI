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


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

class DanDataset(Dataset):
    
    def __init__(self, datadir, split='train', downsample=0.0, is_stack=False, N_vis=-1, rnd_ray=False, args=None):
        
        self.N_vis = N_vis
        self.root_dir = datadir 
        self.split = split
        self.is_stack = is_stack
        self.device = torch.device('cuda')     
        
        # get the number of frames from the args and generate the frame id from the camera numbers:
        
        self.frames_num = args.frames_num
        self.cameras_num = args.cameras_num
        # if args.recon_mode == '3d' or args.recon_mode == 'test_static':
        #     self.frames_num = 1
        if args.recon_mode == 'test_static' and self.split == 'path':
            self.cameras_num = args.render_cam_num
            print("============================>", self.cameras_num)
            
        self.n_images = self.frames_num * self.cameras_num
        # create the number of frame id according to the camera numbers:
        self.fid = np.array([list(range(0, self.frames_num )) for _ in range(self.cameras_num)]).T.reshape(-1, 1).squeeze()  
        self.time_emb_list = (self.fid / self.frames_num *2) - 0.95
        # set the parameters for the image
        self.white_bg = True
        self.near_far = [args.near, args.far]      
        self.downsample=downsample
        
        # define if we want to reconstruct the static scene or the dynamic scene:
        self.recon_mode = args.recon_mode
        print("self.rencon_mode:", self.recon_mode)
        if self.recon_mode == 'test_static' or self.recon_mode == '3d':
            self.recon_frame =args.recon_frame

        self.define_transforms()
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

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
        self.args = args
        
    def read_meta(self):
        
        # read the metadata from the json file
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        # initialize everything we need, rays, rgb, depth, mask, poses, intrinsics, etc.
        self.cameras_path = os.path.join(self.root_dir, self.meta['cameras_path'])
        self.image_paths = []
        self.mask_paths = []
        self.poses = []
        self.all_masks = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = [] 
        self.all_depth = []
        self.raw_poses = []
        self.intrinsics_all = []
        
        if self.split == 'train':
            w = self.meta["cameras"][0]['w']
            h = self.meta["cameras"][0]['h']
            self.img_wh = (w, h)
        elif self.split == 'path':
            w = self.meta['width']
            h = self.meta['height']
            self.img_wh = (w, h)
        # if downsample is 1.0, then we need to change the image size
        # TODO: add this downsample funstion later.
        
        # read all the images
        if self.split == 'train':
            for frame_i in tqdm(range(self.frames_num), desc=f'Loading images {self.split}: [{self.frames_num*self.cameras_num}]'):
                for camera_i in range(self.cameras_num):
                    image_folder_path = os.path.join(self.cameras_path, self.meta['cameras'][camera_i]['camera_id'])
                    mask_folder_path = image_folder_path.replace('cameras_bgs_' + f"{camera_i}", 'mask_' + f"{camera_i}")
                    mask_name = "sil." + f"{camera_i:02d}"  + "." + f"{frame_i:05d}" + ".png"
                    mask_path = os.path.join(mask_folder_path, mask_name)
                    self.mask_paths += [mask_path]
                    mask = self.transform(Image.open(mask_path))
                    mask = mask.view(1, -1).permute(1, 0)
                    bool_mask = mask > 0.5
                    self.all_masks += [bool_mask]
                    # now we aslo have the mask_path now, we can load the mask here.
                    image_name = "bgs_rgbd." + f"{camera_i:02d}"  + "." + f"{frame_i:05d}" + ".png"
                    image_path = os.path.join(image_folder_path, image_name)
                    self.image_paths += [image_path]
                    img = Image.open(image_path)
                    #TODO: downsample here to avoid the memory issue        
                    # if self.downsample!=1.0:
                    #     img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img)
                    if img.shape[0] == 4:
                        img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                    else:
                        img = img.view(3, -1).permute(1, 0)
                    self.all_rgbs += [torch.tensor(img)]

        if self.split == 'path':
            print("the length of the frames of the transform_path:", len(self.meta['cameras']))
            self.frames_num = 1
            for frame_i in tqdm(range(self.frames_num), desc=f'Loading images {self.split}: [{self.frames_num*self.cameras_num}]'):
                for camera_i in range(self.cameras_num):
                    image_file_path = os.path.join(self.cameras_path, self.meta['cameras'][camera_i]['file_path'])
                    image_path = image_file_path + '.png'
                    self.image_paths += [image_path]
                    img = Image.open(image_path)
                    #TODO: downsample here to avoid the memory issue
                    # if self.downsample!=1.0:
                    #     img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img)
                    if img.shape[0] == 4:
                        img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                    else:
                        img = img.view(3, -1).permute(1, 0)
                    self.all_rgbs += [torch.tensor(img)]    
    
        # generate the rays for all the images
        img_eval_interval = 1 if self.N_vis < 0 else len(self.frames_num) // self.N_vis
        idxs = list(range(0, self.cameras_num))
        if 'camera_angle_x' in self.meta:
            self.focal = self.img_wh[0] * 0.5 / np.tan(0.5 * self.meta['camera_angle_x'])

        # generate the rays for all the 8 cameras......
        for i in tqdm(idxs, desc=f'Generating rays {self.split} ({self.cameras_num})'):
            frame = self.meta['cameras'][i]
            w, h =frame.get('w', w), frame.get('h', h)
            fx, fy = frame.get('fl_x', self.focal), frame.get('fl_y', self.focal)
            cx, cy = frame.get('cx', w / 2), frame.get('cy', h / 2)

            self.img_wh = (w, h)
            self.directions, _ = get_ray_directions(h, w, [fx, fy], [cx, cy])  # (h, w, 3)
            self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
            self.intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
            self.intrinsics_all += [torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()]
            
            raw_pose = np.array(frame['transform_matrix'])
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.raw_poses += [torch.FloatTensor(raw_pose)]
            self.poses += [c2w]
            
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)                
            
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)  returns a list of the rays from all the cameras
            # if we stack them together, we can make it like [8, h*w, 6]
                

        # use all the parameters for the initialization of the geometry
        # TODO:downsample the intrinsic if we want to scale down....
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.poses = torch.stack(self.poses)
        self.pose_all = self.poses.to(self.device)
        
        # print("after all the generation and the rgb loading, the length of the all_rgbs:", len(self.all_rgbs), "\n \
        #       the length of the all_rays:", len(self.all_rays))
        
        if self.recon_mode == 'test_static':
            # select the first self.recon_frame*8 to (self.recon_frame+1)*8 rays and images
            # self.all_rgbs = self.all_rgbs[self.recon_frame*self.cameras_num : (self.recon_frame+1)*self.cameras_num]
            # self.all_masks = self.all_masks[self.recon_frame*self.cameras_num : (self.recon_frame+1)*self.cameras_num]
            # print("the length of the all_rgbs after selection:", len(self.all_rgbs))
            
            if not self.is_stack:
                print("using the not stack one, which is:", self.split) # it is the train dataloader....
                # so it is not stack, so we use the concatination.
                
                self.all_rgbs = self.all_rgbs[self.recon_frame*self.cameras_num : (self.recon_frame+1)*self.cameras_num]
                self.all_masks = self.all_masks[self.recon_frame*self.cameras_num : (self.recon_frame+1)*self.cameras_num]    
                self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
                self.all_masks = torch.cat(self.all_masks, 0)  # (len(self.meta['frames])*h*w, 3)
                # mask the all_rays here, we need to mask the rays according to the mask here.
                # import pdb; pdb.set_trace()
                self.all_rgbs = torch.cat(self.all_rgbs, 0) if self.all_rgbs else None

                # maybe filter the ray after the dvgo initialization.....
                # boradcast to -1, 6 shape
                # self.all_masks = self.all_masks.expand(-1, 6)
                # # use the mask to mask the rays and the rgb here
                # self.all_rays = self.all_rays[self.all_masks]
                # self.all_rgbs = self.all_rgbs[self.all_masks[:,:3]]

                
            else:
                print("using the not stack one, which is:", self.split) # it is the render dataloader...... 
                self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3) if self.all_rgbs else None
                # print the shape of the all_rays and all_rgbs here
                print("the shape of the all_rays:", self.all_rays.shape, "\n the shape of the all_rgbs:", self.all_rgbs.shape)
            
        # elif self.recon_mode == '4d':
        #     # if the recon_mode is 4d, we need to extend the rays/rgbds, maybe random pass. sample here....
        #     # also added time here.
        #     # stack the rgbs and the frame_id(fid) together
            
        #     # self.all_rgbs = torch.stack(self.all_rgbs, 0)
        #     # self.fid = torch.tensor(self.fid)
        #     # frame_id = self.fid.unsqueeze(1).unsqueeze(2)
        #     # frame_id = frame_id.expand(-1, self.all_rgbs.shape[1], 1)
        #     # self.all_rgbts = torch.cat((self.all_rgbs, frame_id),dim=2)  # rgbtc ,
        #     # need more rams here so we are not going to do things like that.
        #     # so I switch to the way of generating according to the image idx....
            
        #     raise NotImplementedError("4d recon mode dataloader is not implemented here, you should check the train function!")
        

    def compute_random_rays(self, img_idx, batch_size, init=False):
        # generate the random rays for the image idx and then we can embed the time for that. 
        # for rays for each image, we have 2073600 rays in total.
        
        if self.recon_mode == '3d':
            self.all_rgbs = self.all_rgbs[self.recon_frame*self.cameras_num : (self.recon_frame+1)*self.cameras_num]        
            self.all_masks = self.all_masks[self.recon_frame*self.cameras_num : (self.recon_frame+1)*self.cameras_num]
            
        if init == True:
            self.all_rays = torch.stack(self.all_rays, 0)  # (cameras_nums*frames_num,h*w, 6)
            self.all_rgbs = torch.stack(self.all_rgbs, 0) # (cameras_nums*frames_num,h*w, 3)                        
                                 
        self.train_image = self.all_rgbs[img_idx]
        # the train_rays should use the camera_idx instead of the img_idx
        ray_chunk_idx = img_idx%self.cameras_num
        self.train_rays = self.all_rays[ray_chunk_idx]
        
        trainingSampler = SimpleSampler(self.all_rays.shape[1], batch_size)
        ray_idx = trainingSampler.nextids()
        
        rays_train, rgb_train = self.train_rays[ray_idx].to(self.device), \
                                self.train_image[ray_idx].to(self.device)
        
        output = {
            'rays': rays_train,
            'rgbs': rgb_train,
            'ray_idx': ray_idx,
            'time_emb': self.time_emb_list[img_idx],
        }
        return output
        
    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
    
    def image_downsample(self):
        pass
    