import torch
import numpy as np
import json, random
import os
from tqdm.auto import tqdm
import sys
import time

from opt_hier import config_parser
args = config_parser()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

from renderer import *
from ray_provider import render_ray_provider
from models.core.Strivec4d import Space_vec
from gui import NeRFGUI
from dataLoader.ray_utils import *

renderer = OctreeRender_trilinear_fast

class ModelContainer:
    def __init__(self, args) -> None:
        self.loaded_models = []
        self.device = torch.device('cuda')
        self.args = args
        
    def add_model(self, model):
        self.loaded_models.append(model)
    
    def load(self, args, fid, geo, local_dims):
        path = os.path.join(args.basedir, 'frame_'+f'{fid}', 'frame_'+f'{fid}.th')
        ckpt = torch.load(path, map_location=self.device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': self.device})
        kwargs.update({'geo': geo, "args":args, "local_dims": args.local_dims_final})    
        kwargs.update({'step_ratio': args.step_ratio})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
        self.add_model(tensorf)
        
def add_dim(obj, times, div=False):
    if obj is None:
        return obj
    elif div:
        obj_lst = []
        for j in range(times):
            leng = len(obj) // times
            obj_lst.append([obj[i] for i in range(j*leng, j*leng+leng)])
        return obj_lst
    else:
        assert len(obj) % times == 0, "{} should be times of 3".format(obj)
        obj_lst = []
        for j in range(len(obj) // times):
            obj_lst.append([obj[j*times+i] for i in range(times)])
        return obj_lst

def comp_revise(args):
    args.local_dims_trend = add_dim(args.local_dims_trend, len(args.max_tensoRF), div=True)
    args.local_range = add_dim(args.local_range, 3)
    args.local_dims_init = add_dim(args.local_dims_init, 3)
    args.local_dims_final = add_dim(args.local_dims_final, 3)
    args.n_lamb_sigma = add_dim(args.n_lamb_sigma, 1)
    args.n_lamb_sh = add_dim(args.n_lamb_sh, 1)
    args.vox_range = add_dim(args.vox_range, 3)
    return args

def create_model_container(args):
    logfolder = f'{args.basedir}/{args.expname}'
    device = torch.device('cuda')
    # init the renderfile:
    os.makedirs(f'{logfolder}/time_render', exist_ok=True)
    
    ## check the fid and the times for all the models, query the model and the prior geo
    ## and use the model to render....
    geo_all = []
    for i in range(args.frames_num):
        path_to_geo = os.path.join(args.basedir, 'frame_'+f'{i}')
        geo_lst = []
        for i in range(len(args.vox_range)):
            path = path_to_geo + "/geo_{}_vox".format(args.vox_range[i][0]) + ".npy"
            # read the path using numpys
            geo_lvl = np.load(path)
            geo_lst.append(torch.tensor(geo_lvl, dtype=torch.float32, device=device))
            
        geo = geo_lst
        geo_all.append(geo)
        
    ## also load all the models
    Container = ModelContainer(args=args)
    for i in range(args.frames_num):
        Container.load(args=args, fid=i, geo=geo_all[i], local_dims=args.local_dims_final)
    
    return Container


# adjust the function below to interact with the gui......

@torch.no_grad()
def render_gui(cam_pose, cam_intrinsics, H, W, fid, args, N_samples=-1, white_bg=False, ray_type=0, device='cuda'):

    # get the height, width and the intrinsics

    length_frame = len(container.loaded_models)

    if fid > length_frame:
        fid = length_frame - 1
        
    tensorf = container.loaded_models[fid]
    height, width = H, W
    fx, fy, cx, cy = cam_intrinsics[0], cam_intrinsics[1], cam_intrinsics[2], cam_intrinsics[3]
    directions, _ = get_ray_directions(height, width, [fx, fy], [cx, cy])
    cam_pose = torch.tensor(cam_pose, dtype=torch.float32, device=device)
    directions = directions.to(device)
    rays_o, rays_d = get_rays(directions, cam_pose)
    rays = torch.cat([rays_o, rays_d], dim = -1) 
    near_far = [args.near, args.far]
    
    rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                    ray_type=ray_type, white_bg = white_bg, device=device, return_depth=True)
    
    rgb_map = rgb_map.clamp(0.0, 1.0)
    rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

    depth_map_for_save, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

    rgb_map_for_save = (rgb_map.numpy() * 255).astype('uint8')
    
    saving_path = f'/home/zhenhui/Nerf-Projects/Strivec_GUI/debug'
    
    #save the rgb and depth map to the saving path
    save_img = np.concatenate((rgb_map_for_save, depth_map_for_save), axis=1)
    imageio.imwrite(f'{saving_path}/rgb_depth.png', save_img)
    
    print("rendering running: saving image to:", )
    
    output = {
        'depth': depth_map,
        'image': rgb_map
       }
    
    return output

    
class UI_Renderer(object):
    
    def __init__(self) -> None:
        pass


if __name__ == '__main__':
    
    args = comp_revise(args)    
    # create the render_ray_provider and 
    render_loader = render_ray_provider(args=args)
    container = create_model_container(args=args)
    gui = NeRFGUI(args, render_gui)
    gui.render()
    # the shape of the all rays should be [101, h*w, 6]
    
    

    