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
from off_render import render_ray_provider
from models.core.Strivec4d import Space_vec

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
# Now suppose we have all the models we want, we also have the rays we want, we need to 
# render them right now....

@torch.no_grad()
def render_path(test_dataset, container, renderer, fids, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ray_type=0, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    data = render_loader.get_rays_for_render()
    rays_o, rays_d = data['rays_o'], data['rays_d']
    length = rays_o.shape[0]
    W, H = test_dataset.image_wh    
    all_rays = torch.cat([rays_o, rays_d], dim = -1)  # (h*w, 6)     
    near_far = test_dataset.near_far
    
    length_frame = len(container.loaded_models)
    
    for idx in tqdm(range(all_rays.shape[0])):

        tensorf = container.loaded_models[idx % length_frame]
        rays = all_rays[idx]
        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ray_type=ray_type, white_bg = white_bg, device=device, return_depth=True)
        
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
 
        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    return PSNRs


def render_test(args, test_dataset, container, fids, device):
    white_bg = test_dataset.white_bg
    ray_type = args.ray_type


    os.makedirs(f'{args.basedir}/render_path', exist_ok=True)
    save_dir = f'{args.basedir}/render_path'
    render_path(test_dataset, container, renderer, fids, save_dir,
                            N_vis=-1, N_samples=-1, white_bg = white_bg, ray_type=ray_type,device=device)


if __name__ == '__main__':
    
    args = comp_revise(args)    
    # create the render_ray_provider and 
    render_loader = render_ray_provider(args=args)
    device = render_loader.device
    container = create_model_container(args=args)
    render_test(args, render_loader, container, fids=[0, 1], device=device)

    
    # the shape of the all rays should be [101, h*w, 6]
    
    

    