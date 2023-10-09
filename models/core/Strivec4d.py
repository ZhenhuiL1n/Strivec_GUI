# Construct the Strivec4d here and try to visualize with some dummy tensors to check if it is correct or no
from .Strivec_helper import StrivecBase_hier
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import time
import os

from torch_scatter import segment_coo
from ..apparatus import *
from tqdm import tqdm
from .Strivec_helper import StrivecBase_hier

# add time embedding, split them into 3 tri-vectors and check how their aggregation works....

class Space_vec(StrivecBase_hier):
    # for a static scene, we wrap the Strivec to output the features....
    def __init__(self, aabb, gridSize, device, **kargs):
        super(Space_vec, self).__init__(aabb, gridSize, device, **kargs)
        # here I init to initialize what we need for a strivec_base_hier, but for the time frame....
        self.density_line = self.init_one_svd(self.geo, self.density_n_comp, self.local_dims, 0.2, device, self.lvl)
        self.app_line = self.init_one_svd(self.geo, self.app_n_comp, self.local_dims, 0.2, device, self.lvl)

        # basic liner mapping to the app_dim, for calculation of the color..
        self.basis_mat = torch.nn.ModuleList([torch.nn.Linear(self.app_n_comp[l][0], self.app_dim[l], bias=False).to(device) for l in range(len(self.app_dim))]).to(device)   
        self.theta_line, self.phi_line = None, None
    
    def compute_feature(self, rays_chunk, white_bg=True, is_train=False, ray_type=0, N_samples=-1, return_depth=0,
                        eval=False, rot_step=False, depth_bg=True):
        # get the feactures, the connection between this space plane to the mlp or any shading module...

        self.viewdirs = rays_chunk[:, 3:6]
        dir_gindx_s, dir_gindx_l, dir_gweight_l = None, None, None

        self.N, _ = rays_chunk.shape
        shp_rand = (self.args.shp_rand > 0) and (not eval)
        ji = (self.args.ji > 0) and (not eval)
    
        xyz_sampled, t_min, ray_id, step_id, shift, pnt_rmatrix = self.sample_ray_cvrg_cuda(rays_chunk[:, :3], self.viewdirs, 
                                                                                            use_mask=True, N_samples=N_samples, random=shp_rand, ji=ji)

        self.shift = shift
        
        # I will add things here, to decompose the xyz into x,y x,z y,z and then add t to each of them, and then do the search, hahahah so fucking clear....
        scale = 1.0
        self.matMode = torch.BoolTensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).cuda()  
        matMode = self.matMode    
        self.coordinate_plane = torch.stack((xyz_sampled[..., matMode[0]] * scale, xyz_sampled[..., matMode[1]] * scale, xyz_sampled[..., matMode[2]] * scale)).view(3, -1, 1, 2)
                
        local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id = self.sample_2_tensoRF_cvrg_hier(xyz_sampled, 
                                                                                                                                                pnt_rmatrix=pnt_rmatrix, 
                                                                                                                                                rotgrad=rot_step)
        sigma_feature = self.compute_densityfeature_geo(
                                                        local_gindx_s, 
                                                        local_gindx_l, 
                                                        local_gweight_s, 
                                                        local_gweight_l, 
                                                        local_kernel_dist, 
                                                        tensoRF_id, agg_id, 
                                                        sample_num=len(ray_id))    

        if shift is None:
            alpha = Raw2Alpha.apply(sigma_feature.flatten(), self.density_shift, self.stepSize * self.distance_scale).reshape(sigma_feature.shape)
        else:
            alpha = Raw2Alpha_randstep.apply(sigma_feature.flatten(), self.density_shift, (shift * self.distance_scale)[ray_id].contiguous()).reshape(sigma_feature.shape)
            
        weights, bg_weight = Alphas2Weights.apply(alpha, ray_id, self.N) #
        mask = weights > self.rayMarch_weight_thres
        
        if mask.any() and (~mask).any():
            if return_depth:
                step_id = step_id[mask]
            weights = weights[mask]
            ray_id = ray_id[mask]
            
            holder = torch.zeros((len(mask)), device=mask.device, dtype=torch.int64)
            holder[mask] = torch.arange(0, torch.sum(mask).cpu().item(), device=mask.device, dtype=torch.int64)
            for l in range(self.lvl):
                tensor_mask = mask[agg_id[l]]
                agg_id[l] = holder[agg_id[l][tensor_mask]]
                tensoRF_id[l] = tensoRF_id[l][tensor_mask]
                local_gindx_s[l] = local_gindx_s[l][tensor_mask]
                local_gindx_l[l] = local_gindx_l[l][tensor_mask]
                local_gweight_s[l] = local_gweight_s[l][tensor_mask]
                local_gweight_l[l] = local_gweight_l[l][tensor_mask]
                local_kernel_dist[l] = local_kernel_dist[l][tensor_mask]

        app_features = self.compute_appfeature_geo(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, tensoRF_id, agg_id,
                                                   sample_num=len(ray_id), dir_gindx_s=dir_gindx_s, dir_gindx_l=dir_gindx_l, dir_gweight_l=dir_gweight_l)
        
        # here the wierd thing just happened, the tmin shape is smaller than the ray_id shape,
        # but after the masking operation, the t_min[ray_id].shape become the same as the ray_id shape.
        
        return sigma_feature, app_features, t_min, ray_id, step_id, weights, bg_weight
    

    def compute_outputs(self, rays_chunk, time_emb=None, white_bg=True, is_train=False, ray_type=0, N_samples=-1, return_depth=0,
                        eval=False, rot_step=False, depth_bg=True):

        # white_bg = False
        feat_sigma, feat_color, t_min, ray_id, step_id, weights, bg_weight = self.compute_feature(rays_chunk=rays_chunk, white_bg=white_bg, is_train=is_train, 
                                                                ray_type=ray_type, N_samples=N_samples, return_depth=return_depth,
                                                                eval=eval, rot_step=rot_step, depth_bg=depth_bg)
        
        rgb = self.renderModule(None, self.viewdirs[ray_id], feat_color)
        rgb_map = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id,
            out=torch.zeros([self.N, 3], device=weights.device, dtype=torch.float32),
            reduce='sum')
        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map += (bg_weight.unsqueeze(-1))

        if return_depth:
            with torch.no_grad():
                z_val = t_min[ray_id] + step_id * (self.shift[ray_id] if self.shift is not None else self.stepSize)
                depth_map = segment_coo(
                    src=(weights.unsqueeze(-1) * z_val[..., None]),
                    index=ray_id,
                    out=torch.zeros([self.N, 1], device=weights.device, dtype=torch.float32),
                    reduce='sum')[..., 0]
                depth_map += (bg_weight * 1000) if depth_bg else 0
        else:
            depth_map = None
        rgb_map = rgb_map.clamp(0, 1)
        
        return rgb_map, depth_map, rgb, ray_id, weights # rgb, sigma, alpha, weight, bg_weight