import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time
import os
# from torch_scatter import segment_coo
from torch_scatter import segment_coo
from torch.utils.cpp_extension import load
from plyfile import PlyData, PlyElement
parent_dir = os.path.dirname(os.path.abspath(__file__))

render_utils_cuda = load(
    name='render_utils_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
    verbose=True)

search_geo_cuda = load(
    name='search_geo_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/search_geo.cpp', 'cuda/search_geo.cu']],
    verbose=True)

search_geo_hier_cuda = load(
    name='search_geo_hier_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/search_geo_hier.cpp', 'cuda/search_geo_hier.cu']],
    verbose=True)

grid_sample_1d = load(
    name='grid_sample_1d',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/grid_sample_1d.cpp', 'cuda/grid_sample_1d.cu']],
    verbose=True)


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma * dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb


def filter_ray_by_points(xyz_sampled, geo, half_range):
    # xyz_dist = torch.abs(
    #     xyz_sampled[..., None, :] - geo[None, None, ..., :3])  # chunksize * raysampleN * 4096 * 3
    # mask_inrange = torch.all(xyz_dist <= self.local_range[None, None, None, :],
    #                          dim=-1)  # chunksize * raysampleN * 4096
    # mask_inrange = torch.any(mask_inrange.view(mask_inrange.shape[0], -1), dim=-1)
    mask_inrange = render_utils_cuda.filter_ray_by_points(xyz_sampled, geo.contiguous(), half_range) > 0
    return mask_inrange

def filter_ray_by_projection(rays_o, rays_d, geo, half_range):
    # xyz_dist = torch.abs(
    #     xyz_sampled[..., None, :] - geo[None, None, ..., :3])  # chunksize * raysampleN * 4096 * 3
    # mask_inrange = torch.all(xyz_dist <= self.local_range[None, None, None, :],
    #                          dim=-1)  # chunksize * raysampleN * 4096
    # mask_inrange = torch.any(mask_inrange.view(mask_inrange.shape[0], -1), dim=-1)
    # rays_d has to be unit
    tensoRF_per_ray = render_utils_cuda.filter_ray_by_projection(rays_o.contiguous(), rays_d.contiguous(), geo.contiguous(), torch.square(half_range))
    return tensoRF_per_ray

def filter_ray_by_cvrg(xyz_sampled,mask_inbox, units, xyz_min, xyz_max, tensoRF_cvrg_mask):
    # xyz_dist = torch.abs(
    #     xyz_sampled[..., None, :] - geo[None, None, ..., :3])  # chunksize * raysampleN * 4096 * 3
    # mask_inrange = torch.all(xyz_dist <= self.local_range[None, None, None, :],
    #                          dim=-1)  # chunksize * raysampleN * 4096
    # mask_inrange = torch.any(mask_inrange.view(mask_inrange.shape[0], -1), dim=-1)
    # rays_d has to be unit
    tensoRF_per_ray = search_geo_cuda.filter_ray_by_cvrg(xyz_sampled.contiguous(), mask_inbox.contiguous(), units.contiguous(), xyz_min.contiguous(), xyz_max.contiguous(), tensoRF_cvrg_mask)
    return tensoRF_per_ray

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume, mask_cache_thres=None):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(self.device)
        # mask = (self.alpha_volume >= mask_cache_thres).squeeze(0).squeeze(0)
        # self.register_buffer('mask', mask)
        # self.register_buffer('xyz2ijk_scale', (self.gridSize - 1) / self.aabbSize)
        # self.register_buffer('xyz2ijk_shift', -self.aabbSize[0] * self.xyz2ijk_scale)

    @torch.no_grad()
    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()
        if isinstance(inChanel, list):
            inChanel = sum(inChanel)
        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender_PE(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3) + inChanel  #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + inChanel
        self.viewpe = viewpe

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

def draw_ray(all_rays_vert, all_rays_edge, near, far):

    vertex = np.array([(all_rays_vert[i, 0], all_rays_vert[i, 1], all_rays_vert[i, 2]) for i in range(all_rays_vert.shape[0])],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    #vertex_o = np.array([(all_rays[i, 0]+all_rays[i, 3]*near, all_rays[i, 1]+all_rays[i, 4]*near, all_rays[i, 2]+all_rays[i, 5]*near) for i in range(all_rays.shape[0])],
    #                     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    #vertex_d = np.array([(all_rays[i, 0]+all_rays[i, 3]*far, all_rays[i, 1]+all_rays[i, 4]*far, all_rays[i, 2]+all_rays[i, 5]*far) for i in range(all_rays.shape[0])],
    #                     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    edge = np.array([(2 * a + 0, 2 * a + 1, 255, 165, 0) for a in range(all_rays_edge.shape[0])], dtype = [('vertex1', 'i4'),('vertex2', 'i4'),('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    
    ver = PlyElement.describe(vertex, 'vertex')
    edg = PlyElement.describe(edge, 'edge')
    #log = f'/home/gqk/cloud_tensoRF/log/indoor_scnenes'
    # log = f'/home/gqk/cloud_tensoRF/log/indoor_scnenes/rot_tensoRF'

    with open('{}/{}.ply'.format(log, f'ray'), mode='wb') as f:
        PlyData([ver, edg], text=True).write(f)


def draw_box(center_xyz, local_range, log, step, rot_m=None):
    sx, sy, sz = local_range[0], local_range[1], local_range[2]
    shift = torch.as_tensor([[sx, sy, sz],
                             [-sx, sy, sz],
                             [sx, -sy, sz],
                             [sx, sy, -sz],
                             [sx, -sy, -sz],
                             [-sx, -sy, sz],
                             [-sx, sy, -sz],
                             [-sx, -sy, -sz],
                             ], dtype=center_xyz.dtype, device=center_xyz.device)[None, ...]
                             
    #corner_xyz = center_xyz[..., None, :] + (torch.matmul(shift, rot_m) if rot_m is not None else shift)
    corner_xyz = center_xyz + (torch.matmul(shift, rot_m) if rot_m is not None else shift)

    corner_xyz = corner_xyz.cpu().detach().numpy().reshape(-1, 3)

    vertex = np.array([(corner_xyz[i,0], corner_xyz[i,1], corner_xyz[i,2]) for i in range(len(corner_xyz))],
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])


    edge = np.array([(8 * a + 0, 8 * a + 1, 255, 165, 0) for a in range(len(center_xyz))] +
                    [(8 * b + 1, 8 * b + 5, 255, 165, 0) for b in range(len(center_xyz))] +
                    [(8 * c + 5, 8 * c + 2, 255, 165, 0) for c in range(len(center_xyz))] +
                    [(8 * d + 2, 8 * d + 0, 255, 165, 0) for d in range(len(center_xyz))] +
                    [(8 * e + 6, 8 * e + 7, 0  , 255, 0) for e in range(len(center_xyz))] +
                    [(8 * f + 7, 8 * f + 4, 255, 0,   0) for f in range(len(center_xyz))] +
                    [(8 * g + 4, 8 * g + 3, 255, 165, 0) for g in range(len(center_xyz))] +
                    [(8 * h + 3, 8 * h + 6, 255, 165, 0) for h in range(len(center_xyz))] +
                    [(8 * i + 1, 8 * i + 6, 255, 165, 0) for i in range(len(center_xyz))] +
                    [(8 * j + 5, 8 * j + 7, 0,   0, 255) for j in range(len(center_xyz))] +
                    [(8 * k + 2, 8 * k + 4, 255, 165, 0) for k in range(len(center_xyz))] +
                    [(8 * l + 0, 8 * l + 3, 255, 165, 0) for l in range(len(center_xyz))]
                    ,dtype = [('vertex1', 'i4'),('vertex2', 'i4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    ver = PlyElement.describe(vertex, 'vertex')
    edg = PlyElement.describe(edge, 'edge')
    os.makedirs('{}/rot_tensoRF'.format(log), exist_ok=True)
    with open('{}/rot_tensoRF/{}.ply'.format(log, step), mode='wb') as f:
        PlyData([ver, edg], text=True).write(f)

def mask_split(tensor, indices):
    unique = torch.unique(indices)
    times = len(indices) // len(tensor)
    return [tensor.repeat(times, 1)[indices == i].cpu() for i in unique], unique

def draw_box_pca(center_xyz, pca_cluster, local_range, log, step, args, rot_m=None, subdir="rot_tensoRF"):
    corner_xyz_r = []
   
    for kn in range(local_range.shape[0]):
        sx, sy, sz = local_range[kn, 0], local_range[kn, 1], local_range[kn, 2]
        shift = torch.as_tensor([[sx, sy, sz],
                                 [-sx, sy, sz],
                                 [sx, -sy, sz],
                                 [sx, sy, -sz],
                                 [sx, -sy, -sz],
                                 [-sx, -sy, sz],
                                 [-sx, sy, -sz],
                                 [-sx, -sy, -sz],
                                 ], dtype=center_xyz.dtype, device=center_xyz.device)[None, ...]
        # import pdb;pdb.set_trace()
        corner_xyz_kn = center_xyz[kn, None, :] + (torch.matmul(shift, rot_m[kn].T) if rot_m is not None else shift)
        corner_xyz_kn = corner_xyz_kn.cpu().detach().numpy()#.reshape(-1, 3)
        corner_xyz_r.append(corner_xyz_kn)
    
    corner_xyz = np.array(corner_xyz_r).reshape(-1, 3).squeeze()
    vertex = np.array([(corner_xyz[i,0], corner_xyz[i,1], corner_xyz[i,2]) for i in range(len(corner_xyz))],
                             dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    
    edge = np.array([(8 * a + 0, 8 * a + 1, 255, 165, 0) for a in range(len(center_xyz))] +
                    [(8 * b + 1, 8 * b + 5, 255, 165, 0) for b in range(len(center_xyz))] +
                    [(8 * c + 5, 8 * c + 2, 255, 165, 0) for c in range(len(center_xyz))] +
                    [(8 * d + 2, 8 * d + 0, 255, 165, 0) for d in range(len(center_xyz))] +
                    [(8 * e + 6, 8 * e + 7, 0  , 255, 0) for e in range(len(center_xyz))] +
                    [(8 * f + 7, 8 * f + 4, 255, 0,   0) for f in range(len(center_xyz))] +
                    [(8 * g + 4, 8 * g + 3, 255, 165, 0) for g in range(len(center_xyz))] +
                    [(8 * h + 3, 8 * h + 6, 255, 165, 0) for h in range(len(center_xyz))] +
                    [(8 * i + 1, 8 * i + 6, 255, 165, 0) for i in range(len(center_xyz))] +
                    [(8 * j + 5, 8 * j + 7, 0,   0, 255) for j in range(len(center_xyz))] +
                    [(8 * k + 2, 8 * k + 4, 255, 165, 0) for k in range(len(center_xyz))] +
                    [(8 * l + 0, 8 * l + 3, 255, 165, 0) for l in range(len(center_xyz))]
                    ,dtype = [('vertex1', 'i4'),('vertex2', 'i4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        #np.savetxt(args.pointfile[:-4] + "_{}_{}_vox_tensorfs".format(args.datadir.split("/")[-1], args.vox_range[step][0]) + ".txt", center_xyz.cpu().numpy(), delimiter=";")
    # for k_cl in range(len(pca_cluster)):
    #   if len(pca_cluster[k_cl])>0:
    #     with open(args.pointfile[:-4] + "_{}_{}_vox_pca".format(args.datadir.split("/")[-1], args.vox_range[step][0]) + ".txt", 'a+') as ff:
    #       np.savetxt(ff, pca_cluster[k_cl][0], delimiter=";")
    ver = PlyElement.describe(vertex, 'vertex')
    edg = PlyElement.describe(edge, 'edge')
    os.makedirs('{}/{}'.format(log, subdir), exist_ok=True)
    
    with open('{}/{}/{}_pca.ply'.format(log, subdir, step), mode='wb') as f:
        PlyData([ver, edg], text=True).write(f)


def draw_hier_box(geo_xyz, local_range, log, step=0, rot_m=None):
    for l in range(len(geo_xyz)):
        center_xyz = geo_xyz[l]
        sx, sy, sz = local_range[l][0], local_range[l][1], local_range[l][2]
        shift = torch.as_tensor([[sx, sy, sz],
                                 [-sx, sy, sz],
                                 [sx, -sy, sz],
                                 [sx, sy, -sz],
                                 [sx, -sy, -sz],
                                 [-sx, -sy, sz],
                                 [-sx, sy, -sz],
                                 [-sx, -sy, -sz],
                                 ], dtype=center_xyz.dtype, device=center_xyz.device)[None, ...]

        # corner_xyz = center_xyz[..., None, :] + (torch.matmul(shift, rot_m) if rot_m is not None else shift)
        # print("(torch.matmul(shift, rot_m) if rot_m is not None else shift)", shift.shape, center_xyz.shape)
        corner_xyz = center_xyz[..., None, :] + (torch.matmul(shift, rot_m) if rot_m is not None else shift)

        corner_xyz = corner_xyz.cpu().detach().numpy().reshape(-1, 3)

        vertex = np.array([(corner_xyz[i, 0], corner_xyz[i, 1], corner_xyz[i, 2]) for i in range(len(corner_xyz))],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        edge = np.array([(8 * a + 0, 8 * a + 1, 255, 165, 0) for a in range(len(center_xyz))] +
                        [(8 * b + 1, 8 * b + 5, 255, 165, 0) for b in range(len(center_xyz))] +
                        [(8 * c + 5, 8 * c + 2, 255, 165, 0) for c in range(len(center_xyz))] +
                        [(8 * d + 2, 8 * d + 0, 255, 165, 0) for d in range(len(center_xyz))] +
                        [(8 * e + 6, 8 * e + 7, 0, 255, 0) for e in range(len(center_xyz))] +
                        [(8 * f + 7, 8 * f + 4, 255, 0, 0) for f in range(len(center_xyz))] +
                        [(8 * g + 4, 8 * g + 3, 255, 165, 0) for g in range(len(center_xyz))] +
                        [(8 * h + 3, 8 * h + 6, 255, 165, 0) for h in range(len(center_xyz))] +
                        [(8 * i + 1, 8 * i + 6, 255, 165, 0) for i in range(len(center_xyz))] +
                        [(8 * j + 5, 8 * j + 7, 0, 0, 255) for j in range(len(center_xyz))] +
                        [(8 * k + 2, 8 * k + 4, 255, 165, 0) for k in range(len(center_xyz))] +
                        [(8 * l + 0, 8 * l + 3, 255, 165, 0) for l in range(len(center_xyz))]
                        , dtype=[('vertex1', 'i4'), ('vertex2', 'i4'),
                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        ver = PlyElement.describe(vertex, 'vertex')
        edg = PlyElement.describe(edge, 'edge')
        os.makedirs('{}/rot_tensoRF'.format(log), exist_ok=True)
        with open('{}/rot_tensoRF/{}_lvl_{}.ply'.format(log, step, l), mode='wb') as f:
            PlyData([ver, edg], text=True).write(f)


def draw_sep_box_pca(raw_cluster, center_xyz, pca_cluster, local_range, log, step, args, rot_m=None, subdir="rot_tensoRF"):
    for kn in range(local_range.shape[0]):
        sx, sy, sz = local_range[kn, 0], local_range[kn, 1], local_range[kn, 2]
        shift = torch.as_tensor([[sx, sy, sz],
                                 [-sx, sy, sz],
                                 [sx, -sy, sz],
                                 [sx, sy, -sz],
                                 [sx, -sy, -sz],
                                 [-sx, -sy, sz],
                                 [-sx, sy, -sz],
                                 [-sx, -sy, -sz],
                                 ], dtype=center_xyz.dtype, device=center_xyz.device)[None, ...]
        # import pdb;pdb.set_trace()
        corner_xyz_kn = center_xyz[kn, None, :] + (torch.matmul(shift, rot_m[kn].T) if rot_m is not None else shift)

        corner_xyz = corner_xyz_kn.cpu().detach().numpy().reshape(-1, 3)
        vertex = np.array([(corner_xyz[i, 0], corner_xyz[i, 1], corner_xyz[i, 2]) for i in range(len(corner_xyz))], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        edge = np.array([(0, 1, 255, 165, 0)] +
                    [(1, 5, 255, 165, 0)] +
                    [(5, 2, 255, 165, 0)] +
                    [(2, 0, 255, 165, 0)] +
                    [(6, 7, 0, 255, 0)] +
                    [(7, 4, 255, 0, 0)] +
                    [(4, 3, 255, 165, 0)] +
                    [(3, 6, 255, 165, 0)] +
                    [(1, 6, 255, 165, 0)] +
                    [(5, 7, 0, 0, 255)] +
                    [(2, 4, 255, 165, 0)] +
                    [(0, 3, 255, 165, 0)]
                    , dtype=[('vertex1', 'i4'), ('vertex2', 'i4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # np.savetxt(args.pointfile[:-4] + "_{}_{}_vox_tensorfs".format(args.datadir.split("/")[-1], args.vox_range[step][0]) + ".txt", center_xyz.cpu().numpy(), delimiter=";")
    # for k_cl in range(len(pca_cluster)):
    #   if len(pca_cluster[k_cl])>0:
    #     with open(args.pointfile[:-4] + "_{}_{}_vox_pca".format(args.datadir.split("/")[-1], args.vox_range[step][0]) + ".txt", 'a+') as ff:
    #       np.savetxt(ff, pca_cluster[k_cl][0], delimiter=";")
        ver = PlyElement.describe(vertex, 'vertex')
        edg = PlyElement.describe(edge, 'edge')
        os.makedirs('{}/{}/sep/'.format(log, subdir), exist_ok=True)
        with open('{}/{}/sep/box_{:03d}_{}.ply'.format(log, subdir, kn, step), mode='wb') as f:
            PlyData([ver, edg], text=True).write(f)
        # print('{}/{}/sep/rawcluster_{:03d}_{}.txt'.format(log, subdir, kn, step))
        np.savetxt('{}/{}/sep/rawcluster_{:03d}_{}.txt'.format(log, subdir, kn, step), raw_cluster[kn].reshape(-1,3), delimiter=";")


''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval);
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None


class Raw2Alpha_randstep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha_randstep(density, shift, interval);
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_randstep_backward(exp, grad_back.contiguous(), interval), None, None


class GridSample1dVm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plane, line_1, line_2, line_3, xyz_sampled, aabb_low, aabb_high, units, lvl_units_l, local_range_l, local_dims_l, tensoRF_cvrg_inds_l, tensoRF_countl, tensoRF_topindx_l, geo_xyz_l, K_tensoRF_l, KNN):
        # plane: plane_len (1 or 3), n_component[l][0], (int)(local_dims[l][0] * self.args.vm_dim_factor) + self.plane_add, (int)(local_dims[l][1] * self.args.vm_dim_factor) + self.plane_add)))
        # line: len(self.geo_xyz[l]), n_component[l][0], local_dims[l][i] + self.line_add
        plane_out, line_out, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id, local_norm_xyz = grid_sample_1d.grid_sample_from_tensoRF(plane, line_1, line_2, line_3, xyz_sampled, aabb_low, aabb_high, units, lvl_units_l, local_range_l, local_dims_l, tensoRF_cvrg_inds_l, tensoRF_countl, tensoRF_topindx_l, geo_xyz_l, K_tensoRF_l, KNN)
        if plane.requires_grad:
            ctx.save_for_backward(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id)
            ctx.plane_dim = list(plane.shape)
            ctx.line_dim = list(line_1.shape)
        # print("line_out", line_out.shape, torch.max(line_out, 1)[0])
        # print("plane_out", plane_out.shape, torch.max(plane_out, 1)[0])
        return plane_out, line_out, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id, local_norm_xyz

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_planeout, grad_lineout, grad_local_gindx_s, grad_local_gindx_l, grad_local_gweight_s, grad_local_gweight_l, grad_local_kernel_dist, grad_final_tensoRF_id, grad_final_agg_id, grad_local_norm_xyz):
        local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id = ctx.saved_tensors
        plane_dim_lst = ctx.plane_dim
        line_dim_lst = ctx.line_dim
        planesurf_num, linesurf_num, component_num, res = plane_dim_lst[0], line_dim_lst[0], plane_dim_lst[1], plane_dim_lst[2]

        grad_plane, grad_line_1, grad_line_2, grad_line_3 = grid_sample_1d.grid_sample_from_tensoRF_backward(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id, grad_planeout.contiguous(), grad_lineout.contiguous(), planesurf_num, linesurf_num, component_num, res)
        # print("grad_plane", grad_plane.shape, torch.max(grad_plane, -1)[0])
        # print("grad_line_1", grad_line_1.shape, torch.max(grad_line_1, -1)[0])
        # print("grad_line_2", grad_line_2.shape, torch.max(grad_line_2, -1)[0])
        # print("grad_line_3", grad_line_3.shape, torch.max(grad_line_3, -1)[0])
        return grad_plane, grad_line_1, grad_line_2, grad_line_3, None, None, None, None, None, None, None, None, None, None, None, None, None



class GridSample1dVm_winds(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plane, line_1, line_2, line_3, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id):

        plane_out, line_out = grid_sample_1d.cal_w_inds(plane, line_1, line_2, line_3, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id)
        if plane.requires_grad:
            ctx.save_for_backward(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id)
            ctx.plane_dim = list(plane.shape)
            ctx.line_dim = list(line_1.shape)
        # print("line_out", line_out.shape, torch.max(line_out, 1)[0])
        # print("plane_out", plane_out.shape, torch.max(plane_out, 1)[0])
        return plane_out, line_out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_planeout, grad_lineout):
        local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id = ctx.saved_tensors
        plane_dim_lst = ctx.plane_dim
        line_dim_lst = ctx.line_dim
        planesurf_num, linesurf_num, component_num, res = plane_dim_lst[0], line_dim_lst[0], plane_dim_lst[1], plane_dim_lst[2]

        grad_plane, grad_line_1, grad_line_2, grad_line_3 = grid_sample_1d.grid_sample_from_tensoRF_backward(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id, grad_planeout.contiguous(), grad_lineout.contiguous(), planesurf_num, linesurf_num, component_num, res)
        # print("grad_plane", grad_plane.shape, torch.max(grad_plane, -1)[0])
        # print("grad_line_1", grad_line_1.shape, torch.max(grad_line_1, -1)[0])
        # print("grad_line_2", grad_line_2.shape, torch.max(grad_line_2, -1)[0])
        # print("grad_line_3", grad_line_3.shape, torch.max(grad_line_3, -1)[0])
        return grad_plane, grad_line_1, grad_line_2, grad_line_3, None, None, None, None, None


def raw2alpha_only(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)
    return alpha

def grid_xyz(center, local_range, local_dims):
    xs = torch.linspace(center[0]-local_range[0], center[0]+local_range[0], steps=local_dims[0]+1,
                        device=local_range.device,  dtype=local_range.dtype)
    ys = torch.linspace(center[1]-local_range[1], center[1]+local_range[1], steps=local_dims[1]+1,
                        device=local_range.device,  dtype=local_range.dtype)
    zs = torch.linspace(center[2]-local_range[2], center[2]+local_range[2], steps=local_dims[2]+1,
                        device=local_range.device,  dtype=local_range.dtype)
    xx = xs.view(-1, 1, 1).repeat(1, len(ys), len(zs))
    yy = ys.view(1, -1, 1).repeat(len(xs), 1, len(zs))
    zz = zs.view(1, 1, -1).repeat(len(xs), len(ys), 1)
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
            alpha, weights, T, alphainv_last,
            i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


def randomize_ray(rays_o, rgb_train, alpha, ijs, c2ws, focal, cent):
    b, _ = rays_o.shape
    xyshift = torch.rand(b, 1, 1, 2, device=rgb_train.device)
    inds = torch.round(xyshift).long()
    revers_mask = alpha[torch.arange(b, dtype=torch.int64, device=rgb_train.device), inds[:, 0, 0, 0], inds[:, 0, 0, 1]]
    # print("revers_mask", revers_mask.shape, torch.sum(revers_mask))
    xyshift[revers_mask] = 0.5
    xyshift = xyshift * 2 - 1
    rgbs = torch.nn.functional.grid_sample(rgb_train, xyshift, mode='bilinear', align_corners=True)[..., 0, 0]
    # print("rgbs", rgbs.shape)
    inds = ijs + xyshift[:,0,0,:]
    directions = torch.stack([(inds[...,0] - cent[0]) / focal, (inds[...,1] - cent[1]) / focal, torch.ones_like(inds[...,0])], -1)  # (H, W, 3)
    directions /= torch.norm(directions, dim=-1, keepdim=True)
    rays_d = torch.matmul(directions[:,None,:], c2ws).squeeze(1)
    return torch.cat([rays_o, rays_d], dim=1), rgbs