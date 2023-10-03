import torch
import numpy as np
import json, random
import os
from tqdm.auto import tqdm
import sys
import time

#----------- Getting the arguments------------
from opt_hier import config_parser
args = config_parser()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

from utils import *
from models.init_net.run import get_density_pnts
# from models.init_geo.run import get_density_pnts
from pyhocon import ConfigFactory
from models.masked_adam import MaskedAdam

from renderer import *
from models.apparatus import *
from preprocessing.recon_prior_hier import gen_geo
import datetime
from models.core.Strivec4d import Space_vec
from dataLoader.dan_video import DanDataset
import time

renderer = OctreeRender_trilinear_fast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_image_perm(n_images):
    return torch.arange(0, n_images).long()

def train(args, train_dataset, render_dataset):
    # train_dataset = Dataset(args)
    # render_dataset = train_dataset
    # allrays, allmasks, allrgbs = train_dataset.gen_all_rays_at_time(0, resolution_level=1)
    # sphere_near, sphere_far = train_dataset.near_far_from_sphere(allrays[:3], allrays[3:6])
    # print("sphere near and far ------->", sphere_near, sphere_far)
    
    # Now do the initialization of the geometrys for a n_frames we want....
    pnts = get_density_pnts(args, train_dataset) if args.use_geo < 0 else None
    cluster_dict, geo = gen_geo(args, pnts)

    pnt_xyz = [geo_lvl[..., :3].cuda().contiguous() for geo_lvl in geo]
    local_dims =  torch.as_tensor(args.local_dims_init, device=device, dtype=torch.int64)
    lvl = len(local_dims)

    # draw the initialization of the points....

    logfolder = f'{args.basedir}/{args.expname}'
    pcl_dir = f'{logfolder}/pcl'
    iteration = 105
    # we need to draw the boxes once we complete the
    draw_hier_box(pnt_xyz, args.local_range, pcl_dir, iteration, rot_m=None)    
    
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ray_type = args.ray_type
    
    print("white_bg", white_bg, "near_far", train_dataset.near_far, "start training.......")

    
    # we have intialized the geo and the dataset, now we need to intialize the model.
    
    # init resolution
    update_AlphaMask_list = args.update_AlphaMask_list

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
   
    aabb = train_dataset.scene_bbox.to(device)
    # make aabb to float32, just to be careful to avoid the error
    aabb = aabb.type(torch.float32)
    
    print("=====>aabb<======:", aabb)
    print(" ======> local_dims:",args.local_dims_init)
    
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device, "geo": geo, "local_dims":args.local_dims_final})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, None, device,
            density_n_comp=args.n_lamb_sigma, appearance_n_comp=args.n_lamb_sh,
            app_dim=args.data_dim_color, near_far=near_far, shadingMode=args.shadingMode,
            alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift,
            distance_scale=args.distance_scale, pos_pe=args.pos_pe, view_pe=args.view_pe,
            fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct, local_dims=args.local_dims_init, cluster_dict=cluster_dict, geo=geo, args=args)
    
        # make the ray chunk, output the shape of the ray_chunk
        
        skip_zero_grad = args.skip_zero_grad
        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis, skip_zero_grad=skip_zero_grad > 0)
        
        if args.lr_decay_iters > 0:
            lr_fator = args.lr_decay_target_ratio**(1.0/args.lr_decay_iters)
        else: 
            args.lr_decay_iters = args.n_iters
            lr_factor = args.lr_decay_target_ratio**(1.0/args.n_iters)
            
        print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
        
        optimizer = MaskedAdam(grad_vars, betas=(0.9,0.99)) if skip_zero_grad else torch.optim.Adam(grad_vars, betas=(0.9,0.99))
        
        dim_lst = []
        # set upsample voxel dims
        if args.local_dims_trend is not None:
            assert args.upsamp_list is not None and len(args.upsamp_list) == len(args.local_dims_trend[0]), \
            "args.local_dims_trend and args.upsamp_list mismatch "
            for i in range(len(args.local_dims_trend)):
                level_dim_lst = []
                trend = torch.as_tensor(args.local_dims_trend[i], device="cuda")
                for j in range(len(args.local_dims_init[i])):
                    level_dim_lst.append(torch.floor(trend * args.local_dims_final[i][j] / args.local_dims_final[i][0] + 0.01).long())
                dim_lst.append(torch.stack(level_dim_lst, dim=-1))
        else:
            for i in range(len(args.local_dims_init)):
                level_dim_lst = []
                for j in range(len(args.local_dims_init[i])):
                    level_dim_lst.append((torch.floor(torch.exp2(torch.linspace(np.log2(args.local_dims_init[i][j]-1), np.log2(args.local_dims_final[i][j]-1), len(args.upsamp_list)+1))/2)*2 + 1).long()[1:] if args.upsamp_list is not None else None)
                dim_lst.append(torch.stack(level_dim_lst, dim=-1))
            
        print("dim_lst ============>", dim_lst)
        if args.upsamp_list is not None:
            local_dim_list = torch.stack(dim_lst, dim=1).tolist()
        else:
            local_dim_list = None

        torch.cuda.empty_cache()
        
        PSNRs,PSNRs_test = [],[0]  
        allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
        # we will need to torch cat the ray here, because we need to make it compatiable with the 4d recon mode
        if args.recon_mode == "4d" and not args.recon_mode == 'test_static':
            allrays = torch.cat(allrays, dim=0)
            allrgbs = torch.cat(allrgbs, dim=0)
        
        # should adjust the ray filtering technique here because it is not compatible with the 4d recon mode, we do not want to create too much rays that 
        # occupy all the memorys... so may be put then into the loop, but every time filter the rays will cost a lot of the time...
        # lets fix that tmr....
        # maybe change it to the masks and thus we can reduce the time of filtering and ray and also get the performance.
        # TODO: ZhenhuiLin 22/9/2023
        # after fixing that, I think you still need to use the pdb to check if every ray chunk we get is correct. Xixi
        # doing this, we can futher did it in the data module which could save the memory loading also...
        
        # print all the ray shape 0/ ok change this to the mask filtering...
        
        # print("before filtering allrays shape:", allrays.shape, "allrgbs shape:", allrgbs.shape, allrays.dtype)
        # if args.ray_type != 1:
        #     mask_filtered, tensoRF_per_ray = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
        #     allrays, allrgbs = allrays[mask_filtered], allrgbs[mask_filtered]
            
        # else:
        #     mask_filtered, tensoRF_per_ray = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
        #     allrays, allrgbs = allrays[mask_filtered], allrgbs[mask_filtered]
        
        # print("after filtering allrays shape:", allrays.shape, "allrgbs shape:", allrgbs.shape, allrays.dtype)
        # So it actually filtered out the ray which will not contribute to the training process...
        # start modifying from here, we need to sample according to the img_idx for 4d recon_mode, and 3d recon_mode
        # we use the original loading method
        
        if args.recon_mode == "test_static":
            trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
        
        Ortho_reg_weight = args.Ortho_weight
        print("initial Ortho_reg_weight", Ortho_reg_weight)

        L1_reg_weight = args.L1_weight_inital
        print("initial L1_reg_weight", L1_reg_weight)
        TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
        tvreg = TVLoss()
        print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

        pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)

        shrink_list = [update_AlphaMask_list[0]] if args.shrink_list is None else args.shrink_list
        filter_ray_list = [update_AlphaMask_list[1]] if args.filter_ray_list is None else args.filter_ray_list
        
        new_aabb = None
        cur_rot_step = False
        rot_step = args.rot_step
        upsamp_reset_list = args.upsamp_reset_list if args.upsamp_reset_list is not None else [0 for i in range(len(args.upsamp_list))]

        time_start = time.time()        

        # now we start training, Go Go Go!!
        image_perm = get_image_perm(train_dataset.n_images)
        
        init_4d  =True
        for iteration in pbar:

            img_idx = image_perm[iteration % len(image_perm)]
            # print("image_perm length:", len(image_perm) ,"iteration: ", iteration, "image index:", img_idx) # tested
            # it is true ...
            time_emb = None
            ray_sampler_start = time.time()
            if args.recon_mode == "test_static":
                ray_idx = trainingSampler.nextids()
                rays_train, rgb_train = allrays[ray_idx].to(device), allrgbs[ray_idx].to(device)

            elif args.recon_mode =="4d" or args.recon_mode == "3d":
                output = train_dataset.compute_random_rays(img_idx, args.batch_size, init=init_4d)
                rays_train, rgb_train, ray_idx, time_emb = output["rays"], output["rgbs"], output["ray_idx"], output["time_emb"]
                init_4d = False
            # print("ray sampling time:", time.time() - ray_sampler_start)
                
            # Set the whole things up and start training, check if the output is correct, if it is correct to deal with the static scene,
            # then we can extend to the dynamic scene....

            # get the start time
            time_start_render = time.time()
            rgb_map, weights, depth_map, rgbpers, ray_ids = renderer(rays_train, tensorf,time_emb=time_emb, \
                                                                    chunk=args.batch_size, 
                                                                    N_samples=-1, white_bg = white_bg, 
                                                                    ray_type=ray_type, device=device, 
                                                                    is_train=True, rot_step=cur_rot_step)            

            #------------------------------------------------  loss below ------------------------------------------------
            # all the loss
            # print("the time used for rendering:", time.time() - time_start_render)
            
            loss = torch.mean((rgb_map - rgb_train) ** 2)
            # loss
            total_loss = loss
            if Ortho_reg_weight > 0:
                loss_reg = tensorf.vector_comp_diffs()
                total_loss += Ortho_reg_weight*loss_reg
                # summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
            if L1_reg_weight > 0:
                loss_reg_L1 = tensorf.density_L1()
                total_loss += L1_reg_weight*loss_reg_L1
                # summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

            if TV_weight_density>0:
                TV_weight_density *= lr_factor
                loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
                total_loss = total_loss + loss_tv
                # summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
            if TV_weight_app>0:
                TV_weight_app *= lr_factor
                loss_tv = loss_tv + tensorf.TV_loss_app(tvreg)*TV_weight_app
                total_loss = total_loss + loss_tv
                # summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)
            if args.weight_rgbper > 0:
                total_loss += args.weight_rgbper * ((rgbpers - rgb_train[ray_ids]).pow(2).sum(-1) * weights.detach()).sum() / len(rgb_train)
                # summary_writer.add_scalar('train/rgbper', loss_reg_L1.detach().item(), global_step=iteration)
            # if not rot_step:
            optimizer.zero_grad(set_to_none=True) if skip_zero_grad else optimizer.zero_grad()
            if cur_rot_step:
                geo_optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if cur_rot_step:
                geo_optimizer.step()

            loss = loss.detach().item()
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))


            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_factor

            # Print the current values of the losses.
            if iteration % args.progress_refresh_rate == 0:
                pbar.set_description(
                    f'Iteration {iteration:05d}:'
                    + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                    + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                    + f' mse = {loss:.6f}'
                    + (f' rotx = {tensorf.pnt_rot[0,0] * 180 / np.pi:.6f}' if args.rotgrad > 0 else "")
                    + (f' roty = {tensorf.pnt_rot[0,1] * 180 / np.pi:.6f}' if args.rotgrad > 0 else "")
                    + (f' rotz = {tensorf.pnt_rot[0,2] * 180 / np.pi:.6f}' if args.rotgrad > 0 else "")
                )
                PSNRs = []


            if iteration % args.vis_every == 0 and args.N_vis!=0 and iteration > 0:
                if args.render_test:
                    # test_dataset
                    PSNRs_test = evaluation(render_dataset, tensorf, args, renderer, \
                                            f'{logfolder}/imgs_vis/', N_vis=args.N_vis, prtx=f'{iteration:06d}_',
                                            N_samples=-1, white_bg = white_bg, ray_type=ray_type, compute_extra_metrics=False)
                    # summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
                if args.render_train:
                    train_dataset_1 = DanDataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, rnd_ray=args.rnd_ray, args=args)
                    PSNRs_test = evaluation(train_dataset_1, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis, prtx=f'{iteration:06d}_', N_samples=-1, white_bg = white_bg, ray_type=ray_type, compute_extra_metrics=False)
            if update_AlphaMask_list is not None and iteration in update_AlphaMask_list:
                new_aabb = tensorf.updateAlphaMask()

            if iteration in shrink_list:
                assert new_aabb is not None, "can't shrink before first updateAlphaMask"
                tensorf.shrink(new_aabb)
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)
                
            if args.upsamp_list is not None and iteration in args.upsamp_list:
                print("upsampling......")
                local_dims = local_dim_list.pop(0)
                reset = upsamp_reset_list.pop(0) > 0
                #up_stage+=1
                #tensorf.up_stage = up_stage
                tensorf.upsample_volume_grid(local_dims, reset_feat=reset)

                if args.lr_upsample_reset:
                    print("reset lr to initial")
                    lr_scale = 1 #0.1 ** (iteration / args.n_iters)
                else:
                    lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale, skip_zero_grad = skip_zero_grad > 0)
                optimizer = MaskedAdam(grad_vars, betas=(0.9,0.99)) if skip_zero_grad else torch.optim.Adam(grad_vars, betas=(0.9,0.99))
                if args.rotgrad > 0:
                    geo_optimizer = torch.optim.Adam(tensorf.get_geoparam_groups(args.lr_geo_init * lr_scale), betas=(0.9,0.99), weight_decay=0.0)            
        
            if iteration % (len(image_perm)) == 0:
                image_perm = get_image_perm(train_dataset.n_images)
        
        tensorf.save(f'{logfolder}/{args.expname}.th')
        # render the pixel and then input them into the network, also with the positional encoding...
        
        # optimizer = MaskedAdam(grad_vars, betas=(0.9,0.99)) if skip_zero_grad else torch.optim.Adam(grad_vars, betas=(0.9,0.99))

@torch.no_grad()
def render_test(args, geo, test_dataset, train_dataset):
    white_bg = test_dataset.white_bg
    ray_type = args.ray_type

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    
    # print the kewargs before we update them......
    print("the kwargs before we update them:", kwargs)
    kwargs.update({'device': device})
    kwargs.update({'geo': geo, "args":args, "local_dims": args.local_dims_final})    
    kwargs.update({'step_ratio': args.step_ratio})
    tensorf = eval(args.model_name)(**kwargs)
    # import pdb; pdb.set_trace()
    
    tensorf.load(ckpt)


    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = DanDataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ray_type=ray_type,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/', N_vis=args.N_vis, N_samples=-1, white_bg = white_bg, ray_type=ray_type,device=device)

    if args.render_path:
        c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ray_type=ray_type,device=device)


def main(args):
    
    print("hello wooden")
    args = comp_revise(args)
    
    train_dataset = DanDataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, rnd_ray=False, args=args)
    print("...... finish loading the train dataset ...............")
    render_dataset = DanDataset(args.datadir, split='path', downsample=args.downsample_train, is_stack=True, rnd_ray=False, args=args)
    print("...... finish loading the render dataset ...............")
    
    if args.render_only == True:
        # construct the geo list here:
        print("the vox range:", args.vox_range)
        
        geo_lst = []
        for i in range(len(args.vox_range)):
            path = args.pointfile[:-4] + "_{}_vox".format(args.vox_range[i][0]) + ".npy"
            # read the path using numpys
            geo_lvl = np.load(path)
            geo_lst.append(torch.tensor(geo_lvl, dtype=torch.float32, device=device))
            
        geo = geo_lst
        render_test(args, geo, render_dataset, train_dataset)
    else:
        train(args, train_dataset, render_dataset)

if __name__ == '__main__':
    main(args)

# building the core of the strivec