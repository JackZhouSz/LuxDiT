import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import glob
import cv2
import imageio.v3 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from braceexpand import braceexpand
import itertools
import random
from omegaconf import OmegaConf, ListConfig
from .rendering_utils import (
    envmap_vec, envmap_xfm, rotate_y, uv_mesh, ray2zdepth,
    get_ideal_ball, get_ref_vector, envmap_chrome_ball,
    normalize_depth, cam_intrinsics, read_image, get_cam_matrix, safe_normalize,
    latlong_to_cubemap_torch, cubemap_sample_torch,
    rgb2srgb, reinhard, 
)
from src.utils.util import instantiate_from_config


class LightningLoader(torch.utils.data.DataLoader):
    def __init__(self, data_config, batch_size=1, num_workers=4, **kwargs):
        self.data_config = data_config
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._dataset = instantiate_from_config(self.data_config)

        # distributed_sampler = DistributedSampler(self._dataset, shuffle=True)

        super().__init__(
            self._dataset, 
            # sampler=distributed_sampler,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def train_dataloader(self):
        return self


# put all cubemap functions on CPU, no need for nvdiffrast
class EnvRenderingDataset(Dataset):

    def __init__(
        self,
        hdri_list,
        cam_phi_range_inp=(0, 0), # azimuth
        cam_phi_range_out=(0, 0), # azimuth
        cam_theta_range=(-10, 15), # elevation
        env_fov_range=(60, 120),
        env_scale_range=(0.7, 1.4),
        env_random_roll=True,
        env_random_flip=True,
        ball_resolution=[512, 512],
        pers_resolution=[480, 720],
        erp_resolution=[384, 768],
        out_format='CHW',
        normalize_image=True, # [0, 1] --> [-1, 1]
        sampling_weight=1,
        ball_random_cam=False,
        ball_dir_condition='normal',
        erp_random_cam=False,
        erp_random_roll=False,
        use_cam_space_dir=True,
        inp_type='pers',
        out_type='ball',
        process_idx=None,
        world_size=None,
        random_swap=0,
        append_features=None,
        **kwargs
    ):
        if not isinstance(hdri_list, str):
            hdri_list = [list(braceexpand(urls)) for urls in hdri_list]
            hdri_list = list(itertools.chain.from_iterable(hdri_list))
        else:
            if hdri_list.endswith('.txt'):
                hdri_list = np.loadtxt(hdri_list, dtype=str).tolist()
            else:
                hdri_list = list(braceexpand(hdri_list))

        self.hdri_list = []
        for hdri in hdri_list:
            self.hdri_list.extend(sorted(glob.glob(hdri)))

        if process_idx is not None and world_size is not None:
            print(f"Env data shard: {process_idx} / {world_size}")
            self.hdri_list = self.hdri_list[process_idx::world_size]
        
        self.cam_phi_range_inp = [np.deg2rad(deg) for deg in cam_phi_range_inp]
        self.cam_phi_range_out = [np.deg2rad(deg) for deg in cam_phi_range_out]
        self.cam_theta_range = [np.deg2rad(deg) for deg in cam_theta_range]
        self.env_fov_range = [np.deg2rad(deg) for deg in env_fov_range]
        self.env_scale_range = env_scale_range
        self.env_random_roll = env_random_roll
        self.env_random_flip = env_random_flip

        self.ball_resolution = ball_resolution
        self.pers_resolution = pers_resolution
        self.erp_resolution = erp_resolution

        self.out_format = out_format
        self.normalize_image = normalize_image
        self.use_cam_space_dir = use_cam_space_dir

        self.inp_type = inp_type
        self.out_type = out_type

        self.ball_dir_condition = ball_dir_condition
        self.ball_random_cam = ball_random_cam

        self.erp_random_cam = erp_random_cam
        self.erp_random_roll = erp_random_roll

        self.random_swap = random_swap
        self.sampling_weight = sampling_weight
        self.append_features = append_features

        if self.append_features is not None:
            self.append_features = dict(self.append_features)
            for k, v in self.append_features.items():
                if isinstance(v, ListConfig):
                    self.append_features[k] = np.array(v)

        if 'pers' in [self.inp_type, self.out_type]:
            self.env_uv = uv_mesh(self.pers_resolution[1], self.pers_resolution[0])
        if 'ball' in [self.inp_type, self.out_type]:
            assert self.ball_resolution[0] == self.ball_resolution[1]
            self.normal_map, self.mask = get_ideal_ball(self.ball_resolution[0], flip_x=False)
            self.nrm_chrome_ball = get_ref_vector(self.normal_map, torch.tensor([0, 0, 1], dtype=torch.float32))
        if 'erp' in [self.inp_type, self.out_type]:
            self.erp_vec = envmap_vec(self.erp_resolution).flip(1)
            self.erp_vec = torch.roll(self.erp_vec, self.erp_vec.shape[1] // 2, dims=1) # make pers in the center

    def __len__(self):
        return len(self.hdri_list)

    def process_hdr(self, hdr, max_ldr=16, max_log=10000):
        rgb_ldr = rgb2srgb(reinhard(hdr, max_point=max_ldr)).clip(0, 1) # LDR
        rgb_log = rgb2srgb(torch.log1p(hdr) / np.log1p(max_log)).clip(0, 1) # 0-10000 scale
        if self.normalize_image:
            rgb_ldr = 2 * rgb_ldr - 1
            rgb_log = 2 * rgb_log - 1
        return rgb_ldr, rgb_log

    def get_pers_image(self, c2w, cubemap, rot_azimuth=0):
        c2w_pers = c2w
        if rot_azimuth != 0:
            c2w_pers = rotate_y(rot_azimuth) @ c2w
        
        env_fov = random.uniform(*self.env_fov_range)
        intrinsic = cam_intrinsics(env_fov, self.pers_resolution[1], self.pers_resolution[0])

        pos_cam = self.env_uv @ torch.linalg.inv(intrinsic).T
        ray_dir = pos_cam @ c2w_pers[:3, :3].T
        ray_dir = safe_normalize(ray_dir)
        
        query_dir = (ray_dir.reshape(-1, 3)).reshape_as(ray_dir)
        nrm_pers = -query_dir.flip(1).contiguous()

        env_proj = cubemap_sample_torch(cubemap, nrm_pers.reshape(-1, 3)).reshape_as(nrm_pers)
        nrm_map = nrm_pers
        if self.use_cam_space_dir:
            nrm_map = (nrm_pers.reshape(-1, 3) @ c2w_pers[:3, :3]).reshape_as(nrm_pers) 
        # NOTE: this is HDR

        env_proj_ldr, env_proj_log = self.process_hdr(env_proj)

        return env_proj_ldr, env_proj_log, nrm_map

    def get_ball_image(self, c2w, cubemap, rot_azimuth=0):
        use_fwd_cam = False
        if self.ball_random_cam:
            use_fwd_cam = random.random() < 0.3

        c2w_pers = c2w
        if rot_azimuth != 0:
            c2w_pers = rotate_y(rot_azimuth) @ c2w
    
        if use_fwd_cam:
            vec_ball = self.nrm_chrome_ball
        else:
            vec_ball = (self.nrm_chrome_ball.reshape(-1, 3) @ c2w_pers[:3, :3].T).reshape_as(self.nrm_chrome_ball)
        
        env_ball = cubemap_sample_torch(cubemap, vec_ball.reshape(-1, 3)).reshape_as(vec_ball)

        if self.ball_dir_condition == 'normal':
            normal_map = self.normal_map
        elif self.ball_dir_condition == 'dir':
            normal_map = vec_ball
            if self.use_cam_space_dir and not use_fwd_cam:
                normal_map = (normal_map.reshape(-1, 3) @ c2w_pers[:3, :3]).reshape_as(normal_map)

        env_ball_ldr, env_ball_log = self.process_hdr(env_ball)

        return env_ball_ldr, env_ball_log, normal_map

    def get_erp_image(self, c2w, cubemap):
        use_fwd_cam = True
        if self.erp_random_cam:
            use_fwd_cam = random.random() > 0.25
        if use_fwd_cam:
            erp_vec = self.erp_vec
        else:
            erp_vec = (self.erp_vec.reshape(-1, 3) @ c2w[:3, :3].T).reshape_as(self.erp_vec)

        env_map = cubemap_sample_torch(cubemap, erp_vec.reshape(-1, 3)).reshape_as(erp_vec)
        normal_map = erp_vec
        if self.use_cam_space_dir:
            normal_map = (erp_vec.reshape(-1, 3) @ c2w[:3, :3]).reshape_as(erp_vec)

        if self.erp_random_roll:
            roll_w = env_map.shape[1]
            roll_pix = random.randint(0, roll_w - 1)
            env_map = torch.roll(env_map, roll_pix, dims=1)
            normal_map = torch.roll(normal_map, roll_pix, dims=1)
        
        env_map_ldr, env_map_log = self.process_hdr(env_map)
        return env_map_ldr, env_map_log, normal_map


    def __getitem__(self, idx):
        envmap = self.hdri_list[idx]
        envmap = read_image(envmap)[..., :3]
        envmap = torch.from_numpy(envmap).float()

        env_scale = random.uniform(*self.env_scale_range)
        envmap = envmap * env_scale
        
        # sample envmap_rot, flip
        # envmap_rot = random.uniform(0, 2 * np.pi)
        envmap_w = envmap.shape[1]
        if self.env_random_roll:
            envmap_roll = random.randint(0, envmap_w - 1)
            envmap = torch.roll(envmap, envmap_roll, dims=1)

        envmap_flip = self.env_random_flip and random.random() < 0.5

        if envmap_flip:
            envmap = envmap.flip(1)

        # setup cubemap
        cubemap = latlong_to_cubemap_torch(envmap.contiguous(), [512, 512])

        # sample scene camera
        azimuth_0 = np.pi / 2 # c2w = eye
        elevation = random.uniform(*self.cam_theta_range)
        radius = 1 # TODO: no consideration for camera radius...

        w2c = get_cam_matrix(azimuth_0, elevation, radius=radius) # [4, 4]
        c2w = torch.linalg.inv(w2c)

        rot_azimuth = random.uniform(*self.cam_phi_range_inp)

        if self.inp_type == 'pers':
            inp_ldr, inp_log, inp_nrm = self.get_pers_image(c2w, cubemap, rot_azimuth=rot_azimuth)
        elif self.inp_type == 'ball':
            inp_ldr, inp_log, inp_nrm = self.get_ball_image(c2w, cubemap, rot_azimuth=rot_azimuth)
        elif self.inp_type == 'erp':
            inp_ldr, inp_log, inp_nrm = self.get_erp_image(c2w, cubemap)

        rot_azimuth = random.uniform(*self.cam_phi_range_out)
       
        if self.out_type == 'pers':
            out_ldr, out_log, out_nrm = self.get_pers_image(c2w, cubemap, rot_azimuth)
        elif self.out_type == 'ball':
            out_ldr, out_log, out_nrm = self.get_ball_image(c2w, cubemap, rot_azimuth)
        elif self.out_type == 'erp':
            out_ldr, out_log, out_nrm = self.get_erp_image(c2w, cubemap)

        if self.random_swap > 0:
            if random.random() < self.random_swap:
                inp_ldr, out_ldr = out_ldr, inp_ldr
                inp_log, out_log = out_log, inp_log
                inp_nrm, out_nrm = out_nrm, inp_nrm

        output = {
            'env_ldr': inp_ldr,
            'env_log': inp_log,
            'env_nrm': inp_nrm,
            'rgb': out_ldr,
            'rgb_log': out_log,
            'normal': out_nrm,
        }

        for key in output:
            if self.out_format == 'CHW':
                output[key] = output[key].permute(2, 0, 1).unsqueeze(0).contiguous()
            else:
                output[key] = output[key].unsqueeze(0).contiguous()

        return output    

