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
from .rendering_utils import (
    envmap_vec, envmap_xfm, rotate_y, uv_mesh, ray2zdepth,
    get_ideal_ball, get_ref_vector, envmap_chrome_ball,
    normalize_depth, cam_intrinsics, read_image, get_cam_matrix, safe_normalize,
    latlong_to_cubemap_torch, cubemap_sample_torch,
    rgb2srgb, reinhard, prefilter_cubemap, percentile_tone, ocio_tonemapping
)
from src.utils.util import instantiate_from_config
from src.data.env_rendering_dataset import EnvRenderingDataset
from torchvision.transforms.functional import gaussian_blur

def read_img_example(
        img_path, resolution=[480, 720], env_resolution=[128, 256], 
        frames_per_sample=1, vid_cam_mode='mixed',
        **kwargs
):
    # read envmap
    example = RGBXEnvDataset(
        img_path, resolution=resolution, env_resolution=env_resolution,
        out_format='HWC', num_frames=frames_per_sample, vid_cam_mode=vid_cam_mode,
        **kwargs
    )[0]
    example = {
        k: (v * 0.5 + 0.5).numpy() for k, v in example.items()
    }
    return example

def quantize_f_image(image, normalize=True):
    if normalize:
        image = image * 0.5 + 0.5
    image = (image * 255).long().float() / 255
    if normalize:
        image = image * 2 - 1
    return image


# put all cubemap functions on CPU, no need for nvdiffrast
class RGBXEnvDataset(EnvRenderingDataset):

    def __init__(
        self,
        hdri_list,
        cam_theta_range=(-10, 10), # elevation
        erp_resolution=[128, 256],
        pers_resolution=[480, 720],
        env_fov_range=(60, 95),
        env_scale_range=(0.7, 1.4),
        erp_random_cam=0,
        erp_random_roll=False,
        use_cam_space_dir=True,
        inp_type='pers',
        out_type='erp',
        resolution=None,
        env_resolution=None,
        num_frames=1,
        random_blur=0,
        random_roughness=0,
        random_input_blur=0,
        random_tonemapping=False,
        random_autoexposure=0,
        tonemapping='ocio: AgX Base sRGB',
        ocio_config='configs/colormanagement/config.ocio',
        laval_tone=0,
        aos_tone=False,
        faster_laval_tone=False,
        return_pers=True,
        return_hdr=False,
        return_meta=False,
        random_quantize=0,
        env_roll=0,
        vid_cam_mode='oscil_cam',
        vid_oscil_range=(2, 10),
        vid_span_range=(5, 15),
        vid_random_reverse=False,
        vid_loop_min=24,
        **kwargs
    ):
        if resolution is not None:
            pers_resolution = resolution
        if env_resolution is not None:
            erp_resolution = env_resolution
        super().__init__(
            hdri_list=hdri_list,
            cam_theta_range=cam_theta_range,
            erp_resolution=erp_resolution,
            pers_resolution=pers_resolution,
            use_cam_space_dir=use_cam_space_dir,
            erp_random_cam=erp_random_cam,
            erp_random_roll=erp_random_roll,
            env_fov_range=env_fov_range,
            env_scale_range=env_scale_range,
            inp_type=inp_type,
            out_type=out_type,
            **kwargs
        )
        # ensure the same format as diffusion_renderer

        self.erp_vec = envmap_vec(self.erp_resolution)
        self.random_blur = random_blur
        self.random_roughness = random_roughness
        self.random_tonemapping = random_tonemapping
        self.random_autoexposure = random_autoexposure
        self.random_quantize = random_quantize
        self.random_input_blur = random_input_blur
        self.return_pers = return_pers
        self.return_hdr = return_hdr
        self.return_meta = return_meta
        self.env_roll = env_roll

        # video sample
        self.num_frames = num_frames
        self.vid_modes = ['oscil_cam', 'span_cam', 'static']
        self.vid_cam_mode = vid_cam_mode
        self.vid_oscil_range = vid_oscil_range
        self.vid_span_range = vid_span_range
        self.vid_random_reverse = vid_random_reverse
        self.vid_loop_min = vid_loop_min

        self.laval_tone = laval_tone
        self.aos_tone = aos_tone
        self.faster_laval_tone = faster_laval_tone
        # self.use_ocio = use_ocio
        if laval_tone:
            self.laval_tone_mapping = lambda img: img
    
        if tonemapping.startswith('ocio:'):
            dst_space = tonemapping.split(':')[1].strip()
            self.tone_mapping = ocio_tonemapping(ocio_config=ocio_config, dst_space=dst_space)
        elif tonemapping.startswith('percentile'):
            self.tone_mapping = lambda img: percentile_tone(img, gamma=2.4, percentile=99, max_mapping=0.9)
        else:
            self.tone_mapping = rgb2srgb

        if random_tonemapping:
            self.tone_mappings = [
                ocio_tonemapping(ocio_config=ocio_config, dst_space='AgX Base sRGB'),
                ocio_tonemapping(ocio_config=ocio_config, dst_space='Filmic sRGB'),
                ocio_tonemapping(ocio_config=ocio_config, dst_space='sRGB')
            ]

    def get_pers_w_tune_image(self, c2w, cubemap, rot_azimuth=0, tone_mapping=None, env_fov=1.05): # env_fov ~60deg
        c2w_pers = c2w
        if rot_azimuth != 0:
            c2w_pers = rotate_y(rot_azimuth) @ c2w
        
        intrinsic = cam_intrinsics(env_fov, self.pers_resolution[1], self.pers_resolution[0])

        pos_cam = self.env_uv @ torch.linalg.inv(intrinsic).T
        ray_dir = pos_cam @ c2w_pers[:3, :3].T
        ray_dir = safe_normalize(ray_dir)
        
        query_dir = (ray_dir.reshape(-1, 3)).reshape_as(ray_dir)
        nrm_pers = -query_dir.flip(1).contiguous()

        env_proj = cubemap_sample_torch(cubemap, nrm_pers.reshape(-1, 3)).reshape_as(nrm_pers)

        # nrm_map = nrm_pers
        # if self.use_cam_space_dir:
        #     nrm_map = (nrm_pers.reshape(-1, 3) @ c2w_pers[:3, :3]).reshape_as(nrm_pers) 

        # NOTE: this is HDR            
        # env_proj_ldr, env_proj_log = self.process_hdr(env_proj)
        env_proj_ldr = torch.from_numpy(tone_mapping(env_proj.cpu().numpy()))

        if self.normalize_image:
            env_proj_ldr = env_proj_ldr * 2 - 1

        return env_proj_ldr #, env_proj_log, nrm_map

    def get_erp_w_filter_image(
            self, c2w, cubemap, 
            blur_kernel_size=0, roughness=0, use_fwd_cam=True, erp_roll=0):
        # NOTE: this is different from the base class, but follow the same format as DR

        if use_fwd_cam:
            erp_vec = self.erp_vec
        else:
            erp_vec = (self.erp_vec.reshape(-1, 3) @ c2w[:3, :3].T).reshape_as(self.erp_vec)

        cubemap_filtered = prefilter_cubemap(cubemap, roughness) if roughness > 0 else cubemap
        env_map = cubemap_sample_torch(cubemap_filtered, erp_vec.reshape(-1, 3)).reshape_as(erp_vec)
        normal_map = erp_vec
        if self.use_cam_space_dir:
            normal_map = (erp_vec.reshape(-1, 3) @ c2w[:3, :3]).reshape_as(erp_vec)

        if erp_roll != 0:
            roll_w = env_map.shape[1]
            roll_pix = int(erp_roll * roll_w)
            env_map = torch.roll(env_map, roll_pix, dims=1)
            normal_map = torch.roll(normal_map, roll_pix, dims=1)

        if blur_kernel_size > 0:
            env_map = env_map.permute(2, 0, 1).unsqueeze(0)
            # pad left and right with circular padding
            pad = blur_kernel_size // 2
            env_map_pad = F.pad(env_map, (pad, pad, 0, 0), mode='circular')
            env_map = gaussian_blur(env_map_pad, blur_kernel_size)
            env_map = env_map.squeeze(0).permute(1, 2, 0)
            env_map = env_map[:, pad:-pad, :]

        
        env_map_ldr, env_map_log = self.process_hdr(env_map)
        return env_map_ldr, env_map_log, normal_map, env_map

    def get_sample_stride(self, in_frames):
        sample_stride = self.sample_stride
        if self.sample_stride_max > 1:
            max_stride = min(in_frames // self.num_frames, self.sample_stride_max)
            if max_stride > 1 and random.random() < 0.5:
                sample_stride = random.randint(2, max_stride)
        return sample_stride

    def __getitem__(self, idx):
        envmap = self.hdri_list[idx]
        envmap = read_image(envmap)[..., :3]

        if self.random_input_blur > 0 and random.random() < self.random_input_blur:
            # CV2 downsample
            resolution_list = [(2048, 1024), (1024, 512)]
            resolution_weights = [0.3, 0.7]
            resolution = random.choices(resolution_list, weights=resolution_weights)[0]
            envmap = cv2.resize(envmap, resolution, interpolation=cv2.INTER_AREA)

        env_scale = random.uniform(*self.env_scale_range)
        envmap = envmap * env_scale
        envmap = torch.from_numpy(envmap).float().clip(0, 100000) # [H, W, 3]

        # sample envmap_rot, flip
        # envmap_rot = random.uniform(0, 2 * np.pi)
        envmap_w = envmap.shape[1]
        envmap_roll = 0
        if self.env_random_roll:
            envmap_roll = random.randint(0, envmap_w - 1)
            envmap = torch.roll(envmap, envmap_roll, dims=1)
        elif self.env_roll != 0:
            envmap_roll = int(envmap_w * self.env_roll / 360)
            envmap = torch.roll(envmap, envmap_roll, dims=1)
            
        envmap_flip = self.env_random_flip and random.random() < 0.5
        if envmap_flip:
            envmap = envmap.flip(1)

        # setup cubemap
        cubemap = latlong_to_cubemap_torch(envmap.contiguous(), [512, 512]) # [6, 512, 512, 3]
        cubemap_ldr = cubemap

        laval_tone = self.laval_tone > 0 and random.random() < self.laval_tone
        if laval_tone:
            cubemap_hdr_np = percentile_tone(cubemap.cpu().numpy(), percentile=50, max_mapping=0.5, clip=False, with_gamma=False)
            cubemap = torch.from_numpy(cubemap_hdr_np)
            if self.return_pers and not self.faster_laval_tone:
                percentile, max_mapping = 99, 0.9
                if self.aos_tone:
                    percentile, max_mapping = 90, 0.8
                envmap_ldr_np = percentile_tone(envmap.cpu().numpy(), gamma=2.4, percentile=percentile, max_mapping=max_mapping)
                cubemap_ldr = latlong_to_cubemap_torch(torch.from_numpy(envmap_ldr_np).contiguous(), [512, 512])
            else:
                cubemap_ldr = cubemap

        elif self.random_autoexposure > 0 and random.random() < self.random_autoexposure:
            _, alpha = percentile_tone(
                cubemap.cpu().numpy(), percentile=50, max_mapping=0.5, 
                clip=False, with_gamma=False, ret_alpha=True
            )
            cubemap = cubemap * alpha
            cubemap_ldr = cubemap

        # sample scene camera
        azimuth_0 = np.pi / 2 # c2w = eye
        elevation_0 = random.uniform(*self.cam_theta_range)
        radius = 1 # TODO: no consideration for camera radius...
        env_fov = random.uniform(*self.env_fov_range)

        inp_ldr_list, env_ldr_list, env_log_list, env_nrm_list, env_hdr_list = [], [], [], [], []

        # Data augmentation
        tone_mapping = random.choice(self.tone_mappings) if self.random_tonemapping else self.tone_mapping
        if  laval_tone:
            tone_mapping = self.laval_tone_mapping
        blur_kernel_size, roughness = 0, 0
        
        blur_rand = random.random()
        if self.random_blur > 0 and blur_rand < self.random_blur:
            blur_kernel_size = 3 # random.randint(1, 2) * 2 + 1
        elif self.random_roughness > 0 and (blur_rand - self.random_blur) < self.random_roughness:
            roughness = 1 - random.uniform(0, 0.8) ** 2

        use_fwd_cam = True
        if self.erp_random_cam:
            thres = 0.5 if isinstance(self.erp_random_cam, bool) else self.erp_random_cam
            use_fwd_cam = random.random() > thres
        erp_roll = 0
        if self.erp_random_roll:
            erp_roll = random.uniform(-0.5, 0.5)

        num_frames = self.num_frames
        azimuth_offset, elevation_offset = None, None
        if num_frames > 1:
            vid_cam_mode = random.choice(self.vid_modes) if self.vid_cam_mode == 'mixed' else self.vid_cam_mode
            vid_loop_frames = max(num_frames, self.vid_loop_min)
            if vid_cam_mode == 'oscil_cam':
                cone_angle = random.uniform(*self.vid_oscil_range) / 180 * np.pi
                azimuth_offset = np.sin(np.linspace(0, 2*np.pi, vid_loop_frames, endpoint=False)) * cone_angle
                elevation_offset = np.cos(np.linspace(0, 2*np.pi, vid_loop_frames, endpoint=False)) * cone_angle
            elif vid_cam_mode == 'span_cam':
                span_angle = random.uniform(*self.vid_span_range) / 180 * np.pi
                elevation_offset = np.zeros(vid_loop_frames)
                azimuth_offset = np.linspace(-span_angle, span_angle, vid_loop_frames)

        reverse_vid = random.random() < 0.5 if self.vid_random_reverse else False

        meta = {}
        if self.return_meta:
            meta = {
                'env_path': self.hdri_list[idx],
                'env_scale': env_scale,
                'env_roll': envmap_roll,
                'env_flip': envmap_flip,
                'env_fov': env_fov,
                'frames': []
            }

        for i in range(num_frames):
            if azimuth_offset is not None:
                azimuth = azimuth_0 + azimuth_offset[i]
                elevation = elevation_0 + elevation_offset[i]
            else:
                azimuth, elevation = azimuth_0, elevation_0
            w2c = get_cam_matrix(azimuth, elevation, radius=radius) # [4, 4]
            c2w = torch.linalg.inv(w2c)

            if self.return_pers:
                inp_ldr = self.get_pers_w_tune_image(
                    c2w, cubemap_ldr, rot_azimuth=0, env_fov=env_fov, tone_mapping=tone_mapping
                )
                inp_ldr_list.append(inp_ldr)

            env_ldr, env_log, env_nrm, env_hdr = self.get_erp_w_filter_image(
                c2w, cubemap, blur_kernel_size=blur_kernel_size, roughness=roughness,
                use_fwd_cam=use_fwd_cam, erp_roll=erp_roll
            )

            env_ldr_list.append(env_ldr)
            env_log_list.append(env_log)
            env_nrm_list.append(env_nrm)
            env_hdr_list.append(env_hdr)
            if self.return_meta:
                meta['frames'].append({
                    'azimuth': azimuth,
                    'elevation': elevation,
                })

        env_ldr = torch.stack(env_ldr_list, dim=0)
        env_log = torch.stack(env_log_list, dim=0)
        env_nrm = torch.stack(env_nrm_list, dim=0)
        if self.return_pers:
            inp_ldr = torch.stack(inp_ldr_list, dim=0)
        if self.return_hdr:
            env_hdr = torch.stack(env_hdr_list, dim=0)

        if self.random_quantize > 0 and random.random() < self.random_quantize:
            env_ldr = quantize_f_image(env_ldr, normalize=self.normalize_image)
            env_log = quantize_f_image(env_log, normalize=self.normalize_image)

        output = {
            'env_ldr': env_ldr,
            'env_log': env_log,
            'env_nrm': env_nrm,
        }
        if self.return_pers:
            output['rgb'] = inp_ldr
            
        if self.return_hdr:
            output['env_hdr'] = env_hdr

        for key in output:
            if reverse_vid and self.num_frames > 1:
                output[key] = output[key].flip(0)
                if self.return_meta:
                    meta['frames'] = meta['frames'][::-1]
            if self.out_format == 'CHW':
                output[key] = output[key].permute(0, 3, 1, 2).contiguous()
            else:
                output[key] = output[key].contiguous()

        if self.append_features is not None:
            output.update(self.append_features)

        if self.return_meta:
            output['meta'] = meta

        return output    

