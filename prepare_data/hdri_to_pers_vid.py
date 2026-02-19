import os
import glob
import numpy as np
import cv2
import imageio.v3 as imageio
import imageio as iio
import json
from tqdm import tqdm
import argparse
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from src.data.env_rgbxenv_dataset import RGBXEnvDataset

parser = argparse.ArgumentParser()
parser.add_argument('--hdri_folder', type=str, default=None)
parser.add_argument('--output_folder', type=str, default=None)
parser.add_argument('--fov', type=float, default=60) # in degrees
parser.add_argument('--resolution', type=int, nargs=2, default=(480, 720))
parser.add_argument('--elevation', type=float, default=0) # in degrees
parser.add_argument('--num_frames', type=int, default=25)
parser.add_argument('--vid_cam_mode', type=str, default='mixed', choices=['mixed', 'oscil_cam', 'span_cam', 'static'])

args = parser.parse_args()

hdri_folder = args.hdri_folder
output_folder = args.output_folder
fov = args.fov
resolution = args.resolution
elevation = args.elevation
num_frames = args.num_frames
vid_cam_mode = args.vid_cam_mode

hdri_list = sorted(
    glob.glob(os.path.join(hdri_folder, '*.hdr')) + \
    glob.glob(os.path.join(hdri_folder, '*.exr'))
)

print(f'{len(hdri_list)} hdris found')

output_pers_folder = os.path.join(output_folder, 'pers')
output_hdr_folder = os.path.join(output_folder, 'hdr')
output_vid_folder = os.path.join(output_folder, 'vid')
output_meta_folder = os.path.join(output_folder, 'meta')

os.makedirs(output_pers_folder, exist_ok=True)
os.makedirs(output_hdr_folder, exist_ok=True)
os.makedirs(output_vid_folder, exist_ok=True)
os.makedirs(output_meta_folder, exist_ok=True)

dataset = RGBXEnvDataset(
    hdri_list=hdri_list,
    cam_theta_range=(elevation, elevation), # elevation
    env_fov_range=(fov, fov), # Fov on height side
    env_scale_range=(1., 1.),
    out_format='HWC',
    inp_type='pers',
    out_type='erp',
    erp_random_cam=1, # use distorted erp
    resolution=resolution,
    return_hdr=True,
    return_meta=True,
    num_frames=num_frames,
    vid_cam_mode=vid_cam_mode,
    vid_random_reverse=True,
    vid_oscil_range=(2, 10),
    vid_span_range=(5, 15)
)

dataset.vid_modes = ['oscil_cam', 'span_cam']

for j in tqdm(range(len(hdri_list))):
    example = dataset[j]
    rgb = example['rgb'].numpy() * 0.5 + 0.5 # [f, h, w, 3]
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    hdr = example['env_hdr'].numpy() # [f, h, w, 3]
    ldr = example['env_ldr'].numpy() * 0.5 + 0.5
    ldr = (np.clip(ldr, 0, 1) * 255).astype(np.uint8)
    base_file_name = os.path.basename(hdri_list[j])
    for k in range(rgb.shape[0]):
        rgb_k = rgb[k]
        hdr_k = hdr[k]
        # save the rgb image
        file_name = base_file_name.replace('.hdr', f'.{k:03d}.png')
        save_path = os.path.join(output_pers_folder, file_name)
        imageio.imwrite(save_path, rgb_k)
        # save the hdr image
        file_name = base_file_name.replace('.hdr', f'.{k:03d}.exr')
        save_path = os.path.join(output_hdr_folder, file_name)
        imageio.imwrite(save_path, hdr_k, plugin='opencv')
        # save video
    file_name = base_file_name.replace('.hdr', f'.rgb.mp4')
    save_path = os.path.join(output_vid_folder, file_name)
    iio.mimwrite(save_path, rgb, fps=8)
    file_name = base_file_name.replace('.hdr', f'.env.mp4')
    save_path = os.path.join(output_vid_folder, file_name)
    iio.mimwrite(save_path, ldr, fps=8)
    # save the meta data
    file_name = base_file_name.replace('.hdr', f'.json')
    save_path = os.path.join(output_meta_folder, file_name)
    with open(save_path, 'w') as f:
        json.dump(example['meta'], f, indent=4)

print(f'Done')