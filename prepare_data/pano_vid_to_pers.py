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

from src.data.env_panovideo_dataset import PanoVidDataset

parser = argparse.ArgumentParser()
parser.add_argument('--pano_vid_folder', type=str, default=None)
parser.add_argument('--output_folder', type=str, default=None)
parser.add_argument('--fov', type=float, default=60) # in degrees
parser.add_argument('--resolution', type=int, nargs=2, default=(480, 720))
parser.add_argument('--elevation', type=float, default=0) # in degrees
parser.add_argument('--num_frames', type=int, default=25)

args = parser.parse_args()

pano_vid_folder = args.pano_vid_folder
output_folder = args.output_folder
fov = args.fov
resolution = args.resolution
elevation = args.elevation
num_frames = args.num_frames

hdri_list = sorted(
    glob.glob(os.path.join(pano_vid_folder, '*.mp4'))
)

print(f'{len(hdri_list)} pano vids found')

output_pers_folder = os.path.join(output_folder, 'pers')
output_vid_folder = os.path.join(output_folder, 'vid')
output_meta_folder = os.path.join(output_folder, 'meta')

os.makedirs(output_pers_folder, exist_ok=True)
os.makedirs(output_vid_folder, exist_ok=True)
os.makedirs(output_meta_folder, exist_ok=True)

dataset = PanoVidDataset(
    hdri_list=hdri_list,
    cam_theta_range=(elevation, elevation), # elevation
    env_fov_range=(fov, fov), # Fov on height side
    env_scale_range=(1., 1.),
    out_format='HWC',
    inp_type='pers',
    out_type='erp',
    erp_random_cam=0, # use undistorted erp
    resolution=resolution,
    random_autoexposure=0,
    return_hdr=False,
    return_meta=True,
    num_frames=num_frames,
    vid_cam_mode='static',
    sample_stride=1,
    sample_stride_max=2,
)

for j in tqdm(range(len(hdri_list))):
    example = dataset[j]
    rgb = example['rgb'].numpy() * 0.5 + 0.5 # [f, h, w, 3]
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    ldr = example['env_ldr'].numpy() * 0.5 + 0.5
    ldr = (np.clip(ldr, 0, 1) * 255).astype(np.uint8)
    base_file_name = os.path.basename(hdri_list[j])
    for k in range(rgb.shape[0]):
        rgb_k = rgb[k]
        # save the rgb image
        file_name = base_file_name.replace('.mp4', f'.{k:03d}.png')
        save_path = os.path.join(output_pers_folder, file_name)
        imageio.imwrite(save_path, rgb_k)
        # save the hdr image
        # save video
    file_name = base_file_name.replace('.mp4', f'.rgb.mp4')
    save_path = os.path.join(output_vid_folder, file_name)
    iio.mimwrite(save_path, rgb, fps=8)
    file_name = base_file_name.replace('.mp4', f'.env.mp4')
    save_path = os.path.join(output_vid_folder, file_name)
    iio.mimwrite(save_path, ldr, fps=8)
    # save the meta data
    file_name = base_file_name.replace('.mp4', f'.json')
    save_path = os.path.join(output_meta_folder, file_name)
    with open(save_path, 'w') as f:
        json.dump(example['meta'], f, indent=4)


print(f'Done')