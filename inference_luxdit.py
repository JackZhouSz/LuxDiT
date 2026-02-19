import os
import sys
import glob
import numpy as np
import json
import torch
import argparse
from omegaconf import OmegaConf, ListConfig
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image
import imageio
from diffusers.utils import export_to_video
from braceexpand import braceexpand

from src.models.custom_cogvideox_transformer_3d import CustomCogVideoXTransformer3DModel
from src.models.custom_autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from src.pipelines.pipeline_cogvideox_rgbxenv import RGBXEnvCogVideoXPipeline, T5EncoderModel
from transformers import AutoTokenizer
from src.data.rendering_utils import resize_crop, envmap_vec, read_image, get_cam_matrix
from luxdit_config import TrainingConfig

def group_video_frames(input_list):
    # has to be xxxxx.0001.png, xxxxx.0002.png, ...
    grouped_frames = []
    prev_video = None
    for f in input_list:
        basename = os.path.basename(f)
        basevideo = ".".join(basename.split('.')[:-2])
        if basevideo != prev_video:
            grouped_frames.append([f])
            prev_video = basevideo
        else:
            grouped_frames[-1].append(f)

    return grouped_frames

READ_EXAMPLE_FUNC = None

def main(args, cfg):
    pipeline_cfg = cfg.model_pipeline
    inference_cfg = cfg.inference_kwargs
    data_cfg = cfg.train_dataset if 'train_dataset' in cfg else cfg.train_data
    cond_mode = pipeline_cfg.cond_mode

    if pipeline_cfg.get('text_encoder_path', None) is None:
        pipeline_cfg['text_encoder_path'] = cfg.pretrained_model_name_or_path
    if pipeline_cfg.get('vae_path', None) is None:
        pipeline_cfg['vae_path'] = cfg.pretrained_model_name_or_path
    if pipeline_cfg.get('transformer_path', None) is None:
        pipeline_cfg['transformer_path'] = cfg.pretrained_model_name_or_path

    transformer_kwargs = dict(pipeline_cfg['transformer_kwargs'])
    for k, v in transformer_kwargs.items():
        if isinstance(v, list) or isinstance(v, ListConfig):
            transformer_kwargs[k] = tuple(v)

    text_encoder, image_encoder, vae, env_encoder = None, None, None, None
    tokenizer, feature_extractor = None, None


    device = torch.device('cuda')
    weight_dtype = torch.bfloat16
    if args.precision == "fp16":
        weight_dtype = torch.float16
    elif args.precision == "bf16":
        weight_dtype = torch.bfloat16
    weight_dtype, device

    vae = AutoencoderKLCogVideoX.from_pretrained(
        pipeline_cfg.vae_path, subfolder="vae", revision=cfg.revision, variant=None, # cfg.variant
        torch_dtype=weight_dtype, device=device
    )

    cond_mode = pipeline_cfg.get('cond_mode', 'env')
    if cond_mode == 'text':
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        text_encoder = T5EncoderModel.from_pretrained(
            pipeline_cfg['text_encoder_path'], subfolder="text_encoder", 
            revision=cfg.revision, variant=cfg.variant, torch_dtype=weight_dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pipeline_cfg['text_encoder_path'], subfolder="tokenizer", revision=cfg.revision
        )
        
    transformers = CustomCogVideoXTransformer3DModel.from_pretrained(
        args.transformer_path, subfolder="transformer", revision=cfg.revision, variant=None, # cfg.variant
        torch_dtype=weight_dtype, device=device, 
        **transformer_kwargs
    )

    noise_scheduler = CogVideoXDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")

    pipeline = RGBXEnvCogVideoXPipeline(
        vae=vae, 
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformers,
        scheduler=noise_scheduler
    )

    if args.lora_dir is not None and os.path.exists(args.lora_dir):
        print('Loading lora from', args.lora_dir)
        try:
            pipeline.unload_lora_weights()
        except:
            pass
        pipeline.load_lora_weights(args.lora_dir, adapter_name="env-lora")
    else:
        print('No lora loaded')

    pipeline = pipeline.to(device)

    denoise_type = pipeline_cfg.get('denoise_type', 'additional_hidden_states')
    text_prompt = pipeline_cfg.get('text_prompt', None)
    separate_timesteps = pipeline_cfg.get('separate_timesteps', False)
    additional_rope_time_only = pipeline_cfg.get('additional_rope_time_only', True)

    _target_labels, cond_labels = pipeline_cfg.target_image, pipeline_cfg.cond_images
    additional_cond_labels = pipeline_cfg.get('additional_cond_labels', None)
    _additional_target_labels = pipeline_cfg.get('additional_target_image', [])

    if denoise_type == 'additional_hidden_states':
        target_labels = _additional_target_labels
        viz_input = _target_labels
    else:
        target_labels = _target_labels
        viz_input = _additional_target_labels

    resolution = args.resolution
    height, width = resolution
    env_resolution = args.env_resolution
    data_read_kwargs = {}
    if args.data_type in ['image', 'video']:
        cond_labels.pop('env_ldr', None)
        cond_labels.pop('env_log', None)

        env_nrm_raw = envmap_vec(env_resolution)

        if args.cam_elevation != 0:
            elevation = args.cam_elevation / 180 * np.pi
            c2w = torch.linalg.inv(get_cam_matrix(np.pi / 2, elevation))
            env_nrm = (env_nrm_raw.reshape(-1, 3) @ c2w[:3, :3]).reshape_as(env_nrm_raw)
        else:
            env_nrm = env_nrm_raw
        
        env_nrm = env_nrm * 0.5 + 0.5

        if args.data_type == 'image':    
            def read_img_example(img_path, resolution, **kwargs):
                img = read_image(img_path).astype(np.float32)[..., :3] / 255.0
                img = resize_crop(img, resolution)
                return {'rgb': img[None, :], 'env_nrm': env_nrm[None, :].numpy()}
            READ_EXAMPLE_FUNC = read_img_example
            
        elif args.data_type == 'video':
            data_read_kwargs = {
                'frames_per_sample': args.frames_per_sample,
                'frame_start_idx': args.frame_start_idx,
                'frame_stride': args.frame_stride,
                'camera_pose_file': args.camera_pose_file,
                'sample_stride': args.sample_stride,
            }
            def read_vid_example(
                img_path, resolution, 
                frames_per_sample, frame_start_idx, frame_stride, camera_pose_file, 
                sample_stride=1, **kwargs
            ):
                # Check if input is an MP4 file
                is_mp4 = False
                mp4_path = None
                if isinstance(img_path, str):
                    if img_path.lower().endswith('.mp4'):
                        is_mp4 = True
                        mp4_path = img_path
                elif isinstance(img_path, list) and len(img_path) > 0:
                    if isinstance(img_path[0], str) and img_path[0].lower().endswith('.mp4'):
                        is_mp4 = True
                        mp4_path = img_path[0]
                
                if is_mp4:
                    # Read frames from MP4 video
                    reader = imageio.get_reader(mp4_path, 'ffmpeg')
                    all_frames = []
                    for frame in reader:
                        all_frames.append(frame)
                    reader.close()
                    
                    # Apply sample_stride for 25 frame sampling
                    if sample_stride > 1:
                        all_frames = all_frames[::sample_stride]
                    
                    # Default to 25 frames if frames_per_sample is None
                    if frames_per_sample is None:
                        frames_per_sample = 25
                    
                    # Apply frame_start_idx and limit to frames_per_sample
                    all_frames = all_frames[frame_start_idx:]
                    all_frames = all_frames[:frames_per_sample]
                    
                    img_list = []
                    for frame in all_frames:
                        img = frame.astype(np.float32)[..., :3] / 255.0
                        img = resize_crop(img, resolution)
                        img_list.append(img)
                    
                    img = np.stack(img_list, axis=0)
                    num_frames = img.shape[0]
                    # Generate unique frame names for MP4 files
                    base_name = os.path.splitext(os.path.basename(mp4_path))[0]
                    frame_list = [f"{base_name}_{i:04d}.mp4" for i in range(num_frames)]
                else:
                    # Original behavior for image lists
                    img_list = []
                    img_path = img_path[frame_start_idx::frame_stride]
                    if frames_per_sample is None:
                        frames_per_sample = len(img_path) 
                    img_path = img_path[:frames_per_sample] 

                    for img in img_path:
                        img = read_image(img).astype(np.float32)[..., :3] / 255.0
                        img = resize_crop(img, resolution)
                        img_list.append(img)

                    img = np.stack(img_list, axis=0)
                    num_frames = img.shape[0]
                    frame_list = img_path
                
                if camera_pose_file is not None:
                    if not os.path.isfile(camera_pose_file):
                        base_path = mp4_path if is_mp4 else img_path[0]
                        camera_pose_file = os.path.join(os.path.dirname(base_path), camera_pose_file)
                    with open(camera_pose_file, 'r') as f:
                        c2w_list = json.load(f)['frames']
                    c2w_list = [cam['transform_matrix'] for cam in c2w_list]
                    c2w_list = c2w_list[frame_start_idx::frame_stride][:frames_per_sample]
                    c2w_list = torch.tensor(c2w_list).float() # (N, 4, 4)
                    env_nrm_vid = env_nrm_raw[None, ...].repeat(num_frames, 1, 1, 1) 
                    env_nrm_vid = (env_nrm_vid.reshape(num_frames, -1, 3) @ c2w_list[:, :3, :3]).reshape_as(env_nrm_vid)
                    env_nrm_vid = env_nrm_vid * 0.5 + 0.5
                else:
                    env_nrm_vid = env_nrm[None, :].repeat(num_frames, 1, 1, 1)
                return {'rgb': img, 'env_nrm': env_nrm_vid.numpy(), 'frame_list': frame_list}
            READ_EXAMPLE_FUNC = read_vid_example
   
    input_dir = args.input_dir
    if input_dir.endswith('.txt'):
        input_dir = np.loadtxt(input_dir, dtype=str).tolist()
    else:
        input_dir = list(braceexpand(input_dir))

    input_list = []
    for inp in input_dir:
        if os.path.isdir(inp):
            dir_inp = sorted(glob.glob(os.path.join(inp, '*')))
            dir_inp = [f for f in dir_inp if f.split('.')[-1] in ['png', 'jpg', 'jpeg', 'mp4', 'exr', 'hdr']]
            input_list.extend(dir_inp)
        else:
            input_list.extend(sorted(glob.glob(inp)))

    if args.data_type == 'video':
        # Separate MP4 files from image sequences
        mp4_files = [f for f in input_list if f.lower().endswith('.mp4')]
        image_files = [f for f in input_list if not f.lower().endswith('.mp4')]
        
        # Group image sequences
        grouped_image_frames = group_video_frames(image_files) if image_files else []
        
        # Each MP4 file is treated as a single video
        grouped_mp4_frames = [[f] for f in mp4_files]
        
        # Combine both
        input_list = grouped_image_frames + grouped_mp4_frames

    input_labels = data_cfg.input_labels

    guidance_scale = args.guidance_scale
    use_dynamic_cfg = args.use_dynamic_cfg
    num_inference_steps = args.num_inference_steps
    lora_scale = args.lora_scale
    seed = args.seed

    output_dir = args.output_dir
    output_ll_dir = os.path.join(output_dir, 'ldr_log')
    os.makedirs(output_ll_dir, exist_ok=True)
    output_inp_dir = os.path.join(output_dir, 'input')
    os.makedirs(output_inp_dir, exist_ok=True)
    output_gt_dir = os.path.join(output_dir, 'gt')
    os.makedirs(output_gt_dir, exist_ok=True)
    output_vid_dir = os.path.join(output_dir, 'video')
    if args.data_type == 'video':
        os.makedirs(output_vid_dir, exist_ok=True)
    suffix = args.output_suffix

    for i, input_path in tqdm(enumerate(input_list)):
        frame_list = input_path
        if args.data_type == 'video':
            input_path = frame_list[0]

        input_name = os.path.basename(input_path)
        input_name = os.path.splitext(input_name)[0]
        
        output_name_ldr = os.path.join(output_ll_dir, f"{input_name}{suffix}_ldr.png")
        output_name_log = os.path.join(output_ll_dir, f"{input_name}{suffix}_log.png")
        if os.path.exists(output_name_ldr) and os.path.exists(output_name_log):
            print(f"Skip {input_name}")
            continue

        vid_example = READ_EXAMPLE_FUNC(
            input_path if args.data_type != 'video' else frame_list,
            resolution=resolution, env_resolution=env_resolution,
            input_labels=input_labels, crop_square=False, # img_prefix='0001'
            **data_read_kwargs
        ) # NOTE: FHWC np.float

        frame_list = vid_example.pop('frame_list', frame_list)
        frames_per_sample = vid_example['rgb'].shape[0]

        target_image, cond_images = \
            pipeline.example2input(vid_example, target_labels, cond_labels) # BHWC

        generator = torch.Generator(device=vae.device).manual_seed(seed) if seed else None

        frames_pred = []
        with torch.no_grad():
            pred = pipeline(
                prompt=text_prompt,
                cond_images=cond_images,
                cond_mapping=cond_labels,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                num_inference_steps=num_inference_steps, 
                generator=generator, 
                height=height, width=width, 
                num_frames=frames_per_sample,
                additional_cond_labels=additional_cond_labels,
                attention_kwargs={'scale': lora_scale},
                denoise_type=denoise_type,
                target_labels=_target_labels,
                additional_target_labels=_additional_target_labels,
                num_latents=len(target_labels),
                separate_timesteps=separate_timesteps,
                additional_rope_time_only=additional_rope_time_only,
            ).frames # list of pil images

        frames_ldr = pred[0][0]
        frames_log = pred[1][0]

        if frames_per_sample > 1:
            f_str = ''
            if args.data_type == 'video' and not input_path.endswith('.mp4'):
                f_str = '.' + input_name.split('.')[-1]
            output_name_video = output_name_ldr.replace(f'{f_str}_ldr', '').replace(output_ll_dir, output_vid_dir).replace('.png', '.mp4')
            # concate ldr and log
            frames = [
                np.concatenate([np.asarray(f_ldr), np.asarray(f_log)], axis=1) for f_ldr, f_log in zip(frames_ldr, frames_log)
            ]
            imageio.mimwrite(output_name_video, frames, fps=8)

        if args.data_type != 'video':
            frames_ldr[0].save(output_name_ldr)
            frames_log[0].save(output_name_log)
        else:
            for j, frame in enumerate(frame_list):
                frame_name = os.path.splitext(os.path.basename(frame))[0]
                output_name_ldr = os.path.join(output_ll_dir, f"{frame_name}{suffix}_ldr.png")
                output_name_log = os.path.join(output_ll_dir, f"{frame_name}{suffix}_log.png")
                frames_ldr[j].save(output_name_ldr)
                frames_log[j].save(output_name_log)

        if args.dump_input:
            inp_img = vid_example['rgb'][0]
            inp_img = (inp_img * 255).astype(np.uint8)
            Image.fromarray(inp_img).save(os.path.join(output_inp_dir, f"{input_name}.png"))

        if args.dump_gt:
            gt_ldr = vid_example['env_ldr'][0]
            gt_log = vid_example['env_log'][0]
            gt_ldr = (gt_ldr * 255).astype(np.uint8)
            gt_log = (gt_log * 255).astype(np.uint8)
            Image.fromarray(gt_ldr).save(os.path.join(output_gt_dir, f"{input_name}_ldr.png"))
            Image.fromarray(gt_log).save(os.path.join(output_gt_dir, f"{input_name}_log.png"))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--transformer_path", type=str)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--data_type", type=str, default="image")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument("--resolution", nargs='+', type=int, default=[480, 720])
    parser.add_argument("--env_resolution", nargs='+', type=int, default=[128, 256])
    parser.add_argument("--cam_elevation", type=float, default=0)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--use_dynamic_cfg", action='store_true')
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--lora_scale", type=float, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dump_input", action='store_true')
    parser.add_argument("--dump_gt", action='store_true')
    parser.add_argument("--frames_per_sample", type=int, default=None) # use all frames if None
    parser.add_argument("--frame_start_idx", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--camera_pose_file", type=str, default=None)
    args = parser.parse_args()

    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    missing_keys = set(cfg.keys()) - set(schema.keys())
    for key in missing_keys:
        OmegaConf.update(schema, key, None, force_add=True)
    cfg = OmegaConf.merge(schema, cfg)

    main(args, cfg)
