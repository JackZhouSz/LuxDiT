import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
from omegaconf import OmegaConf, ListConfig, DictConfig

import os
import numpy as np
from PIL import Image
import cv2
import imageio
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import BaseOutput, logging
from src.data.rendering_datasets import READ_EXAMPLE_FUNC
from src.pipelines.pipeline_cogvideox_rgbxenv import RGBXEnvCogVideoXPipeline
from src.utils.utils_cogvideox import compute_prompt_embeddings, prepare_rotary_positional_embeddings

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()
    # u = torch.randn(shape, dtype=dtype, device=device)
    # return (u * scale + loc).exp()

def resize_pad_image(image: np.ndarray, target_size: List[int], pad_val=255) -> np.ndarray:
    """Resize image with original aspect ratio to best fit target_size, then pad to target_size."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    target_aspect_ratio = target_w / target_h
    aspect_ratio = w / h

    if aspect_ratio > target_aspect_ratio:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[pad_val]*3)
    return padded_image


def prepare_dropout_mask(
    latents, labels,
    latent_dim=16,
    cond_dropout={},
    cond_dropout_reuse=[],
    cond_exclusive=None, # not supported yet
    dropout_map={},
    compact_mask=False
):
    bsz = latents.shape[0]
    num_latents = latents.shape[2] // latent_dim
    assert num_latents == len(labels)
    rand_mask_list = []
    for cond_image in labels:
        droprate = cond_dropout.get(cond_image, 0.0)
        if isinstance(droprate, str):
            rand_mask = dropout_map[droprate]
        elif isinstance(droprate, list) or isinstance(droprate, ListConfig):
            # droprate = [label, 0.1]: mask both on previous label and new dropouts
            rand_mask = dropout_map[droprate[0]]
            rand_mask = rand_mask * (torch.rand(bsz, device=latents.device) > droprate[1])
        elif droprate > 0:
            rand_mask = torch.rand(bsz, device=latents.device) > droprate
        else:
            rand_mask = torch.ones(bsz, device=latents.device) > 0

        if cond_image in cond_dropout_reuse:
            dropout_map[cond_image] = rand_mask

        rand_mask_list.append(rand_mask)
    
    rand_dropout_mask = torch.stack(rand_mask_list, dim=1) # B x N
    if not compact_mask:
        rand_dropout_mask = rand_dropout_mask[..., None].repeat(1, 1, latent_dim).reshape(bsz, 1, -1, 1, 1)  # B 1 NF
    return rand_dropout_mask

        
@torch.no_grad()
def prepare_training_conditions(
    cfg, # for simplicity just use cfg dict for now.
    batch,
    latents,
    cond_images,
    model_config=None,
    vae=None,
    tokenizer=None,
    text_encoder=None,
    # env_encoder=None,
    reuse_encoder_hidden_states=True,
    encoder_hidden_states=None, 
    image_rotary_emb=None,
    additional_rotary_emb=None,
    additional_rope_time_only=True,
    additional_cond_labels=None,
    vae_scale_factor_spatial=8,
    cond_dropout={},
    cond_dropout_reuse=[],
    cond_dropout_skip=None, # to skip dropout all
    cond_exclusive=None,
    cond_dropout_all=0,
    accelerator=None,
    weight_dtype=None,
    exclude_labels=[],
    dropout_map={},
):
    bsz, fsz, ch, latents_h, latents_w = latents.shape
    vae_latent_dim = vae.config.latent_channels
    time_context = None

    # # Prepare rotary embeds
    if model_config.use_rotary_positional_embeddings:
        if image_rotary_emb is None:
            image_rotary_emb = prepare_rotary_positional_embeddings(
                height=latents_h * vae_scale_factor_spatial,
                width=latents_w * vae_scale_factor_spatial,
                num_frames=fsz,  # NOTE: fsz is not px_fsz
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                transformer_config=model_config,
                device=accelerator.device,
            )
        
        if  additional_cond_labels is not None and additional_rotary_emb is None:
            _, _, _, env_h, env_w = batch['env_ldr'].shape 
            additional_rotary_emb = prepare_rotary_positional_embeddings(
                height=env_h,
                width=env_w,
                num_frames=fsz,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                transformer_config=model_config,
                time_embed_only=additional_rope_time_only,
                device=accelerator.device,
            )

    # Get cross-attention embeddings for conditioning
    if not reuse_encoder_hidden_states or encoder_hidden_states is None:
        if cfg.cond_mode == 'text':
            # Get the text embedding for conditioning
            encoder_hidden_states = compute_prompt_embeddings(
                tokenizer,
                text_encoder,
                batch["caption"],
                model_config.max_text_seq_length,
                accelerator.device,
                weight_dtype,
                requires_grad=False,
            )
        elif cfg.cond_mode == 'skip':
            encoder_hidden_states = torch.zeros(bsz, 1, 1, model_config.cross_attention_dim).to(latents)

    # time context
    cond_aug = cfg.model_pipeline.get('cond_aug', 0)

    # Sample noise for the condition noise augmentation
    if cond_aug != 0:
        cond_sigma_mean = cfg.model_pipeline.get('cond_sigma_mean', -3.0)
        cond_sigma_std = cfg.model_pipeline.get('cond_sigma_std', 0.5)
        cond_sigmas = rand_log_normal(shape=[bsz,], loc=cond_sigma_mean, scale=cond_sigma_std).to(latents)
        # noise_aug_strength = cond_sigmas[:]
        cond_sigmas = cond_sigmas[:, None, None, None, None]

    # Conditional Latents
    exclude_mask = {}
    if cond_exclusive is not None:
        choice = torch.randint(0, len(cond_exclusive), (bsz,), device=latents.device)
        exclude_mask = {k: i == choice for i, k in enumerate(cond_exclusive)}

    # Prepare additional conditioning latents
    cond_latents = []
    additional_cond_latents = []
    global_rand_mask = None
    if cond_dropout_all > 0:
        global_rand_mask = torch.rand(latents.shape[0], device=latents.device) > cond_dropout_all
        dropout_map['global'] = global_rand_mask

    scale_cond_latents = cfg.model_pipeline.get('scale_cond_latents', False)
    for cond_image, encoding in cond_images.items():
        _latents = None
        if encoding not in ['vae', 'downsample']:
            continue
        if cond_image in exclude_labels:
            continue

        if encoding == 'vae':
            if cond_image in batch:
                cond_image_batch = batch[cond_image]
                if cond_aug != 0:
                    cond_noise = torch.randn_like(cond_image_batch) 
                    cond_image_batch = cond_image_batch + cond_noise * cond_sigmas
                cond_image_batch = rearrange(cond_image_batch, "b f c h w -> b c f h w")
                _latents = vae.encode(cond_image_batch.to(weight_dtype)).latent_dist.sample() # NOTE: mode -> sample (we use mode in SVD-RGBX)
                _latents = rearrange(_latents, "b c f h w -> b f c h w")
                if not vae.config.invert_scale_latents:
                    _latents = vae.config.scaling_factor * _latents
                else:
                    # NOTE: be careful with this, if using v1.5
                    # https://github.com/THUDM/CogVideo/issues/570#issuecomment-2550667997
                    _latents = 1 / vae.config.scaling_factor * _latents
            else:
                _latents = torch.zeros(bsz, fsz, vae_latent_dim, latents_h, latents_w).to(latents)
        elif encoding == 'downsample':
            _latents = torch.nn.functional.interpolate(
                batch[cond_image].to(device=accelerator.device, dtype=weight_dtype),
                size=(latents_h, latents_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

        # Condition Dropout
        droprate = cond_dropout.get(cond_image, 0.0)
        
        rand_mask = None
        if isinstance(droprate, str):
            if droprate in dropout_map:
                rand_mask = dropout_map[droprate]
        elif isinstance(droprate, list) or isinstance(droprate, ListConfig):
            # droprate = [label, 0.1]: mask both on previous label and new dropouts
            rand_mask = torch.rand(_latents.shape[0], device=_latents.device) > droprate[1]
            if droprate[0] in dropout_map:
                rand_mask = rand_mask * dropout_map[droprate[0]]
        elif droprate > 0:
            rand_mask = torch.rand(_latents.shape[0], device=_latents.device) > droprate

        if cond_image in cond_dropout_reuse:
            dropout_map[cond_image] = rand_mask
        
        if global_rand_mask is not None:
            if cond_dropout_skip is not None and cond_image in cond_dropout_skip:
                pass
            else:
                rand_mask = rand_mask * global_rand_mask if rand_mask is not None else global_rand_mask

        if rand_mask is not None:
            _latents = torch.einsum('b...,b->b...', _latents, rand_mask)

        if cond_exclusive is not None:
            for k, v in exclude_mask.items():
                if cond_image.startswith(k):
                    _latents = torch.einsum('b...,b->b...', _latents, v)
                    break
        
        if additional_cond_labels is not None and cond_image in additional_cond_labels:
            additional_cond_latents.append(_latents)
        else:
            cond_latents.append(_latents)

    cond_latents = torch.cat([*cond_latents], dim=2) if len(cond_latents) > 0 else None
    additional_cond_latents = torch.cat([*additional_cond_latents], dim=2) if len(additional_cond_latents) > 0 else None

    # dropout for hidden states
    if cfg.cond_mode != 'text' and cfg.cond_mode != 'skip' and encoder_hidden_states is not None:
        droprate = max(cond_dropout.get('encoder_hidden_states', 0), cond_dropout.get('clip_img', 0))
        rand_mask = None
        if droprate > 0:
            rand_mask = torch.rand(latents.shape[0], device=latents.device) > droprate
        if global_rand_mask is not None:
            rand_mask = rand_mask * global_rand_mask if rand_mask is not None else global_rand_mask

        if isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = torch.einsum('b...,b->b...', encoder_hidden_states, rand_mask)
        else:
            encoder_hidden_states = [torch.einsum('b...,b->b...', v, rand_mask) for v in encoder_hidden_states]

    return (
        encoder_hidden_states, cond_latents, additional_cond_latents,
        image_rotary_emb, additional_rotary_emb
    )

def log_validation(
        vae, 
        transformer, 
        tokenizer=None,
        text_encoder=None,
        cfg=None, 
        accelerator=None, 
        epoch_or_step=0,
        default_text_prompt=None,
        default_negative_text_prompt=None,
        return_pipeline=False
    ):
    logger.info("Running validation... ")

    # or DDIM
    noise_scheduler = CogVideoXDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")

    pipeline = RGBXEnvCogVideoXPipeline(
        vae=vae, 
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=accelerator.unwrap_model(transformer),
        scheduler=noise_scheduler
    )

    pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=True)
    # pipeline.enable_sequential_cpu_offload()
    pipeline.vae.enable_tiling()

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

    image_logs = []

    data_cfg = cfg.train_data
    val_data_cfg = cfg.validation_data
    model_cfg = cfg.model_pipeline
    inference_cfg = cfg.inference_kwargs
    if isinstance(inference_cfg, DictConfig) or isinstance(inference_cfg, dict):
        inference_cfg = [inference_cfg]

    _target_labels, cond_labels = model_cfg.target_image, model_cfg.cond_images
    _additional_target_labels = model_cfg.get('additional_target_image', [])
    additional_cond_labels = model_cfg.get('additional_cond_labels', None)
    additional_rope_time_only = model_cfg.get('additional_rope_time_only', True)
    default_denoise_type = model_cfg.get('denoise_type', 'additional_hidden_states')
    separate_timesteps = model_cfg.get('separate_timesteps', False)
    dataset_name = inference_cfg[0].get('dataset_name', data_cfg.get('dataset_name', 'wds_rendering_data_objaverse_vid'))

    _frames_per_sample = inference_cfg[0].get('frames_per_sample', data_cfg.get('frames_per_sample', 14))
    _num_inference_steps = inference_cfg[0].get('num_inference_steps', 25)
    _lora_scale = inference_cfg[0].get('lora_scale', 0)
    _num_infer_per_image = inference_cfg[0].get('num_infer_per_image', 2)
    _text_prompt = inference_cfg[0].get('text_prompt', default_text_prompt)
    _negative_text_prompt = inference_cfg[0].get('negative_text_prompt', default_negative_text_prompt)
    _guidance_scale = inference_cfg[0].get('guidance_scale', 1.5)
    _use_dynamic_cfg = inference_cfg[0].get('use_dynamic_cfg', False)
    _fps = inference_cfg[0].get('fps', 8)
    _viz_rows = inference_cfg[0].get('viz_rows', 1)
    _resolution = inference_cfg[0].get('resolution', [512, 512])
    _env_resolution = inference_cfg[0].get('env_resolution', _resolution)

    drop_conds = None # TODO: add support for this

    save_dir = os.path.join(cfg.output_dir, "validation", f'{epoch_or_step:06d}')
    os.makedirs(save_dir, exist_ok=True)

    num_validation_images = len(val_data_cfg.validation_images)
    # validation_image can be a list of images or list of list
    # for validation_image in val_data_cfg.validation_images:
    for val_idx in range(num_validation_images):
        validation_image = val_data_cfg.validation_images[val_idx]
        cur_dataset_name = dataset_name
        (
            num_inference_steps, lora_scale, num_infer_per_image, frames_per_sample, 
            text_prompt, negative_text_prompt, guidance_scale, use_dynamic_cfg, fps, 
            resolution, env_resolution, viz_rows, denoise_type
        ) = (
            _num_inference_steps, _lora_scale, _num_infer_per_image, _frames_per_sample, 
            _text_prompt, _negative_text_prompt, _guidance_scale, _use_dynamic_cfg, _fps, 
            _resolution, _env_resolution, _viz_rows, default_denoise_type
        )

        dataset_kwargs = {}
        if isinstance(validation_image, list) or isinstance(validation_image, ListConfig):
            dataset_idx, validation_image = validation_image
            cur_inference_cfg = inference_cfg[dataset_idx]
            cur_dataset_name = cur_inference_cfg.get('dataset_name', dataset_name)
            num_inference_steps = cur_inference_cfg.get('num_inference_steps', num_inference_steps)
            lora_scale = cur_inference_cfg.get('lora_scale', lora_scale)
            num_infer_per_image = cur_inference_cfg.get('num_infer_per_image', num_infer_per_image)
            frames_per_sample = cur_inference_cfg.get('frames_per_sample', frames_per_sample)
            text_prompt = cur_inference_cfg.get('text_prompt', text_prompt)
            guidance_scale = cur_inference_cfg.get('guidance_scale', guidance_scale)
            use_dynamic_cfg = cur_inference_cfg.get('use_dynamic_cfg', use_dynamic_cfg)
            fps = cur_inference_cfg.get('fps', fps)
            viz_rows = cur_inference_cfg.get('viz_rows', viz_rows)
            resolution = cur_inference_cfg.get('resolution', resolution)
            env_resolution = cur_inference_cfg.get('env_resolution', env_resolution)
            denoise_type = cur_inference_cfg.get('denoise_type', denoise_type)
            dataset_kwargs = cur_inference_cfg.get('dataset_kwargs', {})

        if denoise_type == 'additional_hidden_states':
            target_labels = _additional_target_labels
            viz_input = _target_labels
        else:
            target_labels = _target_labels
            viz_input = _additional_target_labels
        if len(viz_input) >= len(target_labels):
            viz_input = viz_input[:len(target_labels)]
        else:
            viz_input = viz_input + viz_input[-1:] * (len(target_labels) - len(viz_input))

        # TODO: may need to change this to multi-dataset format?
        vid_example = READ_EXAMPLE_FUNC[cur_dataset_name](
            validation_image, input_labels=data_cfg.input_labels, 
            frames_per_sample=frames_per_sample, resolution=resolution, env_resolution=env_resolution,
            **dataset_kwargs
        ) # NOTE: FHWC np.float [0, 1]

        for key in vid_example.keys():
            if vid_example[key].ndim == 3:
                vid_example[key] = vid_example[key][None] # 1HWC

        target_image, cond_images = \
            pipeline.example2input(vid_example, target_labels, cond_labels) # BHWC
        
        height, width = resolution
        
        frames_gt = []
        if viz_input is not None:
            for key in viz_input:
                if key in vid_example:
                    frames_gt.append((vid_example[key].clip(0, 1) * 255.0).astype(np.uint8))
        if target_image is not None:
            # frames_gt = (target_image[0].clip(0, 1) * 255.0).astype(np.uint8) # NHWC, uint8
            for key in target_image.keys():
                _frames_gt = (target_image[key][0].clip(0, 1) * 255.0).astype(np.uint8) # NHWC, uint8
                frames_gt.append(_frames_gt)

        frames_pred = []
        for i in range(num_infer_per_image):
            # with torch.autocast("cuda"): # https://github.com/huggingface/transformers/issues/10830
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

            if len(target_labels) == 1:
                frames_pred.append(pred[0])
            else:
                for pred_f in pred:
                    frames_pred.append(pred_f[0])
            
        # export frames here before posting to trackers
        frame_dump_name = f'val_{val_idx:03d}'
        vid_frames = []
        for fid in range(frames_per_sample):
            formatted_images = [resize_pad_image(f_gt[fid], (width, height)) for f_gt in frames_gt] if frames_gt is not None else []
            for frames in frames_pred:
                formatted_images.append(resize_pad_image(np.asarray(frames[fid]), (width, height)))
            if viz_rows > 1:
                num_cols = len(formatted_images) // viz_rows
                formatted_images = [np.concatenate(formatted_images[i*num_cols:(i+1)*num_cols], axis=1) for i in range(viz_rows)]
                formatted_images = np.concatenate(formatted_images, axis=0)
            else:
                formatted_images = np.concatenate(formatted_images, axis=1)
            vid_frames.append(formatted_images)
        if len(vid_frames) > 1:
            save_path = os.path.join(save_dir, f'{frame_dump_name}.mp4')
            imageio.mimsave(save_path, vid_frames, fps=fps, codec='h264')
        else:
            save_path = os.path.join(save_dir, f'{frame_dump_name}.jpg')
            imageio.imwrite(save_path, vid_frames[0])
        image_logs.append(save_path)

    pipeline.vae.disable_tiling()
    
    if return_pipeline:
        return pipeline
    
    del pipeline
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return None