from typing import Dict, Optional, Tuple, List, Union, Any

import torch
from einops import rearrange, repeat
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, T5EncoderModel, T5Tokenizer

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

# CogVideoX utils

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds

def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds

# # FIXME: add v1.5
# def prepare_rotary_positional_embeddings(
#     height: int,
#     width: int,
#     num_frames: int,
#     vae_scale_factor_spatial: int = 8,
#     patch_size: int = 2,
#     attention_head_dim: int = 64,
#     device: Optional[torch.device] = None,
#     base_height: int = 480,
#     base_width: int = 720,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     grid_height = height // (vae_scale_factor_spatial * patch_size)
#     grid_width = width // (vae_scale_factor_spatial * patch_size)
#     base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
#     base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

#     grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
#     freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
#         embed_dim=attention_head_dim,
#         crops_coords=grid_crops_coords,
#         grid_size=(grid_height, grid_width),
#         temporal_size=num_frames,
#         device=device,
#     )

#     return freqs_cos, freqs_sin

# Copied from diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline._prepare_rotary_positional_embeddings
def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int,
    transformer_config: torch.nn.Module,
    device: torch.device,
    time_embed_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    p = transformer_config.patch_size
    p_t = transformer_config.patch_size_t

    base_size_width = transformer_config.sample_width // p
    base_size_height = transformer_config.sample_height // p

    if p_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + p_t - 1) // p_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
            device=device,
        )

    if time_embed_only:
        dim_t = transformer_config.attention_head_dim // 4
        # set [dim_t:] to 1 and 0
        freqs_cos[:, dim_t:] = 1.0
        freqs_sin[:, dim_t:] = 0.0

    return freqs_cos, freqs_sin

def prepare_latents(vae, pixel_values):
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    latents = vae.encode(pixel_values).latent_dist.sample()
    latents = rearrange(latents, "b c f h w -> b f c h w")
    latents = latents * vae.config.scaling_factor
    return latents