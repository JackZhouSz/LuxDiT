# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from einops import rearrange, repeat

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import CogVideoXLoraLoaderMixin
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import (
    CogVideoXImageToVideoPipeline,
    get_resize_crop_region_for_grid,
    retrieve_timesteps,
    retrieve_latents
)

from src.models.custom_cogvideox_transformer_3d import CustomCogVideoXTransformer3DModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    FIXME: Update to the new application.
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import CogVideoXImageToVideoPipeline
        >>> from diffusers.utils import export_to_video, load_image

        >>> pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        ... )
        >>> video = pipe(image, prompt, use_dynamic_cfg=True)
        >>> export_to_video(video.frames[0], "output.mp4", fps=8)
        ```
"""


class RGBXEnvCogVideoXPipeline(CogVideoXImageToVideoPipeline, DiffusionPipeline, CogVideoXLoraLoaderMixin):
    r"""
    Pipeline for RGBX generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CustomCogVideoXTransformer3DModel`]):
            A text conditioned `CustomCogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CustomCogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(
            tokenizer, text_encoder, vae, transformer, scheduler
        )

        # self.register_modules(
        #     tokenizer=tokenizer,
        #     text_encoder=text_encoder,
        #     vae=vae,
        #     transformer=transformer,
        #     scheduler=scheduler,
        # )
        # self.vae_scale_factor_spatial = (
        #     2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        # )
        # self.vae_scale_factor_temporal = (
        #     self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        # )
        # self.vae_scaling_factor_image = (
        #     self.vae.config.scaling_factor if hasattr(self, "vae") and self.vae is not None else 0.7
        # )

        # self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)


    def example2input(self, example, target_label, cond_labels):
        target_image = None
        if isinstance(target_label, str):
            if target_label in example:
                target_image = example[target_label][None, ...] # BFHWC
        else:
            target_image = {}
            for target_label in target_label:
                if target_label in example:
                    target_image[target_label] = example[target_label][None, ...]
                
        # cond_images = [example[cond_label][None, ...] for cond_label in cond_labels]
        cond_images = {}
        for cond_label, op in cond_labels.items():
            conds = cond_label.split('+') if '+' in cond_label else [cond_label]
            cond_images_group = []
            try:
                for cond in conds:
                    if '@' in cond:
                        cond, ch = cond.split('@')
                        ch = int(ch)
                        cond_images_group.append(example[cond][None, ..., ch:ch+1])
                    else:
                        cond_images_group.append(example[cond][None, ...])
                ret = np.concatenate(cond_images_group, axis=-1) # BFHWC
                if op in ['vae', 'clip'] and ret.shape[-1] == 1:
                    ret = ret.repeat(3, axis=-1)
                cond_images[cond_label] = ret
            except Exception as e:
                print(f"Failed to process cond_label {cond_label}: {e}")
        return target_image, cond_images

    # TODO:
    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        # For CogVideoX1.5-I2V, the latent should add 1 for padding (Not use)
        if self.transformer.config.patch_size_t is not None:
            shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    

    def prepare_cond_latents(
        self,
        cond_images: Dict[str, torch.Tensor],
        cond_mapping: Dict[str, str],
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 13,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        cond_latents: Optional[torch.Tensor] = None,
        drop_conds: Optional[List[str]] = None,
        additional_cond_labels: Optional[List[str]] = None,
        add_cond_latents: Optional[torch.Tensor] = None,
        add_cond_only: bool = False,
        exclude_cond_labels: Optional[List[str]] = None,
    ):
        if cond_latents is not None:
            cond_latents = cond_latents.to(device, dtype=dtype)
            add_cond_latents = add_cond_latents.to(device, dtype=dtype) if add_cond_latents is not None else None
        else:
            num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
            shape = (
                batch_size,
                num_frames,
                num_channels_latents,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            )
            if self.transformer.config.patch_size_t is not None:
                shape = shape[:1] + (shape[1] + shape[1] % self.transformer.config.patch_size_t,) + shape[2:]

            cond_latents = [] 
            add_cond_latents = []
            image_latents = None
            for key, encoding in cond_mapping.items():
                if encoding not in ['vae', 'downsample']:
                    continue
                if add_cond_only and key not in additional_cond_labels:
                    continue
                if exclude_cond_labels is not None and key in exclude_cond_labels:
                    continue
                if key in cond_images:
                    image = cond_images[key] # [B, F, C, H, W]
                    image = rearrange(image, "b f c h w -> b c f h w").to(device, dtype=dtype) # [B, C, F, H, W]
                    image_latents = None
                    if encoding == 'vae':
                        if isinstance(generator, list):
                            image_latents = [
                                retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
                            ]
                        else:
                            image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in image]

                        image_latents = torch.cat(image_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

                        if not self.vae.config.invert_scale_latents:
                            image_latents = self.vae_scaling_factor_image * image_latents
                        else:
                            # This is awkward but required because the CogVideoX team forgot to multiply the
                            # scaling factor during training :)
                            image_latents = 1 / self.vae_scaling_factor_image * image_latents

                        # For CogVideoX1.5, the latent should add 1 for padding
                        # Select the first frame along the second dimension
                        if self.transformer.config.patch_size_t is not None:
                            first_frame = image_latents[:, : image_latents.size(1) % self.transformer.config.patch_size_t, ...]
                            image_latents = torch.cat([first_frame, image_latents], dim=1)

                        # assert image_latents.shape == shape, f"Expected shape {shape}, got {image_latents.shape}"
                    else:
                        raise NotImplementedError(f"Encoding {encoding} is not supported for cond_latents")
                
                    if drop_conds is not None and key in drop_conds:
                        image_latents = torch.zeros_like(image_latents)
                else:
                    print(f"Warning: {key} not in cond_images, using zeros")
                    image_latents = torch.zeros_like(image_latents)
                
                if additional_cond_labels is not None and key in additional_cond_labels:
                    add_cond_latents.append(image_latents)
                else:
                    cond_latents.append(image_latents)

            cond_latents = torch.cat(cond_latents, dim=-3) if len(cond_latents) > 0 else None # [B, F, C, H, W]
            add_cond_latents = torch.cat(add_cond_latents, dim=-3) if len(add_cond_latents) > 0 else None # [B, F, C, H, W]

        return cond_latents, add_cond_latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        # ----------------- Custom Inputs -----------------
        cond_images: Optional[Dict[str, torch.Tensor]] = None,
        cond_mapping: Optional[Dict[str, str]] = None,
        cond_latents: Optional[torch.FloatTensor] = None,
        # additional_inputs=None,
        use_cond_guidance=True,
        additional_cond_labels: Optional[List[str]] = None,
        num_latents: int = 1,
        drop_conds: Optional[List[str]] = None,
        denoise_type: str = 'hidden_states',
        target_labels: Optional[Union[str, List[str]]] = None,
        additional_target_labels: Optional[List[str]] = None,
        additional_rope_time_only: bool = True,
        separate_timesteps: bool = False,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            image=[], # just a placeholder
            prompt='',
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        device = self._execution_device
        dtype = self.vae.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        if prompt is not None or prompt_embeds is not None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        


        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        # image = self.video_processor.preprocess(image, height=height, width=width).to(
        #     device, dtype=dtype
        # )

        # Preprocess image
        cond_images_proc = {}
        for key in cond_mapping:
            if key == "input_context":
                continue
            # NOTE: preprocess does not support 5D input
            if key not in cond_images:
                continue
            cond_image = rearrange(cond_images[key], "b f h w c -> (b f) h w c")
            image = self.video_processor.preprocess(cond_image).to(device, dtype=dtype) # Resize + Normalize (-1,1) for all. (bf)chw
            # if noise_aug_strength != 0:
            #     noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
            #     image = image + noise_aug_strength * noise
            cond_images_proc[key] = rearrange(image, "(b f) h w c -> b f h w c", b=batch_size)


        latent_channels = self.vae.config.latent_channels
        exclude_cond_labels = target_labels if denoise_type == 'hidden_states' else additional_target_labels

        cond_latents, add_cond_latents = self.prepare_cond_latents(
            cond_images_proc,
            cond_mapping,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            dtype,
            device,
            generator,
            cond_latents,
            additional_cond_labels=additional_cond_labels,
            drop_conds=drop_conds,
            exclude_cond_labels=exclude_cond_labels,
        ) # [batch, frames, channels, height, width]

        latents = None
        if denoise_type == 'hidden_states':
            if target_labels is not None:
                assert len(target_labels) == num_latents, "The number of target_labels should be equal to num_latents"
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                latent_channels * num_latents,
                num_frames,
                height,
                width,
                dtype,
                device,
                generator,
                latents,
            )
        else:
            a_height, a_width = add_cond_latents.shape[-2:]
            if additional_target_labels is not None:
                assert len(additional_target_labels) == num_latents, "The number of additional_target_labels should be equal to num_latents"
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                latent_channels * num_latents, # make sure this is not None
                num_frames,
                a_height * self.vae_scale_factor_spatial,
                a_width * self.vae_scale_factor_spatial,
                dtype,
                device,
                generator,
                latents, # be careful here, although rarely used
            )
            # add_cond_latents = torch.cat([add_latents, add_cond_latents], dim=-3) # channel concat

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # TODO: Add support for additional token embeddings
        if self.transformer.config.use_rotary_positional_embeddings and add_cond_latents is not None:
            _, add_fsz, _, add_h, add_w = add_cond_latents.size()
            additional_fcos, additional_fsin = self._prepare_rotary_positional_embeddings(
                add_h * self.vae_scale_factor_spatial, add_w * self.vae_scale_factor_spatial,
                add_fsz, device
            )
            if additional_rope_time_only:
                dim_t = additional_fcos[0].size(-1) // 4
                additional_fcos[:, dim_t:] = 1
                additional_fsin[:, dim_t:] = 0
            image_rotary_emb = (
                torch.cat([image_rotary_emb[0], additional_fcos], dim=0),
                torch.cat([image_rotary_emb[1], additional_fsin], dim=0)
            )

        # additional inputs
        hidden_states = None
        num_hidden_latents = len(target_labels) if target_labels is not None else 1
        hidden_states_dim = latent_channels * num_hidden_latents
        additional_hidden_states = None
        additional_model_input = None
        if cond_latents is not None:
            if use_cond_guidance and do_classifier_free_guidance:
                # For classifier-free guidance with conditional guidance
                # First half gets zeros (unconditioned), second half gets the condition
                additional_model_input = torch.cat([torch.zeros_like(cond_latents), cond_latents], dim=0)
                if add_cond_latents is not None:
                    if denoise_type != 'additional_hidden_states':
                        add_cond_latents = torch.cat([torch.zeros_like(add_cond_latents), add_cond_latents], dim=0)
                    else:
                        add_cond_latents = torch.cat([add_cond_latents] * 2, dim=0)
            elif do_classifier_free_guidance:
                # When using classifier-free guidance without conditional guidance
                # Both halves get the condition
                additional_model_input = torch.cat([cond_latents] * 2, dim=0)
                if add_cond_latents is not None:
                    add_cond_latents = torch.cat([add_cond_latents] * 2, dim=0)
            else:
                # No classifier-free guidance case
                additional_model_input = cond_latents
                if add_cond_latents is not None:
                    add_cond_latents = add_cond_latents
        additional_hidden_states = add_cond_latents
        
        if denoise_type == 'additional_hidden_states':
            cond_dim = additional_model_input.shape[-3]
            hidden_states, additional_model_input = additional_model_input.split_with_sizes([hidden_states_dim, cond_dim-hidden_states_dim], dim=-3)

        # 8. Create ofs embeds if required
        ofs_emb = None if self.transformer.config.ofs_embed_dim is None else latents.new_full((1,), fill_value=2.0)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # if additional_inputs is not None:
                #     # don't need adjustment since additional_model_input is already taking care of it (for guidance)
                #     timestep = timestep * (1 - additional_model_input[:, 0, -1, 0, 0]) # Set timestep of conditional video to 0

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) 
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                target_timestep = t.expand(latent_model_input.shape[0])
                # Tricky part here:
                timestep, additional_timestep = target_timestep, None
                if denoise_type == 'hidden_states':
                    hidden_states = latent_model_input
                    if separate_timesteps:
                        additional_timestep = target_timestep * 0
                else:
                    additional_hidden_states = torch.cat([latent_model_input, add_cond_latents], dim=-3)
                    if separate_timesteps:
                        additional_timestep = target_timestep
                        timestep = target_timestep * 0


                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    additional_timestep=additional_timestep,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    additional_inputs=additional_model_input,
                    additional_hidden_states=additional_hidden_states,
                    denoise_type=denoise_type,
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            if num_latents == 1:
                video = self.decode_latents(latents)
                video = self.video_processor.postprocess_video(video=video, output_type=output_type)
            else:
                video_list = []
                for i in range(num_latents):
                    video = self.decode_latents(latents[:, :, i * latent_channels : (i + 1) * latent_channels])
                    video = self.video_processor.postprocess_video(video=video, output_type=output_type)
                    video_list.append(video)
                video = video_list
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
