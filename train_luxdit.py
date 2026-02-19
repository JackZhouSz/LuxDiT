#!/usr/bin/env python

import argparse
import logging
import math
import os
import gc
import glob
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, FullStateDictConfig
)

import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs, AutocastKwargs
from accelerate import FullyShardedDataParallelPlugin, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, DistributedType
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from einops import rearrange, repeat

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, T5EncoderModel, T5Tokenizer
from transformers.utils import ContextManagers

from typing import Dict, Optional, Tuple, List, Union, Any
from omegaconf import OmegaConf, ListConfig
from dataclasses import dataclass

import diffusers
from diffusers.loaders import LoraLoaderMixin
from diffusers import (
    # AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.model_loading_utils import load_state_dict
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.training_utils import cast_training_params, free_memory

from diffusers.utils import (
    check_min_version, deprecate, 
    is_wandb_available, make_image_grid,
    export_to_video, export_to_gif, convert_unet_state_dict_to_peft
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from src.utils.random_state_utils import save_random_state
from src.utils.utils_cogvideox import (
    encode_prompt, compute_prompt_embeddings, prepare_rotary_positional_embeddings, prepare_latents
)
from src.models.custom_cogvideox_transformer_3d import CustomCogVideoXTransformer3DModel
from src.models.custom_autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from dataloader import create_dataloader, create_dataloader_iterators, instantiate_from_config, MultiLoaderIterator
from utils_luxdit import prepare_training_conditions, log_validation, prepare_dropout_mask
from luxdit_config import TrainingConfig

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.3")

logger = get_logger(__name__, log_level="INFO")

def main(
    cfg: TrainingConfig
):

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != cfg.local_rank:
        cfg.local_rank = env_local_rank

    if cfg.report_to == "wandb" and cfg.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(cfg.output_dir, cfg.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
    find_unused_parameters = cfg.find_unused_parameters
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    autocast_kwargs = AutocastKwargs(cache_enabled=cfg.autocast_cache_enabled)

    if cfg.use_fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
            use_orig_params=True, # useless True
        )
        fsdp_plugin.use_orig_params = True # Stupid stupid design in accelerate
        assert not cfg.use_ema, "FSDP does not support EMAModel yet, please consider DeepSpeed"
        assert cfg.validation_steps is None, "FSDP does not support validation yet, please consider DeepSpeed"
    else:
        fsdp_plugin = None
    
    if cfg.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=cfg.deepspeed_config,
            zero_stage=cfg.zero_stage, # this will be forced overwritten by the config if any
        )
        deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = cfg.gradient_accumulation_steps
        deepspeed_plugin.deepspeed_config['gradient_clipping'] = cfg.max_grad_norm
        deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = cfg.train_batch_size
        if cfg.mixed_precision == "fp16":
            deepspeed_plugin.deepspeed_config['fp16'] = {
                "enabled": 'auto',
                "auto_cast": True,
                "initial_scale_power": 16,
            }
        elif cfg.mixed_precision == "bf16":
            deepspeed_plugin.deepspeed_config['bf16'] = {
                "enabled": True
            }
    else:
        deepspeed_plugin = None

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[ddp_kwargs, autocast_kwargs]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        # the seed across all processes to make sure that models initialized in the same way
        set_seed(cfg.seed) 

    # Set paths
    if cfg.model_pipeline.get('text_encoder_path', None) is None:
        cfg.model_pipeline['text_encoder_path'] = cfg.pretrained_model_name_or_path
    if cfg.model_pipeline.get('vae_path', None) is None:
        cfg.model_pipeline['vae_path'] = cfg.pretrained_model_name_or_path
    if cfg.model_pipeline.get('transformer_path', None) is None:
        cfg.model_pipeline['transformer_path'] = cfg.pretrained_model_name_or_path

    cfg.cond_mode = cfg.model_pipeline.get('cond_mode', 'image')
            
    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)
            OmegaConf.save(cfg, os.path.join(cfg.output_dir, "config.yaml"))

    # Load scheduler, tokenizer and models.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_pipeline['text_encoder_path'], subfolder="tokenizer", revision=cfg.revision
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        # return [deepspeed_plugin.zero3_init_context_manager(enable=False)] # NOT working for no reason
        return [transformers.modeling_utils.set_zero3_state()] # a hacky way to disable zero.Init

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    text_encoder, vae, env_encoder = None, None, None
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        if cfg.cond_mode == 'text':
            # https://github.com/huggingface/transformers/issues/5486
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            text_encoder = T5EncoderModel.from_pretrained(
                cfg.model_pipeline['text_encoder_path'], subfolder="text_encoder", 
                revision=cfg.revision, variant=cfg.variant
            )
        elif cfg.cond_mode == 'env':
            pass
        else:
            raise ValueError(f"Unsupported cond_mode: {cfg.cond_mode}")

        vae = AutoencoderKLCogVideoX.from_pretrained(
            cfg.model_pipeline['vae_path'], subfolder="vae", revision=cfg.revision, variant=cfg.variant
        )

        if cfg.enable_slicing:
            vae.enable_slicing()
        if cfg.enable_tiling:
            vae.enable_tiling()
    
    vae_latent_dim = vae.config.latent_channels
    transformer_kwargs = dict(cfg.model_pipeline['transformer_kwargs'])
    for k, v in transformer_kwargs.items():
        if isinstance(v, list) or isinstance(v, ListConfig):
            transformer_kwargs[k] = tuple(v)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        cfg.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        cfg.mixed_precision = accelerator.mixed_precision
    
    # FP16 is not used to load trainable weights
    # transformer_dtype = torch.float32 if cfg.mixed_precision == "fp16" else weight_dtype
    transformer_dtype = torch.float32
    if cfg.transformer_precision is not None:
        transformer_dtype = torch.bfloat16 if cfg.transformer_precision == "bf16" else transformer_dtype
    use_learned_positional_embeddings = transformer_kwargs.pop("use_learned_positional_embeddings", None)
    if cfg.pretrained_transformer is not None:
        transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
            cfg.pretrained_transformer, 
            subfolder="transformer", 
            torch_dtype=transformer_dtype, 
            **transformer_kwargs
        )
    else:
        additional_input_channels = transformer_kwargs.pop("additional_input_channels", None)
        additional_output_channels = transformer_kwargs.pop("additional_output_channels", None)
        additional_patch_embed_channels = transformer_kwargs.pop("additional_patch_embed_channels", None)
        in_channels = transformer_kwargs.pop("in_channels", None) # assume in_channels is only for noise
        out_channels = transformer_kwargs.pop("out_channels", None)
        duplicate_channels = transformer_kwargs.pop("duplicate_channels", False)
        transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
            cfg.model_pipeline['transformer_path'],
            subfolder="transformer",
            torch_dtype=transformer_dtype,
            revision=cfg.revision,
            variant=cfg.variant,
            # additional_input_channels=None, # JON: Adding weights after because can't get the -5B dumps to work with devuce_map=None, low_cpu_mem_usage=False trick
            **transformer_kwargs,
        )
        transformer.additional_input_channels(additional_input_channels) 
        transformer.disentangle_image_channels(vae_latent_dim=vae_latent_dim)
        transformer.init_additional_patch_embed(additional_patch_embed_channels)
        transformer.extend_input_channels(in_channels, duplicate_channels=duplicate_channels)
        transformer.extend_output_channels(out_channels)
        transformer.init_additional_output_layes(additional_output_channels)
        transformer_kwargs['additional_input_channels'] = additional_input_channels
        transformer_kwargs['additional_patch_embed_channels'] = additional_patch_embed_channels
        transformer_kwargs['out_channels'] = out_channels
        transformer_kwargs['in_channels'] = in_channels

        with torch.no_grad():
            if transformer_kwargs.get('copy_weights', True):
                print("Copying weights from pretrained transformer to additional layers")
                # copy weights
                if transformer.additional_patch_embed is not None:
                    num_latents = additional_output_channels // vae_latent_dim
                    patch_embed_proj = transformer.patch_embed.proj
                    transformer.additional_patch_embed.proj.weight.data[:, :vae_latent_dim] = patch_embed_proj.weight.data[:, :vae_latent_dim]
                    if duplicate_channels:
                        for i in range(1, num_latents):
                            transformer.additional_patch_embed.proj.weight.data[:, i*vae_latent_dim:(i+1)*vae_latent_dim] = patch_embed_proj.weight.data[:, :vae_latent_dim]
                        transformer.additional_patch_embed.proj.weight.data[:] = transformer.additional_patch_embed.proj.weight.data / num_latents
                    transformer.additional_patch_embed.proj.bias.data[:] = patch_embed_proj.bias.data
                    num_additional_input = additional_input_channels // vae_latent_dim
                    add_proj_weight = 0
                    for i in range(num_additional_input):
                        add_proj_weight += transformer.additional_input_embed.proj.weight.data[:, i*vae_latent_dim:(i+1)*vae_latent_dim]
                    transformer.additional_patch_embed.proj.weight.data[:, num_latents*vae_latent_dim:(num_latents+1)*vae_latent_dim] = \
                        add_proj_weight
                    # transformer.additional_patch_embed.proj.weight.data[:, num_latents*vae_latent_dim:(num_latents+1)*vae_latent_dim] = \
                    #     transformer.additional_input_embed.proj.weight.data[:, :vae_latent_dim]

            if transformer_kwargs.get('copy_temb_layer', True):
                swap_temb = transformer_kwargs.get('swap_temb', False)
                print("Copying weights from pretrained transformer to additional temb layer")
                for block in transformer.transformer_blocks:
                    linear_dim = block.norm1.linear.weight.shape[0]
                    half_dim = linear_dim // 2
                    for norm_layer in [block.norm1, block.norm2]:
                        if not swap_temb:
                            norm_layer.linear.weight.data[half_dim:] = norm_layer.linear.weight.data[:half_dim]
                        else:
                            half_1, half_2 = norm_layer.linear.weight.data.chunk(2, dim=0)
                            norm_layer.linear.weight.data[:] = torch.cat([half_2, half_1], dim=0)
                        if norm_layer.linear.bias is not None:
                            if not swap_temb:
                                norm_layer.linear.bias.data[half_dim:] = norm_layer.linear.bias.data[:half_dim]
                            else:
                                half_1, half_2 = norm_layer.linear.bias.data.chunk(2, dim=0)
                                norm_layer.linear.bias.data[:] = torch.cat([half_2, half_1], dim=0)

    if use_learned_positional_embeddings is not None:
        transformer.patch_embed.use_learned_positional_embeddings = use_learned_positional_embeddings
        if hasattr(transformer.patch_embed, "pos_embedding") and transformer.patch_embed.pos_embedding is not None:
            transformer.patch_embed.pos_embedding = None
        transformer.register_to_config(use_learned_positional_embeddings=use_learned_positional_embeddings)
        

    # Freeze vae and text_encoder and transformer
    # Move text_encode and vae to gpu and cast to weight_dtype
    transformer.train()
    transformer.requires_grad_(False)
    for module in [vae, text_encoder]:
        if module is not None:
            module.requires_grad_(False)
            module.to(accelerator.device, dtype=weight_dtype) 
            # FIXME: not sure if VAE should be in half precision

    if cfg.compile_frozen_modules:
        vae.encode = torch.compile(vae.encode)
        # vae.decode = torch.compile(vae.decode) # 

    # Add lora support
    with_lora = False
    lora_params = []
    if cfg.lora_config is not None:
        transformer_lora_config = LoraConfig(
            **cfg.lora_config
        )
        transformer.add_adapter(transformer_lora_config)
        with_lora = True

        lora_params = [name for name, p in transformer.named_parameters() if p.requires_grad]

        if cfg.model_pipeline.get('lora_path', None) is not None:
            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(cfg.model_pipeline['lora_path'])
            transformer_state_dict = {f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")}
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
            print(f"Loaded LoRA weights from {cfg.model_pipeline['lora_path']}")

    if not cfg.lora_only:
        transformer.requires_grad_(True)

    # Create EMA for the transformer.
    ema_shard_start, ema_shard_end = None, None
    if cfg.use_ema:
        # split transformer parameters into N parts
        if with_lora and cfg.lora_only:
            transformer_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        else:
            transformer_params = list(transformer.parameters())
        num_params = len(transformer_params)
        num_params_local = (num_params - 1) // accelerator.num_processes + 1
        pid = accelerator.process_index
        ema_shard_start, ema_shard_end = pid * num_params_local, (pid + 1) * num_params_local
        transformer_params_local = transformer_params[ema_shard_start:ema_shard_end]
        ema_transformer = EMAModel(transformer_params_local)
        if accelerator.is_main_process:
            if with_lora and cfg.lora_only:
                param_keys = [n for n, p in transformer.named_parameters() if p.requires_grad]
            else:
                param_keys = list(transformer.state_dict().keys())
            param_keys_file = os.path.join(cfg.output_dir, "transformer_ema_keys.txt")
            np.savetxt(param_keys_file, param_keys, fmt="%s")

        ema_transformer.to(accelerator.device)
        
    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # NOTE: currently only save and load the transformer model
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if cfg.use_ema:
                save_path = os.path.join(output_dir, "transformer_ema", f"pytorch_model_rank{accelerator.process_index}.bin")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(ema_transformer.state_dict(), save_path)
            if accelerator.is_main_process:
                # if cfg.use_ema:
                #     ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                # NOTE: you can remove this if condition to save pytorch transformer ckpt along with deepspeed ckpt.
                if accelerator.distributed_type not in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                    for i, model in enumerate(models):

                        model_ = model
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            model_ = model.module
                        if isinstance(model_, type(unwrap_model(transformer))):
                            if with_lora:
                                transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                                # model.save_pretrained(os.path.join(output_dir, "lora"), transformer_lora_layers=transformer_lora_layers_to_save)
                                CogVideoXImageToVideoPipeline.save_lora_weights(
                                    os.path.join(output_dir, "lora"), transformer_lora_layers=transformer_lora_layers_to_save
                                )
                                if not cfg.lora_only:
                                    model.save_pretrained(os.path.join(output_dir, "transformer"))
                            else:
                                model.save_pretrained(os.path.join(output_dir, "transformer"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights: # NOTE: deepspeed doesn't have weights, you need this check
                        weights.pop()


        def load_model_hook(models, input_dir):
            if cfg.use_ema:
                # load_model = EMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), CustomCogVideoXTransformer3DModel)
                # ema_transformer.load_state_dict(load_model.state_dict())
                # # ema_transformer.to(accelerator.device)
                # del load_model
                ema_path = os.path.join(input_dir, "transformer_ema", f"pytorch_model_rank{accelerator.process_index}.bin")
                ema_transformer.load_state_dict(torch.load(ema_path))
                logger.info(f"Loaded EMA weights from {ema_path}")

            if accelerator.distributed_type not in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    if not with_lora:
                        load_model = CustomCogVideoXTransformer3DModel.from_pretrained(input_dir, subfolder="transformer")
                        model.register_to_config(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                        del load_model
                    else:
                        if not cfg.lora_only:
                            ckpt_list = sorted(glob.glob(os.path.join(input_dir, "transformer", "*.safetensors")))
                            state_dict = {}
                            for ckpt in ckpt_list:
                                state_dict.update(load_state_dict(ckpt))
                            model.load_state_dict(state_dict)
                            del state_dict
                        else: # only load lora
                            lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir, subfolder="lora")
                            transformer_state_dict = {
                                f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
                            }
                            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
                            incompatible_keys = set_peft_model_state_dict(model, transformer_state_dict, adapter_name="default")
                            if incompatible_keys is not None:
                                # check only for unexpected keys
                                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                                if unexpected_keys:
                                    logger.warning(
                                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                                        f" {unexpected_keys}. "
                                    )
                            del lora_state_dict
                    gc.collect()
    


        # if accelerator.distributed_type not in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    use_deepspeed_opt = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            print("Note: You are using 8-bit Adam optimizer from bitsandbytes.")
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    elif use_deepspeed_opt:
        optimizer_cls = accelerate.utils.DummyOptim
    else:
        optimizer_cls = torch.optim.AdamW

    parameters_list = []
    param_names = []

    # Customize the parameters that need to be trained; if necessary, you can uncomment them yourself.
    transformer_trainable_modules = cfg.model_pipeline.get('transformer_trainable_modules', None) # None means all trainable
    if transformer_trainable_modules is not None:
        # Determine whether list is excluding or including a set of params
        exclude = transformer_trainable_modules[0].startswith('-') 
        for module in transformer_trainable_modules:
            if exclude and not module.startswith('-'):
                assert module.startswith('-'), f"Cannot have both include and exclude modules {module}."
            if not exclude and module.startswith('-'):
                assert module.startswith('-'), f"Cannot have both include and exclude modules {module}."
        
        for name, param in transformer.named_parameters():
            for module in transformer_trainable_modules:
                if exclude: 
                    assert module.startswith('-')
                    module = module[1:]
                
                if (not exclude and module in name) or (exclude and module not in name):
                    parameters_list.append(param)
                    param_names.append(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        for name, param in transformer.named_parameters():
            if name in lora_params and name not in param_names:
                parameters_list.append(param)
                param_names.append(name)
                param.requires_grad = True
    else:
        parameters_list = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # fsdp - prepare model in advance of optimizer creation
    if accelerator.distributed_type == DistributedType.FSDP:
        transformer = accelerator.prepare(transformer)

    optimizer = optimizer_cls(
        parameters_list,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    global_batch_size = cfg.train_batch_size * accelerator.num_processes
    num_train_batches = math.ceil(cfg.num_train_examples / global_batch_size)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(num_train_batches / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes # NOTE: accelerate iterates num_proc times per step, not a bug
    num_training_steps=cfg.max_train_steps * accelerator.num_processes

    use_deepspeed_lr_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    if use_deepspeed_lr_scheduler:
        lr_scheduler = accelerate.utils.DummyScheduler(
            name=cfg.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )
    else:
        lr_scheduler = get_scheduler(
            name=cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps, # NOTE: accelerate iterates num_proc times per step, not a bug
            num_training_steps=num_training_steps,
        )

    # Prepare everything with our `accelerator`.
    if accelerator.distributed_type == DistributedType.FSDP:
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    else:
        transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)
    

    # clip_image_mean, clip_image_std = 0, 1
    # if cfg.cond_mode == 'image':
    #     clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:,None,None].to(accelerator.device, dtype=torch.float32)
    #     clip_image_std = torch.as_tensor(feature_extractor.image_std)[:,None,None].to(accelerator.device, dtype=torch.float32)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(num_train_batches / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(cfg))
        pop_keys = []
        for k, v in tracker_config.items():
            if v is not None and not isinstance(v, (int, float, str, bool, torch.Tensor)):
                pop_keys.append(k)
        for k in pop_keys: 
            tracker_config.pop(k)
        accelerator.init_trackers(cfg.tracker_project_name, tracker_config)

    initial_global_step = global_step = 0
    first_epoch = 0
    encoder_hidden_states = None
    
    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        is_latest_resume = False
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            if path is None:
                # This must be our first time since no checkpoint was found
                logger.warning(f"No latest resume checkpoint found, assuming this is our first training session!")
            else:
                is_latest_resume = True

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(cfg.output_dir, path)) # will also resume the random seed states
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                if is_latest_resume:
                    logger.warning("Remove the broken checkpoint and exit.")
                    # if accelerator.is_main_process:
                        # remove the checkpoint if it fails to load
                        # shutil.rmtree(os.path.join(cfg.output_dir, path))
                exit(1)
                
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    # Initialize DataLoaders
    if initial_global_step == 0 and cfg.seed is not None: # reset the seed again, 
        set_seed(cfg.seed, device_specific=True) # differ in each process, for data loading

    target_image = cfg.model_pipeline.get('target_image', 'rgb') # the label that maps to pixel_value
    if isinstance(target_image, str):
        target_image = [target_image]
    additional_target_image = cfg.model_pipeline.get('additional_target_image', [])
    additional_rope_time_only = cfg.model_pipeline.get('additional_rope_time_only', True)
    if isinstance(additional_target_image, str):
        additional_target_image = [additional_target_image]
    denoise_type = cfg.model_pipeline.get('denoise_type', 'additional_hidden_states')
    separate_timesteps = cfg.model_pipeline.get('separate_timesteps', False)
    cond_images = cfg.model_pipeline.get('cond_images', [])
    additional_cond_labels = cfg.model_pipeline.get('additional_cond_labels', None)
    # text prompt
    text_prompt = cfg.model_pipeline.get('text_prompt', '')
    negative_text_prompt = cfg.model_pipeline.get('negative_text_prompt', '')
    # cond dropout
    cond_dropout = cfg.model_pipeline.get('cond_dropout', {})
    cond_dropout_all = cfg.model_pipeline.get('cond_dropout_all', 0)
    cond_dropout_skip = cfg.model_pipeline.get('cond_dropout_skip', None)
    cond_exclusive = cfg.model_pipeline.get('cond_exclusive', None)
    cond_dropout_reuse = set()
    for k, v in cond_dropout.items():
        if isinstance(v, str):
            assert v in cond_dropout.keys(), f"Key {v} not found in cond_dropout"
            cond_dropout_reuse.add(v)
        elif isinstance(v, list) or isinstance(v, ListConfig):
            assert v[0] in cond_dropout.keys(), f"Key {v[0]} not found in cond_dropout"
            cond_dropout_reuse.add(v[0])

    target_dropout = cfg.model_pipeline.get('target_dropout', None)
    
    # Get the training dataset: multi-loader version
    wds_loader = True
    train_dataloaders_list = []
    for train_data_cfg in cfg.train_data['datasets']:
        assert "target" in train_data_cfg
        assert 'LightningLoader' in train_data_cfg['target']
        train_data_cfg["params"]["data_config"]["params"]["n_nodes"] = accelerator.num_processes // 8 + 1
        train_data_cfg["params"]["data_config"]["params"]["world_size"] = accelerator.num_processes
        train_data_cfg["params"]["data_config"]["params"]["process_idx"] = accelerator.process_index 
        train_data_cfg["params"]["global_batch_size"] = cfg.train_batch_size * accelerator.num_processes
        train_data_cfg["params"]["target_image"] = target_image
        # train_data_cfg["params"]["cond_images"] = cond_images
        train_data_instance = instantiate_from_config(train_data_cfg)
        train_dataloaders_list.append(
            (train_data_instance.train_dataloader(), train_data_instance._dataset.sampling_weight)
        )

    train_dataloader = MultiLoaderIterator(train_dataloaders_list,
                                        cfg.train_data.get('multi_loader_mode', 'random'),
                                        rank=accelerator.process_index)      # Note: MultiLoaderIterator assumes taking unlimited dataloaders

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    # Useful variables
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    # we now assume a simple text prompt as the condition, TODO: make it more general
    prompt_embeds = None
    if cfg.cond_mode == 'text':
        prompt_embeds = compute_prompt_embeddings(
            tokenizer,
            text_encoder,
            text_prompt,
            model_config.max_text_seq_length,
            accelerator.device,
            weight_dtype,
            requires_grad=False,
        )
        # no need text_encoder during training
        text_encoder.to(torch.device("cpu"))
    # elif cfg.cond_mode == 'env':
    #     env_labels = [label for label in cond_images if label.startswith('env')]

    image_rotary_emb, additional_rotary_emb = None, None
    _pix_w, _pix_h, _fsz = 0, 0, 0
    _a_pix_w, _a_pix_h = 0, 0
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {cfg.num_train_examples}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    
    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    break_loop = cfg.inference_only
    while not break_loop:
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Convert images to latent space
                if wds_loader:
                    batch_keys = list(batch.keys())
                    for k in batch_keys:
                        if isinstance(batch[k], torch.Tensor) and k not in ['lora_scale', 'loss_mask', 'target_dropout']:                       
                            batch[k] = batch[k].to(accelerator.device, non_blocking=True)
                            if batch[k].ndim == 4:
                                batch[k] = batch[k].unsqueeze(1)

                latents_list = []
                latents = None
                _, video_length, _, pixel_h, pixel_w = batch[target_image[0]].shape
                for target in target_image:
                    pixel_values = batch[target].to(weight_dtype)
                    with torch.no_grad():
                        _latents = prepare_latents(vae, pixel_values)
                        del pixel_values # free up memory
                    latents_list.append(_latents)
                latents = torch.cat(latents_list, dim=2)
                bsz, fsz, ch, latents_h, latents_w = latents.shape
                latent_dim = ch // len(latents_list)

                additional_latents = None
                add_latents, add_latents_list = None, []
                if len(additional_target_image) > 0:
                    _, _, _, a_pixel_h, a_pixel_w = batch[additional_target_image[0]].shape
                    for target in additional_target_image:
                        pixel_values = batch[target].to(weight_dtype)
                        with torch.no_grad():
                            _latents = prepare_latents(vae, pixel_values)
                            del pixel_values
                        add_latents_list.append(_latents)
                    add_latents = torch.cat(add_latents_list, dim=2)

                # Sample a random timestep for each image
                target_timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                timesteps, additional_timesteps = target_timesteps, None
                dropout_map = {}

                cur_denoise_type = denoise_type
                if denoise_type == 'joint':
                    cur_denoise_type = 'hidden_states' if random.random() < 0.5 else 'additional_hidden_states'

                target_dropout_mask = 1
        
                if cur_denoise_type == 'hidden_states':
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = scheduler.add_noise(latents, noise, target_timesteps)
                    # random dropout on multi denoising latents here!
                    if target_dropout is not None:
                        target_dropout_mask = prepare_dropout_mask(noisy_latents, target_image, cond_dropout=target_dropout) # B, 1, ND, 1, 1
                        noisy_latents = noisy_latents * target_dropout_mask 
                    target = latents
                    latents = noisy_latents
                    if add_latents is not None:
                        rand_dropout_mask = prepare_dropout_mask(
                            add_latents, additional_target_image, dropout_map=dropout_map,
                            cond_dropout=cond_dropout, cond_dropout_reuse=cond_dropout_reuse, 
                        )
                    else:
                        rand_dropout_mask = 1
                    if separate_timesteps:
                        additional_timesteps = timesteps * 0
                        

                elif cur_denoise_type == 'additional_hidden_states':
                    # Alternatively, denoise additional_hidden_states
                    # assume noisy latents always come first
                    noise = torch.randn_like(add_latents)
                    noisy_latents = scheduler.add_noise(add_latents, noise, target_timesteps)
                    # random dropout on multi denoising latents here!
                    batch_target_dropout = batch.get('target_dropout', None)
                    if isinstance(batch_target_dropout, torch.Tensor): # B, L
                        batch_target_dropout = batch_target_dropout[0].tolist() # L
                        batch_target_dropout = {k: v for k, v in zip(additional_target_image, batch_target_dropout)}
                    else:
                        batch_target_dropout = target_dropout

                    if batch_target_dropout is not None:
                        target_dropout_mask = prepare_dropout_mask(noisy_latents, additional_target_image, cond_dropout=batch_target_dropout)
                        noisy_latents = noisy_latents * target_dropout_mask
                        
                    target = add_latents
                    add_latents = noisy_latents
                    rand_dropout_mask = prepare_dropout_mask(
                        latents, target_image, dropout_map=dropout_map,
                        cond_dropout=cond_dropout, cond_dropout_reuse=cond_dropout_reuse, 
                    )
                    if separate_timesteps:
                        additional_timesteps = target_timesteps
                        timesteps = target_timesteps * 0

                # Prepare rotary embeds
                if model_config.use_rotary_positional_embeddings:
                    if (pixel_h != _pix_h or pixel_w != _pix_w or fsz != _fsz):
                        image_rotary_emb = None
                    if (len(additional_target_image) > 0) and (a_pixel_h != _a_pix_h or a_pixel_w != _a_pix_w or fsz != _fsz):
                        additional_rotary_emb = None
                
                _pix_h, _pix_w, _fsz = pixel_h, pixel_w, fsz
                
                ###### Process the Conditioning ######
                encoder_hidden_states = prompt_embeds.repeat(bsz, 1, 1) if prompt_embeds is not None else None

                (encoder_hidden_states, additional_inputs, additional_hidden_states,
                 image_rotary_emb, additional_rotary_emb) \
                    = prepare_training_conditions(
                        cfg, batch, latents, cond_images, 
                        text_encoder=text_encoder, vae=vae, 
                        model_config=model_config,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        additional_rotary_emb=additional_rotary_emb,
                        additional_rope_time_only=additional_rope_time_only,
                        additional_cond_labels=additional_cond_labels,
                        cond_dropout=cond_dropout, # TODO: no drop for potential target
                        cond_dropout_reuse=cond_dropout_reuse,
                        cond_dropout_all=cond_dropout_all,
                        cond_dropout_skip=cond_dropout_skip,
                        cond_exclusive=cond_exclusive,
                        accelerator=accelerator, weight_dtype=weight_dtype,
                        exclude_labels=target_image + additional_target_image,
                    )   
                lora_scale = batch.get('lora_scale', None)
                if lora_scale is not None:
                    if isinstance(lora_scale, list):
                        lora_scale = lora_scale[0]
                    elif isinstance(lora_scale, torch.Tensor):
                        lora_scale = lora_scale[0].item()
                attention_kwargs = {'scale': lora_scale} if lora_scale is not None else {}

                loss_mask = batch.get('loss_mask', 1)
                if isinstance(loss_mask, torch.Tensor): # B, L
                    loss_mask = loss_mask[:1].unsqueeze(-1).repeat(1, 1, latent_dim) # 1, L, C
                    loss_mask = loss_mask.reshape(1, 1, -1, 1, 1).to(latents.device) # 1, 1, L*C, 1, 1
                loss_mask = loss_mask * target_dropout_mask

                if cfg.model_offload:
                    vae.cpu()
                # apply rand_dropout_mask here
                global_rand_mask = dropout_map.get('global', 1)
                if cur_denoise_type == 'hidden_states':
                    if add_latents is not None:
                        add_latents = add_latents * global_rand_mask * rand_dropout_mask
                elif cur_denoise_type == 'additional_hidden_states':
                    latents = latents * global_rand_mask * rand_dropout_mask

                if additional_hidden_states is not None:
                    additional_hidden_states = torch.cat([add_latents, additional_hidden_states], dim=2)
                
                # TODO: ofs_emb for v1.5

                if additional_rotary_emb is not None:
                    # cat the additional rotary emb to the image rotary emb
                    rotary_emb = [
                        torch.cat([iemb, aemb], dim=0) 
                        for iemb, aemb in zip(image_rotary_emb, additional_rotary_emb)
                    ]
                else:
                    rotary_emb = image_rotary_emb      
          
                batch.clear()
                del batch # free up memory, not used anymore

                # Predict the noise residual and compute loss
                model_output = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    additional_timestep=additional_timesteps,
                    image_rotary_emb=rotary_emb,
                    return_dict=False,
                    additional_inputs=additional_inputs,
                    additional_hidden_states=additional_hidden_states,
                    denoise_type=cur_denoise_type,
                    attention_kwargs=attention_kwargs,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_latents, target_timesteps)

                # MSE loss
                alphas_cumprod = scheduler.alphas_cumprod[target_timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)
                loss = torch.mean(
                    (weights.float() * loss_mask * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 
                    dim=1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), cfg.max_grad_norm)
                    # double check the nan/inf, sometime doesn't work in DDP for no reason
                    if optimizer.scaler is not None:
                        optimizer.scaler._check_inf_per_device(optimizer.optimizer)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if cfg.model_offload:
                    vae.to(accelerator.device)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.use_ema: # NOTE: 👀
                    if with_lora and cfg.lora_only:
                        transformer_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
                    else:
                        transformer_params = list(transformer.parameters())
                    transformer_params_local = transformer_params[ema_shard_start:ema_shard_end]
                    ema_transformer.step(transformer_params_local)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0],}, step=global_step)
                train_loss = 0.0

                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(cfg.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(cfg.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process or accelerator.distributed_type in [DistributedType.FSDP, DistributedType.DEEPSPEED]:
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if cfg.save_multi_random_states and not accelerator.is_main_process:
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        save_random_state(save_path, accelerator.process_index)

                if cfg.validation_data is not None and \
                    cfg.validation_steps is not None and (global_step % cfg.validation_steps == 0 or global_step==1): # TODO!!!

                    torch.cuda.empty_cache()
                    if cfg.cond_mode == 'text':
                        text_encoder.to(accelerator.device)

                    if accelerator.is_main_process:
                        # if cfg.use_ema:
                        #     # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        #     ema_transformer.store(transformer.parameters())
                        #     transformer_params = [p for n, p in transformer.named_parameters() if 'lora' not in n] if with_lora else transformer.parameters()
                        #     ema_transformer.copy_to(transformer_params)
                        transformer.eval()

                        save_weights = (cfg.save_weights_every_n_validation is not None and
                                global_step % int(cfg.save_weights_every_n_validation * cfg.validation_steps) == 0)

                        pipeline = log_validation(  # TODO:
                            vae, transformer, tokenizer, text_encoder,
                            cfg, accelerator, global_step, 
                            default_text_prompt=text_prompt,
                            default_negative_text_prompt=negative_text_prompt,
                            return_pipeline=save_weights
                        )

                        if save_weights:
                            pipeline.transformer.save_pretrained(os.path.join(cfg.output_dir, f"model_weights-{global_step}", "transformer"))
                            del pipeline

                        transformer.train()
                        # if cfg.use_ema:
                        #     # Switch back to the original UNet parameters.
                        #     ema_transformer.restore(transformer.parameters())

                    if cfg.compile_frozen_modules:
                        torch._dynamo.reset()

                    if cfg.cond_mode == 'text':
                        text_encoder.to(torch.device("cpu"))
                    torch.cuda.empty_cache()

                if cfg.job_stop_steps is not None and global_step % cfg.job_stop_steps == 0:
                    logger.info('Reach Job Stop Steps')
                    break_loop = True
                    break

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if optimizer.step_was_skipped:
                logs["overflow"] = 1
                logs["scaler"] = optimizer.scaler._scale.item() if optimizer.scaler is not None else 1
                logger.warning(f"Gradient overflow.  Skipping step {global_step}, scaler {logs['scaler']}")
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.max_train_steps:
                logger.info('Reach Max Train Steps')
                break_loop = True
                break
            
            del model_pred, loss
            if cfg.free_memory_every is not None and global_step % cfg.free_memory_every == 0:
                del latents, noisy_latents, model_output,
                del encoder_hidden_states, additional_inputs, additional_hidden_states
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            # assert not loss.detach().isnan().any(), f"NaN occur, step: {global_step}"
        if break_loop:
            break


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and (global_step >= cfg.max_train_steps or cfg.inference_only):
        transformer = unwrap_model(transformer)
        # if cfg.use_ema:
        #     transformer_params = [p for n, p in transformer.named_parameters() if 'lora' not in n] if with_lora else transformer.parameters()
        #     ema_transformer.copy_to(transformer_params)
        transformer.eval()
        pipeline = log_validation(  # TODO:
            vae, transformer, tokenizer, text_encoder,
            cfg, accelerator, global_step, 
            default_text_prompt=text_prompt,
            return_pipeline=True
        )
        if cfg.save_final_model:
            # pipeline.save_pretrained(cfg.output_dir)

            if with_lora:
                transformer_lora_layers_to_save = get_peft_model_state_dict(pipeline.transformer)
                # model.save_pretrained(os.path.join(output_dir, "lora"), transformer_lora_layers=transformer_lora_layers_to_save)
                CogVideoXImageToVideoPipeline.save_lora_weights(
                    os.path.join(cfg.output_dir, f"model_weights-{global_step}", "lora"),  
                    transformer_lora_layers=transformer_lora_layers_to_save
                )
                if not cfg.lora_only:
                    transformer.unload_lora()
                    pipeline.transformer.save_pretrained(os.path.join(cfg.output_dir, f"model_weights-{global_step}", "transformer"))
            else:
                pipeline.transformer.save_pretrained(os.path.join(cfg.output_dir, f"model_weights-{global_step}", "transformer"))

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, unknown = parser.parse_known_args()
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(args.config)
    missing_keys = set(cfg.keys()) - set(schema.keys())
    for key in missing_keys:
        OmegaConf.update(schema, key, None, force_add=True)
    cfg = OmegaConf.merge(schema, cfg)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)
    main(cfg)

