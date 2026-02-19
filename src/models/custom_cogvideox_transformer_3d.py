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

from typing import Any, Dict, Optional, Tuple, Union, List

import torch
from torch import nn
from einops import rearrange, repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_3d_sincos_pos_embed # CogVideoXPatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def partial_linear_forward(linear_layer, input_tensor, start_index, end_index):
    weight = linear_layer.weight[start_index:end_index, :]
    if linear_layer.bias is not None:
        bias = linear_layer.bias[start_index:end_index]
        return torch.nn.functional.linear(input_tensor, weight, bias)
    else:
        return torch.nn.functional.linear(input_tensor, weight)


class CogVideoXLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        num_states: int = 2,
    ) -> None:
        super().__init__()

        self.num_states = num_states
        self.ssg_dim = 3 * embedding_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, self.num_states * self.ssg_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states_list: List[torch.Tensor], temb: Union[torch.Tensor, List[torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        # shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=1)
        ssg_list = None
        if isinstance(temb, torch.Tensor):
            ssg_list = self.linear(self.silu(temb)).chunk(self.num_states, dim=1)
        # SSG: shift, scale, gate
        processed_hidden_states_list = []
        gates_list = []
        
        # TODO: how to support different temb per state?
        for i, hidden_states in enumerate(hidden_states_list):
            if hidden_states is not None:
                normalized = self.norm(hidden_states)
                # Use the last state for the rest, if num_states < len(hidden_states_list)
                ssg_id = min(i, self.num_states - 1) 
                if ssg_list is not None:
                    # Extract shift, scale, gate for this state from ssg_list[i]
                    shift, scale, gate = ssg_list[ssg_id].chunk(3, dim=1)
                else:
                    # shift, scale, gate = self.linear(self.silu(temb[i])).chunk(3, dim=1)
                    shift, scale, gate = partial_linear_forward(
                        self.linear, self.silu(temb[ssg_id]),
                        ssg_id * self.ssg_dim, (ssg_id + 1) * self.ssg_dim
                    ).chunk(3, dim=1)
                processed = normalized * (1 + scale)[:, None, :] + shift[:, None, :]
                processed_hidden_states_list.append(processed)
                gates_list.append(gate[:, None, :])
            else:
                processed_hidden_states_list.append(None)
                gates_list.append(None)

        return processed_hidden_states_list, gates_list

@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        additional_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        
        seq_length = hidden_states.size(1)
        text_seq_length = encoder_hidden_states.size(1) if encoder_hidden_states is not None else 0
        additional_seq_length = additional_hidden_states.size(1) if additional_hidden_states is not None else 0

        # norm & modulate
        (norm_hidden_states, norm_encoder_hidden_states, norm_add_hidden_states), \
            (gate_msa, enc_gate_msa, add_gate_msa) = self.norm1(
                (hidden_states, encoder_hidden_states, additional_hidden_states), temb
            )

        if additional_hidden_states is not None:
            norm_hidden_states = torch.cat([norm_hidden_states, norm_add_hidden_states], dim=1)

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        if encoder_hidden_states is not None:       
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
    
        if additional_hidden_states is not None:
            attn_hidden_states, attn_add_hidden_states = attn_hidden_states.split(
                [seq_length, additional_seq_length], dim=1
            )
            additional_hidden_states = additional_hidden_states + add_gate_msa * attn_add_hidden_states
        
        hidden_states = hidden_states + gate_msa * attn_hidden_states

        # norm & modulate
        (norm_hidden_states, norm_encoder_hidden_states, norm_add_hidden_states), \
            (gate_ff, enc_gate_ff, add_gate_ff) = self.norm2(
                (hidden_states, encoder_hidden_states, additional_hidden_states), temb
            )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        if additional_hidden_states is not None:
            norm_hidden_states = torch.cat([norm_hidden_states, norm_add_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)
        ff_length = ff_output.size(1)
        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:ff_length-additional_seq_length]
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        if additional_hidden_states is not None:
            additional_hidden_states = additional_hidden_states + add_gate_ff * ff_output[:, ff_length-additional_seq_length:]

        return hidden_states, encoder_hidden_states, additional_hidden_states


class CogVideoXPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
        use_fixed_pos_embedding: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings
        self.use_fixed_pos_embedding = use_fixed_pos_embedding

        self.skip_warning = False

        if patch_size_t is None:
            # CogVideoX 1.0 checkpoints
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
            )
        else:
            # CogVideoX 1.5 checkpoints
            self.proj = nn.Linear(in_channels * patch_size * patch_size * patch_size_t, embed_dim)

        self.text_proj = nn.Linear(text_embed_dim, embed_dim)

        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)

    def _get_positional_embeddings(
        self, sample_height: int, sample_width: int, sample_frames: int, 
        text_seq_length: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
            device=device,
            output_type="pt",
        )
        pos_embedding = pos_embedding.flatten(0, 1)
        text_seq_length = self.max_text_seq_length if text_seq_length is None else text_seq_length
        joint_pos_embedding = pos_embedding.new_zeros(
            1, text_seq_length + num_patches, self.embed_dim, requires_grad=False
        )
        joint_pos_embedding.data[:, text_seq_length :].copy_(pos_embedding)

        return joint_pos_embedding

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds) if text_embeds is not None else None
        text_seq_length = text_embeds.size(1) if text_embeds is not None else 0
        batch_size, num_frames, channels, height, width = image_embeds.shape

        if self.patch_size_t is None:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)

        if text_embeds is not None:
            embeds = torch.cat(
                [text_embeds, image_embeds], dim=1
            ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]
        else:
            embeds = image_embeds.contiguous()

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if not self.skip_warning and self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                logger.warning(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )
                self.skip_warning = True

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if self.use_fixed_pos_embedding:
                p = self.patch_size
                text_offset = self.max_text_seq_length  if text_embeds is None else 0
                frame_seq_length = self.sample_height * self.sample_width // p // p
                imag_seq_length = image_embeds.size(1)
                assert image_embeds.size(1) % frame_seq_length == 0
                pos_embedding = self.pos_embedding[:, text_offset:self.max_text_seq_length+imag_seq_length]
            else:
                if (
                    self.sample_height != height
                    or self.sample_width != width
                    or self.sample_frames != pre_time_compression_frames
                ):
                    pos_embedding = self._get_positional_embeddings(
                        height, width, pre_time_compression_frames, device=embeds.device, text_seq_length=text_seq_length
                    )


            pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding
        
        else:
            pass # No positional embeddings

        return embeds

class CogVideoXImagePatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        in_channels: int = 16,
        embed_dim: int = 1920,
        temporal_compression_ratio: int = 4,
        bias: bool = True,
        with_pos_embed: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.embed_dim = embed_dim
        self.temporal_compression_ratio = temporal_compression_ratio
        self.with_pos_embed = with_pos_embed

        if patch_size_t is None:
            # CogVideoX 1.0 checkpoints
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
            )
        else:
            # CogVideoX 1.5 checkpoints
            self.proj = nn.Linear(in_channels * patch_size * patch_size * patch_size_t, embed_dim)

    def forward(self, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        batch_size, num_frames, channels, height, width = image_embeds.shape

        if self.patch_size_t is None:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)

        if self.with_pos_embed:
            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1
            post_patch_height = height // self.patch_size
            post_patch_width = width // self.patch_size
            post_time_compression_frames = (pre_time_compression_frames - 1) // self.temporal_compression_ratio + 1
            num_patches = post_patch_height * post_patch_width * post_time_compression_frames
            
            pos_embed = get_3d_sincos_pos_embed(
                self.embed_dim,
                (post_patch_width, post_patch_height),
                post_time_compression_frames,
                spatial_interpolation_scale=1.875,
                temporal_interpolation_scale=1.0,
                device=image_embeds.device,
                output_type="pt",
            )
            pos_embed = pos_embed.flatten(0, 1)
            image_embeds = image_embeds + pos_embed

        return image_embeds

class CustomCogVideoXTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        use_positional_embeddings: bool = False,
        use_fixed_pos_embedding: bool = False,
        patch_bias: bool = True,
        additional_input_channels: Optional[int] = None, 
        additional_output_channels: Optional[int] = None,
        additional_patch_embed_channels: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        # if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
        #     raise ValueError(
        #         "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
        #         "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
        #         "issue at https://github.com/huggingface/diffusers/issues."
        #     )

        # 1. Patch embedding # TODO: for concatenation, u need extend self.patch_embed.proj (Conv2d)
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings and use_positional_embeddings, # False for 1.0 & 1.5
            use_learned_positional_embeddings=use_learned_positional_embeddings, # True for 1.0, but False for 1.5
            use_fixed_pos_embedding=use_fixed_pos_embedding,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False

        ###### Custom part: ######
        # Additive embed: add processed additional input to the hidden states (equavalent to concatenation)
        self.additional_input_embed = None
        if additional_input_channels is not None:
            print("additional_input_channels", additional_input_channels)
            self.additional_input_channels(additional_input_channels)

        self.additional_patch_embed = None
        if additional_patch_embed_channels is not None:
            print("additional_patch_embed_channels", additional_patch_embed_channels)
            self.init_additional_patch_embed(additional_patch_embed_channels)        

        self.additional_norm_out = None
        self.additional_proj_out = None
        if additional_output_channels is not None:
            print("additional_output_channels", additional_output_channels)
            self.init_additional_output_layes(additional_output_channels)
        # Context embed: used for multipass TODO: @zian, do we still need this?


    # Hack to workaround the from_pretained stuff
    def additional_input_channels(self, additional_input_channels = None):
        if additional_input_channels is not None and \
            (self.additional_input_embed is None or self.config.additional_input_channels != additional_input_channels):

            if additional_input_channels == 0:
                self.additional_input_embed = None
                # remove additional_input_embed from config
                self.register_to_config(additional_input_channels=additional_input_channels)
                return

            inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
            patch_size = self.config.patch_size
            patch_size_t = self.config.get("patch_size_t", None)
            bias = self.config.get("patch_bias", True)
            # NOTE: bias term can be merged into self.patch_embed.proj, if we want to try sparse condition.
            additional_input_embed = zero_module(
                CogVideoXImagePatchEmbed(
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                    in_channels=additional_input_channels,
                    embed_dim=inner_dim,
                    bias=bias
                )
            )
            if self.additional_input_embed is not None:
                if patch_size_t is None:
                    additional_input_embed.proj.weight.data[:, :additional_input_channels] = \
                        self.additional_input_embed.proj.weight.data[:, :additional_input_channels]
                    if bias:
                        additional_input_embed.proj.bias.data[:] = self.additional_input_embed.proj.bias.data

            self.additional_input_embed = additional_input_embed

            self.register_to_config(additional_input_channels=additional_input_channels)
            print("additional_input_channels", self.config.additional_input_channels)

    def extend_output_channels(self, out_channels = None, duplicate_channels=True):
        if out_channels is not None and self.config.out_channels != out_channels:
            assert out_channels % self.config.out_channels == 0
            num_latents = out_channels // self.config.out_channels
            in_dim = self.proj_out.in_features
            out_dim = self.proj_out.out_features
            bias = self.proj_out.bias is not None
            proj_out_layer = zero_module(nn.Linear(in_dim, out_dim * num_latents, bias=bias))

            proj_out_layer.weight.data[:out_dim] = self.proj_out.weight.data
            if duplicate_channels:
                proj_out_layer.weight.data[out_dim:] = self.proj_out.weight.data
            if bias:
                proj_out_layer.bias.data[:out_dim] = self.proj_out.bias.data
                if duplicate_channels:
                    proj_out_layer.bias.data[out_dim:] = self.proj_out.bias.data

            # self.additional_proj_out.load_state_dict(self.proj_out.state_dict())
            print("extend output channels from", self.out_channels, "to", out_channels)
            self.register_to_config(out_channels=out_channels)
            self.proj_out = proj_out_layer
      

    def init_additional_patch_embed(self, additional_patch_embed_channels=None, duplicate_channels=True):
        if additional_patch_embed_channels is not None and self.additional_patch_embed is None:
            self.register_to_config(additional_patch_embed_channels=additional_patch_embed_channels)
            inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
            patch_size = self.config.patch_size
            patch_size_t = self.config.get("patch_size_t", None)
            bias = self.config.get("patch_bias", True)
            # NOTE: bias term can be merged into self.patch_embed.proj, if we want to try sparse condition.
            self.additional_patch_embed = zero_module(
                CogVideoXImagePatchEmbed(
                    patch_size=patch_size,
                    patch_size_t=patch_size_t,
                    in_channels=additional_patch_embed_channels,
                    embed_dim=inner_dim,
                    bias=bias
                )
            )
            print("additional_patch_embed_channels", self.config.additional_patch_embed_channels)
    
    def init_additional_output_layes(self, additional_output_channels=None, duplicate_channels=True):
        if additional_output_channels is not None and self.additional_norm_out is None:
            assert additional_output_channels % self.config.out_channels == 0
            num_latents = additional_output_channels // self.config.out_channels

            inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

            self.additional_norm_out = AdaLayerNorm(
                embedding_dim=self.config.time_embed_dim,
                output_dim=2 * inner_dim,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps,
                chunk_dim=1,
            )

            self.additional_norm_out.linear.weight.data[:] = self.norm_out.linear.weight.data
            if self.norm_out.linear.bias is not None:
                self.additional_norm_out.linear.bias.data[:] = self.norm_out.linear.bias.data

            self.additional_norm_out.norm.weight.data[:] = self.norm_out.norm.weight.data
            if self.norm_out.norm.bias is not None:
                self.additional_norm_out.norm.bias.data[:] = self.norm_out.norm.bias.data

            output_dim = self.proj_out.out_features
            bias = self.proj_out.bias is not None
            self.additional_proj_out = zero_module(nn.Linear(inner_dim, output_dim * num_latents, bias=bias))
            self.additional_proj_out.weight.data[:output_dim] = self.proj_out.weight.data
            if duplicate_channels:
                for i in range(1, num_latents):
                    self.additional_proj_out.weight.data[i*output_dim:(i+1)*output_dim] = self.proj_out.weight.data
            if bias:
                self.additional_proj_out.bias.data[:output_dim] = self.proj_out.bias.data
                if duplicate_channels:
                    for i in range(1, num_latents):
                        self.additional_proj_out.bias.data[i*output_dim:(i+1)*output_dim] = self.proj_out.bias.data

            self.register_to_config(additional_output_channels=additional_output_channels)


    # Ruofan: Only for I2V model, hack disentangle the cat image channels from hidden states 
    def disentangle_image_channels(self, vae_latent_dim=16):
        old_in_channels = self.config.get("in_channels", 16)
        if vae_latent_dim != old_in_channels:
            cond_in_channels = old_in_channels - vae_latent_dim
            # assert cond_in_channels <= self.config.get("additional_input_channels", 0)
            old_proj = self.patch_embed.proj

            inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
            patch_size = self.config.patch_size
            patch_size_t = self.config.get("patch_size_t", None)
            bias = self.config.get("patch_bias", True)
            self.register_to_config(in_channels=vae_latent_dim)
            if patch_size_t is None:
                self.patch_embed.proj = zero_module(
                    nn.Conv2d(
                        vae_latent_dim, inner_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
                    )
                )
                self.patch_embed.proj.weight.data[:] = old_proj.weight.data[:, :vae_latent_dim]
                if bias:
                    self.patch_embed.proj.bias.data[:] = old_proj.bias.data
                if self.additional_input_embed is not None:
                    self.additional_input_embed.proj.weight.data[:, :cond_in_channels] = old_proj.weight.data[:, vae_latent_dim:]
            else:
                weight_channels = patch_size * patch_size * patch_size_t
                self.patch_embed.proj = zero_module(
                    nn.Linear(vae_latent_dim * weight_channels, inner_dim)
                )
                self.patch_embed.proj.weight.data[:] = old_proj.weight.data[:, :vae_latent_dim * weight_channels]
                # NOTE: https://github.com/huggingface/diffusers/blob/100142586f82a9410f5bc393b9eb06c12d771006/src/diffusers/models/embeddings.py#L669, does not add bias
                # but config set bias=False ... https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V/blob/9f310b78e4ed32a15fec50712149d4785d3d0fc4/transformer/config.json#L18
                if old_proj.bias is not None:
                    self.patch_embed.proj.bias.data[:] = old_proj.bias.data
                if self.additional_input_embed is not None:
                    self.additional_input_embed.proj.weight.data[:, :cond_in_channels * weight_channels] = \
                        old_proj.weight.data[:, vae_latent_dim * weight_channels:]


    def extend_input_channels(self, in_channels = None, duplicate_channels=True, vae_latent_dim=16):
        old_in_channels = self.config.get("in_channels", 16)
        if in_channels is not None and in_channels != old_in_channels:
            assert in_channels % vae_latent_dim == 0
            assert vae_latent_dim == old_in_channels
            num_latents = in_channels // old_in_channels
            
            old_proj = self.patch_embed.proj

            inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
            patch_size = self.config.patch_size
            patch_size_t = self.config.get("patch_size_t", None)
            bias = self.config.get("patch_bias", True)
            print("extend input noise channels from", old_in_channels, "to", in_channels)
            self.register_to_config(in_channels=vae_latent_dim)
            if patch_size_t is None:
                self.patch_embed.proj = zero_module(
                    nn.Conv2d(
                        num_latents * vae_latent_dim, inner_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
                    )
                )
                self.patch_embed.proj.weight.data[:, :vae_latent_dim] = old_proj.weight.data[:, :vae_latent_dim]
                if duplicate_channels:
                    for i in range(1, num_latents):
                        self.patch_embed.proj.weight.data[:, i*vae_latent_dim:(i+1)*vae_latent_dim] = old_proj.weight.data[:, :vae_latent_dim]
                    self.patch_embed.proj.weight.data[:] = self.patch_embed.proj.weight.data / num_latents

                if bias:
                    self.patch_embed.proj.bias.data[:] = old_proj.bias.data
            else:
                weight_channels = patch_size * patch_size * patch_size_t
                w_offset = vae_latent_dim * weight_channels
                self.patch_embed.proj = zero_module(
                    nn.Linear(num_latents * w_offset, inner_dim)
                )
                self.patch_embed.proj.weight.data[:, :w_offset] = old_proj.weight.data[:, :w_offset]
                if duplicate_channels:
                    for i in range(1, num_latents):
                        self.patch_embed.proj.weight.data[:, i*w_offset:(i+1)*w_offset] = old_proj.weight.data[:, :w_offset]
                    self.patch_embed.proj.weight.data[:] = self.patch_embed.proj.weight.data / num_latents
                if old_proj.bias is not None:
                    self.patch_embed.proj.bias.data[:] = old_proj.bias.data
                

    # Ruofan: alternative solution: extend the patch_embed.proj, without using additional_input_embed 
    # TODO: later, if we preprocess is hard, or pos embed is needed.


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def prepare_time_embedding(self, timestep, ofs=None, timestep_cond=None, dtype=None):
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb
        return emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        additional_inputs: Optional[torch.Tensor] = None,
        additional_hidden_states: Optional[torch.Tensor] = None,
        additional_timestep: Optional[torch.Tensor] = None,
        # additional_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # NOTE: not used
        denoise_type: str = 'hidden_states',
        # **kwargs,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape
        if additional_hidden_states is not None:
            _, _, _, a_height, a_width = additional_hidden_states.shape

        # 1. Time embedding
        emb = self.prepare_time_embedding(timestep, ofs, timestep_cond, hidden_states.dtype)
        if additional_timestep is not None:
            additional_emb = self.prepare_time_embedding(additional_timestep, ofs, timestep_cond, hidden_states.dtype)
            emb = [emb, additional_emb]

        # 2. Patch embedding
        text_seq_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        
        if additional_inputs is not None and self.additional_input_embed is not None:
            additional_input_embs = self.additional_input_embed(additional_inputs)
            hidden_states = torch.cat([
                hidden_states[:, :text_seq_length], 
                hidden_states[:, text_seq_length:] + additional_input_embs # NOTE: addition
            ], dim=1)

        if additional_hidden_states is not None:
            additional_hidden_states = self.additional_patch_embed(additional_hidden_states) # TODO:
            self.embedding_dropout(additional_hidden_states)

        hidden_states = self.embedding_dropout(hidden_states)
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        additional_seq_length = additional_hidden_states.shape[1] if additional_hidden_states is not None else 0

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # NOTE: no support now
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states, additional_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    additional_hidden_states,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states, additional_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    additional_hidden_states=additional_hidden_states,
                )

        norm_out, proj_out = self.norm_out, self.proj_out
        emb_out = emb[0] if isinstance(emb, list) else emb
        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            if additional_hidden_states is not None:
                hidden_states = torch.cat([hidden_states, additional_hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            if denoise_type == 'hidden_states':
                hidden_states = hidden_states[:, text_seq_length:hidden_states.size(1)-additional_seq_length]
            elif denoise_type == 'additional_hidden_states':
                hidden_states = hidden_states[:, -additional_seq_length:]
                norm_out, proj_out = self.additional_norm_out, self.additional_proj_out
                height, width = a_height, a_width
                emb_out = emb[1] if isinstance(emb, list) else emb

        # 4. Final block
        hidden_states = norm_out(hidden_states, temb=emb_out)
        hidden_states = proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)