from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: Optional[str] = None
    pretrained_transformer: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    model_pipeline: Optional[Dict] = None
    lora_config: Optional[Dict] = None  # e.g., rank, init, modules
    lora_only: bool = False

    num_train_examples: int = 100000

    train_data: Optional[Dict] = None
    validation_data: Optional[Dict] = None  # TODO: change to XR style
    inference_kwargs: Optional[Any] = None

    inference_only: bool = False

    cond_mode: Optional[str] = "text"  # text, image, env

    compile_frozen_modules: bool = False
    use_fsdp: bool = False
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    zero_stage: int = 2
    autocast_cache_enabled: bool = False
    enable_slicing: bool = False
    enable_tiling: bool = False

    output_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    seed: Optional[int] = None
    num_frames: int = 14
    resolution: int = 512
    resolution_height: Optional[int] = None
    resolution_width: Optional[int] = None
    train_batch_size: int = 4
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None  # If provided, overrides num_train_epochs.
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    # snr_gamma: Optional[float] = None
    use_8bit_adam: bool = False
    allow_tf32: bool = False
    use_ema: bool = False
    shard_ema: bool = False
    dataloader_num_workers: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    prediction_type: Optional[str] = None
    logging_dir: str = "logs"
    mixed_precision: Optional[str] = None
    transformer_precision: Optional[str] = None
    report_to: Optional[str] = None
    local_rank: int = -1
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    save_multi_random_states: bool = False
    # enable_xformers_memory_efficient_attention: bool = False
    # noise_offset: float = 0.0 # Not used
    # rescale_betas_zero_snr: bool = False # Not used
    validation_epochs: Optional[int] = None
    validation_steps: Optional[int] = None  # If provided, overrides validation_epochs.
    save_weights_every_n_validation: Optional[int] = None
    save_final_model: bool = False
    tracker_project_name: str = "custom-diffusion"

    find_unused_parameters: bool = False

    job_stop_steps: Optional[int] = None
    free_memory_every: Optional[int] = None
    model_offload: bool = False

    # Not implemented
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

