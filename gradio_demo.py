import os
import sys
import glob
import tempfile
import shutil
import time
import numpy as np
import torch
from omegaconf import OmegaConf, ListConfig
from PIL import Image
import imageio
from diffusers.utils import export_to_video
import gradio as gr

from src.models.custom_cogvideox_transformer_3d import CustomCogVideoXTransformer3DModel
from src.models.custom_autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from src.pipelines.pipeline_cogvideox_rgbxenv import RGBXEnvCogVideoXPipeline, T5EncoderModel
from transformers import AutoTokenizer
from src.data.rendering_utils import resize_crop, envmap_vec, read_image, get_cam_matrix
from luxdit_config import TrainingConfig
from hdr_merger import main as hdr_merger_main
import argparse

# Global variables to cache the pipeline
_pipeline_cache = {}
_hdr_model_cache = None

# Local temp directory for Gradio outputs
LOCAL_TEMP_DIR = os.path.join(os.path.dirname(__file__), "gradio_temp")
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

def load_pipeline(model_type, config_path, transformer_path, lora_dir=None):
    """Load and cache the pipeline (always bf16)."""
    cache_key = f"{model_type}_{transformer_path}_{lora_dir}"
    
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.bfloat16
    
    # Load config
    schema = OmegaConf.structured(TrainingConfig)
    cfg = OmegaConf.load(config_path)
    missing_keys = set(cfg.keys()) - set(schema.keys())
    for key in missing_keys:
        OmegaConf.update(schema, key, None, force_add=True)
    cfg = OmegaConf.merge(schema, cfg)
    
    pipeline_cfg = cfg.model_pipeline
    
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
    
    # Load models
    vae = AutoencoderKLCogVideoX.from_pretrained(
        pipeline_cfg.vae_path, subfolder="vae", revision=cfg.revision, variant=None,
        torch_dtype=weight_dtype, device=device
    )
    
    cond_mode = pipeline_cfg.get('cond_mode', 'env')
    text_encoder, tokenizer = None, None
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
        transformer_path, subfolder="transformer", revision=cfg.revision, variant=None,
        torch_dtype=weight_dtype, device=device, 
        **transformer_kwargs
    )
    
    noise_scheduler = CogVideoXDPMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    pipeline = RGBXEnvCogVideoXPipeline(
        vae=vae, 
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformers,
        scheduler=noise_scheduler
    )
    
    if lora_dir and lora_dir.strip() and os.path.exists(lora_dir.strip()):
        print(f'Loading lora from {lora_dir}')
        try:
            pipeline.unload_lora_weights()
        except:
            pass
        pipeline.load_lora_weights(lora_dir, adapter_name="env-lora")
    else:
        print('No lora loaded')
    
    pipeline = pipeline.to(device)
    _pipeline_cache[cache_key] = (pipeline, cfg)
    
    return pipeline, cfg

def process_image_inference(
    image,
    model_type,
    config_path,
    transformer_path,
    lora_dir,
    lora_scale,
    resolution,
    guidance_scale,
    num_inference_steps,
    seed,
    use_hdr_merger,
    hdr_model_path
):
    """Process a single image"""
    if image is None:
        return None, None, None, "Please upload an image"
    
    try:
        # Create temporary directory in local dir for processing
        temp_dir = os.path.join(LOCAL_TEMP_DIR, f"img_{int(time.time() * 1000)}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            # Save uploaded image - ensure RGB mode
            input_path = os.path.join(temp_dir, "input.png")
            if isinstance(image, str):
                shutil.copy(image, input_path)
            else:
                # Convert to RGB if needed (handles RGBA, L, etc.)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(input_path)
            
            # Load pipeline
            lora_dir_clean = lora_dir.strip() if lora_dir and lora_dir.strip() else None
            pipeline, cfg = load_pipeline(model_type, config_path, transformer_path, lora_dir_clean)
            
            pipeline_cfg = cfg.model_pipeline
            data_cfg = cfg.train_dataset if 'train_dataset' in cfg else cfg.train_data
            
            denoise_type = pipeline_cfg.get('denoise_type', 'additional_hidden_states')
            _target_labels, cond_labels = pipeline_cfg.target_image, pipeline_cfg.cond_images
            additional_cond_labels = pipeline_cfg.get('additional_cond_labels', None)
            _additional_target_labels = pipeline_cfg.get('additional_target_image', [])
            
            if denoise_type == 'additional_hidden_states':
                target_labels = _additional_target_labels
            else:
                target_labels = _target_labels
            
            # Parse resolution from dropdown (format: "480x720" or "512x512")
            if resolution:
                height, width = map(int, resolution.split('x'))
            else:
                height, width = 480, 720  # default
            resolution = (height, width)
            env_resolution = (128, 256)
            
            cond_labels = dict(cond_labels)
            cond_labels.pop('env_ldr', None)
            cond_labels.pop('env_log', None)
            
            env_nrm_raw = envmap_vec(env_resolution)
            env_nrm = env_nrm_raw * 0.5 + 0.5
            
            # Read image
            img = read_image(input_path).astype(np.float32)[..., :3] / 255.0
            img = resize_crop(img, resolution)
            vid_example = {'rgb': img[None, :], 'env_nrm': env_nrm[None, :].numpy()}
            
            # Convert to pipeline input
            target_image, cond_images = pipeline.example2input(vid_example, target_labels, cond_labels)
            
            # Run inference - use vae device for generator to match original
            generator = None
            if seed is not None and not (isinstance(seed, float) and np.isnan(seed)):
                try:
                    seed_int = int(seed)
                    # Use vae device to match original script
                    vae_device = pipeline.vae.device if hasattr(pipeline, 'vae') else pipeline.device
                    generator = torch.Generator(device=vae_device).manual_seed(seed_int)
                except:
                    generator = None
            
            # Get text_prompt from config (usually None)
            text_prompt = pipeline_cfg.get('text_prompt', None)
            
            with torch.no_grad():
                pred = pipeline(
                    prompt=text_prompt,
                    cond_images=cond_images,
                    cond_mapping=cond_labels,
                    guidance_scale=guidance_scale,
                    use_dynamic_cfg=False,  # Default in original script unless --use_dynamic_cfg is set
                    num_inference_steps=num_inference_steps, 
                    generator=generator, 
                    height=height, width=width, 
                    num_frames=1,
                    additional_cond_labels=additional_cond_labels,
                    attention_kwargs={'scale': lora_scale},
                    denoise_type=denoise_type,
                    target_labels=_target_labels,
                    additional_target_labels=_additional_target_labels,
                    num_latents=len(target_labels),
                    separate_timesteps=pipeline_cfg.get('separate_timesteps', False),
                    additional_rope_time_only=pipeline_cfg.get('additional_rope_time_only', True),
                ).frames
            
            frames_ldr = pred[0][0]
            frames_log = pred[1][0]
            
            # Save outputs (hdr_merger expects *_ldr.png naming)
            output_ldr_path = os.path.join(temp_dir, "input_ldr.png")
            output_log_path = os.path.join(temp_dir, "input_log.png")
            frames_ldr[0].save(output_ldr_path)
            frames_log[0].save(output_log_path)
            
            # Also save with simpler names for display
            display_ldr_path = os.path.join(temp_dir, "ldr.png")
            display_log_path = os.path.join(temp_dir, "log.png")
            frames_ldr[0].save(display_ldr_path)
            frames_log[0].save(display_log_path)
            
            # Optionally run HDR merger
            hdr_path = None
            if use_hdr_merger:
                hdr_output_dir = os.path.join(temp_dir, "hdr")
                os.makedirs(hdr_output_dir, exist_ok=True)
                
                # Create args for hdr_merger
                hdr_args = argparse.Namespace(
                    input_dir=temp_dir,
                    output_dir=hdr_output_dir,
                    model_type="mlp",
                    model_path=hdr_model_path if hdr_model_path else "checkpoints/hdr_merge_mlp",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    disable_env_roll=False,
                    disable_env_flip=False,
                    disable_seam_blending=False
                )
                
                try:
                    hdr_merger_main(hdr_args)
                    
                    # Find the generated HDR file
                    hdr_files = [f for f in os.listdir(hdr_output_dir) if f.endswith('.exr')]
                    if hdr_files:
                        hdr_source = os.path.join(hdr_output_dir, hdr_files[0])
                        # Copy to local temp directory that Gradio can access
                        hdr_filename = f"hdr_{int(time.time() * 1000)}.exr"
                        hdr_path = os.path.join(LOCAL_TEMP_DIR, hdr_filename)
                        shutil.copy2(hdr_source, hdr_path)
                except Exception as e:
                    print(f"HDR merger error: {e}")
                    hdr_path = None
            
            # Return results
            ldr_img = Image.open(display_ldr_path)
            log_img = Image.open(display_log_path)
            hdr_info = f"HDR saved to: {hdr_path}" if hdr_path else "HDR merger not enabled or failed"
            
            # Verify HDR file exists before returning
            if hdr_path and not os.path.exists(hdr_path):
                hdr_path = None
                hdr_info = "HDR file was generated but could not be accessed"
            
            return ldr_img, log_img, hdr_path, hdr_info
        finally:
            # Clean up temp directory after a delay (let Gradio access files first)
            # Note: We don't delete immediately to allow Gradio to access files
            pass  # Files will be cleaned up by a separate cleanup process if needed
            
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, error_msg

def process_video_inference(
    video,
    model_type,
    config_path,
    transformer_path,
    lora_dir,
    lora_scale,
    resolution,
    guidance_scale,
    num_inference_steps,
    seed,
    frames_per_sample,
    use_hdr_merger,
    hdr_model_path
):
    """Process a video"""
    if video is None:
        return None, None, None, "Please upload a video"
    
    try:
        # Create temporary directory in local dir for processing
        temp_dir = os.path.join(LOCAL_TEMP_DIR, f"vid_{int(time.time() * 1000)}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            # Save uploaded video
            input_path = os.path.join(temp_dir, "input.mp4")
            if isinstance(video, str):
                shutil.copy(video, input_path)
            else:
                shutil.copy(video, input_path)
            
            # Load pipeline
            lora_dir_clean = lora_dir.strip() if lora_dir and lora_dir.strip() else None
            pipeline, cfg = load_pipeline(model_type, config_path, transformer_path, lora_dir_clean)
            
            pipeline_cfg = cfg.model_pipeline
            data_cfg = cfg.train_dataset if 'train_dataset' in cfg else cfg.train_data
            
            denoise_type = pipeline_cfg.get('denoise_type', 'additional_hidden_states')
            _target_labels, cond_labels = pipeline_cfg.target_image, pipeline_cfg.cond_images
            additional_cond_labels = pipeline_cfg.get('additional_cond_labels', None)
            _additional_target_labels = pipeline_cfg.get('additional_target_image', [])
            
            if denoise_type == 'additional_hidden_states':
                target_labels = _additional_target_labels
            else:
                target_labels = _target_labels
            
            # Parse resolution from dropdown (format: "480x720" or "512x512")
            if resolution:
                height, width = map(int, resolution.split('x'))
            else:
                height, width = 480, 720  # default
            resolution = (height, width)
            env_resolution = (128, 256)
            
            cond_labels = dict(cond_labels)
            cond_labels.pop('env_ldr', None)
            cond_labels.pop('env_log', None)
            
            env_nrm_raw = envmap_vec(env_resolution)
            env_nrm = env_nrm_raw * 0.5 + 0.5
            
            # Read video frames
            reader = imageio.get_reader(input_path, 'ffmpeg')
            all_frames = []
            for frame in reader:
                all_frames.append(frame)
            reader.close()
            
            if frames_per_sample is not None and frames_per_sample > 0:
                all_frames = all_frames[:frames_per_sample]
            
            img_list = []
            for frame in all_frames:
                img = frame.astype(np.float32)[..., :3] / 255.0
                img = resize_crop(img, resolution)
                img_list.append(img)
            
            img = np.stack(img_list, axis=0)
            num_frames = img.shape[0]
            env_nrm_vid = env_nrm[None, :].repeat(num_frames, 1, 1, 1)
            
            vid_example = {'rgb': img, 'env_nrm': env_nrm_vid.numpy()}
            
            # Convert to pipeline input
            target_image, cond_images = pipeline.example2input(vid_example, target_labels, cond_labels)
            
            # Run inference - use vae device for generator to match original
            generator = None
            if seed is not None and not (isinstance(seed, float) and np.isnan(seed)):
                try:
                    seed_int = int(seed)
                    # Use vae device to match original script
                    vae_device = pipeline.vae.device if hasattr(pipeline, 'vae') else pipeline.device
                    generator = torch.Generator(device=vae_device).manual_seed(seed_int)
                except:
                    generator = None
            
            # Get text_prompt from config (usually None)
            text_prompt = pipeline_cfg.get('text_prompt', None)
            
            with torch.no_grad():
                pred = pipeline(
                    prompt=text_prompt,
                    cond_images=cond_images,
                    cond_mapping=cond_labels,
                    guidance_scale=guidance_scale,
                    use_dynamic_cfg=False,  # Default in original script unless --use_dynamic_cfg is set
                    num_inference_steps=num_inference_steps, 
                    generator=generator, 
                    height=height, width=width, 
                    num_frames=num_frames,
                    additional_cond_labels=additional_cond_labels,
                    attention_kwargs={'scale': lora_scale},
                    denoise_type=denoise_type,
                    target_labels=_target_labels,
                    additional_target_labels=_additional_target_labels,
                    num_latents=len(target_labels),
                    separate_timesteps=pipeline_cfg.get('separate_timesteps', False),
                    additional_rope_time_only=pipeline_cfg.get('additional_rope_time_only', True),
                ).frames
            
            frames_ldr = pred[0][0]
            frames_log = pred[1][0]
            
            # Save outputs
            output_ldr_dir = os.path.join(temp_dir, "ldr_log")
            os.makedirs(output_ldr_dir, exist_ok=True)
            
            for j in range(num_frames):
                output_ldr_path = os.path.join(output_ldr_dir, f"frame_{j:04d}_ldr.png")
                output_log_path = os.path.join(output_ldr_dir, f"frame_{j:04d}_log.png")
                frames_ldr[j].save(output_ldr_path)
                frames_log[j].save(output_log_path)
            
            # Create video output
            frames_concat = [
                np.concatenate([np.asarray(f_ldr), np.asarray(f_log)], axis=1) 
                for f_ldr, f_log in zip(frames_ldr, frames_log)
            ]
            output_video_path = os.path.join(temp_dir, "output.mp4")
            imageio.mimwrite(output_video_path, frames_concat, fps=8)
            
            # Optionally run HDR merger
            hdr_info = "HDR merger not enabled"
            hdr_files_list = None
            if use_hdr_merger:
                hdr_output_dir = os.path.join(temp_dir, "hdr")
                os.makedirs(hdr_output_dir, exist_ok=True)
                
                hdr_args = argparse.Namespace(
                    input_dir=output_ldr_dir,
                    output_dir=hdr_output_dir,
                    model_type="mlp",
                    model_path=hdr_model_path if hdr_model_path else "checkpoints/hdr_merge_mlp",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    disable_env_roll=False,
                    disable_env_flip=False,
                    disable_seam_blending=False
                )
                
                try:
                    hdr_merger_main(hdr_args)
                    hdr_files = [os.path.join(hdr_output_dir, f) for f in os.listdir(hdr_output_dir) if f.endswith('.exr')]
                    if hdr_files:
                        # Copy HDR files to local temp directory that Gradio can access
                        persistent_hdr_files = []
                        for idx, hdr_file in enumerate(hdr_files):
                            hdr_filename = f"hdr_{int(time.time() * 1000)}_{idx}.exr"
                            persistent_path = os.path.join(LOCAL_TEMP_DIR, hdr_filename)
                            shutil.copy2(hdr_file, persistent_path)
                            persistent_hdr_files.append(persistent_path)
                        hdr_files_list = persistent_hdr_files[0] if len(persistent_hdr_files) == 1 else persistent_hdr_files
                        hdr_info = f"HDR files generated: {len(persistent_hdr_files)} files"
                    else:
                        hdr_files_list = None
                        hdr_info = "HDR merger completed but no files found"
                except Exception as e:
                    hdr_files_list = None
                    hdr_info = f"HDR merger error: {str(e)}"
            
            return output_video_path, hdr_files_list, hdr_info
        finally:
            # Clean up temp directory after a delay (let Gradio access files first)
            # Note: We don't delete immediately to allow Gradio to access files
            pass  # Files will be cleaned up by a separate cleanup process if needed
            
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, error_msg

def _example_path(rel_path):
    """Resolve example path relative to this script."""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel_path)


def _discover_examples():
    """Discover existing image and video examples under examples/."""
    base = os.path.dirname(os.path.abspath(__file__))
    ex_root = os.path.join(base, "examples")
    
    def glob_sorted(*patterns):
        out = []
        for p in patterns:
            path = os.path.join(ex_root, p)
            for f in sorted(glob.glob(path)):
                if os.path.isfile(f):
                    out.append(f)
        return out
    
    image_examples = []
    # Synthetic images (no LoRA)
    for f in glob_sorted("input_demo/synthetic_images/*.png", "input_demo/synthetic_images/*.jpg")[:2]:
        image_examples.append((f, "base", "configs/luxdit_base.yaml", "checkpoints/luxdit_base",
            "", 0.0, "480x720", 2.5, 50, 33, True, "checkpoints/hdr_merge_mlp"))
    # Real scene images (with LoRA)
    for f in glob_sorted("input_demo/scene_images/*.png", "input_demo/scene_images/*.jpg")[:3]:
        image_examples.append((f, "base", "configs/luxdit_base.yaml", "checkpoints/luxdit_base",
            "checkpoints/luxdit_base/lora", 0.8, "480x720", 2.5, 50, 33, True, "checkpoints/hdr_merge_mlp"))
    
    video_examples = []
    # Synthetic videos
    for f in glob_sorted("input_demo/synthetic_videos/*.mp4")[:2]:
        video_examples.append((f, "video", "configs/luxdit_base.yaml", "checkpoints/luxdit_video",
            "", 0.0, "480x720", 2.5, 40, 33, 25, True, "checkpoints/hdr_merge_mlp"))
    # Real scene videos
    for f in glob_sorted("input_demo/scene_videos/*.mp4", "360vid/*.mp4")[:2]:
        video_examples.append((f, "video", "configs/luxdit_base.yaml", "checkpoints/luxdit_video",
            "checkpoints/luxdit_video/lora", 0.8, "480x720", 2.5, 40, 33, 25, True, "checkpoints/hdr_merge_mlp"))
    
    return image_examples, video_examples


def create_demo():
    """Create the Gradio interface"""
    
    image_examples_list, video_examples_list = _discover_examples()
    
    with gr.Blocks(title="LuxDiT: Lighting Estimation Demo") as demo:
        gr.Markdown("""
        # LuxDiT: Lighting Estimation with Video Diffusion Transformer
        
        This demo allows you to estimate high-quality HDR environment maps from images or videos.
        Use the **Quick start** examples below to try with sample inputs, or upload your own.
        """)
        
        with gr.Tabs():
            with gr.Tab("Image Inference"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Input Image", type="pil")
                        
                        with gr.Accordion("Model Settings", open=True):
                            model_type = gr.Radio(
                                choices=["base", "video"],
                                value="base",
                                label="Model Type"
                            )
                            config_path = gr.Textbox(
                                value="configs/luxdit_base.yaml",
                                label="Config Path"
                            )
                            transformer_path = gr.Textbox(
                                value="checkpoints/luxdit_base",
                                label="Transformer Path"
                            )
                            lora_dir = gr.Textbox(
                                value="",
                                label="LoRA Directory (optional, for real scenes)",
                                placeholder="Leave empty for synthetic scenes. Example: checkpoints/luxdit_base/lora"
                            )
                            lora_scale = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                                label="LoRA Scale"
                            )
                        
                        with gr.Accordion("Inference Parameters", open=True):
                            resolution = gr.Dropdown(
                                choices=["480x720", "512x512"],
                                value="480x720",
                                label="Resolution (Height x Width)"
                            )
                            guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=2.5,
                                step=0.1,
                                label="Guidance Scale"
                            )
                            num_inference_steps = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5,
                                label="Number of Inference Steps"
                            )
                            seed = gr.Number(
                                value=33,
                                label="Random Seed (leave empty for random)",
                                precision=0
                            )
                        
                        with gr.Accordion("HDR Merger", open=False):
                            use_hdr_merger = gr.Checkbox(
                                value=True,
                                label="Enable HDR Merger"
                            )
                            hdr_model_path = gr.Textbox(
                                value="checkpoints/hdr_merge_mlp",
                                label="HDR Model Path"
                            )
                        
                        image_submit = gr.Button("Generate Environment Map", variant="primary")
                    
                    with gr.Column():
                        ldr_output = gr.Image(label="LDR Environment Map")
                        log_output = gr.Image(label="Log Environment Map")
                        hdr_output = gr.File(label="HDR File (if enabled)")
                        status_output = gr.Textbox(label="Status", lines=5)
                
                image_submit.click(
                    fn=process_image_inference,
                    inputs=[
                        image_input,
                        model_type,
                        config_path,
                        transformer_path,
                        lora_dir,
                        lora_scale,
                        resolution,
                        guidance_scale,
                        num_inference_steps,
                        seed,
                        use_hdr_merger,
                        hdr_model_path
                    ],
                    outputs=[ldr_output, log_output, hdr_output, status_output]
                )
                
                # Quick-start examples from examples/ (synthetic + scene images)
                if image_examples_list:
                    gr.Examples(
                        examples=[list(ex) for ex in image_examples_list],
                        inputs=[
                            image_input,
                            model_type,
                            config_path,
                            transformer_path,
                            lora_dir,
                            lora_scale,
                            resolution,
                            guidance_scale,
                            num_inference_steps,
                            seed,
                            use_hdr_merger,
                            hdr_model_path,
                        ],
                        label="Quick start (click an example to load)",
                        run_on_click=False,
                    )
            
            with gr.Tab("Video Inference"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Input Video")
                        
                        with gr.Accordion("Model Settings", open=True):
                            model_type_video = gr.Radio(
                                choices=["base", "video"],
                                value="video",
                                label="Model Type"
                            )
                            config_path_video = gr.Textbox(
                                value="configs/luxdit_base.yaml",
                                label="Config Path"
                            )
                            transformer_path_video = gr.Textbox(
                                value="checkpoints/luxdit_video",
                                label="Transformer Path"
                            )
                            lora_dir_video = gr.Textbox(
                                value="",
                                label="LoRA Directory (optional, for real scenes)",
                                placeholder="Leave empty for synthetic scenes. Example: checkpoints/luxdit_video/lora"
                            )
                            lora_scale_video = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                                label="LoRA Scale"
                            )
                        
                        with gr.Accordion("Inference Parameters", open=True):
                            resolution_video = gr.Dropdown(
                                choices=["480x720", "512x512"],
                                value="480x720",
                                label="Resolution (Height x Width)"
                            )
                            guidance_scale_video = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=2.5,
                                step=0.1,
                                label="Guidance Scale"
                            )
                            num_inference_steps_video = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=40,
                                step=5,
                                label="Number of Inference Steps"
                            )
                            frames_per_sample = gr.Number(
                                value=25,
                                label="Frames per Sample (leave empty for all frames)"
                            )
                            seed_video = gr.Number(
                                value=33,
                                label="Random Seed (leave empty for random)",
                                precision=0
                            )
                        
                        with gr.Accordion("HDR Merger", open=False):
                            use_hdr_merger_video = gr.Checkbox(
                                value=True,
                                label="Enable HDR Merger"
                            )
                            hdr_model_path_video = gr.Textbox(
                                value="checkpoints/hdr_merge_mlp",
                                label="HDR Model Path"
                            )
                        
                        video_submit = gr.Button("Generate Environment Maps", variant="primary")
                    
                    with gr.Column():
                        video_output = gr.Video(label="Output Video (LDR + Log concatenated)")
                        hdr_output_video = gr.File(label="HDR Files (if enabled)")
                        status_output_video = gr.Textbox(label="Status", lines=5)
                
                video_submit.click(
                    fn=process_video_inference,
                    inputs=[
                        video_input,
                        model_type_video,
                        config_path_video,
                        transformer_path_video,
                        lora_dir_video,
                        lora_scale_video,
                        resolution_video,
                        guidance_scale_video,
                        num_inference_steps_video,
                        seed_video,
                        frames_per_sample,
                        use_hdr_merger_video,
                        hdr_model_path_video
                    ],
                    outputs=[video_output, hdr_output_video, status_output_video]
                )
                
                # Quick-start examples from examples/ (synthetic + scene videos)
                if video_examples_list:
                    gr.Examples(
                        examples=[list(ex) for ex in video_examples_list],
                        inputs=[
                            video_input,
                            model_type_video,
                            config_path_video,
                            transformer_path_video,
                            lora_dir_video,
                            lora_scale_video,
                            resolution_video,
                            guidance_scale_video,
                            num_inference_steps_video,
                            seed_video,
                            frames_per_sample,
                            use_hdr_merger_video,
                            hdr_model_path_video,
                        ],
                        label="Quick start (click an example to load)",
                        run_on_click=False,
                    )
        
        gr.Markdown("""
        ## Notes
        
        - **Model Type**: Choose `base` for single images or `video` for video sequences
        - **LoRA Scale**: Adjust how much the input scene is merged into the estimated envmap (0.0-1.0)
        - **Resolution**: The model is trained mainly with 512x512 and 480x720 resolutions
        - **Video Frames**: The model is trained with videos of 7, 17, and 25 frames. Longer videos may not work as well.
        - **HDR Merger**: Enable to merge the dual-tone mapped envmap into a single HDR envmap
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
