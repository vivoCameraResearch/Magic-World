import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
# 修正导入
from moviepy.editor import VideoFileClip, concatenate_videoclips

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKLWan,
    AutoTokenizer as WanAutoTokenizer,
    CLIPModel,
    WanT5EncoderModel
)
from videox_fun.models.wan_transformer3d_magicworld_base import WanTransformer3DModel
from videox_fun.data.dataset_image_video import process_pose_file
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline.pipeline_magicworld_base import WanFunControlPipeline
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, get_image_latent, save_videos_grid
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ========================
# 参数配置（沿用 2.1）
# ========================
GPU_memory_mode     = ""  # sequential_cpu_offload
ulysses_degree      = 1
ring_degree         = 1
fsdp_dit            = False
fsdp_text_encoder   = True
compile_dit         = False

# TeaCache / CFG Skip
enable_teacache      = True
teacache_threshold   = 0.10
num_skip_start_steps = 5
teacache_offload     = False
cfg_skip_ratio       = 0

# Riflex（如需）
enable_riflex       = False
riflex_k            = 6

# Config & Model
config_path         = "config/wan2.1/wan_civitai.yaml"
model_name          = "checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"
transformer_name    = "checkpoints/MagicWorld/MagicWorld-Base"

# 采样器与 shift
sampler_name        = "Flow"  # ["Flow", "Flow_Unipc", "Flow_DPM++"]
shift               = 3

# 可选权重
transformer_path    = None
vae_path            = None
lora_path           = None
lora_weight         = 0.55

# I/O & 画质
sample_size         = [480, 832]   # [H, W]
video_length        = 33
fps                 = 16
weight_dtype        = torch.bfloat16

# 控制输入
control_camera_txt  = ""
num_chunks          = 3

# ========================
# 单张图片 + 单条 prompt
# ========================
start_image_path = ""

prompt = ""


negative_prompt = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

guidance_scale      = 6.0
seed                = 43
num_inference_steps = 20
save_root           = "samples/wan-videos-magicworld_base"
os.makedirs(save_root, exist_ok=True)


# ========================
# 工具函数
# ========================
def concat_all_segments(video_dir: str, output_path: str):
    seg_files = [f for f in os.listdir(video_dir) if f.startswith("segment_") and f.endswith(".mp4")]
    seg_files = sorted(seg_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if len(seg_files) == 0:
        raise RuntimeError("No segment_XXX.mp4 found to concatenate.")

    clips = [VideoFileClip(os.path.join(video_dir, f)) for f in seg_files]
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, codec="libx264", audio=False)

    for c in clips:
        c.close()
    final_clip.close()


def save_video_segment(sample: torch.Tensor, save_dir: str, step_idx: int, fps: int):
    os.makedirs(save_dir, exist_ok=True)
    seg_path = os.path.join(save_dir, f"segment_{step_idx:03d}.mp4")
    save_videos_grid(sample, seg_path, fps=fps)
    return seg_path


def _tensor_to_pil_frame(frame_chw: torch.Tensor) -> Image.Image:
    """
    这里假设 sample 的像素范围是 [0,1]。
    如果你的 save_videos_grid 内部对应的是 [-1,1]，这里改成：
        x = ((x + 1.0) / 2.0).clamp(0, 1)
    """
    x = frame_chw.detach().float().clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def save_last_frame_png(sample_videos: torch.Tensor, save_dir: str, step_idx: int) -> str:
    os.makedirs(save_dir, exist_ok=True)
    last_frame = sample_videos[0, :, -1]  # [C,H,W]
    pil_img = _tensor_to_pil_frame(last_frame)
    out_path = os.path.join(save_dir, f"prev_last_frame_step_{step_idx:03d}.png")
    pil_img.save(out_path)
    return out_path


# ========================
# 构建与准备
# ========================
if not os.path.isfile(start_image_path):
    raise FileNotFoundError(f"start_image_path not found: {start_image_path}")

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(
        transformer_name,
        config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')
    ),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)
    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)
    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

tokenizer = WanAutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).eval()

clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype).eval()

Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]

if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
    config['scheduler_kwargs']['shift'] = 1

scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

pipeline = WanFunControlPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder,
)

if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation"], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients,
        num_inference_steps,
        teacache_threshold,
        num_skip_start_steps=num_skip_start_steps,
        offload=teacache_offload
    )

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)


# ========================
# 单张图片：多段生成并拼接
# ========================
generator = torch.Generator(device=device).manual_seed(seed)

with torch.no_grad():
    # 对齐 VAE 时域压缩
    video_length_adj = (
        int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
        if video_length != 1 else 1
    )
    if video_length_adj != video_length:
        print(
            f"[Info] Adjust video_length from {video_length} to {video_length_adj} "
            f"(VAE temporal compression ratio={vae.config.temporal_compression_ratio})."
        )
        video_length = video_length_adj

    # 载入整段相机轨迹并检查足够的帧数
    ctrl_cam_full = process_pose_file(control_camera_txt, sample_size[1], sample_size[0])
    total_needed = video_length * num_chunks
    if ctrl_cam_full.shape[0] < total_needed:
        old_chunks = num_chunks
        num_chunks = ctrl_cam_full.shape[0] // video_length
        print(
            f"[Warning] Trajectory frames {ctrl_cam_full.shape[0]} < {total_needed}. "
            f"Reduce num_chunks from {old_chunks} to {num_chunks}."
        )
        total_needed = video_length * num_chunks

    if num_chunks <= 0:
        raise RuntimeError("num_chunks becomes 0 because trajectory is too short.")

    ctrl_cam_full = ctrl_cam_full[:total_needed]

    stem = Path(start_image_path).stem
    per_image_dir = os.path.join(save_root, stem)
    os.makedirs(per_image_dir, exist_ok=True)
    final_path = os.path.join(save_root, f"{stem}.mp4")

    print(f"\n========== Start image: {start_image_path} ==========")
    print("Prompt:\n", prompt if len(prompt) < 300 else (prompt[:300] + " ..."))

    prev_last_frame_path = None

    for step in range(num_chunks):
        s_idx, e_idx = step * video_length, (step + 1) * video_length
        control_camera_video = ctrl_cam_full[s_idx:e_idx]  # [T,H,W,C]
        control_camera_video = control_camera_video.permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,T,H,W]

        # 第一段用原始 start_image，后续用上一段最后一帧
        if step == 0:
            current_start_image_path = start_image_path
        else:
            if prev_last_frame_path is None or (not os.path.isfile(prev_last_frame_path)):
                raise FileNotFoundError(f"Prev last frame for step {step} not found: {prev_last_frame_path}")
            current_start_image_path = prev_last_frame_path

        # 2.1 管线输入
        start_image_latent = get_image_latent(current_start_image_path, sample_size=sample_size)
        clip_image = Image.open(current_start_image_path).convert("RGB")
        ref_latent = None

        print(f"[{stem}] [Step {step+1}/{num_chunks}] Generating ...")
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            control_video=None,
            control_camera_video=control_camera_video,
            ref_image=ref_latent,
            start_image=start_image_latent,
            clip_image=clip_image,
            shift=shift,
        ).videos

        seg_path = save_video_segment(sample, per_image_dir, step, fps)
        print(f"[{stem}] Saved {seg_path}")

        # 保存最后一帧供下一段使用
        prev_last_frame_path = save_last_frame_png(sample, per_image_dir, step)

    concat_all_segments(per_image_dir, final_path)
    print(f"[{stem}] Final video saved to {final_path}")

# 清理 LoRA（如使用）
if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)
