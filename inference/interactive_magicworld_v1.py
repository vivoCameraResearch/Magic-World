import os
import sys
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer
from uni3c_cam_render_api import render_from_image_and_traj

# 片段拼接与截帧
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer as WanAutoTokenizer, CLIPModel,
                               WanT5EncoderModel)
from videox_fun.models.wan_transformer3d_magicworld_v1 import WanTransformer3DModel
from videox_fun.data.dataset_image_video import process_pose_file
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline.pipeline_magicworld_v1 import WanFunControlPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_latent,
                                    get_video_to_video_render_latent,
                                    save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# ========================
# 参数配置（沿用 2.1 脚手架）
# ========================
GPU_memory_mode     = ""  # If video memory overflows, use sequential_cpu_offload
ulysses_degree      = 1
ring_degree         = 1
fsdp_dit            = False
fsdp_text_encoder   = True
compile_dit         = False

# TeaCache / CFG Skip
enable_teacache     = True
teacache_threshold  = 0.10
num_skip_start_steps = 5
teacache_offload    = False
cfg_skip_ratio      = 0

# Riflex（如需）
enable_riflex       = False
riflex_k            = 6

# Config & Model
config_path         = "config/wan2.1/wan_civitai.yaml"
model_name          = "Wan2.1-Fun-V1.1-1.3B-Control-Camera"
transformer_name    = ""

# 采样器与 shift
sampler_name        = "Flow"  # ["Flow", "Flow_Unipc", "Flow_DPM++"]
shift               = 3

# 可选权重
transformer_path    = None
vae_path            = None
lora_path           = None
lora_weight         = 0.55

# I/O & 画质
sample_size         = [480, 832]  # [H, W]
video_length        = 33
fps                 = 16
weight_dtype        = torch.bfloat16

# 控制输入
control_video           = None
control_camera_txt      = "asset/bench/W/trajectory_full.txt"

# === 新增：指定“对应的视频”用于截取 start_image === 
start_image_path = ""  # <-- 改成你的起始图片
# 兼容保留：基准 ref_image（可为 None）
ref_image               = None

# 交互次数
num_chunks          = 4

prompt = ""
negative_prompt = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 20
save_root               = "samples/wan-videos-magicworld-v1"
os.makedirs(save_root, exist_ok=True)

# ========================
# 工具函数
# ========================
def to_rgb_image(start_image):
    # 如果是字符串或 Path，认为是文件路径
    if isinstance(start_image, str):
        img = Image.open(start_image)
    else:
        # 否则认为已经是图像对象，直接使用
        img = start_image
    # 统一转为 RGB
    return img.convert("RGB")

def concat_all_segments(video_dir: str, output_path: str):
    seg_files = [f for f in os.listdir(video_dir) if f.startswith("segment_") and f.endswith(".mp4")]
    seg_files = sorted(seg_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if len(seg_files) == 0:
        raise RuntimeError("No segment_XXX.mp4 found to concatenate.")
    clips = [VideoFileClip(os.path.join(video_dir, f)) for f in seg_files]
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, codec="libx264", audio=False)


def save_video_segment(sample: torch.Tensor, save_dir: str, step_idx: int, fps: int):
    os.makedirs(save_dir, exist_ok=True)
    seg_path = os.path.join(save_dir, f"segment_{step_idx:03d}.mp4")
    save_videos_grid(sample, seg_path, fps=fps)
    return seg_path


def _ensure_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    assert arr.shape[-1] == 3, f"Unexpected frame shape: {arr.shape}"
    return arr


def extract_frame_from_video(video_path: str, frame_index: int, save_dir: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total:
        cap.release()
        raise IndexError(f"Requested frame {frame_index} >= total frames {total} for video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = _ensure_rgb_uint8(frame_rgb)
    image_pil = Image.fromarray(frame_rgb)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"start_frame_{frame_index:06d}.png")
    image_pil.save(out_path)
    return out_path


def _tensor_to_pil_frame(frame_chw: torch.Tensor) -> Image.Image:
    """将 [-1,1] 的 [C,H,W] 张量转为 PIL RGB。"""
    with torch.no_grad():
        # x = frame_chw.detach().float().cpu().clamp(-1, 1)
        # x = (x + 1.0) / 2.0 # [0,1]
        x = frame_chw
        x = (x * 255.0).round().to(torch.uint8)
        x = x.permute(1, 2, 0).numpy() # [H,W,C]
    return Image.fromarray(x)

def save_last_frame_png(sample_videos: torch.Tensor, save_dir: str, step_idx: int) -> str:
    """
    将当前生成结果 sample 的最后一帧另存为 PNG，供下一段作为 start_image 使用。
    sample_videos: [B,C,T,H,W]，值域约 [-1,1]
    返回保存路径。
    """
    os.makedirs(save_dir, exist_ok=True)
    last_frame = sample_videos[0, :, -1] # [C,H,W]
    pil_img = _tensor_to_pil_frame(last_frame)
    out_path = os.path.join(save_dir, f"prev_last_frame_step_{step_idx:03d}.png")
    pil_img.save(out_path)
    return out_path

def load_traj_params_from_txt(txt_path):
    traj = []
    with open(txt_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            vals = [float(x) for x in ln.split()]  # 每行：K 个浮点
            traj.append(vals)
    return traj  # [T, K]

# ========================
# 构建与准备
# ========================

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(transformer_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
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

# Tokenizer & Text Encoder
tokenizer = WanAutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).eval()

# Clip Image Encoder
clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype).eval()

# Scheduler
Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(**filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs'])))

# Pipeline
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

# TeaCache / CFG Skip
coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold,
        num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

# LoRA
if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)



# ========================
# 生成 7 段并拼接
# ========================

generator = torch.Generator(device=device).manual_seed(seed)

with torch.no_grad():
    # 对齐 VAE 时域压缩
    video_length_adj = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    if video_length_adj != video_length:
        print(f"[Info] Adjust video_length from {video_length} to {video_length_adj} (VAE temporal compression ratio={vae.config.temporal_compression_ratio}).")
        video_length = video_length_adj

    # 读取整段相机轨迹并切块（T,H,W,C）
    ctrl_cam_full = process_pose_file(control_camera_txt, sample_size[1], sample_size[0])
    traj_all = load_traj_params_from_txt(control_camera_txt)  # [T, K]
    # 校验足够帧数
    total_needed = video_length * num_chunks
    if ctrl_cam_full.shape[0] < total_needed:
        old_chunks = num_chunks
        num_chunks = ctrl_cam_full.shape[0] // video_length
        print(f"[Warning] Trajectory frames {ctrl_cam_full.shape[0]} < {total_needed}. Reduce num_chunks from {old_chunks} to {num_chunks}.")
        total_needed = video_length * num_chunks
    ctrl_cam_full = ctrl_cam_full[:total_needed]

    # 起始视频帧检查
    # if not os.path.isfile(start_image_source_video):
    #     raise FileNotFoundError(f"start_image_source_video not found: {start_image_source_video}")
    # cap_check = cv2.VideoCapture(start_image_source_video)
    # if not cap_check.isOpened():
    #     raise RuntimeError(f"Cannot open start_image_source_video: {start_image_source_video}")
    # total_frames_src = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
    # cap_check.release()
    # 起始图片检查
    if not os.path.isfile(start_image_path):
        raise FileNotFoundError(f"start_image_path not found: {start_image_path}")

    prev_last_frame_path = None

    for step in range(num_chunks):
        # 当前段相机视频
        s_idx, e_idx = step * video_length, (step + 1) * video_length

        control_camera_video = ctrl_cam_full[s_idx:e_idx]            # [T,H,W,C]
        control_camera_video = control_camera_video.permute(3, 0, 1, 2).unsqueeze(0)  # -> [1,C,T,H,W]
        traj_params_slice = traj_all[s_idx:e_idx]   # 这是个 list[list[float]]

        # 确定本段起始帧：第 1 段用源视频抽帧；其余段用“上一段生成结果的最后一帧”
        if step == 0:
            current_start_image_path = start_image_path
            start_source = "provided_start_image"
        else:
            if prev_last_frame_path is None or (not os.path.isfile(prev_last_frame_path)):
                raise FileNotFoundError(
                    f"Prev last frame not found for step {step}. Expected: {prev_last_frame_path}"
                )
            current_start_image_path = prev_last_frame_path
            start_source = "prev_last_frame"
        
        point_img = to_rgb_image(current_start_image_path)
        render_frames, mask_frames = render_from_image_and_traj(
        reference_image=point_img,
        traj_params=traj_params_slice ,
        output_path="outputs/point_video_result",
        traj_type="free1",
        nframe=video_length,
        )

        # 为 2.1 管线准备的 start_image latent & clip_image
        start_image_latent = get_image_latent(current_start_image_path, sample_size=sample_size)
        clip_image = Image.open(current_start_image_path).convert("RGB")
        ref_latent = get_image_latent(ref_image, sample_size=sample_size) if ref_image is not None else None

        render_video, render_video_mask, _ , _, _ = get_video_to_video_render_latent(render_frames, mask_frames, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None)


        print(f"\n[Step {step}] Generating chunk {step+1}/{num_chunks} (start source = {'video' if step==0 else 'prev_last_frame'}) ...")
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,

            current_step=step,
            control_video=None,
            render_video  = render_video,
            render_video_mask = render_video_mask,
            control_camera_video=control_camera_video,
            ref_image=ref_latent,
            start_image=start_image_latent,
            clip_image=clip_image,
            shift=shift,
        ).videos

        # 保存本段视频
        save_video_segment(sample, save_root, step, fps)
        print(f"[Step {step}] Saved segment_{step:03d}.mp4")

        # 关键：导出最后一帧，供下一段作为 start_image 使用（图像域→再编码为 latent）
        prev_last_frame_path = save_last_frame_png(sample, save_root, step)

# 拼接输出
final_path = os.path.join(save_root, "final_concat_video.mp4")
concat_all_segments(save_root, final_path)
print(f"Final video saved to {final_path}")

# 清理 LoRA（如使用）
if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)
