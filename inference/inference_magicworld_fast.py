import argparse
import os
import sys
sys.path.insert(0, "./Magic-World")
import re
import json
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torchvision.transforms.functional as TF
from PIL import Image

from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalInferencePipeline,
)

from wan.modules.wan_image_encoder import CLIPModel
from utils.dataset_image_video import process_pose_file
from utils.misc import set_seed

from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller


# -------------------------
# Image & Prompt helpers
# -------------------------
def apply_transform(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image)


def load_prompt_index(json_path: str) -> dict:
    """
    JSON expected: list of {"name": "...png", "describe": "..."}.
    Build:
      - exact filename -> describe
      - stem -> describe
    """
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    by_name = {}
    by_stem = {}
    for it in items:
        name = str(it.get("name", "")).strip()
        desc = str(it.get("describe", "")).strip()
        if not name:
            continue
        by_name[name] = desc
        stem = Path(name).stem
        if stem and stem not in by_stem:
            by_stem[stem] = desc
    return {"by_name": by_name, "by_stem": by_stem, "items": items}


def query_prompt(prompt_index: dict, image_path: str, default_prompt: str = "") -> str:
    img_name = Path(image_path).name
    img_stem = Path(image_path).stem

    if img_name in prompt_index["by_name"]:
        return prompt_index["by_name"][img_name] or default_prompt

    if img_stem in prompt_index["by_stem"]:
        return prompt_index["by_stem"][img_stem] or default_prompt

    # loose hit: stem contained in JSON "name"
    for it in prompt_index["items"]:
        n = str(it.get("name", ""))
        if img_stem and (img_stem in n):
            d = str(it.get("describe", "")).strip()
            if d:
                return d

    return default_prompt


def safe_filename(s: str, max_len: int = 120) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:max_len] if len(s) > max_len else s


# -------------------------
# Dataset
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: str, exts=(".png", ".jpg", ".jpeg", ".webp")):
        self.image_dir = Path(image_dir)
        self.paths = []
        for p in sorted(self.image_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                self.paths.append(str(p))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in: {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.paths[idx]


# -------------------------
# AR helpers
# -------------------------
def video_last_frame_to_pil(video_btchw: torch.Tensor) -> Image.Image:
    """
    video_btchw: [B,T,C,H,W].
    Robustly map to uint8 RGB. Supports outputs in [-1,1] or [0,1].
    """
    assert video_btchw.ndim == 5
    frame = video_btchw[0, -1].detach().float().cpu()  # [C,H,W]

    # infer range
    if frame.min().item() < 0:
        frame = (frame * 0.5 + 0.5).clamp(0, 1)
    else:
        frame = frame.clamp(0, 1)

    frame_u8 = (frame * 255.0).round().to(torch.uint8)      # [C,H,W]
    frame_u8 = frame_u8.permute(1, 2, 0).numpy()            # [H,W,C]
    return Image.fromarray(frame_u8, mode="RGB")


def build_i2v_conditions_from_pil(
    start_pil: Image.Image,
    pipeline,
    clip_image_encoder,
    sampled_noise: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Returns: y_input, clip_context
    Exactly matches your original conditioning logic.
    """
    # clip context
    clip_image = start_pil.convert("RGB")
    clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device=device, dtype=dtype)
    clip_context = clip_image_encoder([clip_image[:, None, :, :]])

    # y_input from VAE latent of resized start image
    start_tensor = apply_transform(start_pil.convert("RGB")).squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=dtype)
    start_latent = pipeline.vae.encode_to_latent(start_tensor).to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)

    # noise is [B,T,16,60,104] in your code; conv_in expects [B,16,T,60,104]
    start_latents_conv_in = torch.zeros_like(sampled_noise).permute(0, 2, 1, 3, 4)
    if sampled_noise.size(1) != 1:
        start_latents_conv_in[:, :, :1] = start_latent
    y_input = start_latents_conv_in
    return y_input, clip_context


def build_camera_latents_for_segment(
    control_camera_video_full: torch.Tensor,
    seg_id: int,
    camera_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    control_camera_video_full: output of process_pose_file(...)
      expected shape like [L, H, W, C] (your original usage)
    returns y_camera_input tensor ready for pipeline.
    """
    seg_cam = control_camera_video_full[seg_id * camera_length:(seg_id + 1) * camera_length]  # [L,H,W,C]
    seg_cam = seg_cam.permute([3, 0, 1, 2]).unsqueeze(0)  # [B,C,L,H,W]

    control_camera_latents = torch.concat(
        [
            torch.repeat_interleave(seg_cam[:, :, 0:1], repeats=4, dim=2),
            seg_cam[:, :, 1:]
        ],
        dim=2
    ).transpose(1, 2)

    b, f, c, h, w = control_camera_latents.shape
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
    control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
    return control_camera_latents.to(device=device, dtype=dtype)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Folder containing start images")
    parser.add_argument("--extended_prompt_path", type=str, required=True, help="JSON prompt index")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--control_camera_txt", type=str, required=True, help="One fixed camera txt used for all images")

    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--i2v", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_with_index", action="store_true")
    args = parser.parse_args()

    # -------- distributed init --------
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        set_seed(args.seed + local_rank)
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda")
        set_seed(args.seed)

    torch.set_grad_enabled(False)

    print(f"[rank{local_rank}] Free VRAM {get_cuda_free_memory_gb(device)} GB")
    low_memory = get_cuda_free_memory_gb(device) < 40

    # -------- config --------
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # -------- init pipeline --------
    pipeline = CausalInferencePipeline(config, device=device)
    

    # load checkpoint
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict["generator" if not args.use_ema else "generator_ema"])
    checkpoint_step = os.path.basename(os.path.dirname(args.checkpoint_path))
    checkpoint_step = checkpoint_step.split("_")[-1]

    pipeline = pipeline.to(dtype=torch.bfloat16)

    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)

    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    # -------- clip image encoder --------
    clip_config_path = "config/wan2.1/wan_civitai.yaml"
    clip_name = "checkpoints/Wan2.1-Fun-V1.1-1.3B-InP"
    clip_config = OmegaConf.load(clip_config_path)

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(clip_name, clip_config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder")),
    ).to(device=device, dtype=torch.bfloat16)
    clip_image_encoder = clip_image_encoder.eval()

    # -------- prompt index --------
    prompt_index = load_prompt_index(args.extended_prompt_path)

    # -------- camera (fixed) --------
    sample_size = [480, 832]
    temporal_compression_ratio = 4
    camera_length = (args.num_output_frames - 1) * temporal_compression_ratio + 1

    control_camera_video_full = process_pose_file(args.control_camera_txt, sample_size[1], sample_size[0])
    segments = 1
    need_len = segments * camera_length
    if control_camera_video_full.shape[0] < need_len:
        raise ValueError(
            f"Fixed camera pose too short: got {control_camera_video_full.shape[0]}, need >= {need_len} "
            f"(segments={segments}, camera_length={camera_length})."
        )

    # -------- output dir --------
    out_dir = os.path.join(args.output_folder, checkpoint_step)
    if local_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # -------- dataset & loader --------
    dataset = ImageFolderDataset(args.data_path)
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    else:
        sampler = SequentialSampler(dataset)

    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=True)

    # -------- run --------
    if not args.i2v:
        raise ValueError("This script is written for I2V mode. Please pass --i2v.")

    if args.num_samples != 1:
        raise ValueError("This script currently supports --num_samples 1 (your original assumption).")

    default_prompt = ""
    for idx, img_path in enumerate(loader):
        img_path = img_path[0]
        img_name = Path(img_path).name
        img_stem = Path(img_path).stem

        prompt = query_prompt(prompt_index, img_path, default_prompt=default_prompt)
        if not prompt:
            # 保底：避免空 prompt 导致 text encoder 行为不可控
            prompt = "A high-quality cinematic video."

        prompts = [prompt] * args.num_samples

        # 自回归交互：start 图像会被更新为上一段 last frame (RGB)
        current_start_pil = Image.open(img_path).convert("RGB")

        all_video = []
        for seg_id in range(segments):
            # noise per segment
            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16
            )

            # build conditions from current_start_pil
            y_input, clip_context = build_i2v_conditions_from_pil(
                current_start_pil,
                pipeline=pipeline,
                clip_image_encoder=clip_image_encoder,
                sampled_noise=sampled_noise,
                device=device,
                dtype=torch.bfloat16
            )

            # fixed camera segment
            y_camera_input = build_camera_latents_for_segment(
                control_camera_video_full,
                seg_id=seg_id,
                camera_length=camera_length,
                device=device,
                dtype=torch.bfloat16
            )

            # inference
            video, _latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=None,
                y_input=y_input,
                y_camera_input=y_camera_input,
                clip_context=clip_context,
                low_memory=low_memory,
            )

            # update start_image for next segment: last RGB frame
            current_start_pil = video_last_frame_to_pil(video)

            # concat: keep full seg0, drop first frame for seg1-3 to avoid duplication
            if seg_id == 0:
                seg_video = rearrange(video, "b t c h w -> b t h w c").cpu()
            else:
                seg_video = rearrange(video[:, 1:], "b t c h w -> b t h w c").cpu()

            all_video.append(seg_video)

        # final output
        video_out = 255.0 * torch.cat(all_video, dim=1)  # [B, T_total, H, W, C]
        pipeline.vae.model.clear_cache()

        # filename
        if args.save_with_index:
            base = f"{idx:06d}_{img_stem}"
        else:
            base = img_stem

        base = safe_filename(base)
        output_path = os.path.join(out_dir, f"{base}.mp4")

        # write
        write_video(output_path, video_out[0], fps=16)

        if local_rank == 0:
            print(f"[OK] {img_name} -> {output_path}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
