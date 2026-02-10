# uni3c_cam_render_api.py
import os
import sys
sys.path.insert(0, "the path of Uni3C-main")
import warnings
from typing import List, Sequence, Union

import einops
import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from diffusers.utils import export_to_video
from pytorch3d.renderer import PointsRasterizationSettings
from torchvision.transforms import ToTensor, ToPILImage

import third_party.depth_pro.depth_pro as depth_pro
from src.pointcloud import point_rendering

# 如果你不再使用 traj_map / build_cameras，可省略；这里保留以兼容签名
# from src.utils import traj_map, points_padding, np_points_padding, set_initial_camera, build_cameras

warnings.filterwarnings("ignore")


def _to_torch_image_0_1(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    """
    将输入统一成 PIL.Image（RGB），内部只做必要的格式兼容。
    """
    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
    elif isinstance(img, np.ndarray):
        # 支持 HxWx3 或 HxWx1（uint8 或 float[0,1]）
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        pil = Image.fromarray(img[:, :, :3], mode="RGB")
    elif isinstance(img, torch.Tensor):
        # 支持 CHW 或 HWC，范围 [0,1] 或 [0,255]
        ten = img.detach().cpu()
        if ten.ndim == 3 and ten.shape[0] in (1, 3):  # CHW
            ten = ten.permute(1, 2, 0)
        if ten.max() <= 1.0:
            ten = (ten * 255.0).round().clamp(0, 255).to(torch.uint8)
        else:
            ten = ten.clamp(0, 255).to(torch.uint8)
        npimg = ten.numpy()
        pil = Image.fromarray(npimg[:, :, :3], mode="RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    return pil


def _camera_from_param(cam_param: Sequence[float]) -> torch.Tensor:
    """
    cam_param:
      - 长度 25：前 9 个是 K（3x3），后 16 个是 w2c（4x4）展平。
      - 长度 19：后 12 个是 w2c（3x4）展平。
    返回 torch.Tensor(4,4) 的 w2c。
    """
    n = len(cam_param)
    if n == 25:
        w2c_flat = np.array(cam_param[9:], dtype=np.float32)  # 16
        w2c = w2c_flat.reshape(4, 4)
        return torch.from_numpy(w2c)
    elif n == 19:
        w2c_flat = np.array(cam_param[-12:], dtype=np.float32)  # 12
        w2c_3x4 = w2c_flat.reshape(3, 4)
        # 补齐为 4x4 齐次矩阵
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
        w2c_4x4 = np.vstack([w2c_3x4, bottom])
        return torch.from_numpy(w2c_4x4)
    else:
        raise ValueError(f"Unsupported cam_param length: {n}. Expected 25 or 19.")


def _points_padding_torch(points2d_xy: torch.Tensor) -> torch.Tensor:
    """
    输入 [H,W,2] 的 (x,y)，返回 [H,W,3] 的 (x,y,1)（torch 版本）
    """
    ones = torch.ones_like(points2d_xy[..., :1])
    return torch.cat([points2d_xy, ones], dim=-1)


def _points_padding_numpy(points3d_hw3: np.ndarray) -> np.ndarray:
    """
    输入 [N,3]，返回 [N,4] 的齐次坐标 (x,y,z,1)（numpy 版本）
    """
    ones = np.ones((points3d_hw3.shape[0], 1), dtype=points3d_hw3.dtype)
    return np.concatenate([points3d_hw3, ones], axis=1)


_MODELS = {}

def get_models(device):
    if device not in _MODELS:
        print("[cam_render_api] Init depth model (singleton)")
        depth_model, depth_transform = depth_pro.create_model_and_transforms(device=device)
        depth_model = depth_model.eval()  # 统一FP32更稳

        print("[cam_render_api] Init matting/seg model (singleton)")
        from huggingface_hub import hf_hub_download
        seg_net = TracerUniversalB7(
            device=device, batch_size=1,
            model_path=hf_hub_download(repo_id="Carve/tracer_b7", filename="tracer_b7.pth")
        ).eval()

        _MODELS[device] = (depth_model, depth_transform, seg_net)
    return _MODELS[device]

@torch.no_grad()
def render_from_image_and_traj(
    reference_image: Union[Image.Image, np.ndarray, torch.Tensor],
    traj_params: Union[List[Sequence[float]], np.ndarray],
    output_path: str,
    *,
    traj_type: str = "custom",
    nframe: int = 199,
    d_r: float = 1.0,
    d_theta: float = 0.0,
    d_phi: float = 0.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    focal_length: float = 1.0,
    start_elevation: float = 5.0,
    device: str = "cuda",
    depth_radius: float = 0.008,
    ppp: int = 8,
    fps: int = 30,
) -> tuple[List[Image.Image], List[Image.Image]]:   # ← 新增显式返回类型
    """
    将原 cam_render.py 的主流程封装成函数：
    - reference_image: 直接给图片对象（PIL/np/tensor）
    - traj_params: 直接给轨迹数据（list of 25-floats per frame 或 numpy[N,25]）
    - 其他参数保持与原命令行一致，便于无缝迁移

    结果：
      output_path/render.mp4, output_path/render_mask.mp4, output_path/pcd.ply
    """
    os.makedirs(output_path, exist_ok=True)

    # === 1) 预处理图像 ===
    image = _to_torch_image_0_1(reference_image)
    image = ImageOps.exif_transpose(image)
    w_origin, h_origin = image.size

    # 与原脚本一致的分辨率选择逻辑（当前固定到 hw_list[6]）
    hw_list = [[480, 768], [512, 720], [608, 608], [720, 512], [768, 480], [720, 1280], [512, 896], [480, 832]]
    height, width = hw_list[7]
    print(f"[cam_render_api] Resolution: {h_origin}x{w_origin} -> {height}x{width}")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    validation_image = ToTensor()(image)[None]  # [1,c,h,w], 0~1

    # === 2) 处理轨迹 ===
    if isinstance(traj_params, np.ndarray):
        cam_params = traj_params.tolist()
    else:
        cam_params = [list(map(float, fr)) for fr in traj_params]

    if nframe is None or nframe <= 0:
        nframe = len(cam_params)
    else:
        # 与传入 nframe 一致（若不一致按 nframe 截断/保护）
        cam_params = cam_params[:nframe]

    w2cs = [ _camera_from_param(p) for p in cam_params ]
    w2cs = torch.stack(w2cs, dim=0)                 # [F,4,4]
    c2ws = w2cs.inverse()                           # [F,4,4]

    # === 3) 深度模型 & 前景分割 ===
    print("[cam_render_api] Init depth model")
    depth_model, depth_transform = depth_pro.create_model_and_transforms(device=device)
    depth_model = depth_model.eval()

    print("[cam_render_api] Init matting/seg model")
    from huggingface_hub import hf_hub_download
    seg_net = TracerUniversalB7(
        device=device, batch_size=1,
        model_path=hf_hub_download(repo_id="Carve/tracer_b7", filename="tracer_b7.pth")
    ).eval()

    depth_image_np = np.array(image)
    depth_image = depth_transform(depth_image_np)   # depth_pro 的 transform
    prediction = depth_model.infer(depth_image, f_px=None)
    depth = prediction["depth"]  # [H,W] in meters
    depth = depth[None, None]    # [1,1,H,W]

    focallength_px = prediction["focallength_px"].item()
    K = torch.tensor([
        [focallength_px, 0,            width / 2],
        [0,             focallength_px, height / 2],
        [0,             0,              1]
    ], dtype=torch.float32)
    K_inv = K.inverse()
    intrinsic = K[None].repeat(nframe, 1, 1)        # [F,3,3]

    # === 4) 构建点云（相机0为世界） ===
    # 像素网格
    xs = torch.arange(width, dtype=torch.float32)
    ys = torch.arange(height, dtype=torch.float32)
    points2d = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1)  # [H,W,2]
    points3d_cam = _points_padding_torch(points2d).reshape(height * width, 3)  # [HW,3] (x,y,1)
    points3d_cam = (K_inv @ points3d_cam.T * depth.reshape(1, height * width).cpu()).T  # -> [HW,3]

    colors = ((depth_image + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).reshape(height * width, 3)

    c2w_0 = c2ws[0].to(torch.float32)
    # [HW,3] -> [HW,4] 齐次 & 乘以 c2w_0[:3,:]
    points3d_world = (c2w_0[:3, :] @ _points_padding_numpy(points3d_cam.cpu().numpy()).T).T  # [HW,3]
    pcd = trimesh.PointCloud(vertices=points3d_world, colors=colors.cpu().numpy())
    _ = pcd.export(os.path.join(output_path, "pcd.ply"))

    # === 5) 前景 mask（用于估计前景深度的中位数） ===
    origin_w_, origin_h_ = image.size
    image_pil_512 = image.resize((512, 512))
    fg_mask = seg_net([image_pil_512])[0]
    fg_mask = fg_mask.resize((origin_w_, origin_h_))
    fg_mask = np.array(fg_mask) > 127.5
    fg_mask_t = torch.tensor(fg_mask)
    if fg_mask_t.float().mean() < 0.05:
        fg_mask_t[...] = True
    depth_avg = torch.median(depth[0, 0, fg_mask_t]).item()

    # === 6) 渲染序列 ===
    # with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device.startswith("cuda"))):
    with torch.no_grad():
    # 强制禁用 autocast，避免被转成 half/bf16
        with torch.autocast("cuda", enabled=False):
            control_imgs, render_masks = point_rendering(
                K=intrinsic.float(),
                w2cs=w2cs.float(),
                depth=depth.float(),
                image=ToTensor()(image)[None].float() * 2 - 1,  # [-1,1]
                raster_settings=PointsRasterizationSettings(
                    image_size=(height, width),
                    radius=depth_radius,
                    points_per_pixel=ppp
                ),
                device=device,
                background_color=[0, 0, 0],
                sobel_threshold=0.35,
                sam_mask=None
            )

    control_imgs = einops.rearrange(control_imgs, "(b f) c h w -> b c f h w", f=nframe)
    render_masks = einops.rearrange(render_masks, "(b f) c h w -> b c f h w", f=nframe)

    render_video, mask_video = [], []
    control_imgs = control_imgs.to(torch.float32)
    to_pil = ToPILImage()
    for i in range(nframe):
        img = to_pil((control_imgs[0][:, i] + 1) / 2)  # [0,1]
        render_video.append(img)
        mask = to_pil(render_masks[0][:, i])
        mask_video.append(mask)

    export_to_video(render_video, os.path.join(output_path, "render.mp4"), fps=fps)
    export_to_video(mask_video, os.path.join(output_path, "render_mask.mp4"), fps=fps)

    print(f"[cam_render_api] Done. Saved to: {output_path}")

    # === 新增：返回帧序列 ===
    return render_video, mask_video
