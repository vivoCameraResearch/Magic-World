import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


# -------------------------
# 1) layout + resize
# -------------------------
def pixel_values_to_bfchw_01(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Return video in [B,F,3,H,W] float in [0,1]
    Accepts:
      - [B,F,3,H,W]
      - [B,3,F,H,W]
      - [B,F,H,W,3]
    """
    x = pixel_values
    if x.ndim != 5:
        raise ValueError(f"pixel_values must be 5D, got {x.shape}")

    if x.shape[2] == 3:  # [B,F,3,H,W]
        bfchw = x
    elif x.shape[1] == 3:  # [B,3,F,H,W]
        bfchw = x.permute(0, 2, 1, 3, 4).contiguous()
    elif x.shape[-1] == 3:  # [B,F,H,W,3]
        bfchw = x.permute(0, 1, 4, 2, 3).contiguous()
    else:
        raise ValueError(f"Cannot infer channel dim for pixel_values shape {x.shape}")

    if bfchw.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        bfchw = bfchw.float()

    if bfchw.max() > 1.5:
        bfchw = bfchw / 255.0
    return bfchw.clamp(0, 1)


def resize_short_side_bfchw(video_bfchw: torch.Tensor, short_side: int = 320) -> torch.Tensor:
    B, Fm, C, H, W = video_bfchw.shape
    if min(H, W) == short_side:
        return video_bfchw
    scale = short_side / float(min(H, W))
    newH = max(2, int(round(H * scale)))
    newW = max(2, int(round(W * scale)))
    x = video_bfchw.reshape(B * Fm, C, H, W)
    x = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)
    return x.reshape(B, Fm, C, newH, newW)


# -------------------------
# 2) pad/unpad to multiple of 8 (torchvision RAFT requirement)
# -------------------------
def pad_to_multiple_of_8_btchw(x: torch.Tensor):
    _, _, H, W = x.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    pads = (0, pad_w, 0, pad_h)  # (left,right,top,bottom)
    x_pad = F.pad(x, pads, mode="replicate")
    return x_pad, pads


def unpad_btchw(x: torch.Tensor, pads):
    l, r, t, b = pads
    if r == 0 and b == 0 and l == 0 and t == 0:
        return x
    return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]


# -------------------------
# 3) warp helpers
# -------------------------
def flow_to_grid(flow_b2hw: torch.Tensor):
    B, _, H, W = flow_b2hw.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=flow_b2hw.device),
        torch.arange(W, device=flow_b2hw.device),
        indexing="ij",
    )
    base = torch.stack([x, y], dim=0).float().unsqueeze(0).expand(B, -1, -1, -1)
    coords = base + flow_b2hw
    gx = 2.0 * (coords[:, 0] / (W - 1 + 1e-6)) - 1.0
    gy = 2.0 * (coords[:, 1] / (H - 1 + 1e-6)) - 1.0
    return torch.stack([gx, gy], dim=-1)  # [B,H,W,2]


def warp_bchw(x_bchw: torch.Tensor, flow_b2hw: torch.Tensor):
    grid = flow_to_grid(flow_b2hw)
    return F.grid_sample(x_bchw, grid, mode="bilinear", padding_mode="border", align_corners=True)


# -------------------------
# 4) RAFT flows (dynamic F)
# -------------------------
@torch.no_grad()
def raft_flow_pair(of_model, of_transforms, img1_bchw, img2_bchw):
    img1_pad, pads = pad_to_multiple_of_8_btchw(img1_bchw)
    img2_pad, _    = pad_to_multiple_of_8_btchw(img2_bchw)
    img1p, img2p = of_transforms(img1_pad, img2_pad)
    out = of_model(img1p, img2p)
    flow = out[-1] if isinstance(out, (list, tuple)) else out
    flow = unpad_btchw(flow, pads)
    return flow  # [B,2,H,W]


@torch.no_grad()
def compute_raft_adjacent_flows(video_bfchw, of_model, of_transforms, need_backward=True):
    """
    video: [B,F,3,H,W], F>=2
    returns:
      fwd_px: [B,F-1,2,H,W] (t->t+1)
      bwd_px: [B,F-1,2,H,W] (t+1->t) or None
    """
    B, Fm, C, H, W = video_bfchw.shape
    if Fm < 2:
        raise ValueError(f"Need at least 2 frames, got {Fm}")

    fwd, bwd = [], [] if need_backward else None
    for t in range(Fm - 1):
        prev = video_bfchw[:, t]
        cur  = video_bfchw[:, t + 1]
        fwd.append(raft_flow_pair(of_model, of_transforms, prev, cur).unsqueeze(1))
        if need_backward:
            bwd.append(raft_flow_pair(of_model, of_transforms, cur, prev).unsqueeze(1))

    fwd_px = torch.cat(fwd, dim=1)  # [B,F-1,2,H,W]
    bwd_px = torch.cat(bwd, dim=1) if need_backward else None
    return fwd_px, bwd_px


# -------------------------
# 5) compose flows with fixed stride=4
# -------------------------
def compose_flows_forward_fixed_stride(flows_fwd_px: torch.Tensor, stride: int = 4):
    """
    flows_fwd_px: [B,T,2,H,W], T=F_px-1, require T%stride==0
    return: [B,K,2,H,W], K=T/stride = F_lat-1
    """
    B, T, _, H, W = flows_fwd_px.shape
    assert T % stride == 0, f"(F_px-1)={T} must be divisible by stride={stride}"
    K = T // stride
    out = []
    for k in range(K):
        base = k * stride
        Fk = torch.zeros((B, 2, H, W), device=flows_fwd_px.device, dtype=flows_fwd_px.dtype)
        for s in range(stride):
            f = flows_fwd_px[:, base + s]
            f_in_k = warp_bchw(f, Fk)
            Fk = Fk + f_in_k
        out.append(Fk.unsqueeze(1))
    return torch.cat(out, dim=1)  # [B,K,2,H,W]


def compose_flows_backward_fixed_stride(flows_bwd_px: torch.Tensor, stride: int = 4):
    """
    flows_bwd_px: [B,T,2,H,W], each (t+1->t), T=F_px-1, require T%stride==0
    return: [B,K,2,H,W], each (base+stride -> base)
    """
    B, T, _, H, W = flows_bwd_px.shape
    assert T % stride == 0
    K = T // stride
    out = []
    for k in range(K):
        base = k * stride
        Fk = torch.zeros((B, 2, H, W), device=flows_bwd_px.device, dtype=flows_bwd_px.dtype)
        for s in range(stride):
            f = flows_bwd_px[:, base + (stride - 1 - s)]
            f_in_k = warp_bchw(f, Fk)
            Fk = Fk + f_in_k
        out.append(Fk.unsqueeze(1))
    return torch.cat(out, dim=1)


def resize_and_scale_flow(flow_bf2hw: torch.Tensor, target_hw):
    B, Fm, _, H, W = flow_bf2hw.shape
    Hl, Wl = target_hw
    x = flow_bf2hw.reshape(B * Fm, 2, H, W)
    x = F.interpolate(x, size=(Hl, Wl), mode="bilinear", align_corners=False)
    x = x.reshape(B, Fm, 2, Hl, Wl)
    x[:, :, 0] *= (Wl / float(W))
    x[:, :, 1] *= (Hl / float(H))
    return x


# -------------------------
# 6) occlusion mask + proxy loss (dynamic latent F)
# -------------------------
def fb_occlusion_mask(flow_fwd_b2hw: torch.Tensor, flow_bwd_b2hw: torch.Tensor, thresh=1.0):
    b_warp = warp_bchw(flow_bwd_b2hw, flow_fwd_b2hw)
    fb = flow_fwd_b2hw + b_warp
    err = torch.sqrt((fb ** 2).sum(dim=1, keepdim=True) + 1e-6)
    return (err < thresh).float()


def latent_flow_proxy_loss(x0_pred_bcfhw: torch.Tensor,
                           flow_fwd_lat_bf2hw: torch.Tensor,
                           flow_bwd_lat_bf2hw: torch.Tensor,
                           occl_thresh: float = 1.0):
    """
    x0_pred: [B,C,F_lat,H,W]
    flow_* : [B,F_lat-1,2,H,W]
    """
    B, C, F_lat, H, W = x0_pred_bcfhw.shape
    K = F_lat - 1
    assert flow_fwd_lat_bf2hw.shape[1] == K
    assert flow_bwd_lat_bf2hw.shape[1] == K

    total = 0.0
    for t in range(K):
        z_t   = x0_pred_bcfhw[:, :, t]
        z_tp1 = x0_pred_bcfhw[:, :, t + 1]
        f = flow_fwd_lat_bf2hw[:, t]
        b = flow_bwd_lat_bf2hw[:, t]

        z_tp1_warp = warp_bchw(z_tp1, f)
        mask = fb_occlusion_mask(f, b, thresh=occl_thresh)

        diff = z_t - z_tp1_warp
        per = torch.sqrt(diff * diff + 1e-6)  # Charbonnier
        total = total + (per * mask).mean()

    return total / max(K, 1)


# -------------------------
# 7) wrapper class: fixed stride=4, dynamic frames
# -------------------------
class OnlineRAFTFlowProxy:
    """
    Fixed temporal compression ratio (stride=4), dynamic frame count.
    Requires:
      (F_px - 1) % 4 == 0
      F_lat == (F_px - 1) / 4 + 1  (enforced)
    """
    def __init__(self, device, short_side=320, stride=4):
        self.device = device
        self.short_side = short_side
        self.stride = stride

        weights = Raft_Large_Weights.DEFAULT
        self.of_transforms = weights.transforms()
        self.of_model = raft_large(weights=weights).to(device).eval()
        self.of_model.requires_grad_(False)

    @torch.no_grad()
    def compute_comp_flows_to_latent(self, pixel_values, F_lat: int, latent_hw):
        video = pixel_values_to_bfchw_01(pixel_values).to(self.device)  # [B,F_px,3,H,W]
        video = resize_short_side_bfchw(video, self.short_side)         # [B,F_px,3,h,w]
        B, F_px, _, h, w = video.shape

        # sanity check: temporal compression ratio is fixed
        T = F_px - 1
        assert T % self.stride == 0, f"(F_px-1)={T} must be divisible by stride={self.stride}"
        expected_F_lat = T // self.stride + 1
        assert F_lat == expected_F_lat, f"Temporal mismatch: pixel F={F_px} -> expected latent F={expected_F_lat}, but got F_lat={F_lat}"

        fwd_px, bwd_px = compute_raft_adjacent_flows(video, self.of_model, self.of_transforms, need_backward=True)
        fwd_comp = compose_flows_forward_fixed_stride(fwd_px, stride=self.stride)  # [B,F_lat-1,2,h,w]
        bwd_comp = compose_flows_backward_fixed_stride(bwd_px, stride=self.stride)

        fwd_lat = resize_and_scale_flow(fwd_comp, latent_hw)  # [B,F_lat-1,2,Hl,Wl]
        bwd_lat = resize_and_scale_flow(bwd_comp, latent_hw)
        return fwd_lat, bwd_lat

    def __call__(self, pixel_values, x0_pred_latent, occl_thresh=1.0):
        F_lat = x0_pred_latent.shape[2]
        Hl, Wl = x0_pred_latent.shape[-2], x0_pred_latent.shape[-1]
        with torch.no_grad():
            fwd_lat, bwd_lat = self.compute_comp_flows_to_latent(pixel_values, F_lat=F_lat, latent_hw=(Hl, Wl))
        return latent_flow_proxy_loss(x0_pred_latent, fwd_lat, bwd_lat, occl_thresh=occl_thresh)
