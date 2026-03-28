from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
import cv2
from torchvision import transforms
from typing import Dict, Any, Optional, List


def apply_transform(image: Image.Image) -> torch.Tensor:
    """
    Apply a series of transformations to an input image.
    Returns: torch.Tensor [C, H, W]
    """
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 按你的要求保持不变
    ])
    transformed_image = transform(image)
    return transformed_image

class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }

class WorldModelPairTensorDataset_NEW(Dataset):
    """
    从列表型 JSON 读取条目（不再区分 _0/_1，逐条使用）。
    每条 JSON 结构示例：
        {
            "file_path": "xxx/xxx.mp4",      # 视频路径
            "first_frame": "xxx/xxx.png",   # 对应首帧图像路径
            "text": "some prompt"           # 文本描述
        }

    返回：
        - rgb:   [C, T, H, W]，来自 file_path 视频
        - image: [C, H, W]，来自 first_frame 图像（注意已不再强制 [C, 1, H, W]）
        - image_path: 首帧图像的路径（便于调试/可视化）
        - prompts: 文本
        - idx: 样本索引
    """

    def __init__(self, index_json_path: str, root: Optional[str] = None, verbose: bool = True):
        self.index_json_path = index_json_path
        self.root = root

        # 读取 JSON
        with open(index_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items: List[Dict[str, Any]] = data if isinstance(data, list) else data.get("items", [])
        if not isinstance(items, list):
            raise ValueError(
                f"Unsupported JSON structure in {index_json_path}. "
                f"Expect list or {{'items': [...]}}."
            )

        # 现在不再按 _0/_1 做配对，直接全部保留
        self.items: List[Dict[str, Any]] = items

        if verbose:
            print(
                f"[Image2VideoPairTensorDataset] Loaded from {index_json_path}\n"
                f"  - entries: {len(self.items)}",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.items)

    def _join(self, p: str) -> str:
        """如果提供了 root 且 p 为相对路径，则拼接 root。"""
        if self.root and not os.path.isabs(p):
            return os.path.join(self.root, p)
        return p

    @staticmethod
    def _read_video_to_tensor(path: str) -> torch.Tensor:
        """
        读取视频为张量 [C, T, H, W]，逐帧用 apply_transform。
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

        frames: List[torch.Tensor] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            t = apply_transform(img)  # [C, H, W]
            frames.append(t)
        cap.release()

        if len(frames) == 0:
            # 空视频容错：构造一个 0 帧的张量
            return torch.empty(0)

        # 堆叠成 [T, C, H, W] 再转 [C, T, H, W]
        vid = torch.stack(frames, dim=0).permute(1, 0, 2, 3).contiguous()
        return vid

    @staticmethod
    def _read_image_to_tensor(path: str) -> torch.Tensor:
        """
        读取单张图像为张量 [C, H, W]，用 apply_transform。
        如需 [C, 1, H, W]，可以在这里加一行 t = t.unsqueeze(1)。
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        t = apply_transform(img)  # [C, H, W]
        # 如果你确实希望是 [C, 1, H, W]，取消下面这行注释：
        # t = t.unsqueeze(1)
        return t
    
    @staticmethod
    def _png_to_txt(p: str) -> str:
        base, _ = os.path.splitext(p)
        return base + ".txt"
    
    @staticmethod
    def _txt_to_png(p: str) -> str:
        base, _ = os.path.splitext(p)
        return base + ".png"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.items[idx]

        # 路径
        file_path = self._join(r["file_path"])        # 视频
        cam_path = self._join(r["control_file_path"])       # 首帧图像
        img_path = self._txt_to_png(cam_path)

        # 读取为张量
        rgb = self._read_video_to_tensor(file_path)   # [C, T, H, W]
        image = self._read_image_to_tensor(img_path)  # [C, H, W] 或 [C, 1, H, W]（取决于上面是否 unsqueeze）
        # print("aaaaaaaa", cam_path, img_path)

        return {
            "rgb": rgb,
            "image": image,
            "image_path": img_path,
            "camera_path": cam_path,
            "prompts": r["text"],
            "idx": idx,
        }

class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data
