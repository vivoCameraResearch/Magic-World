
<div align=center>
<img src="asset/magicworld.png" width="300px">
</div>

<h2 align="center"> 
  <a href="https://arxiv.org/abs/2505.21325v2">
    MagicWorld: Towards Long-Horizon Stability for Interactive Video World Exploration
  </a>
</h2>

<h5 align="center">⭐ 100x faster than LingBot-World-Base, achieves better results under VBench on the RealWM120K-Val dataset. ⭐</h5>

<a href="https://arxiv.org/abs/2505.21325v2"><img src='https://img.shields.io/badge/arXiv-2501.11325-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'></a>&nbsp;
<a href="https://vivocameraresearch.github.io/magicworld/"><img src='https://img.shields.io/badge/Project-Page-Green' alt='Project'></a>&nbsp;
<a href="https://huggingface.co/LuckyLiGY/MagicWorld"><img src='https://img.shields.io/badge/🤗-HuggingFace-blue' alt='HuggingFace'></a>&nbsp;
<a href=""><img src='https://img.shields.io/badge/🤖-ModelScope-purple' alt='ModelScope'></a>&nbsp;
<a href="http://www.apache.org/licenses/LICENSE-2.0"><img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'></a>&nbsp;

This repository is the official implementation of our **MagicWorld**, an interactive video world model that supports exploring a dynamic scene created from a single scene
image through continuous keyboard actions (W, A, S, D), while maintaining structural and temporal consistency. 


## :bulb: Highlights 
- **Motion Drift Constrint**：We introduce a flow-guided motion preservation constrain that enforces temporal coherence in dynamic regions to prevent motion drift and ensure realistic motion evolution of dynamic subjects
- **Long-Horizon Stability**: We design a history cache retrieval strategy to preserve historical scene states during autoregressive rollout, and an enhanced interactive training strategy based on multi-shot aggregated DMD with dual-reward weighting, jointly improving long-horizon stability and reducing error accumulation.
- **RealWM120K Dataset**: We build the RealWM120K dataset with diverse citywalk videos and multimodal annotations for real-world video world modeling. 


## 📣 News 
- **`2026/03/20`**: We open-source the MagicWorld v1.5 codebase, including training and inference scripts.
- **`2026/02/10`**: We open-source the MagicWorld v1 codebase, including training and inference scripts.
- **`2025/11/24`**: Our [**Paper on ArXiv**](https://arxiv.org/abs/2511.18886) is available 🥳!


## ✅ To-Do List for MagicTryOn Release
- ✅ Release the source code of v1
- ✅ Release the source code of v1.5
- [  ] Update training training configuration and instructions
- [  ] Release the RealWM120K dataset and processing tools.
- [  ] Release the MagicWorld v1 pretrained weights
- [  ] Release the MagicWorld v1.5 pretrained weights

## :computer: Installation

Create a conda environment & Install requirments 
```shell
# python==3.12.9 cuda==12.3 torch==2.2
conda create -n magicworld python==3.12.9
conda activate magicworld
pip install -r requirements.txt
```
If you encounter an error while installing Flash Attention, please [**manually download**](https://github.com/Dao-AILab/flash-attention/releases) the installation package based on your Python version, CUDA version, and Torch version, and install it using `pip install flash_attn-2.7.3+cu12torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`.

## :package: Pretrained Model Weights
| Models           | Download |   Features |
|------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| MagicWorld-v1.5      | 🤗 [Huggingface](https://huggingface.co/LuckyLiGY/MagicWorld) 🤖 [ModelScope]()         | Full framework with geometry condition, motion constraint, history cache retrieval and multi-shot aggerated DMD.          |
| MagicWorld-v1.0      | 🤗 [Huggingface](https://huggingface.co/LuckyLiGY/MagicWorld) 🤖 [ModelScope]()         | Basic framework with geometry condition and history cache retrieval.                                                      |  
| MagicWorld-Base      | 🤗 [Huggingface](https://huggingface.co/LuckyLiGY/MagicWorld) 🤖 [ModelScope]()         | Basic framework.                                                      | 
 

## 😉 Demo Inference
Before inference, you need to do two things:
(1) install the [Uni3C](https://github.com/alibaba-damo-academy/Uni3C) library and its environment, then import the path to your installed Uni3C in uni3c_cam_render_api.py.
(2) run **action2traj.py** to map your keyboard actions to a camera trajectory and generate the trajectory .txt file.

```PowerShell
python inference/inference_magicworld_base.py
```
```PowerShell
python inference/inference_magicworld_v1.py
```

## 🚀 Training
We can choose whether to use deep speed in MagicWorld, which can save a lot of video memory.
The data format is shown as follows.
```json
[
    {
      "file_path": "train/00000001.mp4",
      "control_file_path": "camera/trajectory.txt",
      "point_video_path": "render/00000001_render.mp4"
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    .....
]
```
Some parameters in the sh file can be confusing, and they are explained in this document:

- `enable_bucket` is used to enable bucket training. When enabled, the model does not crop the images and videos at the center, but instead, it trains the entire images and videos after grouping them into buckets based on resolution.
- `random_frame_crop` is used for random cropping on video frames to simulate videos with different frame counts.
- `random_hw_adapt` is used to enable automatic height and width scaling for images and videos. When `random_hw_adapt` is enabled, the training images will have their height and width set to `image_sample_size` as the maximum and `min(video_sample_size, 512)` as the minimum. For training videos, the height and width will be set to `image_sample_size` as the maximum and `min(video_sample_size, 512)` as the minimum.
  - For example, when `random_hw_adapt` is enabled, with `video_sample_n_frames=49`, `video_sample_size=1024`, and `image_sample_size=1024`, the resolution of image inputs for training is `512x512` to `1024x1024`, and the resolution of video inputs for training is `512x512x49` to `1024x1024x49`.
  - For example, when `random_hw_adapt` is enabled, with `video_sample_n_frames=49`, `video_sample_size=1024`, and `image_sample_size=256`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49`.
- `training_with_video_token_length` specifies training the model according to token length. For training images and videos, the height and width will be set to `image_sample_size` as the maximum and `video_sample_size` as the minimum.
  - For example, when `training_with_video_token_length` is enabled, with `video_sample_n_frames=49`, `token_sample_size=1024`, `video_sample_size=1024`, and `image_sample_size=256`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49` to `1024x1024x49`.
  - For example, when `training_with_video_token_length` is enabled, with `video_sample_n_frames=49`, `token_sample_size=512`, `video_sample_size=1024`, and `image_sample_size=256`, the resolution of image inputs for training is `256x256` to `1024x1024`, and the resolution of video inputs for training is `256x256x49` to `1024x1024x9`.
  - The token length for a video with dimensions 512x512 and 49 frames is 13,312. We need to set the `token_sample_size = 512`.
    - At 512x512 resolution, the number of video frames is 49 (~= 512 * 512 * 49 / 512 / 512).
    - At 768x768 resolution, the number of video frames is 21 (~= 512 * 512 * 49 / 768 / 768).
    - At 1024x1024 resolution, the number of video frames is 9 (~= 512 * 512 * 49 / 1024 / 1024).
    - These resolutions combined with their corresponding lengths allow the model to generate videos of different sizes.
- `resume_from_checkpoint` is used to set the training should be resumed from a previous checkpoint. Use a path or `"latest"` to automatically select the last available checkpoint.

When train model with multi machines, please set the params as follows:
```sh
export MASTER_ADDR="your master address"
export MASTER_PORT=10086
export WORLD_SIZE=1 # The number of machines
export NUM_PROCESS=8 # The number of processes, such as WORLD_SIZE * 8
export RANK=0 # The rank of this machine

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK scripts/xxx.py
```
You can run the following command:
```PowerShell
bash train_magicworld_v1.sh
```

## 📕 RealWM120K Dataset


## :star: Acknowledgement
Our code is modified based on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main). We adopt [Wan2.1-I2V-14B](https://github.com/Wan-Video/Wan2.1) as the base model. We use [Uni3C](https://github.com/alibaba-damo-academy/Uni3C) to generate 3D points. The style of the logo is borrowed from [Helios](https://github.com/PKU-YuanGroup/Helios).
We thank [Siming Zheng](https://scholar.google.com/citations?user=zfYLLggAAAAJ&hl=zh-CN) and [Shuolin Xu](https://scholar.google.com/citations?user=Jr-Vn8AAAAAJ&hl=zh-CN) for their initial support and suggestions  for our basic framework. Thanks to all the contributors!

## :mag: Related Works
[Infinite-World](https://github.com/MeiGen-AI/Infinite-World)  
[Matrix-Game 2.0](https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2)  
[LingBot-World](https://github.com/robbyant/lingbot-world)  
[YUME 1.5](https://github.com/stdstu12/YUME)  
[Self-Forcing](https://github.com/guandeh17/Self-Forcing)  
[LongLive](https://github.com/NVlabs/LongLive)


## :scroll: License
All the materials, including code, checkpoints, and demo, are made available under the [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. You are free to copy, redistribute, remix, transform, and build upon the project for non-commercial purposes, as long as you give appropriate credit and distribute your contributions under the same license.

## :mortar_board: Citation

```bibtex
@article{li2025magicworld,
  title={MagicWorld: Towards Long-Horizon Stability for Interactive Video World Exploration},
  author={Li, Guangyuan and Li, Bo and Chen, Jinwei and Hu, Xiaobin and Zhao, Lei and Jiang, Peng-Tao},
  journal={arXiv preprint arXiv:2511.18886v2},
  year={2026}
}
```
