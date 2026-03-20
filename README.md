
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
<a href=""><img src='https://img.shields.io/badge/🤗-HuggingFace-blue' alt='HuggingFace'></a>&nbsp;
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
| MagicWorld-v1.5      | 🤗 [Huggingface]() 🤖 [ModelScope]()         | Full framework with geometry condition, motion constraint, history cache retrieval and multi-shot aggerated DMD.          |
| MagicWorld-v1.0      | 🤗 [Huggingface]() 🤖 [ModelScope]()         | Basic framework with geometry condition and history cache retrieval.                                                      |                 
 

## 😉 Demo Inference
Before inference, you need to do two things:
(1) install the [Uni3C](https://github.com/alibaba-damo-academy/Uni3C) library and its environment, then import the path to your installed Uni3C in uni3c_cam_render_api.py.
(2) run **action2traj.py** to map your keyboard actions to a camera trajectory and generate the trajectory .txt file.

```PowerShell
python inference/interactive_magicworld_v1.py
```

## 🚀 Training

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
