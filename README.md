
<h2 align="center"> 
  <a href="https://arxiv.org/abs/2505.21325v2">
    MagicWorld: Interactive Geometry-driven Video World Exploration
  </a>
</h2>

<a href="https://arxiv.org/abs/2511.18886"><img src='https://img.shields.io/badge/arXiv-2501.11325-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'></a>&nbsp;
<a href="https://vivocameraresearch.github.io/magicworld/"><img src='https://img.shields.io/badge/Project-Page-Green' alt='GitHub'></a>&nbsp;
<a href="http://www.apache.org/licenses/LICENSE-2.0"><img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'></a>&nbsp;


We introduce **MagicWorld**, an interactive video world model that supports exploring a dynamic scene created from a single scene
image through continuous keyboard actions, while maintaining structural and temporal consistency. MagicWorld generates action-driven
point clouds from user inputs (W, A, S, D) to provide geometric constraints for stable viewpoint transitions.
![method](asset/magicworld.png)

## 📣 News 
- **`2026/02/10`**: We open-source the MagicWorld v1 codebase, including training and inference scripts.
- **`2025/11/24`**: Our [**Paper on ArXiv**](https://arxiv.org/abs/2511.18886) is available 🥳!

## ✅ To-Do List for MagicTryOn Release
- ✅ Release the source code
- [  ] Release the MagicWorld v1 pretrained weights
- [  ] Release the MagicWorld v2

## 😍 Installation

Create a conda environment & Install requirments 
```shell
# python==3.12.9 cuda==12.3 torch==2.2
conda create -n magicworld python==3.12.9
conda activate magicworld
pip install -r requirements.txt
```
If you encounter an error while installing Flash Attention, please [**manually download**](https://github.com/Dao-AILab/flash-attention/releases) the installation package based on your Python version, CUDA version, and Torch version, and install it using `pip install flash_attn-2.7.3+cu12torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`.


## 😉 Demo Inference
Before inference, you need to do two things:
(1) install the [Uni3C](https://github.com/alibaba-damo-academy/Uni3C) library and its environment, then import the path to your installed Uni3C in uni3c_cam_render_api.py.
(2) run **action2traj.py** to map your keyboard actions to a camera trajectory and generate the trajectory .txt file.

```PowerShell
python inference/interactive_magicworld_v1.py
```


## 😘 Acknowledgement
Our code is modified based on [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main). We adopt [Wan2.1-I2V-14B](https://github.com/Wan-Video/Wan2.1) as the base model. We use [Uni3C](https://github.com/alibaba-damo-academy/Uni3C) to generate 3D points. Thanks to all the contributors!

## 😊 License
All the materials, including code, checkpoints, and demo, are made available under the [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. You are free to copy, redistribute, remix, transform, and build upon the project for non-commercial purposes, as long as you give appropriate credit and distribute your contributions under the same license.

## 🤩 Citation

```bibtex
@article{li2025magicworld,
  title={Magicworld: Interactive geometry-driven video world exploration},
  author={Li, Guangyuan and Zheng, Siming and Xu, Shuolin and Chen, Jinwei and Li, Bo and Hu, Xiaobin and Zhao, Lei and Jiang, Peng-Tao},
  journal={arXiv preprint arXiv:2511.18886},
  year={2025}
}
```
