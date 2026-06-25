# SFVNet: A Physics-Guided Spatial-Frequency Two-Stage Network for Video Snow Removal

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the official PyTorch implementation for the paper: **"SFVNet: A Physics-Guided Spatial-Frequency Two-Stage Network for Video Snow Removal"**.

---

## 🌟 Introduction
`SFVNet` introduces a novel **two-stage collaborative paradigm** that integrates physical imaging priors with deep learning components in both spatial and frequency domains. 

Instead of treating frames in isolation, `SFVNet` utilizes a **sliding-window mechanism** (default: 5 frames) to continuously process spatiotemporal contexts, effectively distinguishing high-speed drifting snow flakes from static backgrounds.

```
---

## 🛠️ Installation

To replicate the experimental environment:

```bash
# Install core dependencies via pip
pip install -r requirements.txt
```

## 📂 Dataset Preparation

Before training or testing, please arrange your benchmarks (e.g., KITTI-Snow or RVSD) according to the expected sliding-window data layout:

Plaintext

```
data/
├── train/
│   ├── snow/     # Degraded video sequence directories (00000/, 00001/, ...)
│   └── clean/    # Ground-truth video sequence directories
└── val/
    ├── snow/
    └── clean/
```

## 🚀 Training

We provide specialized training scripts tailored for different video datasets. Please specify your data path and parameters inside the command arguments or configuration logic.

### 1. Train on KITTI Dataset

Bash

```
python train_kitti.py --data_root ./data/kitti_snow --batch_size 4 --epochs 120
```

### 2. Train on RVSD (Realistic Video Snow Dataset)

Bash

```
python train_rvsd.py --data_root ./data/rvsd_snow --batch_size 4 --epochs 120
```

## 🔬 Testing & Evaluation

You can evaluate the models using the following testing protocols depending on the target dataset and validation scale.

### 1. Evaluate on KITTI Evaluation Benchmarks

Bash

```
python test_kitti.py --data_root ./data/kitti_snow --weights ./checkpoints/sfvnet_kitti_best.pth
```

### 2. Evaluate on RVSD Evaluation Benchmarks

To run inference on raw video frames without center or random cropping, execute the dedicated full-size data loader pipeline:

Bash

```
python test_rvsd.py --data_root ./data/rvsd_snow --weights ./checkpoints/sfvnet_rvsd_best.pth
```

## 📥 数据集下载链接 (Download Links)

为了方便国内研究人员和复现代码，数据集均已上传至百度网盘：

| 数据集名称 (Dataset) | 网盘下载链接 (Baidu Wangpan)                                 | 提取码 (Code) | 说明 (Note)                              |
| :------------------- | :----------------------------------------------------------- | :------------ | :--------------------------------------- |
| **KITTI_snow**       | [百度网盘下载直达](https://pan.baidu.com/s/1wXVgr13j0r8STQMsTAAqgw) | **8ayh**      | 适配自动驾驶场景的合成视频去雪数据集     |
| **RVSD_snow**        | [百度网盘下载直达](https://pan.baidu.com/s/1Y-sI3ZA_DpCUFrEB4ZwZeg) | **ypkp**      | 包含丰富降雪动态时空红利的视频去雪数据集 |



