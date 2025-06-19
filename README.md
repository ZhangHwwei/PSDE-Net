# PSDE-Net: Dual Encoder Network with Parallel Strip Convolution for Road Extraction from Remote Sensing Images

![License](https://img.shields.io/badge/license-MIT-green)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-blue)

**PSDE-Net** PSDE-Net is a deep learning network designed for road extraction from remote sensing images. It addresses the challenges of accurately extracting road features, especially slender and winding roads, and ensuring road connectivity. The method incorporates a dual encoder structure and parallel strip convolutions, significantly improving road feature extraction and connectivity.<br>
**The corresponding paper is currently under review for publication in the journal _Computers and Geosciences_.**

## 🔍 Overview

PSDE-Net is designed to enhance road extraction tasks, especially in the presence of complex, winding, and disconnected road networks. The model leverages two key innovations:
- __Dual Encoder Network:__ Integrates a ResNet-34 branch and a parallel strip convolution branch to extract multi-scale road features and capture long-range contextual information.
- __Road Connectivity Module (RCM):__ Utilizes graph convolution to improve the connectivity of extracted roads by inferring the spatial distribution of roads.

## 🚀 Key Features

- 📏 **Strip Convolution Modules**: Extract directional and linear road features
- 🐍 **Dynamic Snake Convolutions**: Capture winding, narrow roads
- 🧠 **Graph Convolution Network**: Strengthen road connectivity

## 🛠️ Installation

```bash
git clone https://github.com/ZhangHwwei/PSDE-Net.git
cd PSDE-Net

# (Optional) Create a conda virtual environment(Python 3.8 or later)
conda create -n psdenet python=3.8 -y
conda activate psdenet

# Install dependencies
pip install -r requirements.txt
```
## 📁 Dataset
We use two publicly available datasets to evaluate the performance of PSDE-Net:
### 1. [DeepGlobe Road Extraction Dataset](http://deepglobe.org/challenge.html)
### 2. [CHN6-CUG Road Dataset](https://grzy.cug.edu.cn/zhuqiqi/zh_CN/yjgk/32368/list/index.htm)
> Please ensure your dataset is organized in the following directory structure:
