# PlantCCC: A Spatial-aware Graph Deep Learning Framework for Plant Cell-Cell Communication

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20DGL-red)](https://pytorch.org/)

## 📖 简介 (Introduction)

**PlantCCC** 是一个融合空间转录组信息和图深度学习（Graph Deep Learning）的植物细胞通讯分析工具。

传统的植物细胞通讯推断方法主要依赖配体-受体共表达，忽略了细胞间的**空间邻近关系 (Spatial Proximity)**。PlantCCC 通过构建空间邻接图，结合 **多头图注意力网络 (GAT)** 和 **深度图信息最大化 (DGI)** 策略，能够从噪声充斥的候选集中识别出具有显著空间共定位趋势的高置信度通讯对。

主要特性：
- **空间感知**：将空间位置信息显式编码进细胞通讯推断。
- **降噪增强**：针对植物空间转录组数据的定制化表达增强。
- **高可信度**：利用 Attention Score 作为互作可信度指标。
- **动态分析**：支持发育时序数据的通讯动态重编程分析。

## 🛠️ 环境依赖 (Requirements)

建议使用 Anaconda 创建环境：

```bash
conda create -n plantccc python=3.9
conda activate plantccc
pip install -r requirements.txt
