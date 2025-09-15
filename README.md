# DA-Mamba

**DA-Mamba: Dialogue-Aware Selective State-Space Model for Multimodal Engagement Estimation**

---

## 简介

DA-Mamba 是一种面向对话的多模态模型，旨在进行**会话场景下的人类参与度（engagement）估计**。模型用选择性 **state-space models (SSMs)** 替代传统的二次复杂度自注意力模块，从而实现线性时间与内存复杂度，同时保持跨模态与跨参与者的表达能力。

核心模块：
- **Dialogue-Aware Encoder**
- **Modality-Group Fusion**（模态组融合）
- **Partner-Group Fusion**（伙伴组融合）
- 基于 Mamba 设计的局部 chunked attention + SSM 全局分支的混合块

主要贡献：
- 提出可扩展的选择性 SSM 主干（结合局部块注意力与全局 SSM），实现长程建模的线性复杂度；
- 通过模态组与伙伴组的分层建模实现高效的多方对话融合；
- 引入帧级对齐损失（对称 InfoNCE）与轻量跨模态 cross-attention，提高时序对齐与鲁棒性。

---

## 数据集

在论文中评估使用的基准数据集：
- **NoXi Base** — 150 个二人对话（英语/法语/德语），逐帧参与度标签（25 fps）。  
- **NoXi-Add** — 12 条测试对话（阿拉伯语、意大利语、印尼语、西班牙语）。  
- **MPIIGroupInteraction (MPIIGI)** — 12 个 4 人小组讨论（德语），逐帧参与度标签。  

> 数据集引用请参见论文中的 `\cite{...}`。

---

## 输入特征（示例）
每位参与者使用五类线对齐特征（插值到 100 Hz 并投影到相同维度）：
- **Ege** — 100 Hz wav2vec2.0（语音 embedding）
- **W2v** — 100 Hz 情感 token embedding（预训练 EGE 模型）
- **CLIP** — 25 Hz 视觉语言 embedding（面部 ROI）
- **OF** — 25 Hz 密集光流（面部）
- **OF2** — 25 Hz 光流幅值（面部关键点聚合）

将这些特征分组为 `audio-group`（Ege + W2v）和 `visual-group`（CLIP + OF + OF2），并分别输入模态专属 Mamba 堆栈处理。

---

## 训练设置（论文复现关键超参）
- 框架：PyTorch  
- 优化器：AdamW；初始 lr = 5e-5  
- Batch：128 windows（单 RTX 4090, 24GB 在论文中使用）  
- 窗口长度：96 帧（32 中心预测帧 + 两侧各 32 上下文）  
- 学习率调度：cosine annealing，500 warm-up steps  
- 梯度裁剪：max-norm 5.0；EMA decay = 0.999  
- 损失：$\mathcal{L} = \lambda_{\mathrm{CCC}}\mathcal{L}_{\mathrm{CCC}} + \lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}$，论文中 $\lambda_{\text{CCC}}=1.0$, $\lambda_{\text{align}}=0.4$。

---

## 定量结果（论文中的表格）

### 主结果（Test-set CCC）
| Method | NoXi Base | NoXi-Add | MPIIGI | Global |
|---|---:|---:|---:|---:|
| MM24 Baseline \cite{Muller2024_MultiMediate} | 0.64 | 0.51 | 0.09 | 0.41 |
| YLYJ \cite{Muller2024_MultiMediate} | 0.60 | 0.52 | 0.30 | 0.47 |
| nox \cite{Muller2024_MultiMediate} | 0.68 | 0.70 | 0.31 | 0.56 |
| SP-team \cite{Muller2024_MultiMediate} | 0.68 | 0.65 | 0.34 | 0.56 |
| YKK \cite{Muller2024_MultiMediate} | 0.68 | 0.66 | 0.40 | 0.58 |
| Xpace \cite{Muller2024_MultiMediate} | 0.70 | 0.70 | 0.34 | 0.58 |
| ashk \cite{Muller2024_MultiMediate} | 0.72 | 0.69 | 0.42 | 0.61 |
| Kumar et al. \cite{Kumar2024_Engagement} | 0.72 | 0.69 | 0.50 | 0.64 |
| DAT \cite{Li2024_DAT} | 0.76 | 0.67 | 0.49 | 0.64 |
| AI-lab \cite{Muller2024_MultiMediate} | 0.69 | 0.72 | 0.54 | 0.65 |
| **DA-Mamba (Ours)** | **0.77** | **0.70** | **0.52** | **0.66** |

> **说明**：Global 为三数据集的宏平均（macro-average）。

---

### 内存与效率分析（Peak memory vs. sequence length）
| Sequence Length | 64 | 96 | 128 | 192 |
|---:|---:|---:|---:|---:|
| DAT Memory (GB) | 8.2 | 18.4 | OOM | OOM |
| DA-Mamba Memory (GB) | 6.1 | 11.2 | 15.8 | 22.1 |
| Memory Reduction | 26% ↓ | 39% ↓ | 100% ↓ | 100% ↓ |
| DAT Params (M) | 42.5 | 42.5 | 42.5 | 42.5 |
| DA-Mamba Params (M) | 28.7 | 28.7 | 28.7 | 28.7 |

> 注：表中 “OOM” 表示在该序列长度下 DAT 出现显存溢出；Memory Reduction 行反映相对 DAT 的峰值内存下降（论文原表格展示方式）。

---

### 消融实验（Ablation）
| Mamba | Trans. | Mod-F | Part-F | NoXi B | NoXi A | MPIIGI | Global |
|---:|:---:|:---:|:---:|---:|---:|---:|---:|
| ✓ | × | ✓ | ✓ | 0.77 | 0.70 | 0.52 | 0.66 |
| ✓ | × | × | ✓ | 0.75 | 0.66 | 0.48 | 0.63 |
| ✓ | × | ✓ | × | 0.74 | 0.66 | 0.51 | 0.64 |
| ✓ | × | × | × | 0.68 | 0.65 | 0.45 | 0.59 |
| × | ✓ | ✓ | ✓ | 0.75 | 0.64 | 0.49 | 0.63 |

**注释**：`Trans.` 表示是否使用 Transformer；`Mod-F` = Modality-group Fusion；`Part-F` = Partner-group Fusion。表格展示组件对全局 CCC 的贡献。

---




## 引用
如果你使用了 DA-Mamba，请引用论文（参考 bib 条目写入 `refs.bib`）：

