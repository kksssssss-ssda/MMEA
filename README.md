# DA-Mamba

**DA-Mamba: Dialogue-Aware Selective State-Space Model for Multimodal Engagement Estimation**

---

## Overview

DA-Mamba is a dialogue-aware multimodal model for **engagement estimation in conversational scenarios**. The model replaces quadratic-complexity self-attention modules with selective **state-space models (SSMs)** to achieve linear time and memory complexity while preserving cross-modal and cross-participant representational power.

**Core components**

* Dialogue-Aware Encoder
* Modality-Group Fusion
* Partner-Group Fusion
* Hybrid Mamba blocks (local chunked attention + global SSM branch)

**Key contributions**

* A scalable selective-SSM backbone combining local chunked attention with global SSMs for long-range modeling at linear complexity.
* Hierarchical fusion via modality-group and partner-group modeling for efficient multi-party dialogue integration.
* Frame-wise alignment supervision (symmetric InfoNCE) and lightweight cross-attention to improve temporal alignment and robustness.

---

## Datasets

Benchmarks used in the paper:

* **NoXi Base** — 150 dyadic conversations (English/French/German) with framewise engagement annotations (25 fps).
* **NoXi-Add** — 12 test-only dialogues (Arabic, Italian, Indonesian, Spanish).
* **MPIIGroupInteraction (MPIIGI)** — 12 four-person group discussions (German) with frame-level engagement labels.

(See the paper for dataset citations.)

---

## Input features (example)

Each participant uses five frame-aligned cues (interpolated to 100 Hz and projected to a common dimensionality):

* **Ege** — 100 Hz wav2vec2.0 speech embeddings
* **W2v** — 100 Hz emotion token embeddings (pretrained EGE model)
* **CLIP** — 25 Hz vision–language embeddings from face ROI
* **OF** — 25 Hz dense optical flow maps (face)
* **OF2** — 25 Hz optical-flow magnitude aggregated over facial landmarks

Features are grouped into `audio-group` (Ege + W2v) and `visual-group` (CLIP + OF + OF2), then processed by modality-specific Mamba stacks.

---

## Training setup (key hyperparameters from the paper)

* Framework: PyTorch
* Optimizer: AdamW, initial lr = 5e-5
* Batch: 128 windows (experiments reported on a single RTX 4090)
* Window length: 96 frames (32 central prediction frames + 32 context frames on each side)
* LR schedule: cosine annealing with 500 warm-up steps
* Gradient clipping: max-norm = 5.0; EMA decay = 0.999
* Loss: \$\mathcal{L} = \lambda\_{\mathrm{CCC}}\mathcal{L}*{\mathrm{CCC}} + \lambda*{\mathrm{align}}\mathcal{L}*{\mathrm{align}}\$ with \$\lambda*{\text{CCC}}=1.0\$, \$\lambda\_{\text{align}}=0.4\$

---

## Quantitative results

### Main results (Test-set CCC)

| Method              | NoXi Base | NoXi-Add |   MPIIGI |   Global |
| ------------------- | --------: | -------: | -------: | -------: |
| MM24 Baseline       |      0.64 |     0.51 |     0.09 |     0.41 |
| YLYJ                |      0.60 |     0.52 |     0.30 |     0.47 |
| nox                 |      0.68 |     0.70 |     0.31 |     0.56 |
| SP-team             |      0.68 |     0.65 |     0.34 |     0.56 |
| YKK                 |      0.68 |     0.66 |     0.40 |     0.58 |
| Xpace               |      0.70 |     0.70 |     0.34 |     0.58 |
| ashk                |      0.72 |     0.69 |     0.42 |     0.61 |
| Kumar et al.        |      0.72 |     0.69 |     0.50 |     0.64 |
| DAT                 |      0.76 |     0.67 |     0.49 |     0.64 |
| AI-lab              |      0.69 |     0.72 |     0.54 |     0.65 |
| **DA-Mamba (Ours)** |  **0.77** | **0.70** | **0.52** | **0.66** |

> *Global is the macro-average across the three corpora.*

---

### Memory & efficiency analysis (Peak memory vs. sequence length)

|      Sequence Length |    64 |    96 |    128 |    192 |
| -------------------: | ----: | ----: | -----: | -----: |
|      DAT Memory (GB) |   8.2 |  18.4 |    OOM |    OOM |
| DA-Mamba Memory (GB) |   6.1 |  11.2 |   15.8 |   22.1 |
|     Memory Reduction | 26% ↓ | 39% ↓ | 100% ↓ | 100% ↓ |
|       DAT Params (M) |  42.5 |  42.5 |   42.5 |   42.5 |
|  DA-Mamba Params (M) |  28.7 |  28.7 |   28.7 |   28.7 |

> “OOM” indicates out-of-memory for DAT at that sequence length. Memory Reduction reports relative peak memory decrease versus DAT as shown in the paper.

---

### Ablation study

| Mamba | Trans. | Mod-F | Part-F | NoXi B | NoXi A | MPIIGI | Global |
| ----: | :----: | :---: | :----: | -----: | -----: | -----: | -----: |
|     ✓ |    ×   |   ✓   |    ✓   |   0.77 |   0.70 |   0.52 |   0.66 |
|     ✓ |    ×   |   ×   |    ✓   |   0.75 |   0.66 |   0.48 |   0.63 |
|     ✓ |    ×   |   ✓   |    ×   |   0.74 |   0.66 |   0.51 |   0.64 |
|     ✓ |    ×   |   ×   |    ×   |   0.68 |   0.65 |   0.45 |   0.59 |
|     × |    ✓   |   ✓   |    ✓   |   0.75 |   0.64 |   0.49 |   0.63 |

**Notes:** `Trans.` indicates use of Transformer; `Mod-F` = Modality-group Fusion; `Part-F` = Partner-group Fusion. The table shows each component's contribution to global CCC.

---

## References (example bib entry)

If citing the paper, use an appropriate bib entry (add the actual bib data to your bibliography file):

```bibtex
@inproceedings{your2025damamba,
  title = {DA-Mamba: Dialogue-Aware Selective State-Space Model for Multimodal Engagement Estimation},
  author = {Shenwei Kang and Xin Zhang and Wen Liu and Bin Li and Yujie Liu and Bo Gao},
  year = {2025},
  note = {arXiv / conference TBD}
}
```

---

*If you would like a shorter one-page summary or a different formatting, tell me and I will produce it.*
