# 🔍 Factorized Vision Transformer with ProbSparse Attention (FaViT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**From Pixels to Predictions: A Detective Story with Math**

A novel implementation of **Factorized Vision Transformers (FaViT)** that integrates **ProbSparse self-attention** mechanisms to improve computational efficiency. This project bridges the gap between the global context of Transformers and the efficiency of CNNs, achieving competitive accuracy on CIFAR-10 with significantly reduced computational cost.

---

## 🚀 Performance Benchmarks

| Model Variant | Dataset | Accuracy | Epochs | Status |
| :--- | :--- | :--- | :--- | :--- |
| **FaViT-B1** | CIFAR-10 | **86.16%** | 30 | ✅ Tested |
| **FaViT-B3** | CIFAR-10 | **96.00%+** | 50 | 🔮 Projected |

> **Note:** The B1 model achieves >86% accuracy in just 30 epochs from scratch (no pre-training), outperforming standard ViT baselines which typically lag at ~70-80% in early training stages without massive datasets.

---

## 🎬 The Complete Image Journey

### ACT 1: Breaking Down the Image (Patch Embedding)
**The Setup:** A Cat Photo Arrives 🐱
*   **Input:** 224×224 pixels (3 channels)
*   **Total Data:** 150,528 numbers

#### 🔪 Step 1: Cutting the Image
We use a sliding window approach to create overlapping tiles.
*   **Patch Size:** 7×7
*   **Stride:** 4 (Overlapping!)
*   **Result:** 56×56 = **3,136 patches**

#### 🎨 Step 2: Feature Transformation
Each raw tile (7×7×3 = 147 pixels) is projected into a **64-dimensional feature vector**.
> **Analogy:** Instead of storing every brick of a house (pixels), we store a summary: "Red roof, two windows, brick texture."

---

### 🎯 ACT 2: The Attention Mechanism (The Detective Work)
Now the magic happens. Patches need to talk to each other to understand the context. We split the work into three specialized teams:

1.  🌍 **Global Team (ProbSparse):** 8 channels
2.  🏘️ **Local Team (Window):** 28 channels
3.  🔍 **Dilated Team:** 28 channels

#### 🌍 PART 1: Global Attention with ProbSparse
**The Problem:** Comparing every patch to every other patch (3,136²) is too expensive (~9.8M comparisons).
**The ProbSparse Solution:** Smart Sampling.

1.  **The Sampling:** Each patch randomly picks 25 "districts" (key/value pairs) to investigate.
2.  **The Sparsity Score (M):** We measure how "focused" the attention is.
    *   *High M:* The patch found something specific/important. **KEEP IT!**
    *   *Low M:* The patch sees everything as similar. **DROP IT!**
3.  **The Selection:** We keep only the **Top-40 most focused patches** to perform full attention.

> **Result:** 98% memory reduction with minimal accuracy loss. We don't need to check every blade of grass to know it's a lawn!

#### 🏘️ PART 2: Local Window Attention
After global context, we look at the neighbors.
*   **The Grid:** Image divided into 64 windows (8×8 grid).
*   **The Task:** Each patch only talks to its 49 immediate neighbors.
*   **The Goal:** Capture textures, edges, and local patterns (like CNNs do).

---

### 🔗 ACT 3: The Grand Finale
We combine the insights from all three teams:
1.  **Global:** "Where is the cat in the image?"
2.  **Local:** "What is the texture of the fur?"
3.  **Dilated:** "What are the medium-scale patterns?"

All features are concatenated and passed through a final MLP to produce the classification: **"It's a cat!"** 🐱

---

## 🛠️ Installation & Usage

### Prerequisites
*   Python 3.8+
*   PyTorch 2.0+
*   CUDA 11.8+ (Recommended for training)

### Quick Start
1. Clone the repository
git clone https://github.com/AmrutiyaKirtan/Factorized-vit-Probsparsing.git
cd Factorized-vit-Probsparsing

2. Install dependencies
pip install torch torchvision numpy matplotlib

3. Train the model
python New_WN_PY.py
