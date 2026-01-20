# Technical Specification: LLM-Based Channel Prediction Architecture

## 1. Overview

This document details the mathematical architecture of the **"LLM for Channel Prediction" (LLM4CP)** framework. The system treats Channel State Information (CSI) prediction as a **sequence modeling task**, leveraging the self-attention mechanism of Large Language Models to capture long-range dependencies in both time and frequency domains.

**Objective:** Map historical Uplink CSI sequences to future Downlink CSI sequences.

$$ f_{\Omega}: \mathcal{H}_{in} \rightarrow \mathcal{H}_{out} $$

**Key Innovation:** The implementation features a **dual-domain processing architecture** that operates in both frequency and delay domains, combined with a pre-trained GPT-2 backbone for transfer learning.

---

## 2. Module-by-Module Mathematical Design

### Module A: Data Preprocessing & Input Representation

#### A.1 Raw Data Format (from .mat file)

The original CSI data stored in `.mat` files has the following physical structure:

**Original tensor shape:** `[v, n, L, K, a, b, c]` = `[900, 10, 16, 48, 4, 4, 2]`

| Dimension | Symbol | Size | Physical Meaning |
|-----------|--------|------|------------------|
| Speed points | `v` | 900 | 10.1:0.1:100 km/h (velocity range) |
| Samples per speed | `n` | 10 | Number of samples at each speed point |
| Historical time slots | `L` | 16 | Past CSI observations (input sequence) |
| Subcarriers | `K` | 48 | OFDM frequency domain resources |
| Antenna rows | `a` | 4 | Vertical dimension of UPA (antenna panel) |
| Antenna columns | `b` | 4 | Horizontal dimension of UPA (antenna panel) |
| **Polarizations** | `c` | **2** | **Dual-polarization (+45° / -45°)** |

**Key Physical Interpretation:**

The antenna array is a **UPA (Uniform Planar Array)** with dual-polarized elements:

$$ N_{antennas} = a \times b \times c = 4 \times 4 \times 2 = 32 \text{ spatial channels} $$

- **Dimensions `a` and `b`**: Physical antenna positions (4×4 grid)
- **Dimension `c=2`**: Dual-polarization - two orthogonal antenna elements at each physical position
  - Index 0: +45° polarized signal (complex-valued)
  - Index 1: -45° polarized signal (complex-valued)

**Data Type:** The original data is stored as `complex128` (each value is already a complex number with real and imaginary parts).

#### A.2 Data Preprocessing Pipeline

##### Step 1: Rearrange (Flatten Spatial Dimensions)

```python
H = rearrange(H, 'v n L k a b c -> (v n) L (k a b c)')
```

**Input:** `[900, 10, 16, 48, 4, 4, 2]` (complex128)
**Output:** `[9000, 16, 1536]` (complex128)

The spatial dimensions are flattened: `K × a × b × c = 48 × 4 × 4 × 2 = 1536`

##### Step 2: LoadBatch_ofdm (Spatial Expansion)

```python
H = rearrange(H, 'b t (k a) -> (b a) t k', a=32)
```

**Input:** `[9000, 16, 1536]` (complex)
**Output:** `[288000, 16, 48]` (complex)

This operation decomposes `1536 = 48 × 32` and expands the batch dimension:
- `mul = 1536` is split into `K=48` (subcarriers) and `spatial=32` (antennas)
- Batch size multiplies: `9000 × 32 = 288000` samples

##### Step 3: Complex to Real Conversion

```python
H_real = np.zeros([B * 32, T, 48, 2])
H_real[:, :, :, 0] = H.real  # Real part of complex signal
H_real[:, :, :, 1] = H.imag  # Imaginary part of complex signal
H_real = H_real.reshape([B * 32, T, 48 * 2])
```

**Input:** `[288000, 16, 48]` (complex128)
**Output:** `[288000, 16, 96]` (float32)

The complex-valued signal is decomposed into real and imaginary parts for neural network processing.

#### A.3 Model Input Format

*   **Model Input:** $\mathbf{X} \in \mathbb{R}^{B \times L \times D_{in}}$
    *   $B$: Batch size (e.g., 1024)
    *   $L$: Sequence length (historical time slots, default: 16)
    *   $D_{in} = 2 \times K$: Input feature dimension = `2 × 48 = 96`
        *   $K = 48$: Number of subcarriers (after LoadBatch_ofdm spatial expansion)
        *   Factor of 2: **Real and Imaginary parts** of the complex CSI signal

#### A.4 Normalization

**Implementation:** Per-batch normalization (dynamic, not pre-computed)

$$ \mathbf{X}_{norm} = \frac{\mathbf{X}_{raw} - \mu_{batch}}{\sigma_{batch}} $$

Where $\mu_{batch}$ and $\sigma_{batch}$ are computed dynamically for each batch:

```python
mean = torch.mean(x_enc)  # Scalar mean over entire batch
std = torch.std(x_enc)    # Scalar std over entire batch
x_enc = (x_enc - mean) / std
```

**Code Location:** [GPT4CP.py:130-132](../models/GPT4CP.py#L130-L132)

---

### Module B: Dual-Domain Feature Extraction

**NOVEL ARCHITECTURAL COMPONENT** - This is a key enhancement not found in standard ViT-based approaches.

#### B.1 Frequency Domain Processing Path

The frequency domain path processes the normalized CSI directly:

$$ \mathbf{X}^{(fre)} = \text{PatchProcess}(\mathbf{X}_{norm}) $$

**Steps:**

1. **Temporal Patching:** Reshape sequence into patches of size `patch_size` (default: 4)
   $$ \mathbf{X}^{(fre)} \in \mathbb{R}^{B \times \frac{L}{p} \times p \times D_{in}} $$
   where $p = \text{patch_size}$

2. **Patch Linear Projection:**
   $$ \mathbf{X}^{(fre)} = \mathbf{W}_{patch} \mathbf{X}^{(fre)} $$
   Applies a linear transformation to each temporal patch.

3. **Residual CNN Processing (RB_e):**
   $$ \mathbf{X}^{(fre)}_{out} = \text{ResBlock}(\text{reshape}_{2D}(\mathbf{X}^{(fre)})) $$

**Code Location:** [GPT4CP.py:145-149](../models/GPT4CP.py#L145-L149)

#### B.2 Delay Domain Processing Path

The delay domain leverages the frequency-delay duality in wireless communications:

1. **Complex Reconstruction:**
   $$ \mathbf{H}_{complex} = \mathbf{X}_{norm}^{real} + j \cdot \mathbf{X}_{norm}^{imag} $$

2. **Inverse FFT (Frequency → Delay):**
   $$ \mathbf{H}_{delay} = \text{IFFF}(\mathbf{H}_{complex}, \text{dim}=k) $$

3. **Real-Imaginary Decomposition:**
   $$ \mathbf{X}_{delay} = [\Re(\mathbf{H}_{delay}), \Im(\mathbf{H}_{delay})] $$

4. **Same Patching + Residual CNN (RB_f):**
   $$ \mathbf{X}^{(delay)}_{out} = \text{ResBlock}(\text{reshape}_{2D}(\mathbf{X}_{delay})) $$

**Code Location:** [GPT4CP.py:135-143](../models/GPT4CP.py#L135-L143)

#### B.3 Domain Fusion

$$ \mathbf{X}_{fused} = \mathbf{X}^{(fre)}_{out} + \mathbf{X}^{(delay)}_{out} $$

**Code Location:** [GPT4CP.py:151-152](../models/GPT4CP.py#L151-L152)

---

### Module B+: Residual CNN Blocks (Channel Attention)

Before tokenization, the model applies residual convolutional blocks with channel attention.

#### B+.1 Residual Block Architecture

Each residual block consists of:

1. **Two 3×3 Convolutional Layers:**
   $$ \mathbf{F}_1 = \text{ReLU}(\text{Conv2D}_1(\mathbf{X})) $$
   $$ \mathbf{F}_2 = \text{Conv2D}_2(\mathbf{F}_1) $$

2. **Channel Attention Module:**
   $$ \mathbf{A}_{ch} = \sigma(\text{MLP}_{avg}(\mathbf{F}_2) + \text{MLP}_{max}(\mathbf{F}_2)) $$
   where:
   - $\text{MLP}_{avg}$: Global average pooling → FC → ReLU → FC
   - $\text{MLP}_{max}$: Global max pooling → FC → ReLU → FC
   - $\sigma$: Sigmoid activation

3. **Attention-weighted Feature + Residual:**
   $$ \mathbf{Y} = \mathbf{X} + \mathbf{A}_{ch} \odot \mathbf{F}_2 $$

**Architecture Parameters:**
- Input/Output channels: `res_dim` (default: 64)
- Number of blocks: `res_layers` (default: 4)
- Attention reduction ratio: 1 (no reduction)

**Code Location:** [GPT4CP.py:16-50](../models/GPT4CP.py#L16-L50), [GPT4CP.py:121-127](../models/GPT4CP.py#L121-L127)

---

### Module C: Token Embedding (Convolution-based)

**DIFFERENCE FROM SPEC:** Uses 1D Convolution with circular padding instead of simple Linear projection.

#### C.1 Token Embedding via Conv1D

$$ \mathbf{E}_{token} = \text{Conv1D}_{circular}(\mathbf{X}_{fused}) + \mathbf{E}_{pos} $$

**Convolution Parameters:**
- Kernel size: 3
- Padding: Circular (periodic boundary condition)
- Stride: 1
- Activation: None (linear)

**Circular Padding Rationale:** Treats the frequency domain as periodic, which is physically meaningful for OFDM subcarriers.

**Initialization:** Kaiming Normal (He initialization) for LeakyReLU

$$ \mathbf{W} \sim \mathcal{N}(0, \sqrt{\frac{2}{(1 + a^2) \cdot n_{in}}}) $$

where $a$ is the negative slope of LeakyReLU.

**Code Location:** [Embed.py:30-45](../Embed.py#L30-L45)

#### C.2 Positional Embedding

**IMPLEMENTATION:** Fixed (non-learnable) sinusoidal encoding

$$ PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$
$$ PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) $$

**Parameters:**
- Maximum sequence length: 5000
- Dimension: $d_{model}$ (default: 768)
- Requires gradient: False (fixed)

**Code Location:** [Embed.py:8-27](../Embed.py#L8-L27)

#### C.3 Data Embedding Composition

$$ \mathbf{Z}_0 = \text{Dropout}(\mathbf{E}_{token} + \mathbf{E}_{pos}) $$

**Dropout rate:** 0.1 (default)

**Code Location:** [Embed.py:112-130](../Embed.py#L112-L130)

---

### Module D: Pre-trained GPT-2 Backbone

**KEY ARCHITECTURAL CHOICE:** Uses pre-trained GPT-2 with selective fine-tuning.

#### D.1 Model Variants

| Model | Layers | Hidden Dim | Heads | Parameters |
|-------|--------|------------|-------|------------|
| GPT-2 (small) | 12 | 768 | 12 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 350M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |

#### D.2 Parameter Freezing Strategy

**Default Strategy (mlp=0):** Only LayerNorm and positional embeddings are trainable

```python
for name, param in gpt2.named_parameters():
    if 'ln' in name or 'wpe' in name:
        param.requires_grad = True  # Trainable
    elif 'mlp' in name and mlp == 1:
        param.requires_grad = True  # Optional: unfreeze MLP
    else:
        param.requires_grad = False  # Frozen
```

**Frozen Components:**
- Multi-head attention weights ($W_Q, W_K, W_V, W_O$)
- Feed-forward network weights
- Embedding layer (token embeddings)

**Trainable Components:**
- LayerNorm ($\gamma, \beta$ in each layer)
- Positional embeddings ($W_{pe}$)

**Code Location:** [GPT4CP.py:101-107](../models/GPT4CP.py#L101-L107)

#### D.3 GPT-2 Forward Pass

$$ \mathbf{Z}_{L} = \text{GPT2}(\mathbf{Z}_0) $$

Each GPT-2 layer performs:

1. **Layer Normalization:**
   $$ \mathbf{z}'_l = \text{LayerNorm}(\mathbf{z}_{l-1}) $$

2. **Multi-Head Self-Attention:**
   $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
   $$ \mathbf{a}_l = \text{MSA}(\mathbf{z}'_l) $$

3. **Residual Connection:**
   $$ \mathbf{z}''_l = \mathbf{z}_{l-1} + \mathbf{a}_l $$

4. **Layer Normalization:**
   $$ \mathbf{z}'''_l = \text{LayerNorm}(\mathbf{z}''_l) $$

5. **Feed-Forward Network (GELU activation):**
   $$ \text{FFN}(x) = \text{GELU}(x\mathbf{W}_1 + b_1)\mathbf{W}_2 + b_2 $$
   $$ \mathbf{f}_l = \text{FFN}(\mathbf{z}'''_l) $$

6. **Residual Connection:**
   $$ \mathbf{z}_l = \mathbf{z}''_l + \mathbf{f}_l $$

**Code Location:** [GPT4CP.py:159](../models/GPT4CP.py#L159)

#### D.4 Layer Configuration

The model uses the first $L_{layers}$ GPT-2 layers:

```python
gpt2.h = gpt2.h[:gpt_layers]  # Default: 6 layers
```

**Default:** 6 layers (half of GPT-2 small)

---

### Module E: Prediction Head (Output Projection)

The final step maps the latent representations back to the physical CSI dimensions.

#### E.1 Dimension Projection

$$ \mathbf{Y}_{dim} = \mathbf{Z}_{L_{layers}} \mathbf{W}_{out} + \mathbf{b}_{out} $$

*   $\mathbf{W}_{out} \in \mathbb{R}^{d_{ff} \times D_{output}}$
*   $D_{output} = c_{out} \times 2 = K \times N_t \times N_r \times 2$
    *   Factor of 2 for real/imaginary parts

**Code Location:** [GPT4CP.py:116](../models/GPT4CP.py#L116)

#### E.2 Temporal Projection (Time Expansion)

$$ \mathbf{Y}_{time} = \text{Linear}_{L \rightarrow L'}(\mathbf{Y}_{dim}^T) $$

*   Projects from `prev_len` (input sequence length) to `pred_len` (output sequence length)
*   Default: $16 \rightarrow 4$ time steps

**Code Location:** [GPT4CP.py:117-119](../models/GPT4CP.py#L117-L119)

#### E.3 Inverse Normalization

$$ \hat{\mathbf{H}}^d = \mathbf{Y}_{time} \cdot \sigma_{batch} + \mu_{batch} $$

Uses the same batch statistics from Module A for denormalization.

**Code Location:** [GPT4CP.py:165](../models/GPT4CP.py#L165)

#### E.4 Output Slicing

$$ \mathbf{H}_{final} = \hat{\mathbf{H}}^d[:, -L':, :] $$

Returns only the last `pred_len` time steps of the prediction.

**Code Location:** [GPT4CP.py:167](../models/GPT4CP.py#L167)

**Final Output Shape:** $[B, L', 2 \cdot K \cdot N_t \cdot N_r]$

---

## 3. Complete Forward Pass Tensor Shape Tracing

### Default Configuration

```python
batch_size = 1024
prev_len = 16      # Historical time slots
pred_len = 4       # Future time slots
K = 48             # Subcarriers (after LoadBatch_ofdm expansion)
UQh, UQv, BQh, BQv = 1, 1, 1, 1
enc_in = K * UQh * UQv * BQh * BQv = 48
d_model = 768
d_ff = 768
patch_size = 4
res_dim = 64
```

### Input Data Flow Summary

**Raw Data (.mat file)** → **Model Input**:

| Stage | Shape | Data Type | Description |
|-------|-------|-----------|-------------|
| 1. Raw `.mat` | `[900, 10, 16, 48, 4, 4, 2]` | complex128 | Original: 900 speeds × 10 samples × 16 time × 48 subcarriers × 4×4 UPA × 2 polarizations |
| 2. Rearranged | `[9000, 16, 1536]` | complex128 | Flatten spatial: 48×4×4×2 = 1536 |
| 3. LoadBatch_ofdm | `[288000, 16, 48]` | complex128 | Split 1536 = 48×32, expand batch ×32 |
| 4. Real conversion | `[288000, 16, 96]` | float32 | Complex→Real: 48×2 (real+imag) |
| 5. Batch sample | `[1024, 16, 96]` | float32 | Take one batch for model |

**Note:** After LoadBatch_ofdm, the spatial dimension (32 antennas) is absorbed into the batch dimension. The model operates on **48 subcarriers per spatial channel**, treating each antenna-position combination as an independent sample.

### Tensor Evolution Through Model

| Stage | Operation | Tensor Shape | Code Location |
|-------|-----------|--------------|---------------|
| **Input** | Raw CSI | `[1024, 16, 96]` | [train.py:54](../train.py#L54) |
| **A** | Normalization | `[1024, 16, 96]` | [GPT4CP.py:132](../models/GPT4CP.py#L132) |
| B1 | Split real/imag | `[1024, 16, 48, 2]` | [GPT4CP.py:135](../models/GPT4CP.py#L135) |
| B1 | Complex IFFT | `[1024, 16, 48]` (complex) | [GPT4CP.py:137](../models/GPT4CP.py#L137) |
| B1 | Split after IFFT | `[1024, 16, 96]` | [GPT4CP.py:138](../models/GPT4CP.py#L138) |
| B1 | Patch reshape | `[1024, 4, 4, 96]` | [GPT4CP.py:139](../models/GPT4CP.py#L139) |
| B1 | Patch layer | `[1024, 4, 96, 4]` | [GPT4CP.py:140](../models/GPT4CP.py#L140) |
| B1 | Reshape back | `[1024, 16, 96]` | [GPT4CP.py:141](../models/GPT4CP.py#L141) |
| B1 | 2D reshape for CNN | `[1024, 2, 16, 48]` | [GPT4CP.py:142](../models/GPT4CP.py#L142) |
| B1 | ResBlock (delay) | `[1024, 2, 16, 48]` | [GPT4CP.py:143](../models/GPT4CP.py#L143) |
| B2 | Frequency path | `[1024, 2, 16, 48]` | [GPT4CP.py:149](../models/GPT4CP.py#L149) |
| **B3** | Domain fusion | `[1024, 2, 16, 48]` | [GPT4CP.py:151](../models/GPT4CP.py#L151) |
| **B3** | Flatten | `[1024, 16, 96]` | [GPT4CP.py:152](../models/GPT4CP.py#L152) |
| **C** | Token Embedding | `[1024, 16, 768]` | [Embed.py:42](../Embed.py#L42) |
| **C** | + Positional | `[1024, 16, 768]` | [Embed.py:126](../Embed.py#L126) |
| **C** | Dropout | `[1024, 16, 768]` | [Embed.py:126](../Embed.py#L126) |
| **D** | Pre-GPT linear | `[1024, 16, 768]` | [GPT4CP.py:156](../models/GPT4CP.py#L156) |
| **D** | Pad to GPT dim | `[1024, 16, 768]` | [GPT4CP.py:157](../models/GPT4CP.py#L157) |
| **D** | GPT-2 forward | `[1024, 16, 768]` | [GPT4CP.py:159](../models/GPT4CP.py#L159) |
| **E** | Dim projection | `[1024, 16, 96]` | [GPT4CP.py:162](../models/GPT4CP.py#L162) |
| **E** | Time projection | `[1024, 4, 96]` | [GPT4CP.py:163](../models/GPT4CP.py#L163) |
| **E** | Inverse norm | `[1024, 4, 96]` | [GPT4CP.py:165](../models/GPT4CP.py#L165) |
| **Output** | Slice last L' | `[1024, 4, 96]` | [GPT4CP.py:167](../models/GPT4CP.py#L167) |

---

## 4. Loss Function

The model is trained to minimize the **Normalized Mean Square Error (NMSE)** between the predicted downlink CSI ($\hat{\mathbf{H}}^d$) and the ground truth downlink CSI ($\mathbf{H}^d$).

### 4.1 NMSE Loss

$$ \mathcal{L}_{NMSE} = \frac{1}{M} \sum_{m=1}^{M} \frac{\| \hat{\mathbf{H}}^d_m - \mathbf{H}^d_m \|_F^2}{\| \mathbf{H}^d_m \|_F^2} $$

*   $M$: Batch size
*   $\|\cdot\|_F$: Frobenius norm
*   Reduction: Mean (default)

**Implementation:**

```python
def NMSE_cuda(x_hat, x):
    power = torch.sum(x ** 2)
    mse = torch.sum((x - x_hat) ** 2)
    nmse = mse / power
    return nmse
```

**Code Location:** [metrics.py:57-75](../metrics.py#L57-L75)

### 4.2 Optional: Spectral Efficiency Loss

An alternative loss function based on Spectral Efficiency (SE):

$$ \mathcal{L}_{SE} = -\log_2 \det\left(\mathbf{I} + \frac{\mathbf{H}^H \mathbf{H}}{\sigma^2}\right) $$

**Code Location:** [metrics.py:16-54](../metrics.py#L16-L54)

---

## 5. Training Configuration

### 5.1 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.0001 | Adam optimizer |
| Batch Size | 1024 | Training batch |
| Epochs | 500 | Maximum epochs |
| Optimizer | Adam | betas=(0.9, 0.999) |
| Weight Decay | 0.0001 | L2 regularization |
| Dropout | 0.1 | Embedding dropout |

### 5.2 Training Loop

```python
for epoch in range(epochs):
    for batch in training_data_loader:
        pred_t, prev = batch[0], batch[1]
        pred_m = model(prev, None, None, None)
        loss = criterion(pred_m, pred_t)
        loss.backward()
        optimizer.step()
```

**Code Location:** [train.py:45-83](../train.py#L45-L83)

---

## 6. Key Implementation Differences from Theoretical Specification

| Aspect | Theoretical Specification | Actual Implementation |
|--------|---------------------------|----------------------|
| **Positional Encoding** | Learnable matrix | Fixed sinusoidal encoding |
| **Patch Embedding** | 2D spatial patches + Linear | 1D temporal patches + Conv1d with circular padding |
| **Normalization** | Pre-computed $\mu, \sigma$ | Per-batch dynamic statistics |
| **Domain Processing** | Frequency domain only | **Dual-domain** (Frequency + Delay via IFFT) |
| **Pre-LLM Processing** | None | **Residual CNN blocks** with Channel Attention |
| **LLM Architecture** | Generic Transformer | **Pre-trained GPT-2** with frozen weights |
| **Token Embedding** | Simple Linear | 1D Convolution (kernel=3, circular padding) |
| **Trainable Parameters** | All parameters | **Selective fine-tuning** (only LayerNorm + position embeddings) |

---

## 7. Architecture Diagram

```
Input CSI [B, L, 2*K*Nt*Nr]
         │
         ▼
    Normalization (per-batch)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Freq Path  Delay Path (IFFT)
    │         │
    ▼         ▼
 Temporal  Temporal
 Patching  Patching
    │         │
    ▼         ▼
  Linear   Linear
    │         │
    ▼         ▼
ResBlock  ResBlock
 (RB_e)    (RB_f)
    │         │
    └────┬────┘
         │
         ▼
     Fusion (+)
         │
         ▼
   Conv1d Token Embedding
   (circular padding)
         │
         ▼
   + Positional Encoding
   (fixed sinusoidal)
         │
         ▼
      Dropout
         │
         ▼
   Pre-trained GPT-2
   (frozen attention/MLP,
    trainable LayerNorm)
         │
         ▼
   Output Projection
   (dim + time)
         │
         ▼
   Inverse Normalization
         │
         ▼
Output CSI [B, L', 2*K*Nt*Nr]
```

---

## 8. File Structure Reference

| Component | File | Lines |
|-----------|------|-------|
| Main Model | [models/GPT4CP.py](../models/GPT4CP.py) | 53-181 |
| Channel Attention | [models/GPT4CP.py](../models/GPT4CP.py) | 16-32 |
| Residual Block | [models/GPT4CP.py](../models/GPT4CP.py) | 35-50 |
| Token Embedding | [Embed.py](../Embed.py) | 30-45 |
| Positional Embedding | [Embed.py](../Embed.py) | 8-27 |
| Data Embedding | [Embed.py](../Embed.py) | 112-130 |
| NMSE Loss | [metrics.py](../metrics.py) | 57-75 |
| Training Loop | [train.py](../train.py) | 45-104 |

---

## Appendix: Model Parameters

### Constructor Parameters

```python
Model(
    gpt_type='gpt2',        # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
    d_ff=768,               # Feed-forward dimension
    d_model=768,            # Model dimension
    gpt_layers=6,           # Number of GPT-2 layers to use
    pred_len=4,             # Prediction length
    prev_len=16,            # Previous/historical length
    use_gpu=1,
    gpu_id=0,
    mlp=0,                  # Whether to unfreeze GPT-2 MLP
    res_layers=4,           # Number of residual CNN blocks
    K=48,                   # Number of subcarriers
    UQh=4, UQv=1,           # User antenna dimensions
    BQh=2, BQv=1,           # BS antenna dimensions
    patch_size=4,           # Temporal patch size
    stride=1,
    res_dim=64,             # Residual block hidden dimension
    embed='timeF',
    freq='h',
    dropout=0.1
)
```

### Parameter Count

For default configuration (GPT-2 small, UQh=1, UQv=1, BQh=1, BQv=1):

- **Total Parameters:** ~124M (mostly from frozen GPT-2)
- **Learnable Parameters:** ~2M (LayerNorm + position embeddings + custom heads)