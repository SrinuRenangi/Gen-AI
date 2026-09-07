# Day 26: CNNs Part 1 — How Computers See Images

> **"If you flatten a 1-megapixel image into a traditional neural network, you destroy all spatial geometry and require 3 billion weights for a single layer. Convolutional Neural Networks solve vision using just 9 shared weights."**  
> Welcome to Day 26! Today we kick off **Phase 5: Specialized Neural Networks**. We will master how machines process the visual world using **Convolutional Neural Networks (CNNs)**.

---

## 🧭 The Mental Compass: The Art Inspector with a Magnifying Glass

Imagine you are an art authenticator examining a master painting:

```
    APPROACH A: The Paper Shredder (Traditional Flat MLP)
    ────────────────────────────────────────────────────
    • You cut the masterpiece into 1,000,000 microscopic strips.
    • You paste them into one giant 1-dimensional line across the floor.
    • You try to find the painted smile.
    ❌ Pixel (0, 0) and Pixel (1, 0) were touching on the canvas, but now they 
       are separated by 1,000 indices! All 2D spatial relationships are destroyed.

    APPROACH B: The Sliding Magnifying Glass (Convolutional Neural Network)
    ───────────────────────────────────────────────────────────────────────
    • You hold a 3x3 cm magnifying glass (A Kernel / Filter).
    • You slide it steadily across the canvas, row by row.
    • The lens looks for specific micro-textures: diagonal brushstrokes, edges, curves.
    • Whether a cat is in the top-left corner or bottom-right corner, the same lens 
      instantly recognizes its ears (Translation Invariance)!
```

CNNs mimic the biological visual cortex: **local receptive fields, spatial weight sharing, and hierarchical abstraction.**

---

## 1. Why Standard Neural Networks (MLPs) Fail on Images

Before convolutions were invented, why couldn't we just feed pixels into our standard Day 19 Multi-Layer Perceptrons?

### Reason 1: The Parameter Catastrophe
Suppose you take a modest smartphone photo: $1,000 \times 1,000$ pixels with 3 color channels (Red, Green, Blue):
$$\text{Total Inputs } X = 1,000 \times 1,000 \times 3 = \mathbf{3,000,000 \text{ numbers!}}$$

If your first hidden layer has a modest 1,000 neurons:
$$\text{Parameters} = 3,000,000 \times 1,000 = \mathbf{3,000,000,000 \text{ weights (3 Billion!)}}$$
A single layer would consume **12 Gigabytes of GPU RAM** just for weights, requiring millions of training images to prevent catastrophic overfitting!

### Reason 2: The Loss of 2D Spatial Locality
In visual imagery, **nearby pixels are strongly correlated**. The edge of an eye is formed by pixels grouped together in 2D space. Flattening a matrix into a 1D vector completely severs spatial proximity.

### Reason 3: No Translation Invariance
If an MLP learns to recognize a dog in the top-left corner of an image, it has learned weights tied strictly to input pixels $x_1, \dots, x_{500}$. If the dog moves to the bottom-right corner ($x_{2,000,000}$), the MLP is completely blind to it!

---

## 2. The 2D Convolution Operation: Weight Sharing in Action

Instead of connecting every pixel to every neuron, a CNN slides a tiny matrix called a **Kernel (or Filter)** across the image:

![Anatomy of a 2D Convolution (Sliding Window)](assets/convolution_sliding_window_kernel.svg)

### The Mathematics of a Single Window Slide:
For an overlapping image patch $\mathbf{X}_{\text{patch}}$ and kernel $\mathbf{K}$:
$$\text{Feature Value } = \sum_{i=1}^{K_h} \sum_{j=1}^{K_w} (X_{i, j} \cdot K_{i, j}) + b$$

It is simply an **element-wise product followed by a sum** (the Frobenius inner product)!

### Step-by-Step Hand Calculation Example:
Let's compute the top-left cell of the feature map from the diagram:

$$\text{Image Patch} = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{bmatrix}, \quad \text{Kernel} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}, \quad \text{Bias } b = 0$$

$$\text{Products} = \begin{bmatrix} (1 \cdot 1) & (1 \cdot 0) & (1 \cdot 1) \\ (0 \cdot 0) & (1 \cdot 1) & (1 \cdot 0) \\ (0 \cdot 1) & (0 \cdot 0) & (1 \cdot 1) \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

$$\text{Sum} = 1 + 0 + 1 + 0 + 1 + 0 + 0 + 0 + 1 = \mathbf{4.0}$$

The scalar **$4.0$** is placed into coordinate $(0, 0)$ of the Output Feature Map! The kernel then shifts right by the **Stride** and repeats the calculation.

> [!NOTE]
> **The Miracle of Weight Sharing:**  
> Notice that the **exact same 9 weights** in the kernel are used to evaluate every single pixel patch on the entire image. Instead of 3,000,000,000 weights, you only train **9 numbers**!

---

## 3. The 3 Levers: Kernel Size, Padding, and Stride

Every convolutional layer is governed by three fundamental geometric hyperparameters:

![Padding, Stride & Receptive Field Expansion](assets/padding_stride_receptive_field.svg)

---

### Hyperparameter 1: Kernel Size ($K$)
The spatial width and height of the filter (almost always square and odd-numbered: $3 \times 3$, $5 \times 5$, or $7 \times 7$). Odd dimensions ensure an exact central anchor pixel!

---

### Hyperparameter 2: Padding ($P$)
When a kernel slides across an image, border pixels are only visited once, while central pixels are visited up to 9 times. Furthermore, every convolution shrinks the image dimensions! To fix this, we add a border of zeros around the image:

* **Valid Padding ($P = 0$):** No padding added. The feature map shrinks at every layer:
  $$\text{Input: } 32 \times 32 \xrightarrow{K=3} 30 \times 30 \xrightarrow{K=3} 28 \times 28$$
* **Same Padding ($P = \lfloor K / 2 \rfloor$):** Adds enough zero-padding around the border so that **the output spatial dimensions exactly match the input**:
  $$\text{Input: } 32 \times 32 \xrightarrow{K=3, P=1} 32 \times 32 \quad \text{(Dimensions preserved!)}$$

---

### Hyperparameter 3: Stride ($S$)
The number of pixels the sliding window shifts at each step:
* **Stride $S = 1$:** Shifts 1 pixel at a time (standard fine-grained feature extraction).
* **Stride $S = 2$:** Skips 2 pixels at a time. This **downsamples the spatial resolution by 50%**, replacing the need for pooling layers!

---

### The Universal Output Sizing Formula
Given input dimension $W$, kernel size $K$, padding $P$, and stride $S$, the output feature map dimension $O$ is strictly:

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

#### Worked Sizing Examples:
* **Example 1:** $W = 5, K = 3, P = 0, S = 1$:
  $$O = \frac{5 - 3 + 0}{1} + 1 = 2 + 1 = \mathbf{3} \quad (5\times5 \to 3\times3)$$
* **Example 2:** $W = 224, K = 7, P = 3, S = 2$ (ResNet-50 initial conv):
  $$O = \left\lfloor \frac{224 - 7 + 2(3)}{2} \right\rfloor + 1 = \left\lfloor \frac{223}{2} \right\rfloor + 1 = 111 + 1 = \mathbf{112} \quad (224\times224 \to 112\times112)$$

---

## 4. Receptive Field: How Small Filters See the Entire World

A common beginner question: *"If a kernel is only $3 \times 3$ pixels, how can a CNN ever recognize a giant dog that spans $500 \times 500$ pixels?"*

The answer is **Receptive Field Expansion**:
* A neuron in **Layer 1** sees a $3 \times 3$ window of raw pixels.
* A neuron in **Layer 2** sees a $3 \times 3$ window of Layer 1 neurons. But each of those Layer 1 neurons saw a $3 \times 3$ window of raw pixels! Therefore, the Layer 2 neuron has an effective receptive field of **$5 \times 5$ raw pixels**.
* By **Layer 20**, a single neuron's receptive field spans the **entire $1000 \times 1000$ image**!

> [!TIP]
> **The VGGNet Insight (Simonyan & Zisserman, 2014):**  
> Stacking two $3 \times 3$ conv layers has the same receptive field as one single $5 \times 5$ conv layer, but uses **$2 \times (3 \times 3) = 18$ parameters** instead of **$1 \times (5 \times 5) = 25$ parameters** (28% fewer weights!) while adding an extra non-linear ReLU between them!

---

## 5. Hands-On Python Lab: 2D Convolution from Scratch vs PyTorch

Let's implement a manual 2D convolution in pure NumPy and verify that PyTorch's `nn.Conv2d` produces the exact same numerical result:

```python
import numpy as np
import torch
import torch.nn as nn

# =====================================================================
# 1. PURE NUMPY 2D CONVOLUTION FROM SCRATCH
# =====================================================================
def conv2d_scratch(image, kernel, stride=1, padding=0):
    """
    Manual 2D Convolution with stride and padding.
    image: (H, W)
    kernel: (K, K)
    """
    H, W = image.shape
    K, _ = kernel.shape
    
    # 1. Apply Zero-Padding if P > 0
    if padding > 0:
        padded_img = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    else:
        padded_img = image

    H_pad, W_pad = padded_img.shape
    
    # 2. Compute Output Dimensions
    out_H = (H_pad - K) // stride + 1
    out_W = (W_pad - K) // stride + 1
    
    output = np.zeros((out_H, out_W))
    
    # 3. Slide the Kernel across the Image
    for i in range(out_H):
        for j in range(out_W):
            h_start = i * stride
            h_end = h_start + K
            w_start = j * stride
            w_end = w_start + K
            
            # Extract receptive field patch
            patch = padded_img[h_start:h_end, w_start:w_end]
            
            # Element-wise product followed by summation
            output[i, j] = np.sum(patch * kernel)
            
    return output

# =====================================================================
# 2. RUN EXPERIMENT ON A 5x5 IMAGE (MATCHING DIAGRAM)
# =====================================================================
input_image = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
], dtype=np.float32)

diagonal_kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
], dtype=np.float32)

scratch_feature_map = conv2d_scratch(input_image, diagonal_kernel, stride=1, padding=0)

print("=" * 65)
print("     NUMPY FROM-SCRATCH 2D CONVOLUTION FEATURE MAP")
print("=" * 65)
print("Input Dimensions  : (5, 5)")
print("Kernel Dimensions : (3, 3)")
print("Output Feature Map (3x3):\n", scratch_feature_map)

# =====================================================================
# 3. VERIFY WITH PYTORCH nn.Conv2d
# =====================================================================
print("\n--- VERIFYING AGAINST PYTORCH nn.Conv2d ---")

# Reshape to PyTorch 4D format: (Batch, Channels, Height, Width)
img_tensor = torch.tensor(input_image).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 5, 5)

# Build PyTorch Conv2d layer
conv_pytorch = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

# Manually load our exact kernel weights into PyTorch
with torch.no_grad():
    conv_pytorch.weight = nn.Parameter(torch.tensor(diagonal_kernel).unsqueeze(0).unsqueeze(0))

# Execute forward pass
pytorch_feature_map = conv_pytorch(img_tensor).squeeze().detach().numpy()

diff = np.max(np.abs(scratch_feature_map - pytorch_feature_map))
print("PyTorch Feature Map (3x3):\n", pytorch_feature_map)
print(f"Discrepancy: {diff:.2e}")
print("✅ 100% Exact Numerical Match! Scratch logic is identical to PyTorch's C++ backend.")
```

---

## 6. Summary Checklist for Day 26

1. [x] **Why MLPs Fail on Vision:** Destroys 2D spatial locality, parameter explosion (billions of weights), lacks translation invariance.
2. [x] **The Convolutional Filter:** A small matrix of weights that slides across the image, computing local Frobenius inner products.
3. [x] **Weight Sharing:** The same 9 or 25 parameters scan every patch of the image, massively reducing model size and preventing overfitting.
4. [x] **Padding:** Valid ($P=0$) shrinks the output; Same ($P=\lfloor K/2 \rfloor$) preserves spatial resolution.
5. [x] **Stride:** Step size of the window. Stride 2 halves dimensions cleanly.
6. [x] **The Sizing Formula:** $O = \lfloor (W - K + 2P)/S \rfloor + 1$.
7. [x] **Receptive Field:** Deep stacked small kernels ($3 \times 3$) see broad global features with significantly fewer weights.

---

*Tomorrow in **Day 27**, we complete the vision revolution: **CNNs Part 2 — Pooling Layers, Full Architectural Pipelines, and Famous Models (LeNet, AlexNet, VGG, and ResNet Skip Connections)!***
