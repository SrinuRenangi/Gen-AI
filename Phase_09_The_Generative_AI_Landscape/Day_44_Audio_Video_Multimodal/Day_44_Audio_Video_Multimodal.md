# Day 44: Text-to-Audio, Video & Multimodal Models

Welcome to **Day 44 of our 50-Day Generative AI Masterclass**! Over [Day 42](../Day_42_What_is_GenAI/Day_42_What_is_GenAI.md) and [Day 43](../Day_43_Text_to_Image_Diffusion/Day_43_Text_to_Image_Diffusion.md), you learned how generative models explore high-dimensional data distributions and how diffusion models synthesize photorealistic 2D images.

Today, we expand beyond static images into the dynamic frontier of **Audio, Video, and Multimodal Intelligence**:
1. **Audio AI**: How models synthesize human speech and symphonies using Mel-spectrograms and Neural Audio Codecs (Suno, Bark, AudioLDM).
2. **Video Generation**: How OpenAI Sora, Runway Gen-3, and Stable Video Diffusion expand 2D diffusion into the temporal dimension using **factorized spatio-temporal attention**.
3. **Multimodal Vision-Language Models (VLMs)**: How CLIP, LLaVA, and GPT-4o fuse sight with language by projecting image patches directly into an LLM's token stream.

---

## 1. The Core Mental Model: The Hollywood Film Director

Consider how a film director produces a motion picture:

```
+-----------------------------------------------------------------------------------+
|                           THE FILM DIRECTOR'S MULTIVERSE                          |
+-----------------------------------------------------------------------------------+
|  1. The Script (Text / LLM)       --> "A detective steps into a rainy alley under |
|                                       a flickering neon sign."                    |
|                                                                                   |
|  2. The Camera (Video / DiT)      --> Captures 24 frames per second. Must enforce |
|                                       physics: rain falls downward, the trench    |
|                                       coat doesn't change color between frames.   |
|                                                                                   |
|  3. The Score (Audio / Codec)     --> The sound of footsteps splashing in water,  |
|                                       the low hum of neon, moody jazz saxophone.  |
|                                                                                   |
|  4. The Vision-Language Critic    --> Evaluates whether the scene matches the     |
|     (Multimodal VLM / CLIP)           original script and provides feedback.      |
+-----------------------------------------------------------------------------------+
```

True artificial general intelligence cannot live in a text-only vacuum. The real physical world is multimodal, continuous, and temporal.

---

## 2. Text-to-Audio: How Machines Hear and Speak

Generating audio is fundamentally different from generating text or images because of the **sampling rate**:

$$\text{Standard CD-Quality Audio} = 44.1 \text{ kHz} = \mathbf{44,100} \text{ amplitude numbers per second}$$

Generating a 3-minute song requires predicting $44,100 \times 180 = \mathbf{7,938,000}$ continuous float values! An autoregressive language model predicting one sample at a time would grind to a halt.

To make audio generation tractable, researchers use two breakthroughs:

### 1. Mel-Spectrograms & Spectrogram Diffusion (AudioLDM, Stable Audio)
Audio waveforms are converted via the **Short-Time Fourier Transform (STFT)** into a 2D time-frequency heatmap called a **Mel-Spectrogram**:
- The horizontal axis represents **Time**.
- The vertical axis represents **Frequencies (Pitch)** scaled to match human ear sensitivity (Mel scale).
- The pixel intensity represents **Loudness (Amplitude)**.

```
Raw 1D Waveform (44,100 samples/sec)
           |
           v [Short-Time Fourier Transform]
2D Mel-Spectrogram Image (80 frequency bins × T time steps)
           |
           v [Latent Diffusion Model]  <--- Treats audio as a 2D image!
Denoised Mel-Spectrogram
           |
           v [Neural Vocoder: HiFi-GAN]
Reconstructed High-Fidelity Audio Waveform
```

Because a spectrogram looks like an image, we can use the exact same **Latent Diffusion U-Net** from [Day 43](../Day_43_Text_to_Image_Diffusion/Day_43_Text_to_Image_Diffusion.md) to generate music and sound effects!

---

### 2. Neural Audio Codecs & Residual Vector Quantization (EnCodec, DAC, Suno)

Modern music generators (like **Suno** and **Bark**) use **Neural Audio Codecs** (such as Meta's EnCodec or SoundStream):
1. A convolutional encoder compresses raw 24kHz/44.1kHz audio by a factor of $320\times$ down to a compact 50 Hz latent stream.
2. **Residual Vector Quantization (RVQ)** quantizes continuous latents across multiple codebooks:
   - Codebook 1 captures coarse harmonic structure (pitch and rhythm).
   - Codebooks 2–8 capture fine-grained timbre and acoustic acoustic nuances.
3. An autoregressive Transformer then predicts these discrete audio tokens just like words in a sentence!

---

## 3. Video Generation: The Temporal Frontier

A video is not just a sequence of unrelated images; it is a continuous volume of space evolving through time:

$$\text{Video Tensor} \in \mathbb{R}^{B \times C \times T \times H \times W}$$

Where:
- $B$: Batch size
- $C$: Color channels ($3$ for RGB)
- $T$: Number of frames in time (e.g., $24$ frames for a 1-second clip)
- $H, W$: Height and width resolution (e.g., $512 \times 512$)

### The Naive Trap: Frame-by-Frame Generation
If you take a 2D image diffusion model and generate 24 frames independently using identical prompts, the output is horrifying:
- The character's face shifts shape every 40 milliseconds.
- Clothing colors morph randomly.
- Background trees appear and disappear.
This catastrophic failure is known as **temporal flickering** and a complete lack of **object permanence**.

---

### The Solution: Factorized Spatio-Temporal Attention (Sora, Stable Video Diffusion)

To generate silky-smooth video, models decouple attention into two distinct, alternating operations:

![Video Diffusion and Spatio-Temporal Attention](assets/video_diffusion_temporal_attention.svg)

```
Video Latent Tensor: Shape (Batch, Time, Height, Width, Channels)

               +------------------------------------------------+
               |           2D SPATIAL SELF-ATTENTION            |
               | • Reshape to: (Batch * Time, H * W, Channels)  |
               | • Operates purely within each 2D frame         |
               | • Learns: Shapes, lighting, textures, anatomy  |
               +------------------------------------------------+
                                       |
                                       v
               +------------------------------------------------+
               |          1D TEMPORAL SELF-ATTENTION            |
               | • Reshape to: (Batch * H * W, Time, Channels)  |
               | • Fixes (x, y) and attends across all T frames |
               | • Learns: Velocity, gravity, fluid flow, motion|
               +------------------------------------------------+
```

By alternating Spatial Attention with Temporal Attention, the model ensures that:
1. Every individual frame is a crisp, high-resolution photograph.
2. Every pixel coordinate across consecutive frames obeys physical laws of motion and momentum!

### Spacetime Patches (OpenAI Sora)
In 2024, OpenAI introduced **Sora**, which treats video as a collection of **spacetime patches**:
- Instead of cutting 2D images into flat square patches (like a standard ViT), video is carved into small **3D space-time cubes** (e.g., $2 \text{ frames} \times 16 \times 16 \text{ pixels}$).
- A **Diffusion Transformer (DiT)** processes these 3D patch tokens directly, scaling effortlessly across variable resolutions, aspect ratios, and durations.

---

## 4. Multimodal Intelligence: Vision-Language Models (VLMs)

How can an AI look at a medical scan and write a diagnosis, or look at a screenshot of a broken website and debug the React code?

To achieve this, we must build a bridge between two wildly different sensory worlds: **Vision (continuous 2D pixel grids)** and **Language (discrete 1D symbolic tokens)**.

### Step 1: CLIP — The Shared Multimodal Latent Space

In 2021, OpenAI published **CLIP (Contrastive Language-Image Pre-training)**:

```
Batch of N Images ----------------> [ Image Encoder (ViT) ] -------> Image Vectors I_1, ..., I_N
                                                                              |
                                                                    Compute Cosine Similarity
                                                                    Dot Product Matrix (N × N)
                                                                              |
Batch of N Matching Captions -----> [ Text Encoder (Transformer) ] -> Text Vectors T_1, ..., T_N
```

#### The InfoNCE Contrastive Loss
CLIP trains both encoders simultaneously on 400 million (image, text) pairs scraped from the web. 

For a batch of $N$ pairs:
- The $N$ diagonal elements $(I_i, T_i)$ are positive matches (an image and its true caption).
- The $N^2 - N$ off-diagonal elements $(I_i, T_j \text{ where } i \ne j)$ are negative imposters.

The model minimizes the symmetric cross-entropy loss:

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2} \left( \mathcal{L}_{\text{image}\to\text{text}} + \mathcal{L}_{\text{text}\to\text{image}} \right)$$

$$\mathcal{L}_{\text{image}\to\text{text}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp\left( \frac{I_i \cdot T_i}{\tau} \right)}{\sum_{j=1}^N \exp\left( \frac{I_i \cdot T_j}{\tau} \right)}$$

Where $\tau$ is a learnable temperature parameter (typically initialized around $0.07$).

---

### Step 2: The Modern VLM Architecture (LLaVA / GPT-4o / Claude 3.5 Sonnet)

While CLIP can calculate how similar an image is to a text label, it **cannot converse, reason, or answer complex questions**.

The modern blueprint for conversational vision models was established by **LLaVA (Large Language and Vision Assistant)**:

![Multimodal VLM Architecture](assets/multimodal_vlm_architecture_llava.svg)

The VLM architecture consists of three elegant components:

```
+-----------------------------------------------------------------------------------+
|                          THE 3 COMPONENTS OF A MODERN VLM                         |
+-----------------------------------------------------------------------------------+
|  1. The Visual Eyes: Vision Transformer (ViT-L/14)                                |
|     • Splits a 336×336 image into 576 patches of size 14×14.                      |
|     • Outputs 576 visual feature vectors: Z_v ∈ ℝ^{576 × 1024}.                   |
|                                                                                   |
|  2. The Modality Projector: Linear Layer / 2-Layer MLP                            |
|     • The LLM cannot understand 1024-dim vision vectors directly because its      |
|       hidden embedding dimension is 4096!                                         |
|     • The projector maps visual vectors: W_proj : ℝ^{1024} → ℝ^{4096}.            |
|     • Visual patches are now identical in shape to word embeddings!               |
|                                                                                   |
|  3. The Reasoning Brain: Autoregressive Decoder LLM (Llama 3 / Vicuna)            |
|     • Concatenates the 576 visual tokens with the user's prompt tokens:           |
|       [ <image>, v_1, v_2, ..., v_576, </image>, "What", "breed", "is", "this?" ]|
|     • Standard causal self-attention runs over both vision and text tokens        |
|       simultaneously, generating the answer token-by-token!                       |
+-----------------------------------------------------------------------------------+
```

> [!NOTE]
> ### Why Modern VLMs Do Not Need a Separate "Vision Model" to Reason
> Notice the elegance of this design: once the visual patches are projected into the LLM's embedding space ($d = 4096$), the LLM treats each $14\times 14$ pixel patch **exactly like a word in its vocabulary**!
> The profound reasoning capabilities of the language model—its knowledge of biology, physics, medicine, and common sense—are applied directly to the visual tokens via standard self-attention.

---

## 5. Production Hands-On Lab: Implementing CLIP Loss & Spatio-Temporal Video Reshaping

Let's write a runnable PyTorch script that implements:
1. **CLIP InfoNCE Contrastive Loss** from scratch with temperature scaling.
2. **Spatio-Temporal Tensor Reshaping** for factorized video attention.

### Python Script: `multimodal_clip_and_video_lab.py`

```python
"""
multimodal_clip_and_video_lab.py
Hands-on implementation of:
1. CLIP Symmetric Contrastive InfoNCE Loss from scratch
2. Video Spatio-Temporal Factorized Attention Tensor Reshaping
Author: GenAI 50-Day Masterclass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. CLIP CONTRASTIVE LOSS (INFONCE)
# =====================================================================
class CLIPContrastiveLoss(nn.Module):
    """
    Computes symmetric contrastive loss between image and text embeddings:
    L = 0.5 * (CrossEntropy(logits, targets) + CrossEntropy(logits.T, targets))
    """
    def __init__(self, initial_temperature: float = 0.07):
        super().__init__()
        # Learnable temperature parameter log(1 / tau)
        self.log_temperature = nn.Parameter(torch.tensor([torch.log(torch.tensor(1.0 / initial_temperature))]))

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [Batch_size, Embed_dim] (L2-normalized)
            text_features:  [Batch_size, Embed_dim] (L2-normalized)
        """
        # Ensure L2 normalization: ||v||_2 = 1.0
        image_embeddings = F.normalize(image_features, p=2, dim=-1)
        text_embeddings = F.normalize(text_features, p=2, dim=-1)

        # Temperature parameter tau = 1 / exp(log_temp)
        temperature = torch.exp(self.log_temperature)

        # Compute cosine similarity matrix: [Batch_size, Batch_size]
        logits = torch.matmul(image_embeddings, text_embeddings.T) * temperature

        # Ground truth targets: diagonal indices [0, 1, 2, ..., Batch_size - 1]
        batch_size = image_features.shape[0]
        targets = torch.arange(batch_size, device=image_features.device)

        # Symmetric cross-entropy
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)

        total_loss = (loss_i2t + loss_t2i) / 2.0
        return total_loss, logits


# =====================================================================
# 2. VIDEO SPATIO-TEMPORAL ATTENTION RESHAPING
# =====================================================================
def demonstrate_spatio_temporal_reshaping():
    """
    Shows how a 5D video tensor is decoupled into:
    1. 2D Spatial Attention format:  (B * T, H * W, C)
    2. 1D Temporal Attention format: (B * H * W, T, C)
    """
    batch_size = 2
    time_frames = 8
    channels = 64
    height, width = 16, 16

    # 5D Video Latent Tensor: [B, C, T, H, W]
    video_tensor = torch.randn(batch_size, channels, time_frames, height, width)
    print(f"Original Video Tensor Shape: [B={batch_size}, C={channels}, T={time_frames}, H={height}, W={width}]")

    # --- OPERATION A: PREPARE FOR 2D SPATIAL ATTENTION ---
    # Permute to: [B, T, H, W, C] -> Reshape to: [B * T, H * W, C]
    spatial_view = video_tensor.permute(0, 2, 3, 4, 1).contiguous()
    spatial_tokens = spatial_view.view(batch_size * time_frames, height * width, channels)
    print(f"\n1. Spatial Attention View:   {list(spatial_tokens.shape)}")
    print(f"   -> {batch_size * time_frames} individual frames, each with {height * width} spatial tokens.")

    # --- OPERATION B: PREPARE FOR 1D TEMPORAL ATTENTION ---
    # Permute to: [B, H, W, T, C] -> Reshape to: [B * H * W, T, C]
    temporal_view = video_tensor.permute(0, 3, 4, 2, 1).contiguous()
    temporal_tokens = temporal_view.view(batch_size * height * width, time_frames, channels)
    print(f"\n2. Temporal Attention View:  {list(temporal_tokens.shape)}")
    print(f"   -> {batch_size * height * width} pixel trajectories, each with {time_frames} time steps.")


# =====================================================================
# 3. VERIFICATION & NUMERICAL RUN
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("1. CLIP CONTRASTIVE INFONCE LOSS VERIFICATION")
    print("=" * 65)

    clip_loss_fn = CLIPContrastiveLoss(initial_temperature=0.07)

    # Simulate batch of 4 paired image and text embeddings (dim = 512)
    torch.manual_seed(42)
    batch_size = 4
    embed_dim = 512

    # Matched vectors (positive pairs share high similarity)
    base_vectors = torch.randn(batch_size, embed_dim)
    img_feats = base_vectors + torch.randn(batch_size, embed_dim) * 0.1
    txt_feats = base_vectors + torch.randn(batch_size, embed_dim) * 0.1

    loss, similarity_logits = clip_loss_fn(img_feats, txt_feats)

    print(f"Batch Size: {batch_size}, Embedding Dim: {embed_dim}")
    print(f"Similarity Logits (scaled by temperature):\n{similarity_logits.detach().numpy().round(2)}")
    print(f"\nComputed CLIP Loss: {loss.item():.4f}")
    print("Notice: Diagonal elements are dominant, leading to low cross-entropy loss!\n")

    print("=" * 65)
    print("2. VIDEO SPATIO-TEMPORAL FACTORIZED TENSOR RESHAPING")
    print("=" * 65)
    demonstrate_spatio_temporal_reshaping()
    print("=" * 65)
```

---

## 6. Multimodal & Generative Landscape Comparison Cheat Sheet

| Domain | Leading Models | Core Architecture | Data Representation | Primary Engineering Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **Audio AI** | Suno, Bark, AudioLDM 2 | Autoregressive Transformer or 2D Latent Diffusion | 2D Mel-Spectrograms or Discrete RVQ Codec Tokens | Long sequences ($44.1\text{kHz}$); audio phase fidelity |
| **Video AI** | OpenAI Sora, Runway Gen-3, SVD | Diffusion Transformer (DiT) with Spatiotemporal Patches | 5D Video Latents $(B, C, T, H, W)$ | VRAM footprint; temporal coherence; physical causality |
| **Multimodal VLMs** | LLaVA, GPT-4o, Claude 3.5 Sonnet | Vision Transformer (ViT) + MLP Projector + LLM Backbone | 576–1152 Visual Patch Tokens interleaved with text | High token consumption per image; high resolution scaling |

---

## 7. Self-Check Exercises & Solutions

### Question 1: The CLIP Cosine Similarity Diagonal
In a batch of $N = 4$ images and captions, the diagonal cosine similarities are:
$$\text{Diag} = [0.90, 0.85, 0.88, 0.92]$$
All off-diagonal similarities are $0.10$. With temperature $\tau = 0.10$, compute the image-to-text probability $P(\text{Match} \mid \text{Image}_0)$ for the first image.

**Solution**:
1. Scale similarities by temperature ($z = \frac{\text{sim}}{\tau} = \frac{\text{sim}}{0.1} = 10 \times \text{sim}$):
   - Diagonal logit $z_0 = 10 \times 0.90 = 9.0$.
   - Three off-diagonal logits $z_j = 10 \times 0.10 = 1.0$.
2. Apply softmax:
   $$e^{z_0} = e^{9.0} \approx 8103.08$$
   $$e^{z_j} = e^{1.0} \approx 2.718 \quad (3 \text{ terms} \implies 3 \times 2.718 = 8.154)$$
   $$\sum_{j=0}^3 e^{z_j} = 8103.08 + 8.154 = 8111.23$$
   $$P(\text{Match} \mid \text{Image}_0) = \frac{8103.08}{8111.23} \approx \mathbf{0.9990} \text{ (99.9\% confidence)}$$

---

### Question 2: Why Not Flatten the Video Tensor into a Single 1D Attention Sequence?
Consider a 4-second video clip at 24 frames per second ($T = 96$ frames) with $32 \times 32 = 1,024$ spatial latent tokens per frame.
1. What is the total sequence length $L$ if all tokens across space and time attend to each other simultaneously?
2. Why is factorized spatio-temporal attention mandatory here?

**Solution**:
1. Total sequence length:
   $$L = T \times (H \times W) = 96 \times 1024 = \mathbf{98,304} \text{ tokens}$$
   Standard self-attention memory scales quadratically:
   $$L^2 = (98,304)^2 \approx \mathbf{9.66 \times 10^9} \text{ attention elements per head per layer}$$
   This would consume hundreds of gigabytes of VRAM for just the attention matrix of a single layer!
2. **Factorized Attention**:
   - Spatial attention: $96$ operations of length $1024$ $\implies 96 \times (1024)^2 \approx 1.0 \times 10^8$.
   - Temporal attention: $1024$ operations of length $96$ $\implies 1024 \times (96)^2 \approx 9.4 \times 10^6$.
   - Total attention elements $\approx 1.1 \times 10^8$, which is **nearly $90\times$ cheaper computationally**, fitting easily inside modern GPU VRAM!

---

### Question 3: The Role of the Modality Projector in VLMs
Why can't you directly feed the 1024-dimensional output vectors of a CLIP Vision Transformer into the attention blocks of a LLaMA 3 language model?

**Solution**:
Neural network attention layers require strict matrix shape compatibility: the Query, Key, and Value projection matrices inside LLaMA 3 expect input vectors of dimension $d_{\text{LLM}} = 4096$. Because the Vision Transformer produces vectors of dimension $d_{\text{ViT}} = 1024$, matrix multiplication is mathematically impossible without dimension alignment. The modality projector (a linear layer or 2-layer MLP with weights $W \in \mathbb{R}^{1024 \times 4096}$) bridges this dimensional gap while translating visual features into the linguistic semantic space of the LLM.

---

## 8. Summary of Phase 9 & Looking Ahead to Phase 10

Congratulations! You have completed **Phase 9: The Generative AI Landscape (Days 42–44)**!

```
+-----------------------------------------------------------------------------------+
|                        PHASE 9 COMPLETE LEARNING JOURNEY                          |
+-----------------------------------------------------------------------------------+
|  Day 42: The Generative Paradigm Shift (P(Y|X) vs P(X)), Trilemma, and 4 Families  |
|  Day 43: Text-to-Image Diffusion (DDPM, Latent Diffusion, and CFG Steering)       |
|  Day 44: Audio Synthesis (Spectrograms/Codecs), Video Diffusion, and Modern VLMs  |
+-----------------------------------------------------------------------------------+
```

You have now mastered the theoretical and architectural engines behind every major generative modality: text, code, audio, image, and video.

Tomorrow, we inaugurate our final, capstone phase: **Phase 10: Practical GenAI Engineering (Days 45–50)**. We transition from architectural theory to **production software engineering**:
- Prompt engineering patterns (Chain-of-Thought, ReAct, Few-Shot).
- Interacting with OpenAI, Anthropic, and local Ollama APIs.
- Building high-accuracy **Retrieval-Augmented Generation (RAG)** systems.
- Autonomous **AI Agents & Function Calling**.
- Local **Fine-Tuning with LoRA, QLoRA, and Quantization**.

See you in [Day 45](../../Phase_10_Practical_GenAI/Day_45_Prompt_Engineering/Day_45_Prompt_Engineering.md)!
