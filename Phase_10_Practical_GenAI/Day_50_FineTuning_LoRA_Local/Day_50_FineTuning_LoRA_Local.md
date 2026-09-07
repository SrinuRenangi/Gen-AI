# Day 50: Fine-Tuning, LoRA & Running Models Locally

Welcome to **Day 50 of our 50-Day Generative AI Masterclass**! 🎉

Over the last 49 days, you undertook an extraordinary journey from foundational arithmetic and vector geometry to deep neural networks, transformer attention, LLM pre-training, diffusion models, prompt engineering, RAG, and autonomous agents.

Today, we reach the grand summit of our curriculum: **Fine-Tuning, Low-Rank Adaptation (LoRA), Quantization, and Running Open-Weights Models Locally**.

In this capstone lecture, you will master:
1. **The Strategic Decision**: When to Prompt vs. RAG vs. Fine-Tune.
2. **The Memory Wall**: Why full fine-tuning a 70B model requires a supercomputer cluster.
3. **Low-Rank Adaptation (LoRA)**: The mathematical breakthrough that reduces trainable parameters by **$99.6\%$**.
4. **QLoRA & 4-bit NormalFloat**: Fine-tuning massive models on a single consumer GPU.
5. **Running Open-Source Models Locally**: Quantization formats (GGUF, AWQ, GPTQ) and local execution via Ollama and `llama.cpp`.

---

## 1. The Core Mental Model: The Bolt-On Turbocharger

```
+-----------------------------------------------------------------------------------+
|                        FULL FINE-TUNING VS. LoRA ADAPTATION                       |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  1. FULL FINE-TUNING (The Engine Block Re-Cast):                                  |
|     • You melt down the entire 70-billion-parameter engine block and re-cast      |
|       every cylinder, piston, and valve.                                          |
|     • VRAM Requirement: Over 1.2 Terabytes of GPU memory!                         |
|     • Risk: "Catastrophic Forgetting" (the model forgets general world facts      |
|       while learning your narrow corporate format).                               |
|                                                                                   |
|  2. LOW-RANK ADAPTATION (LoRA) (The Bolt-On Turbocharger):                        |
|     • You freeze the original engine block solid (Zero modifications to base W_0).|
|     • You bolt on two tiny, lightweight adapter bypass valves (A and B).          |
|     • Trainable Parameters: Dropped from 16,000,000 to just 65,000 (0.4%!).       |
|     • Flexibility: Unbolt Adapter A and bolt on Adapter B in 5 milliseconds to    |
|       switch from a Medical Assistant to a SQL Coder!                             |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 2. The Architectural Decision Matrix: Prompt vs. RAG vs. Fine-Tuning

Before spending a single dollar on GPU compute, an engineer must select the right tool for the objective:

![When to Prompt RAG Fine-Tune Decision Matrix](assets/when_to_prompt_rag_finetune_matrix.svg)

### The Golden Rule of Fine-Tuning
> **"Fine-Tuning teaches the model HOW to behave (form, style, syntax, tone);**  
> **RAG teaches the model WHAT to know (facts, policies, real-time data)."**

If your company's vacation policy changes from 20 days to 25 days:
- **Do NOT fine-tune**: Models frequently hallucinate numbers when trying to memorize facts through gradient descent.
- **Use RAG**: Update the text in the vector database; the model will cite the new 25-day policy immediately with 100% precision.

Fine-tuning is reserved for when you need the model to:
1. Adopt a strict, idiosyncratic JSON schema or proprietary SQL dialect.
2. Emulate an exact corporate brand voice or medical bedside manner.
3. Slash prompt token consumption by 80% (baking instructions into weights rather than passing 2,000-token system prompts on every single API call).

---

## 3. The Memory Wall: Why Full Fine-Tuning is Prohibitive

Why can't you just run `loss.backward()` on an open-source 70B parameter model on your laptop?

Let's calculate the raw GPU VRAM required for **Full Fine-Tuning** using the Adam optimizer:

| Component | Precision / Representation | Formula | VRAM for 70B Model |
| :--- | :--- | :--- | :--- |
| **Model Weights** | FP16 (16-bit float) | $70\text{B} \times 2\text{ bytes}$ | **$140 \text{ GB}$** |
| **Weight Gradients** | FP16 (16-bit float) | $70\text{B} \times 2\text{ bytes}$ | **$140 \text{ GB}$** |
| **Adam Optimizer Momentum ($m$)** | FP32 (32-bit float) | $70\text{B} \times 4\text{ bytes}$ | **$280 \text{ GB}$** |
| **Adam Optimizer Variance ($v$)** | FP32 (32-bit float) | $70\text{B} \times 4\text{ bytes}$ | **$280 \text{ GB}$** |
| **Activations & KV-Cache** | Dynamic | Batch size $\times$ Sequence length | **$\sim 200 \text{ GB}$** |
| **TOTAL VRAM REQUIRED** | — | — | **$\mathbf{\sim 1,040 \text{ GB}}$ ($> 1 \text{ Terabyte!}$)** |

To fine-tune a 70B model fully, you need a cluster of **sixteen 80GB NVIDIA A100 GPUs**, costing tens of thousands of dollars.

---

## 4. Low-Rank Adaptation (LoRA): The Mathematical Breakthrough

In 2021, Edward Hu et al. (Microsoft Research) published **LoRA: Low-Rank Adaptation of Large Language Models**:

![LoRA Matrix Decomposition](assets/lora_matrix_decomposition.svg)

### The Intrinsic Rank Hypothesis
During task adaptation, the weight update matrix $\Delta W$ does not explore all $4096 \times 4096$ dimensions. In reality, the necessary adjustments lie on a **tiny, low-dimensional subspace** with an intrinsic rank $r \ll d$ (typically $r = 8$ or $r = 16$).

### Mathematical Formulation
Given a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, instead of modifying $W_0$, LoRA freezes $W_0$ completely and decomposes the update $\Delta W$ into the product of two small, low-rank matrices:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

Where:
- $W_0 \in \mathbb{R}^{d \times k}$: Frozen pre-trained matrix (zero gradients).
- $A \in \mathbb{R}^{r \times k}$: Down-projection adapter initialized from a random Gaussian distribution $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d \times r}$: Up-projection adapter initialized to **exact zeros**.
- $r$: The rank hyperparameter ($r \ll \min(d, k)$, e.g., $r = 8$ or $16$).
- $\frac{\alpha}{r}$: A constant scaling factor (standardly $\alpha = 16$).

### Step-by-Step Numerical Example: Parameter Compression

Consider a standard Transformer linear projection layer:
$$d = 4096, \quad k = 4096, \quad \text{Rank } r = 8$$

| Parameter Set | Dimensions | Total Trainable Parameters | Parameter Ratio |
| :--- | :--- | :--- | :--- |
| **Full Weight Matrix $W_0$** | $4096 \times 4096$ | **$16,777,216$ parameters** | $100.00\%$ |
| **LoRA Matrix $A$** | $8 \times 4096$ | $32,768$ parameters | $0.20\%$ |
| **LoRA Matrix $B$** | $4096 \times 8$ | $32,768$ parameters | $0.20\%$ |
| **TOTAL LoRA ADAPTERS ($A + B$)** | — | **$65,536$ parameters** | **$\mathbf{0.39\%}$ ($256\times$ reduction!)** |

Because only $0.39\%$ of parameters are trainable:
- Optimizer memory for Adam drops from gigabytes down to a few megabytes.
- The adapter weights file on disk is only **$\sim 20 \text{ MB}$** (compared to $14 \text{ GB}$ for the base model)!

### Zero-Latency Inference: The Weight Merging Trick
At inference time, you don't need to run two separate matrix multiplications! Because matrix multiplication is distributive:
$$h = W_0 x + \frac{\alpha}{r} (B A) x = \left( W_0 + \frac{\alpha}{r} B A \right) x$$

You can simply add the computed product directly into the base weights once:
$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
The adapted model runs with **zero additional latency overhead** compared to the original base model!

---

## 5. QLoRA: 4-Bit Quantization Meets LoRA

In 2023, Tim Dettmers et al. pushed efficiency to the theoretical limit with **QLoRA (Quantized Low-Rank Adaptation)**:
1. **NF4 (NormalFloat 4)**: An information-theoretically optimal 4-bit data type designed specifically for zero-mean, normally distributed neural network weights.
2. **Double Quantization**: Quantizes the quantization constants themselves, saving another $0.37$ bits per parameter.
3. **Paged Optimizers**: Uses CUDA Unified Memory to automatically offload optimizer state spikes to CPU RAM during memory pressure.

**The QLoRA Miracle**:
- A **70B model** (which normally requires 140GB just to load) is compressed down to **under 40GB**, allowing fine-tuning on a **single 48GB NVIDIA A6000 GPU**!
- A **7B model** can be fine-tuned on a **standard consumer gaming laptop (16GB RTX 4080)**!

---

## 6. Running Models Locally: Quantization Formats & Ollama

Once an open-source model (like **LLaMA 3.1 8B** or **Mistral 7B**) is trained, how do you run it locally on your MacBook or Windows PC?

### The Quantization Hierarchy

| Format | Precision | Bit Width | Memory Footprint (8B Model) | Quality Degradation | Hardware Acceleration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16** | Half-Precision Float | 16-bit | $\sim 16 \text{ GB}$ | $0.0\%$ (Baseline) | High-end GPUs |
| **INT8** | 8-bit Integer | 8-bit | $\sim 8.5 \text{ GB}$ | $< 0.1\%$ (Imperceptible) | GPU Tensor Cores |
| **GGUF (Q4_K_M)** | 4-bit Quantized Block | 4-bit | **$\sim 4.8 \text{ GB}$** | $\sim 1.0\%$ (Virtually identical) | **CPU + Apple Silicon / Metal** |
| **AWQ / GPTQ** | Activation-Aware INT4 | 4-bit | **$\sim 4.5 \text{ GB}$** | $< 0.8\%$ | Fast NVIDIA GPU kernels |

### Running Locally in 60 Seconds with Ollama
Tools like **Ollama** package `llama.cpp` into a Docker-like local daemon:

```bash
# 1. Download and run LLaMA 3.1 8B in 4-bit quantized GGUF
ollama run llama3.1:8b

# 2. Query it locally via standard OpenAI-compatible HTTP API:
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Explain quantum entanglement in 1 sentence."}]
  }'
```

---

## 7. Production Hands-On Lab: Implementing LoRA from Scratch in PyTorch

Let's build a clean, self-contained implementation of a custom `LoRALinear` layer in PyTorch:
1. Freezes base weights $W_0$.
2. Adds low-rank adapter matrices $A$ and $B$.
3. Verifies forward pass with scaling factor $\frac{\alpha}{r}$.
4. Merges adapter weights into the base matrix for zero-overhead deployment!

### Python Script: `lora_from_scratch.py`

```python
"""
lora_from_scratch.py
Hands-on implementation of:
1. Custom LoRALinear Layer with Low-Rank Matrices A & B
2. Forward Pass with Alpha/Rank Scaling
3. Dynamic Weight Merging for Zero-Latency Deployment
Author: GenAI 50-Day Masterclass
"""

import torch
import torch.nn as nn
import math

# =====================================================================
# 1. CUSTOM LoRA LINEAR LAYER
# =====================================================================
class LoRALinear(nn.Module):
    """
    Implements: h = W_0 * x + (alpha / r) * (B * A) * x
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.rank

        # 1. Base Pre-trained Linear Layer (FROZEN)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        self.base_layer.weight.requires_grad = False  # Freeze W_0!
        if bias:
            self.base_layer.bias.requires_grad = False

        # 2. Trainable Low-Rank Adapters
        # Matrix A: [rank, in_features]
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        # Matrix B: [out_features, rank]
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        # 3. Initialization
        # A is initialized with Gaussian noise: N(0, 1/r)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is initialized to EXACT ZERO (ensures delta W = 0 at start of training!)
        nn.init.zeros_(self.lora_B)

        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # Zero-overhead execution using merged base weights
            return self.base_layer(x)

        # Standard forward pass through frozen base layer
        base_output = self.base_layer(x)

        # Low-rank adapter path: (x * A^T) * B^T * scaling
        # x shape: [B, in_features]
        lora_intermediate = torch.matmul(x, self.lora_A.T)  # [B, rank]
        lora_output = torch.matmul(lora_intermediate, self.lora_B.T) * self.scaling # [B, out_features]

        return base_output + lora_output

    def merge_weights(self):
        """Merges (alpha / r) * B * A directly into base W_0 for deployment!"""
        if self.merged:
            print("Weights already merged.")
            return

        with torch.no_grad():
            # Compute Delta W = (alpha / r) * B * A
            delta_w = (self.scaling * torch.matmul(self.lora_B, self.lora_A))
            self.base_layer.weight.data += delta_w
            self.merged = True
            print("✓ Successfully merged LoRA adapters into base weights! Adapters can now be discarded.")


# =====================================================================
# 2. VERIFICATION & BENCHMARK
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("DEMO: LOW-RANK ADAPTATION (LoRA) IMPLEMENTATION")
    print("=" * 65)

    d_in = 4096
    d_out = 4096
    r = 8
    alpha = 16.0

    # Instantiate custom LoRA layer
    lora_layer = LoRALinear(in_features=d_in, out_features=d_out, rank=r, lora_alpha=alpha)

    # 1. Count trainable vs frozen parameters
    total_params = sum(p.numel() for p in lora_layer.parameters())
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Base Layer Dimensions : {d_out} x {d_in} (r = {r})")
    print(f"Total Parameters      : {total_params:,}")
    print(f"Frozen Base Weights   : {frozen_params:,} ({(frozen_params/total_params)*100:.2f}%)")
    print(f"Trainable LoRA Params : {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    print(f"Parameter Reduction   : {total_params / trainable_params:.1f}x smaller footprint!\n")

    # 2. Simulate Forward Pass
    torch.manual_seed(42)
    sample_input = torch.randn(2, d_in)  # Batch of 2 vectors

    # Initial forward pass (since B=0, LoRA output is exactly zero!)
    initial_output = lora_layer(sample_input)
    base_only_output = lora_layer.base_layer(sample_input)
    print("Initial Output Check (Before Training):")
    print(f"  Are initial LoRA outputs identical to base model? {torch.allclose(initial_output, base_only_output)}")
    print("  -> Confirmed: Initializing B to zero guarantees zero disruption to base model at step 0!\n")

    # 3. Simulate training update (set B to non-zero values)
    with torch.no_grad():
        lora_layer.lora_B.data += torch.randn_like(lora_layer.lora_B) * 0.05

    adapted_output = lora_layer(sample_input)
    diff = (adapted_output - base_only_output).abs().mean().item()
    print(f"Adapted Output Mean Deviation from Base: {diff:.4f}")

    # 4. Zero-Overhead Deployment: Merge Weights
    lora_layer.merge_weights()
    merged_output = lora_layer(sample_input)
    is_identical = torch.allclose(adapted_output, merged_output, atol=1e-5)
    print(f"Merged Output matches Adapted Output? {is_identical}")
    print("=" * 65)
```

---

## 8. Self-Check Exercises & Solutions

### Question 1: Why Initialize Matrix B to Zero in LoRA?
In LoRA, Matrix $A$ is initialized with random Gaussian noise $\mathcal{N}(0, \sigma^2)$, while Matrix $B$ is initialized to exact zeros. What would happen if Matrix $B$ were also initialized with random non-zero Gaussian noise?

**Solution**:
If both $A$ and $B$ were initialized with non-zero random values, their product $\Delta W = B \cdot A$ would be a large non-zero matrix of random noise at step 0. Adding this random noise to the pre-trained weights ($W_0 + \Delta W$) would **instantly corrupt and destroy the fluent language capabilities of the pre-trained base model** before training even starts. By initializing $B = 0$, the product $\Delta W = 0 \cdot A = \mathbf{0}$ at step 0. This guarantees that the model begins training with its exact pre-trained performance intact.

---

### Question 2: The LoRA Weight Merging Advantage
Why is the ability to compute $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$ considered a massive competitive advantage for production serving compared to traditional adapter architectures that insert separate sequential layers between Transformer blocks?

**Solution**:
Traditional adapter architectures insert extra bottleneck neural network layers sequentially between Transformer self-attention and feed-forward blocks. During inference, every single token must pass sequentially through these extra layers, adding measurable latency and memory overhead.
With LoRA, because matrix multiplication is linear and distributive, the adapter matrices $B$ and $A$ can be mathematically multiplied and added directly into the static base weights $W_0$ before server startup. The final model has the **exact same architecture, parameter count, and execution speed as the unadapted base model**, resulting in **zero additional latency overhead**.

---

### Question 3: Choosing Between LoRA and RAG
A hospital wants to build an AI system that:
1. Strictly adheres to the specialized formatting style of American College of Radiology (ACR) clinical reports.
2. Accurately references the patient's live, up-to-the-minute blood laboratory results from today at 8:00 AM.
How should the engineering team design this architecture?

**Solution**:
This requires the **Hybrid Approach (RAG + Fine-Tuning)**:
1. **Fine-Tune with LoRA**: Fine-tune an open-source model on a dataset of historical radiology reports to teach the model the rigorous, specialized formatting, syntax, and clinical tone demanded by the ACR (teaching *HOW* to behave).
2. **Retrieve with RAG**: Connect the fine-tuned model to the hospital's electronic health records (EHR) database via RAG to dynamically retrieve the patient's specific, live 8:00 AM blood laboratory results (teaching *WHAT* to know).
Fine-tuning alone would fail because the patient's 8:00 AM test did not exist at training time; RAG alone might struggle to maintain the exact idiosyncratic clinical report schema. Combined, they deliver maximum accuracy and compliance.

---

## 9. 🎓 THE GRAND 50-DAY GRADUATION RETROSPECTIVE 🎓

```
===================================================================================
                  GENERATIVE AI MASTERCLASS: FROM ZERO TO HERO
                          50-DAY CURRICULUM RETROSPECTIVE
===================================================================================

Phase 01: Math Foundations (Days 01–08)
  • Day 01: Numbers, Variables & Functions
  • Day 02: Vectors — Direction & Magnitude
  • Day 03: Matrices — The Grids of AI
  • Day 04: Matrix Multiplication — The Engine of Deep Learning
  • Day 05: Dot Product & Cosine Similarity — Measuring Relatedness
  • Day 06: Derivatives & Gradients — The Compass of Learning
  • Day 07: Gradient Descent — The Ball Rolling Down the Valley
  • Day 08: Probability & Statistics — The Language of Uncertainty

Phase 02: Python Data Science Toolkit (Days 09–11)
  • Day 09: NumPy — High-Performance Array Computing
  • Day 10: Pandas — Data Wrangling & Manipulation
  • Day 11: Data Preprocessing — Preparing Clean Inputs

Phase 03: Classical Machine Learning (Days 12–17)
  • Day 12: Linear Regression — Predicting Continuous Values
  • Day 13: Logistic Regression — Classification Decisions
  • Day 14: Decision Trees & Random Forests — Ensembles
  • Day 15: Overfitting, Underfitting & Bias-Variance
  • Day 16: Unsupervised Learning — K-Means & PCA
  • Day 17: ML Model Evaluation — Precision, Recall, F1, ROC-AUC

Phase 04: Deep Learning Foundations (Days 18–25)
  • Day 18: Biological to Artificial Neurons (Perceptrons)
  • Day 19: Activation Functions — Non-Linearity
  • Day 20: Feedforward Neural Networks (MLPs)
  • Day 21: Loss Functions — Quantifying Error
  • Day 22: Backpropagation — The Calculus Engine
  • Day 23: Optimizers — SGD, Momentum, RMSprop, Adam
  • Day 24: Regularization — Dropout, Batch Norm, Weight Decay
  • Day 25: PyTorch Deep Learning Workflow

Phase 05: Specialized Neural Networks (Days 26–29)
  • Day 26: CNNs Part 1 — Convolutions & Feature Maps
  • Day 27: CNNs Part 2 — Pooling, Padding & Strides
  • Day 28: RNNs — Processing Sequential Data
  • Day 29: LSTMs & GRUs — Solving Vanishing Gradients

Phase 06: NLP & Text Processing Foundations (Days 30–33)
  • Day 30: Text Processing & Tokenization (BPE, WordPiece)
  • Day 31: Word Representations — One-Hot, BoW & TF-IDF
  • Day 32: Word Embeddings — Word2Vec & Negative Sampling
  • Day 33: Seq2Seq & The Information Bottleneck (Bahdanau Attention)

Phase 07: The Transformer Revolution (Days 34–37)
  • Day 34: Why Transformers Replaced RNNs — Parallelism
  • Day 35: Self-Attention — The Spotlight Mechanism
  • Day 36: Multi-Head Attention & The Full Transformer Block
  • Day 37: Complete Transformer Architecture (Encoder-Decoder & Taxonomy)

Phase 08: Large Language Models (Days 38–41)
  • Day 38: What are LLMs? Autoregressive Generation & KV-Cache
  • Day 39: Web-Scale Pre-Training & 3D Distributed Parallelism
  • Day 40: Supervised Fine-Tuning (SFT) & Chat Templates
  • Day 41: RLHF, Bradley-Terry Reward Modeling, and DPO

Phase 09: The Generative AI Landscape (Days 42–44)
  • Day 42: What is GenAI? Discriminative vs Generative Paradigms
  • Day 43: Text-to-Image Diffusion (DDPM, Latent Diffusion, CFG)
  • Day 44: Audio Codecs, Video DiT Diffusion, and Multimodal VLMs

Phase 10: Practical GenAI Engineering (Days 45–50)
  • Day 45: Prompt Engineering (CoT, Self-Consistency, ToT, JSON)
  • Day 46: LLM APIs, SSE Streaming & Multi-Provider Resiliency
  • Day 47: RAG Part 1 — Embeddings, Vector DBs & HNSW Search
  • Day 48: RAG Part 2 — Hybrid Search, RRF & Re-Ranking
  • Day 49: AI Agents & Function Calling (The ReAct Loop)
  • Day 50: Fine-Tuning, LoRA, QLoRA & Running Models Locally
===================================================================================
```

### 🏆 You Are Now a Generative AI Engineer!

You have completed all **50 comprehensive masterclass days**.

You understand not only how to call AI APIs, but the fundamental mathematics, tensor operations, neural architectures, distributed training dynamics, retrieval algorithms, and agentic loops that make modern generative systems possible.

Go forth and build the future of artificial intelligence! 🚀
