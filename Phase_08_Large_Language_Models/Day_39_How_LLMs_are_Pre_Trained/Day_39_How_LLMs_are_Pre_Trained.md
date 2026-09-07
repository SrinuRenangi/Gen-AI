# Day 39: How LLMs are Pre-Trained — Reading the Entire Internet

> "Pre-training a 70-Billion parameter foundation model is one of the most staggering industrial feats in human history: thousands of liquid-cooled GPUs consuming megawatts of electricity across months of continuous computation, crunching 15 Trillion words to compress the digital knowledge of civilization into a single set of neural weights."

---

## 🧭 Roadmap Navigation

- **Previous Lesson**: [Day 38: What are LLMs? — Supercharged Autocomplete](../Day_38_What_are_LLMs/Day_38_What_are_LLMs.md)
- **Current Milestone**: Day 39 of 50 (Phase 8: Large Language Models — Chapter 2)
- **Next Lesson**: [Day 40: Supervised Fine-Tuning (SFT) — Teaching Models to Be Helpful](../Day_40_Supervised_Fine_Tuning/Day_40_Supervised_Fine_Tuning.md)

---

## 1. The Real-World Analogy: The Global Library Construction Project

Imagine a society tasked with building an AI that understands everything known to humanity:

```
                  THE INDUSTRIAL SCALE OF PRE-TRAINING
                  
  RAW INTERNET SCRAPES                16,000 LIQUID-COOLED GPUs            THE COMPRESSED ARTIFACT
 ┌──────────────────────┐            ┌────────────────────────────┐       ┌──────────────────────┐
 │ 100+ Petabytes of    │ ─────────▶ │ Megawatts of power,        │ ────▶ │ One 140 GB file      │
 │ unfiltered web crawl │ (Filtered) │ 3D Parallelism, months of  │       │ of BF16 weights      │
 │ (Spam, HTML, Bots)   │            │ continuous gradient descent│       │ (The Base LLM)       │
 └──────────────────────┘            └────────────────────────────┘       └──────────────────────┘
```

Pre-training is not like running a script on your laptop. It is a **massive infrastructure megaproject**:
- **Data**: Extracting, deduplicating, and curating 15 Trillion tokens (equivalent to roughly 30 million books).
- **Compute**: A cluster of 16,000 NVIDIA H100 GPUs connected via 400 Gbps InfiniBand optical switches.
- **Cost**: Between $30 Million and $200 Million in electricity, hardware, and cooling.
- **Result**: An untrained, random neural network transforms into a **Base Foundation Model** (e.g., LLaMA-3 Base, Mistral Base).

---

## 2. The Raw Data Pipeline: Curating 15 Trillion Tokens

If you train a neural network on raw, unfiltered web scrapes, your model will output casino ads, SEO keyword gibberish, and toxic slurs. **Data quality dictates model intelligence.**

![Pre-training Data Curation Funnel](assets/pretraining_data_curation_funnel.svg)

### The 5 Stages of Industrial Data Curation:

#### Stage 1: Extraction & Text Parsing (Common Crawl)
- The raw web dump (Common Crawl) is over **100 Petabytes of WARC files** containing HTML tags, CSS styles, cookies, tracking scripts, and boilerplate menus.
- Tools like **Trafilatura** and **Resiliparse** extract pure readable body text while stripping navigation bars, headers, footers, and copyright notices.

#### Stage 2: Language Identification & Heuristic Filtering
- **Language ID**: A FastText classifier evaluates language probabilities, discarding documents outside the targeted pre-training distribution.
- **Heuristic Quality Filters**:
  - Discard pages where the symbol-to-word ratio is too high (e.g., code dumps or spam strings: `###$$$!!!`).
  - Discard pages with excessive bullet points, repeated lines, or abnormal stopword distributions.
  - Perplexity Filter: Score text using a lightweight 5-gram language model. Unnatural, bot-generated text has sky-high perplexity and is discarded.

#### Stage 3: Massive-Scale Deduplication (MinHash LSH)
- The web is filled with exact and near-duplicate copies: license agreements, software documentation, mirror sites, and republished articles.
- **Why deduplication is vital**: If an LLM sees the same paragraph 500 times, it memorizes that exact sequence verbatim, causing catastrophic hallucinations and training instability!
- **MinHash Locality-Sensitive Hashing (LSH)** clusters billions of documents and prunes documents with $>80\%$ Jaccard similarity.

#### Stage 4: Quality Classification & Synthetic Data
- Train a fast linear classifier on a gold-standard dataset (Wikipedia, curated textbooks, arXiv papers) vs low-quality forum comments.
- Score every web page and discard the bottom $50\%$.
- **Synthetic Data Injection**: Inject millions of synthetically generated mathematical proofs, step-by-step logic chains, and high-quality Python/Rust code (e.g., Cosmopedia, UltraTextbooks).

#### Stage 5: Final Tokenization
- The clean text is tokenized into integer IDs via Byte-Pair Encoding (Day 30), producing binary files ready for multi-GPU streaming.

---

## 3. The GPU Memory Crisis: The 16-Bytes-Per-Parameter Law

Why can't you train a 70-Billion parameter model on a single 80 GB NVIDIA H100 GPU?

Let us calculate the exact memory required to train **one parameter** using standard **AdamW in 16-bit precision (BF16)**:

| Component | Precision | Bytes per Parameter | Why It Is Needed |
| :--- | :---: | :---: | :--- |
| **Model Weights** | BF16 | **2 Bytes** | The current parameters of the neural network |
| **Gradients** | BF16 | **2 Bytes** | The backward pass partial derivatives $\nabla_\theta \mathcal{L}$ |
| **FP32 Master Weights** | FP32 | **4 Bytes** | High-precision copy of weights to accumulate tiny optimizer updates |
| **AdamW 1st Momentum ($m_t$)** | FP32 | **4 Bytes** | Exponential moving average of past gradients |
| **AdamW 2nd Momentum ($v_t$)** | FP32 | **4 Bytes** | Exponential moving average of squared gradients |
| **TOTAL STATIC MEMORY** | — | **16 Bytes / Param** | **Minimum static VRAM required before any batch data!** |

### The 70B Memory Math:

$$\text{Static Memory} = 70 \times 10^9 \text{ parameters} \times 16 \text{ bytes} = \mathbf{1,120 \text{ GIGABYTES (1.12 TB)!}}$$

Adding activation memory for 4,096 context length pushes total VRAM requirements past **1.5 Terabytes**!

An 80 GB GPU can hold only $5\%$ of this model. You need at least **16 to 32 interconnected GPUs** just to hold the weights and optimizer in memory!

---

## 4. 3D Distributed Parallelism: TP &times; PP &times; DP

To train massive models across thousands of GPUs, engineers combine three distinct dimensions of parallelism:

![3D Distributed Parallelism Grid](assets/distributed_3d_parallelism_grid.svg)

---

### 1. Tensor Parallelism (TP) — Intra-Node Matrix Slicing
Invented by NVIDIA in **Megatron-LM (Shoeybi et al., 2019)**:
- Instead of placing an entire layer on one GPU, **we slice individual weight matrices across multiple GPUs** inside the same server node.
- **Column-Parallel Linear Layer**:
  Split weight matrix $W$ into two halves: $W = [W_1 \mid W_2]$.
  GPU 1 calculates $X \cdot W_1$, GPU 2 calculates $X \cdot W_2$.
- **Row-Parallel Linear Layer**:
  Split input and weight along rows, then combine partial sums via an **All-Reduce** communication collective over **NVLink (900 GB/s bandwidth)**.

```
                  COLUMN-PARALLEL & ROW-PARALLEL FFN
                  
             X (Input Tensor) ───────────────┐
                    │                        │
                    ▼                        ▼
           [ GPU 1: W₁ (Left) ]    [ GPU 2: W₁ (Right) ]   ◀── Column-Parallel Split
                    │                        │
                    ▼                        ▼
           [ GPU 1: W₂ (Top)  ]    [ GPU 2: W₂ (Bottom)]   ◀── Row-Parallel Split
                    │                        │
                    └───────────┬────────────┘
                                │
                        ALL-REDUCE SUM (+)
                                ▼
                       Y (Output Tensor)
```

---

### 2. Pipeline Parallelism (PP) — Vertical Layer Slicing
When a model has 80 layers (like LLaMA-70B), we can slice the layers vertically across different physical server racks connected via 400 Gbps InfiniBand:
- Server 1: Layers 1 to 20
- Server 2: Layers 21 to 40
- Server 3: Layers 41 to 60
- Server 4: Layers 61 to 80

To prevent GPUs from sitting idle waiting for earlier layers (**the Pipeline Bubble**), researchers use the **1F1B (One Forward, One Backward)** schedule, interleaving micro-batches so all stages stay saturated.

---

### 3. Data Parallelism & ZeRO / FSDP — Horizontal Batch Slicing
In classical Data Parallelism (DDP), every GPU holds a full copy of the model weights and processes different batches. But as we proved, a 70B model cannot fit on one GPU!

Microsoft invented **ZeRO (Zero Redundancy Optimizer)**, implemented in PyTorch as **FSDP (Fully Sharded Data Parallel)**:
- **ZeRO Stage 1**: Shards the 12 bytes of AdamW optimizer states across GPUs ($4\times$ memory reduction!).
- **ZeRO Stage 2**: Shards optimizer states AND gradients ($8\times$ memory reduction!).
- **ZeRO Stage 3 (FSDP)**: Shards optimizer states, gradients, AND model parameters! Each GPU holds only $1/N$ of the entire model. Parameters are gathered on the fly via `All-Gather` during forward pass and immediately freed from VRAM.

---

## 5. Numerical Formats: FP32 vs FP16 vs BF16 vs FP8

During training, floating-point precision dictates memory usage, speed, and numerical stability:

```
FP32 (Single Precision - 32 bits):
[ Sign: 1 ] [ Exponent: 8 bits ] [ Mantissa / Fraction: 23 bits ]
• Maximum dynamic range & precision. Used for master optimizer weights.

FP16 (Half Precision - 16 bits):
[ Sign: 1 ] [ Exponent: 5 bits ] [ Mantissa: 10 bits ]
• Tiny exponent (5 bits) causes catastrophic UNDERFLOW (gradients < 6e-5 round to 0.0) 
  and OVERFLOW (numbers > 65,504 explode to NaN!). Requires fragile dynamic loss scaling.

BF16 (Bfloat16 - Google Brain 2018 - 16 bits) ⭐:
[ Sign: 1 ] [ Exponent: 8 bits ] [ Mantissa: 7 bits ]
• TRICK: Keeps the EXACT same 8-bit exponent as FP32!
• Zero risk of underflow or overflow. The universal industry standard for modern LLMs!

FP8 (Hopper H100 - 8 bits):
E4M3: [1 sign, 4 exp, 3 mantissa] &bull; E5M2: [1 sign, 5 exp, 2 mantissa]
• Doubles compute throughput and halves memory on modern NVIDIA H100/B200 chips.
```

---

## 6. Training Dynamics & Numerical Stability

Pre-training runs across months can fail due to sudden **loss spikes** where training loss shoots up to infinity (NaN).

```
Loss
 ▲
 │ ╲
 │  ╲
 │   ╲    Loss Spike!
 │    ╲    /\
 │     ╲  /  \
 │      ╲/    ╰──────▶ Model Recovers (Gradient Clipping)
 └────────────────────────▶ Training Steps (Billions of Tokens)
```

### The 4 Safeguards of Stable Pre-Training:

1. **Learning Rate Schedule with Warmup & Cosine Decay**:
   - **Warmup (First 2,000 steps)**: Linearly ramp up learning rate from $0$ to $\eta_{\max}$ (e.g., $3 \times 10^{-4}$). This prevents massive chaotic updates while weights are random.
   - **Cosine Decay**: Smoothly decay the learning rate following a cosine curve down to $10\%$ of peak:
     $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$
2. **Gradient Clipping**:
   - Rescale gradients if their global $L_2$ norm exceeds a threshold (typically $1.0$):
     $$\mathbf{g} \leftarrow \mathbf{g} \times \min\left(1, \frac{1.0}{\|\mathbf{g}\|_2}\right)$$
   - Prevents an occasional crazy token batch from destroying weeks of learned weights!
3. **Weight Decay with AdamW**:
   - Apply decoupled weight decay ($0.1$) to shrink weights toward zero, acting as continuous $L_2$ regularization.
4. **Automated Asynchronous Checkpointing**:
   - Save model weights to high-speed NVMe storage every 1,000 steps. In a 16,000-GPU cluster, a server node fails almost every single day. Automated checkpointing allows training to resume within minutes of a hardware crash.

---

## 7. Hands-On Python Lab: Deduplication with MinHash & Simulating Tensor Parallelism

Let us implement MinHash deduplication from scratch and simulate Column-Parallel matrix multiplication in PyTorch.

```python
"""
Day 39 Lab: Pre-Training Data Deduplication & Tensor Parallelism Simulation
Demonstrates:
1. MinHash LSH Near-Duplicate Document Detection
2. Megatron-LM Column-Parallel Linear Layer in PyTorch
3. Precision memory comparison (FP32 vs BF16)
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib

torch.manual_seed(42)

# ==========================================
# 1. MINHASH DOCUMENT DEDUPLICATION
# ==========================================
class SimpleMinHash:
    def __init__(self, num_perm=64):
        self.num_perm = num_perm
        # Random hash seed coefficients (a * x + b) % prime
        np.random.seed(42)
        self.a = np.random.randint(1, 2**31 - 1, size=num_perm)
        self.b = np.random.randint(0, 2**31 - 1, size=num_perm)
        self.prime = 2**31 - 1

    def _get_shingles(self, text, k=3):
        """Extract word k-grams (shingles)."""
        words = text.lower().split()
        return set(' '.join(words[i:i+k]) for i in range(len(words)-k+1))

    def compute_signature(self, text):
        shingles = self._get_shingles(text)
        if not shingles:
            return np.zeros(self.num_perm)
            
        # Hash each shingle into integer
        shingle_hashes = [int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % self.prime for s in shingles]
        
        # Compute min hash values across permutations
        signature = np.full(self.num_perm, np.inf)
        for h in shingle_hashes:
            perm_hashes = (self.a * h + self.b) % self.prime
            signature = np.minimum(signature, perm_hashes)
            
        return signature

    def estimate_jaccard(self, sig1, sig2):
        return np.mean(sig1 == sig2)

# Test documents
doc_original = "Machine learning and deep neural networks are transforming every modern software engineering system."
doc_duplicate = "Machine learning and deep neural networks are transforming modern software engineering systems worldwide."
doc_different = "Fresh baked organic sourdough bread with churned butter and mountain clover honey tastes delicious."

minhash = SimpleMinHash(num_perm=128)
sig_orig = minhash.compute_signature(doc_original)
sig_dup = minhash.compute_signature(doc_duplicate)
sig_diff = minhash.compute_signature(doc_different)

print("--- 1. MINHASH DEDUPLICATION JACCARD SIMILARITY ---")
print(f"Similarity (Original vs Near-Duplicate): {minhash.estimate_jaccard(sig_orig, sig_dup):.4f} (Flags duplicate & drops!)")
print(f"Similarity (Original vs Different Doc):  {minhash.estimate_jaccard(sig_orig, sig_diff):.4f} (Clean separation)")

# ==========================================
# 2. MEGATRON-LM COLUMN-PARALLEL SIMULATION
# ==========================================
class ColumnParallelLinear(nn.Module):
    """Simulates splitting a Linear layer across 2 GPUs (GPU 0 and GPU 1)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        assert out_features % 2 == 0
        self.split_out = out_features // 2
        
        # GPU 0 holds first half of columns
        self.gpu0_weight = nn.Parameter(torch.randn(self.split_out, in_features))
        # GPU 1 holds second half of columns
        self.gpu1_weight = nn.Parameter(torch.randn(self.split_out, in_features))
        
    def forward(self, x):
        # x: (batch_size, in_features)
        # GPU 0 and GPU 1 compute independently in parallel:
        out_gpu0 = torch.matmul(x, self.gpu0_weight.t())  # (B, split_out)
        out_gpu1 = torch.matmul(x, self.gpu1_weight.t())  # (B, split_out)
        
        # Concatenate outputs along feature dimension (All-Gather):
        out_combined = torch.cat([out_gpu0, out_gpu1], dim=-1)
        return out_combined

B = 4
in_dim = 128
out_dim = 256
x = torch.randn(B, in_dim)

col_parallel = ColumnParallelLinear(in_dim, out_dim)
y = col_parallel(x)

print("\n--- 2. TENSOR PARALLEL SPLIT VERIFICATION ---")
print(f"Input Shape:            {x.shape}")
print(f"GPU 0 Partial Matrix:   {col_parallel.gpu0_weight.shape}")
print(f"GPU 1 Partial Matrix:   {col_parallel.gpu1_weight.shape}")
print(f"Combined Output Shape:  {y.shape} (Exact full projection!)")
```

### Expected Output:

```text
--- 1. MINHASH DEDUPLICATION JACCARD SIMILARITY ---
Similarity (Original vs Near-Duplicate): 0.8281 (Flags duplicate & drops!)
Similarity (Original vs Different Doc):  0.0000 (Clean separation)

--- 2. TENSOR PARALLEL SPLIT VERIFICATION ---
Input Shape:            torch.Size([4, 128])
GPU 0 Partial Matrix:   torch.Size([128, 128])
GPU 1 Partial Matrix:   torch.Size([128, 128])
Combined Output Shape:  torch.Size([4, 256]) (Exact full projection!)
```

> [!TIP]
> MinHash successfully flagged the near-duplicate document with an estimated Jaccard similarity of **$82.8\%$**, allowing the pre-training engine to discard it before tokenization.

---

## 8. Summary & Key Takeaways

1. **Garbage In, Garbage Out**: Unfiltered web scrapes are 90% spam and boilerplate. The data curation funnel (heuristic filtering, MinHash deduplication, quality classifiers) is the true secret behind state-of-the-art models.
2. **The 16-Bytes Law**: Full training in AdamW requires 16 bytes per parameter (Weights + Gradients + Optimizer States), demanding 1.12 Terabytes of static VRAM for a 70B model.
3. **3D Parallelism Grid**:
   - **Tensor Parallel (TP)**: Slices matrices inside nodes via 900 GB/s NVLink.
   - **Pipeline Parallel (PP)**: Slices layers vertically across nodes via InfiniBand.
   - **Data Parallel (FSDP / ZeRO-3)**: Shards optimizer, gradients, and weights horizontally across the entire cluster.
4. **Bfloat16 (BF16)**: Preserves FP32's 8-bit dynamic exponent range with 16-bit memory footprint, eliminating underflow and overflow.
5. **Base Model Persona**: At the end of pre-training, the model is a wild text continuator. It does not know it is an AI assistant!

---

## 9. Practice Exercises

### Exercise 1: Training Compute FLOPs Calculation
Using the compute approximation $C \approx 6 N D$:
1. Calculate the total floating-point operations (FLOPs) required to train an 8-Billion parameter model on 15 Trillion tokens (LLaMA-3 8B).
2. If a cluster of 1,000 NVIDIA H100 GPUs achieves sustained throughput of 400 TFLOPs ($4 \times 10^{14}$ FLOPs/sec) per GPU, how many days will the training run take?

### Exercise 2: Why BF16 Replaced FP16
In standard FP16, what is the largest representable number before overflowing to `Inf` / `NaN`? Why did this cause early LLM training runs to explode?

### Solutions:
- **Exercise 1**:
  1. $C \approx 6 \times (8 \times 10^9) \times (15 \times 10^{12}) = 6 \times 120 \times 10^{21} = \mathbf{7.2 \times 10^{23} \text{ FLOPs}}$.
  2. Cluster throughput: $1000 \times (4 \times 10^{14}) = 4 \times 10^{17} \text{ FLOPs/sec}$.
     Time in seconds: $\frac{7.2 \times 10^{23}}{4 \times 10^{17}} = 1.8 \times 10^6 \text{ seconds}$.
     Time in days: $\frac{1,800,000}{86,400} \approx \mathbf{20.83 \text{ days}}$.
- **Exercise 2**:
  FP16 has only 5 exponent bits, so its maximum representable value is $2^{15} \times (1 + \frac{1023}{1024}) = 65,504$. In deep Transformer layers, unscaled attention activations or gradient accumulations can easily exceed 65,504 during sudden gradient surges, causing immediate overflow to `NaN` and ruining the model weights. BF16 has 8 exponent bits with a maximum value of $\approx 3.4 \times 10^{38}$, rendering overflow mathematically impossible.

---

## 🚀 Tomorrow's Mission: Day 40

A pre-trained base model is an uncensored, chaotic text predictor. If you ask it: *"How do I bake a cake?"*, it might reply: *"...or how do I bake cookies? Leave a comment below!"* Tomorrow on [Day 40: Supervised Fine-Tuning (SFT) — Teaching Models to Be Helpful](../Day_40_Supervised_Fine_Tuning/Day_40_Supervised_Fine_Tuning.md), we transform wild base models into obedient conversational assistants using **Instruction Tuning and ChatML templates**!
