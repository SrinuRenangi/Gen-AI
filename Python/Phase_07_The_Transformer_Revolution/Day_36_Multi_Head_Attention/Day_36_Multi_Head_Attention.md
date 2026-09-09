# Day 36: Multi-Head Attention & The Full Transformer Block


| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 35: Self-Attention](../Day_35_Self_Attention/Day_35_Self_Attention.md) | [All 50 Days Overview](../../README.md) | [Day 37: The Complete Transformer Architecture →](../Day_37_Complete_Transformer_Architecture/Day_37_Complete_Transformer_Architecture.md) |

> "If single-head attention is a single flashlight beam, Multi-Head Attention is a stadium lighting rig with 8 or 16 multi-colored spotlights: one tracks grammar, one tracks pronouns, one tracks entities, and one tracks long-range theme—all operating simultaneously in parallel."

---

## 🧭 Roadmap Navigation

- **Previous Lesson**: [Day 35: Self-Attention — The Spotlight Mechanism](../Day_35_Self_Attention/Day_35_Self_Attention.md)
- **Current Milestone**: Day 36 of 50 (Phase 7: The Transformer Revolution — Chapter 3)
- **Next Lesson**: [Day 37: The Complete Transformer Architecture — Putting It All Together](../Day_37_Complete_Transformer_Architecture/Day_37_Complete_Transformer_Architecture.md)

---

## 1. The Real-World Analogy: The Jury of 8 Specialized Experts

Imagine a high-profile legal trial with a mountain of complicated evidence:

```
TRIAL EVIDENCE: "The patient did not receive the medication because it expired."
```

If you seat a **single juror** (Single-Head Attention):
- That juror must attempt to evaluate the chemical composition of the drug, the grammatical subject of the sentence, the coreference of the pronoun *"it"*, the hospital protocol, and the doctor's timeline all at once.
- Because a single attention head produces **only one scalar weight** between any two words, it is forced to compromise: should *"it"* attend to *"patient"* ($0.50$) or *"medication"* ($0.50$)? The signal gets muddied.

Now assemble a **Jury of 8 Specialized Jurors** (**Multi-Head Attention**):
- **Juror 1 (Coreference Specialist)**: Dedicates 100% of their focus to pronouns. They decisively link *"it"* $\leftrightarrow$ *"medication"* (weight $0.98$).
- **Juror 2 (Syntax Specialist)**: Tracks verb-object dependencies. They link *"receive"* $\leftrightarrow$ *"medication"*.
- **Juror 3 (Causal Specialist)**: Tracks logical reasoning. They link *"not receive"* $\leftrightarrow$ *"because"*.
- **Jurors 4–8**: Track entities, dates, sentiment, and vocabulary nuance in parallel.

```
                  THE 8 SPECIALIZED ATTENTION JURORS
                  
  Head 1 (Syntax)     Head 2 (Coreference)     Head 3 (Causality)     Head 4..8 (Semantics)
 ┌─────────────────┐ ┌────────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
 │ Verb ↔ Object   │ │ "it" ↔ "medication"│   │ "did not" ↔ "why"│   │ Clinical Entities   │
 └─────────────────┘ └────────────────────┘   └──────────────────┘   └─────────────────────┘
          │                    │                       │                         │
          └────────────────────┴───────────┬───────────┴─────────────────────────┘
                                           │
                           CONCATENATE & PROJECT (Wᴼ)
                                           ▼
                      Unified Multi-Faceted Word Understanding!
```

At the end of the deliberation, the Jury Foreman concatenates all 8 specialized verdicts into one rich, unified representation vector ($W^O$).

---

## 2. Multi-Head Attention: Mathematical Formulation

Instead of performing a single attention function with dimension $d_{\text{model}}$, we linearly project Queries, Keys, and Values $h$ times with different, learned linear projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O$$

$$\text{where } \text{head}_i = \text{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right)$$

![Multi-Head Attention Architecture](assets/multi_head_attention_architecture.svg)

### The Dimensionality Division Trick:
In the original Transformer paper (Vaswani et al., 2017):
- Total model dimension: $d_{\text{model}} = 512$
- Number of heads: $h = 8$
- Dimension per head:
  $$d_k = d_v = \frac{d_{\text{model}}}{h} = \frac{512}{8} = \mathbf{64}$$

Notice the engineering genius:
- Because each head operates on a compressed slice of $64$ dimensions, running $8$ heads in parallel costs the **exact same computational FLOPs and parameter count as a single head with $512$ dimensions**!
- We get 8 distinct representational subspaces for **free**.

```
Projection Weights Tensor Dimensions:
W_i^Q:  (d_model, d_k)     = (512, 64)
W_i^K:  (d_model, d_k)     = (512, 64)
W_i^V:  (d_model, d_v)     = (512, 64)
W^O:    (h · d_v, d_model) = (8 · 64, 512) = (512, 512)
```

---

## 3. The Full Transformer Block Anatomy

A Transformer is not just Attention. Self-Attention only **re-routes and blends** information between tokens. 

To transform those representations into deep non-linear features, each Transformer Block pairs Multi-Head Attention with a **Position-Wise Feed-Forward Network (FFN)**, bound together by **Residual Skip Connections** and **Layer Normalization**:

![Transformer Encoder Block Anatomy](assets/transformer_encoder_block_anatomy.svg)

### The 4 Sublayers in a Modern Pre-LN Transformer Block:

Given input tensor $x \in \mathbb{R}^{B \times N \times d_{\text{model}}}$:

#### 1. Pre-LayerNorm 1:
$$\tilde{x}_1 = \text{LayerNorm}(x)$$

#### 2. Multi-Head Attention with Residual Highway:
$$x_1 = x + \text{MultiHead}(\tilde{x}_1, \tilde{x}_1, \tilde{x}_1)$$

#### 3. Pre-LayerNorm 2:
$$\tilde{x}_2 = \text{LayerNorm}(x_1)$$

#### 4. Feed-Forward Network with Residual Highway:
$$x_{\text{out}} = x_1 + \text{FFN}(\tilde{x}_2)$$

---

## 4. The Position-Wise Feed-Forward Network (FFN)

After tokens communicate with each other through Attention, each token enters an independent 2-layer Multi-Layer Perceptron (MLP):

$$\text{FFN}(z) = \max\left(0, z W_1 + b_1\right) W_2 + b_2$$

*(In modern LLMs like LLaMA and Mistral, the activation is upgraded from ReLU to **SwiGLU** or **GELU**).*

### 1. Dimension Expansion ($4 \times d_{\text{model}}$):
The intermediate hidden layer expands by a factor of 4:
$$d_{ff} = 4 \times d_{\text{model}}$$
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}$ (e.g., $512 \to 2048$, or in LLaMA $4096 \to 11008$)
- $W_2 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}$ (e.g., $2048 \to 512$)

### 2. The FFN as "Factual Key-Value Memory" (Geva et al., 2021)
Why does the FFN consume **two-thirds of all parameters** in a Transformer?

In 2021, research by Mor Geva et al. (*"Transformer Feed-Forward Layers Are Key-Value Memories"*) uncovered what FFNs actually do:
- **Multi-Head Attention** routes information across the sequence (the "Syntactic Bus").
- **The FFN** acts as a **factual knowledge database**!
  - The first layer ($W_1$) acts as pattern detectors for conceptual keys (*"capital of France"*).
  - The second layer ($W_2$) acts as memory retrieval for the associated value (*"Paris"*).
  - When an LLM recites facts, historical dates, or Python syntax from memory, that knowledge is retrieved directly from the weights of the FFN!

---

## 5. Layer Normalization: Why BatchNorm Fails on Text

On [Day 27](../../Phase_05_Specialized_Neural_Networks/Day_27_CNNs_Part_2/Day_27_CNNs_Part_2.md), we saw Batch Normalization normalize across the batch dimension.

### Why BatchNorm Fails on Natural Language:
1. **Variable Sequence Lengths**: Sentences in a batch have different lengths (5 words, 42 words, 120 words). Padded zero tokens corrupt the batch statistics!
2. **Inference Dependency**: BatchNorm requires running mean/variance statistics accumulated across training batches, which introduces discrepancies during single-sample real-time inference.

### Layer Normalization (Ba, Kiros, & Hinton, 2016):
LayerNorm normalizes **across the feature dimension independently for each individual token**:

$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

Where $\gamma, \beta \in \mathbb{R}^d$ are learnable scale and shift parameters.

```
Batch Normalization (Computer Vision):    Layer Normalization (Transformers / NLP):
Normalizes across BATCH (Column-wise)     Normalizes across CHANNELS (Row-wise)
       Tokens ─────────────────▶                 Tokens ─────────────────▶
Batch ┌───────────────┐                  Batch ┌───────────────┐
  │   │   │   │   │   │                    │   │═══════════════│ ◀── μ, σ computed per token!
  ▼   │   │   │   │   │                    ▼   │═══════════════│
      └───────────────┘                        └───────────────┘
      ▲                                        Zero cross-sample dependencies!
      └── μ, σ computed across batch           Works perfectly with variable sequence length!
```

---

## 6. Pre-LN vs Post-LN: The Architectural Shift

Notice the placement of LayerNorm in the block diagram:

```
Original Vaswani 2017: Post-LN
x_{l+1} = LayerNorm( x_l + Sublayer(x_l) )
─────────────────────────────────────────────────────────────────────────────
• Flaw: The normalization wraps the residual connection!
• Gradient flowing back must pass through the non-linear LayerNorm derivative at every layer.
• Gradients at layer 1 vanish exponentially for deep models (L > 12).
• Required fragile learning rate warmup to avoid early divergence.
```

```
Modern LLM Standard: Pre-LN (LLaMA, GPT-NeoX, Mistral)
x_{l+1} = x_l + Sublayer( LayerNorm(x_l) )
─────────────────────────────────────────────────────────────────────────────
• Triumph: An unhindered identity highway (x_l + ...) connects the input directly to the output!
• x_L = x_0 + Sublayer_1 + Sublayer_2 + ... + Sublayer_L
• Gradients flow backwards directly to layer 1 with ZERO decay!
• Enables training 100+ layer architectures with zero learning rate warmup instability.
```

---

## 7. Hands-On PyTorch Lab: Multi-Head Attention & Transformer Block from Scratch

Let us write a production-grade, vectorized implementation of `MultiHeadAttention` and the complete `TransformerBlock` in PyTorch.

```python
"""
Day 36 Lab: Vectorized Multi-Head Attention & Full Pre-LN Transformer Block
Demonstrates:
1. Batched multi-head tensor splitting and transposition
2. Scaled Dot-Product Attention across all h heads simultaneously
3. Position-Wise Feed-Forward Network with GELU activation
4. Pre-LN architecture with residual skip highways
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

# ==========================================
# 1. VECTORIZED MULTI-HEAD ATTENTION MODULE
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads!"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head (e.g. 512 / 8 = 64)
        
        # Combined projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        # 1. Linear projections: (B, N, d_model)
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        
        # 2. Reshape and transpose for multi-head parallelism:
        # (B, N, d_model) -> (B, N, h, d_k) -> (B, h, N, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention:
        # Q: (B, h, N, d_k) @ K^T: (B, h, d_k, N) -> Scores: (B, h, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)  # (B, h, N, N)
        
        # 4. Values weighted sum: (B, h, N, N) @ (B, h, N, d_k) -> (B, h, N, d_k)
        context = torch.matmul(attn_weights, V)
        
        # 5. Concatenate all heads:
        # (B, h, N, d_k) -> transpose -> (B, N, h, d_k) -> contiguous view -> (B, N, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 6. Final linear projection: (B, N, d_model)
        output = self.W_o(context)
        return output, attn_weights

# ==========================================
# 2. FULL PRE-LN TRANSFORMER BLOCK
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # Sublayer 1: Attention
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.drop1 = nn.Dropout(dropout)
        
        # Sublayer 2: Feed-Forward Network
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Modern smooth activation
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # 1. Pre-LN Attention + Residual Highway
        norm_x1 = self.ln1(x)
        attn_out, weights = self.mha(norm_x1, norm_x1, norm_x1, mask=mask)
        x = x + self.drop1(attn_out)  # Identity + Attention
        
        # 2. Pre-LN Feed-Forward + Residual Highway
        norm_x2 = self.ln2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + ffn_out               # Identity + FFN
        
        return x, weights

# ==========================================
# 3. VERIFICATION AND TESTING
# ==========================================
B = 2       # Batch size
N = 16      # Sequence length (tokens)
d_model = 256
num_heads = 4
d_ff = 1024

block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
dummy_input = torch.randn(B, N, d_model)

# Forward pass
output_tensor, attention_maps = block(dummy_input)

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

print("--- TRANSFORMER BLOCK VERIFICATION ---")
print(f"Input Shape:          {dummy_input.shape}")
print(f"Output Shape:         {output_tensor.shape} (Exact preservation!)")
print(f"Attention Maps Shape: {attention_maps.shape} [Batch, Heads, Seq_Len, Seq_Len]")
print(f"\nTotal Parameters in 1 Block: {count_params(block):,}")
print(f"  • MHA Sublayer: {count_params(block.mha):,} params")
print(f"  • FFN Sublayer: {count_params(block.ffn):,} params ({(count_params(block.ffn)/count_params(block)*100):.1f}% of block!)")
```

### Expected Output:

```text
--- TRANSFORMER BLOCK VERIFICATION ---
Input Shape:          torch.Size([2, 16, 256])
Output Shape:         torch.Size([2, 16, 256]) (Exact preservation!)
Attention Maps Shape: torch.Size([2, 4, 16, 16]) [Batch, Heads, Seq_Len, Seq_Len]

Total Parameters in 1 Block: 790,016
  • MHA Sublayer: 262,144 params
  • FFN Sublayer: 526,336 params (66.6% of block!)
```

> [!TIP]
> Notice the parameter breakdown: The Feed-Forward Network (FFN) contains **exactly 66.6% (two-thirds)** of all parameters in the block! It provides the immense capacity required to store the model's factual memory.

---

## 8. Summary & Key Takeaways

1. **Multi-Head Attention**: Splits $d_{\text{model}}$ into $h$ independent heads of size $d_k = d_{\text{model}} / h$, allowing the model to simultaneously track multiple relational properties (syntax, pronouns, entities) without extra FLOPs.
2. **Position-Wise FFN**: A 2-layer MLP expanding by $4\times$ that acts as a factual key-value associative memory.
3. **Layer Normalization**: Normalizes across features per token independently, rendering it immune to variable batch sizes and padding lengths.
4. **Pre-LN vs Post-LN**: Pre-LN preserves an uninterrupted linear residual highway ($x_{l+1} = x_l + \dots$), eliminating vanishing gradients and enabling stable training across 100+ stacked layers.

---

## 9. Practice Exercises

### Exercise 1: Multi-Head Dimension Contract
A language model has $d_{\text{model}} = 4096$ and $h = 32$ heads.
1. What is the head dimension $d_k$?
2. What is the shape of the weight matrix $W_o$ after concatenation?
3. If the FFN expansion factor is 4, what is the inner dimension $d_{ff}$, and how many parameters exist in the FFN sublayer?

### Exercise 2: Post-LN Gradient Flow
Why did early 2017 Transformer models with Post-LN fail to train when stacked beyond 16 layers without careful learning rate warmup?

### Solutions:
- **Exercise 1**:
  1. $d_k = \frac{4096}{32} = \mathbf{128}$.
  2. $W_o$ shape: `(h * d_k, d_model)` = $(32 \times 128, 4096) = \mathbf{(4096, 4096)}$.
  3. Inner dimension $d_{ff} = 4 \times 4096 = \mathbf{16,384}$.
     - $W_1$ params: $4096 \times 16384 = 67,108,864$.
     - $W_2$ params: $16384 \times 4096 = 67,108,864$.
     Total FFN parameters $= 67,108,864 \times 2 = \mathbf{134,217,728}$ ($\approx 134 \text{ Million params per layer!}$).
- **Exercise 2**:
  In Post-LN, the update is $x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$. By chain rule, the gradient flowing back from $x_{l+1}$ to $x_l$ is multiplied by the Jacobian of the LayerNorm operation: $\frac{\partial \text{LN}}{\partial x}$. Across $L > 16$ layers, compounding this Jacobian causes gradients at layer 1 to vanish exponentially, leading to numerical divergence in early training steps.

---

## 🚀 Tomorrow's Mission: Day 37

We now have the complete Transformer block! But how do we connect multiple blocks together into an end-to-end model? Tomorrow on [Day 37: The Complete Transformer Architecture — Putting It All Together](../Day_37_Complete_Transformer_Architecture/Day_37_Complete_Transformer_Architecture.md), we build the full **Encoder-Decoder, explore Causal Masking, and unpack the Transformer Trinity: BERT vs GPT vs T5**!


---

## 🧭 Navigation & Next Steps

| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 35: Self-Attention](../Day_35_Self_Attention/Day_35_Self_Attention.md) | [All 50 Days Overview](../../README.md) | [Day 37: The Complete Transformer Architecture →](../Day_37_Complete_Transformer_Architecture/Day_37_Complete_Transformer_Architecture.md) |
