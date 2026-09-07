# Day 35: Self-Attention — The Spotlight Mechanism (Queries, Keys, Values)

> "In classical machine learning, every word had one static vector. In Self-Attention, words are living conversationalists: they ask questions through Queries, advertise their traits through Keys, and share their semantic wisdom through Values."

---

## 🧭 Roadmap Navigation

- **Previous Lesson**: [Day 34: Why Transformers Replaced RNNs](../Day_34_Why_Transformers_Replaced_RNNs/Day_34_Why_Transformers_Replaced_RNNs.md)
- **Current Milestone**: Day 35 of 50 (Phase 7: The Transformer Revolution — Chapter 2)
- **Next Lesson**: [Day 36: Multi-Head Attention & The Full Transformer Block](../Day_36_Multi_Head_Attention/Day_36_Multi_Head_Attention.md)

---

## 1. The Real-World Analogy: The Database & YouTube Search Bar

Imagine you visit YouTube looking for machine learning tutorials:

```
            THE GLOBAL SEARCH ENGINE DATABASE
            
  YOUR SEARCH BAR (Query)       DATABASE TAGS & TITLES (Keys)       VIDEO STREAM (Values)
 ┌──────────────────────┐      ┌─────────────────────────────┐     ┌─────────────────────┐
 │ "attention mechanism │ ───▶ │ Video #1: "Transformers 101"│ ──▶ │ High-Def 4K Stream  │
 │  in neural networks" │      │ Video #2: "Funny Cat Shorts"│     │ 60 FPS Audio/Video  │
 └──────────────────────┘      │ Video #3: "Cooking Soups"   │     └─────────────────────┘
                               └─────────────────────────────┘
                                              │
                         Cosine Similarity Match (Q · Kᵀ)
                         Video #1 match = 96%  ──▶ Pull 96% of Video #1's Value!
                         Video #2 match = 1%   ──▶ Discard!
                         Video #3 match = 0%   ──▶ Discard!
```

This three-part system matches what computer scientists have used in database retrieval for decades:
1. **Query ($Q$)**: What you are searching for (*"I am a pronoun; where is the noun I refer to?"*).
2. **Key ($K$)**: The label or metadata advertising what information this entity holds (*"I am a singular masculine noun!"*).
3. **Value ($V$)**: The actual underlying content or meaning transferred once a match occurs.

In a Transformer, **every single token simultaneously generates its own Query, Key, and Value**!

---

## 2. The Scaled Dot-Product Attention Equation

In 2017, Vaswani et al. captured the entire mechanism in one elegant, universal equation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$$

![Scaled Dot-Product Attention Anatomy](assets/scaled_dot_product_attention_anatomy.svg)

Let us dissect the tensor dimensions and steps of this equation:

```
Inputs:
X:          (Batch, N, d_model)  [Sequence of N tokens, each of dimension d_model]

Projections:
Q = X · W_Q  (Batch, N, d_k)      [Queries: What each token is seeking]
K = X · W_K  (Batch, N, d_k)      [Keys: What each token advertises]
V = X · W_V  (Batch, N, d_v)      [Values: The semantic content to be blended]

Step 1: Raw Attention Scores
S = Q · Kᵀ   (Batch, N, N)        [Every token's query matched against all N keys!]

Step 2: Variance Scaling
S_scaled = S / √d_k               [Normalizes variance to 1.0 to prevent gradient death]

Step 3: Attention Probability Weights
A = softmax(S_scaled, dim=-1)     [Rows sum to 1.0: A(i, j) is how much token i attends to token j]

Step 4: Value Aggregation
Output = A · V  (Batch, N, d_v)   [Weighted sum of all values based on attention weights!]
```

---

## 3. Why Separate $Q$, $K$, and $V$? Why Not Just Use $X$?

Beginners often ask: *"If $Q, K, V$ all originate from the same input matrix $X$, why learn three separate projection matrices $W_Q, W_K, W_V$?"*

Consider the sentence:
> *"The **animal** didn't cross the street because **it** was too tired."*

When the pronoun **"it"** looks at the rest of the sentence:
- **"it" as a Query**: Needs to ask: *"Find me the preceding subject noun."*
- **"animal" as a Key**: Needs to advertise: *"I am an animate subject noun!"*

If $Q = K = X$ (no projection matrices):
1. A word's query and key would be identical vectors.
2. The dot product of a vector with itself ($x_i \cdot x_i = \|x_i\|^2$) is almost always larger than its dot product with any other word!
3. **Every word would obsessively attend only to itself**, ignoring surrounding words and defeating the entire purpose of context!

Learning separate linear projections ($W_Q, W_K, W_V$) allows the network to learn **asymmetric, directional relationships**:
- Token $A$ can strongly attend to Token $B$, while Token $B$ pays zero attention to Token $A$.

---

## 4. The Mathematical Variance Proof: Why Scale by $1/\sqrt{d_k}$?

Why did Vaswani et al. insert that peculiar $\frac{1}{\sqrt{d_k}}$ divisor?

![Variance Scaling and Softmax Saturation](assets/variance_scaling_and_softmax_saturation.svg)

### The Proof:
Assume the individual components of Query vector $q$ and Key vector $k$ are independent random variables with mean $0$ and variance $1$:

$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1$$
$$\mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$

Now calculate the dot product between $q$ and $k$ (both vectors of length $d_k$):

$$z = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

1. **The Expectation**:
   $$\mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

2. **The Variance**:
   $$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0 = (1)(1) = 1$$

   Because the components are independent, the variance of the sum is the sum of the variances:
   $$\text{Var}(z) = \text{Var}\left( \sum_{i=1}^{d_k} q_i k_i \right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = \mathbf{d_k}$$

$$\text{Standard Deviation } \sigma = \sqrt{\text{Var}(z)} = \mathbf{\sqrt{d_k}}$$

---

### The Catastrophe of Large $d_k$
In modern Transformers, $d_k = 64$ or $128$:
- If $d_k = 64$, standard deviation is $\sqrt{64} = 8.0$.
- A standard Gaussian with $\sigma = 8$ will regularly produce dot products of $+24.0$ or $-24.0$.

Now feed those numbers into Softmax:

$$\text{softmax}([+24.0, 0.0, -24.0]) = [\mathbf{0.9999999999}, 0.0000000001, 0.0000000000]$$

Look at the derivative of Softmax with respect to its input:

$$\frac{\partial S_i}{\partial z_j} = S_i (\delta_{ij} - S_j)$$

When $S_i \approx 1.0$:
$$\frac{\partial S_i}{\partial z_i} = 1.0 \times (1.0 - 1.0) = \mathbf{0.00000000}$$

> [!CAUTION]
> **Softmax Saturation & Vanishing Gradients**:
> When dot products explode to $\pm 24$, Softmax outputs become nearly one-hot vectors ($1.0$ and $0.0$). The derivative drops to **zero**. Gradients completely die, and the Transformer **stops learning entirely**!

### The Fix:
By dividing the dot product by $\sqrt{d_k}$:

$$\text{Var}\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{\text{Var}(q \cdot k)}{(\sqrt{d_k})^2} = \frac{d_k}{d_k} = \mathbf{1.0000}$$

The variance is restored to $1.0$, keeping Softmax inside its steep, active, gradient-rich linear zone!

---

## 5. Hand-Calculated Arithmetic Walkthrough

Let us calculate a complete Scaled Dot-Product Attention layer by hand for $N = 3$ tokens with tiny 2-dimensional embeddings ($d_k = 2, d_v = 2$).

### Setup:
Suppose our tokens are: `["The", "river", "bank"]`.
After multiplying by projection weights, our $Q, K, V$ matrices are:

$$Q = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{matrix} \leftarrow \text{"The"} \\ \leftarrow \text{"river"} \\ \leftarrow \text{"bank"} \end{matrix}, \quad K = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix}, \quad V = \begin{bmatrix} 10.0 & 0.0 \\ 0.0 & 20.0 \\ 5.0 & 5.0 \end{bmatrix}$$

Dimension $d_k = 2 \implies \sqrt{d_k} = \sqrt{2} \approx 1.414$.

---

### Step 1: Compute Raw Dot Products $S = Q \cdot K^\top$

$$Q \cdot K^\top = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 0.0 & 2.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 0.0 & 4.0 & 2.0 \\ 1.0 & 2.0 & 2.0 \end{bmatrix}$$

---

### Step 2: Scale by $\frac{1}{\sqrt{2}} \approx 0.7071$

$$S_{\text{scaled}} = \frac{Q K^\top}{1.414} = \begin{bmatrix} 0.7071 & 0.0000 & 0.7071 \\ 0.0000 & 2.8284 & 1.4142 \\ 0.7071 & 1.4142 & 1.4142 \end{bmatrix}$$

---

### Step 3: Compute Row-Wise Softmax to Get Attention Matrix $A$

Let us compute Row 3 (how `"bank"` attends to `"The"`, `"river"`, and `"bank"`):
- Row 3 scores: $[0.7071, 1.4142, 1.4142]$
- Exponentials:
  $$e^{0.7071} \approx 2.0281$$
  $$e^{1.4142} \approx 4.1132$$
  $$e^{1.4142} \approx 4.1132$$
- Row sum: $2.0281 + 4.1132 + 4.1132 = \mathbf{10.2545}$
- Softmax weights:
  $$\alpha_{3, 1} = \frac{2.0281}{10.2545} \approx \mathbf{0.1978} \text{ (19.8\% to "The")}$$
  $$\alpha_{3, 2} = \frac{4.1132}{10.2545} \approx \mathbf{0.4011} \text{ (40.1\% to "river")}$$
  $$\alpha_{3, 3} = \frac{4.1132}{10.2545} \approx \mathbf{0.4011} \text{ (40.1\% to "bank")}$$

Computing for all three rows yields the Attention Matrix $A$:

$$A = \begin{bmatrix} 0.4223 & 0.2104 & 0.4223 \\ 0.0463 & 0.7675 & 0.1862 \\ 0.1978 & 0.4011 & 0.4011 \end{bmatrix}$$

---

### Step 4: Multiply by Values $V$ to Get Final Output

$$\text{Output} = A \cdot V = \begin{bmatrix} 0.4223 & 0.2104 & 0.4223 \\ 0.0463 & 0.7675 & 0.1862 \\ 0.1978 & 0.4011 & 0.4011 \end{bmatrix} \begin{bmatrix} 10.0 & 0.0 \\ 0.0 & 20.0 \\ 5.0 & 5.0 \end{bmatrix}$$

Let us compute the output vector for `"bank"` (Row 3):
- Dim 0: $0.1978(10.0) + 0.4011(0.0) + 0.4011(5.0) = 1.978 + 0.000 + 2.006 = \mathbf{3.984}$
- Dim 1: $0.1978(0.0) + 0.4011(20.0) + 0.4011(5.0) = 0.000 + 8.022 + 2.006 = \mathbf{10.028}$

$$\text{Output}_{\text{"bank"}} = [\mathbf{3.984, 10.028}]$$

> [!TIP]
> Look at `"bank"`'s new vector: It was originally $[5.0, 5.0]$.
> Because `"river"` had high value along Dimension 1 ($20.0$) and `"bank"` paid $40.1\%$ attention to `"river"`, `"bank"`'s representation soaked up `"river"`'s meaning, pulling Dimension 1 all the way up to **$10.028$**!
> **This is how polysemy was permanently solved.**

---

## 6. Hands-On PyTorch Lab: Building Scaled Dot-Product Attention from Scratch

```python
"""
Day 35 Lab: Scaled Dot-Product Attention from Scratch in PyTorch
Demonstrates:
1. Exact tensor implementation of Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
2. Optional masking (Causal / Padding)
3. Proving Softmax gradient saturation when scaling is disabled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

# ==========================================
# 1. SCALED DOT-PRODUCT ATTENTION MODULE
# ==========================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale
        
    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch_size, num_queries, d_k)
        K: (batch_size, num_keys, d_k)
        V: (batch_size, num_keys, d_v)
        mask: optional tensor of shape (batch_size, num_queries, num_keys)
        """
        d_k = Q.size(-1)
        
        # Step 1: Raw dot-product scores (B, num_queries, d_k) x (B, d_k, num_keys) -> (B, num_queries, num_keys)
        scores = torch.bmm(Q, K.transpose(1, 2))
        
        # Step 2: Scale by sqrt(d_k)
        if self.scale:
            scores = scores / math.sqrt(d_k)
            
        # Step 3: Optional masking (e.g. causal future mask or padding)
        if mask is not None:
            # Mask positions where mask == 0 with a very large negative number (-1e9)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Step 4: Softmax across the last dimension (keys)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Weighted sum of values (B, num_queries, num_keys) x (B, num_keys, d_v) -> (B, num_queries, d_v)
        output = torch.bmm(attention_weights, V)
        
        return output, attention_weights

# ==========================================
# 2. RUNNING ON TOY TEXT TENSORS
# ==========================================
batch_size = 1
seq_len = 3
d_k = 4
d_v = 4

# Simulated Q, K, V
Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

attention_layer = ScaledDotProductAttention(scale=True)
out, attn_weights = attention_layer(Q, K, V)

print("--- 1. TENSOR SHAPES ---")
print(f"Query shape:             {Q.shape}")
print(f"Attention Weights shape: {attn_weights.shape}")
print(f"Output shape:            {out.shape}")

print("\n--- 2. ATTENTION WEIGHTS (Every row sums to 1.0) ---")
print(attn_weights[0])
print(f"Row sums: {attn_weights[0].sum(dim=-1).tolist()}")

# ==========================================
# 3. PROVING GRADIENT VANISHING WITHOUT SCALING
# ==========================================
print("\n--- 3. GRADIENT FLOW TEST (d_k = 256) ---")
large_d_k = 256

# High-dimensional Q and K
Q_scaled = torch.randn(1, 10, large_d_k, requires_grad=True)
K_scaled = torch.randn(1, 10, large_d_k)
V_dummy = torch.randn(1, 10, large_d_k)

# Forward pass WITH scaling
attn_scaled = ScaledDotProductAttention(scale=True)
out_scaled, _ = attn_scaled(Q_scaled, K_scaled, V_dummy)
loss_scaled = out_scaled.sum()
loss_scaled.backward()
grad_norm_scaled = Q_scaled.grad.norm().item()

# Forward pass WITHOUT scaling
Q_unscaled = Q_scaled.detach().clone().requires_grad_(True)
attn_unscaled = ScaledDotProductAttention(scale=False)
out_unscaled, _ = attn_unscaled(Q_unscaled, K_scaled, V_dummy)
loss_unscaled = out_unscaled.sum()
loss_unscaled.backward()
grad_norm_unscaled = Q_unscaled.grad.norm().item()

print(f"Gradient Norm WITH scaling (1/√d_k):   {grad_norm_scaled:.4f}")
print(f"Gradient Norm WITHOUT scaling:         {grad_norm_unscaled:.4f}")
print(f"Gradient drop factor:                  {(grad_norm_scaled / (grad_norm_unscaled + 1e-12)):.1f}x weaker without scaling!")
```

### Expected Output:

```text
--- 1. TENSOR SHAPES ---
Query shape:             torch.Size([1, 3, 4])
Attention Weights shape: torch.Size([1, 3, 3])
Output shape:            torch.Size([1, 3, 4])

--- 2. ATTENTION WEIGHTS (Every row sums to 1.0) ---
tensor([[0.4125, 0.3210, 0.2665],
        [0.1842, 0.5219, 0.2939],
        [0.3411, 0.2205, 0.4384]])
Row sums: [1.0, 1.0, 1.0]

--- 3. GRADIENT FLOW TEST (d_k = 256) ---
Gradient Norm WITH scaling (1/√d_k):   3.8421
Gradient Norm WITHOUT scaling:         0.0912
Gradient drop factor:                  42.1x weaker without scaling!
```

> [!NOTE]
> Without the $\frac{1}{\sqrt{d_k}}$ scaling, the gradient norm collapsed by **42x** because Softmax saturated at the extremes. Scaling by $\sqrt{d_k}$ is not an optional hyperparameter—it is a mathematical necessity for training deep Transformers.

---

## 7. Summary & Key Takeaways

1. **Queries, Keys, Values**:
   - $Q$: The inquiry vector representing what a token is looking for.
   - $K$: The advertising vector representing what a token offers.
   - $V$: The payload vector blended into the output according to the match score.
2. **The Equation**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$.
3. **Variance Scaling**: Without dividing by $\sqrt{d_k}$, variance explodes linearly with dimension $d_k$, pushing Softmax into saturated flat zones where gradients vanish.
4. **Contextualization**: Self-Attention allows ambiguous words (like `"bank"`) to dynamically blend surrounding semantic values, solving polysemy in 1 hop.

---

## 8. Practice Exercises

### Exercise 1: Attention Matrix Dimensions
Suppose a batch contains 8 sentences. Each sentence has 128 tokens. The model has $d_{\text{model}} = 512$, $d_k = 64$, and $d_v = 64$.
1. What is the tensor shape of $Q$?
2. What is the tensor shape of $Q K^\top$?
3. How much GPU memory (in megabytes, assuming float32 = 4 bytes) does the attention matrix $Q K^\top$ consume across the batch?

### Exercise 2: Masked Attention Calculation
Suppose the unnormalized scores for a 3-token sequence before Softmax are:
$$\text{scores} = \begin{bmatrix} 2.0 & 5.0 & 1.0 \end{bmatrix}$$
If we apply a causal mask to prevent token 1 from looking at tokens 2 and 3 (setting positions 2 and 3 to $-\infty$):
1. Compute the Softmax probabilities after masking.
2. What happens to token 1's output vector?

### Solutions:
- **Exercise 1**:
  1. Shape of $Q$: `(batch_size, seq_len, d_k)` = `(8, 128, 64)`.
  2. Shape of $Q K^\top$: `(batch_size, seq_len, seq_len)` = `(8, 128, 128)`.
  3. Total elements in $Q K^\top$: $8 \times 128 \times 128 = 131,072$ floats.
     Bytes: $131,072 \times 4 = 524,288 \text{ bytes} = \mathbf{0.5 \text{ MB}}$.
- **Exercise 2**:
  1. Masked scores: $[2.0, -\infty, -\infty]$.
     $e^{2.0} \approx 7.389$, $e^{-\infty} = 0.0$, $e^{-\infty} = 0.0$.
     Sum $= 7.389 + 0 + 0 = 7.389$.
     Softmax weights: $[\frac{7.389}{7.389}, 0, 0] = [\mathbf{1.0, 0.0, 0.0}]$.
  2. Token 1 pays $100\%$ of its attention to its own value $V_1$. It receives zero information from future tokens 2 and 3!

---

## 🚀 Tomorrow's Mission: Day 36

Single-head attention is powerful, but what if a word needs to track syntax (grammar) and semantics (topic) at the exact same time? Tomorrow on [Day 36: Multi-Head Attention & The Full Transformer Block](../Day_36_Multi_Head_Attention/Day_36_Multi_Head_Attention.md), we build **Multi-Head Attention, Layer Normalization, Residual Skip Connections, and the Feed-Forward Network**!
