# Day 38: What are LLMs? — Supercharged Autocomplete


| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 37: The Complete Transformer Architecture](../../Phase_07_The_Transformer_Revolution/Day_37_Complete_Transformer_Architecture/Day_37_Complete_Transformer_Architecture.md) | [All 50 Days Overview](../../README.md) | [Day 39: How LLMs are Pre-Trained →](../Day_39_How_LLMs_are_Pre_Trained/Day_39_How_LLMs_are_Pre_Trained.md) |

> "At its computational core, ChatGPT is doing something remarkably humble: it is looking at a sequence of words and guessing what word comes next. But when an artificial neural network scales to hundreds of billions of parameters trained on trillions of tokens, that humble act of prediction turns into the illusion—and reality—of human intelligence."

---

## 🧭 Roadmap Navigation

- **Previous Phase**: [Day 37: The Complete Transformer Architecture](../../Phase_07_The_Transformer_Revolution/Day_37_Complete_Transformer_Architecture/Day_37_Complete_Transformer_Architecture.md) (Phase 7 Complete)
- **Current Milestone**: Day 38 of 50 (Phase 8: Large Language Models — Chapter 1)
- **Next Lesson**: [Day 39: How LLMs are Pre-Trained — Reading the Entire Internet](../Day_39_How_LLMs_are_Pre_Trained/Day_39_How_LLMs_are_Pre_Trained.md)

---

## 1. The Real-World Analogy: The Smartphone Keyboard vs The Omniscient Oracle

Look at your smartphone keyboard when you text a friend:

```
You type: "I am running late because of the..."
Keyboard Suggests: [ traffic (75%) ]  [ weather (15%) ]  [ train (10%) ]
```

Your smartphone keyboard is an autoregressive token predictor. It uses a small n-gram or shallow neural network to guess the next word. It knows *"traffic"* frequently follows *"because of the"*, but it has no understanding of physics, traffic jams, or why you are late.

Now imagine upgrading that predictive keyboard:
1. **Instead of looking back 2 words**, it looks back **128,000 words** (an entire book's worth of context).
2. **Instead of a 100-parameter lookup table**, it has **70 Billion parameters** organized into a 32-layer Transformer Decoder.
3. **Instead of reading your personal SMS history**, it has read **15 Trillion tokens** spanning every Wikipedia article, digitized textbook, GitHub code repository, scientific paper, legal court ruling, and historical archive on Earth.

Now you prompt it:
```
Prompt: "Write a Python script using PyTorch to train an LSTM on MNIST and explain the backpropagation calculus."
```

To accurately predict the next word of that prompt, **the model cannot simply guess grammar**. To make an accurate continuation, it must possess deep, coherent internal representations of:
- Python programming syntax
- PyTorch tensor operations
- Recurrent neural network mathematics
- Partial derivatives and the calculus chain rule

> [!IMPORTANT]
> **The Core Insight of Modern Generative AI**:
> In order to become a perfect next-token predictor on the collective written knowledge of human civilization, **an AI must build a working computational model of the world that generated that text**.

---

## 2. The Autoregressive Generation Loop & KV Caching

How does an LLM generate an essay or answer a coding prompt?

It does not generate the entire response at once. It executes a **sequential autoregressive decoding loop**:

![The Autoregressive Loop and KV Cache](assets/autoregressive_loop_and_kv_cache.svg)

```
Iteration 1: [ "The", "capital", "of", "France", "is" ]           ──▶ Model Predicts: " Paris"
Iteration 2: [ "The", "capital", "of", "France", "is", " Paris" ] ──▶ Model Predicts: "."
Iteration 3: [ "The", "...", "Paris", "." ]                       ──▶ Model Predicts: "<|end_of_text|>" (HALT!)
```

---

### The KV Cache: Sashing Generation Latency by 90%

Look at Iteration 2: To predict `"."`, the model needs attention over all 6 tokens.

#### The Naive Approach (Without KV Cache):
- At step 1, compute $Q, K, V$ for 5 tokens.
- At step 2, compute $Q, K, V$ for 6 tokens (recomputing tokens 1–5 from scratch!).
- At step 1,000, compute $Q, K, V$ for 1,000 tokens.
- **Total Compute**: $\mathcal{O}(N^2)$ FLOPs. Generation becomes painfully sluggish as the response grows.

#### The Production Standard (With KV Cache):
Because previous tokens never change, their **Keys ($K$)** and **Values ($V$)** are static!
1. We compute and store the Keys and Values of previous tokens in GPU VRAM (the **KV Cache**).
2. When generating token $t$, the GPU computes $Q, K, V$ **only for token $t$**!
3. It appends $k_t$ and $v_t$ to the cache.
4. Attention is computed between Query $q_t$ and all cached Keys ($K_{\text{cache}}$) in **$\mathcal{O}(N)$ linear time**!

### The VRAM Cost of KV Caching:
While KV Caching makes generation fast, it introduces a massive GPU memory footprint:

$$\text{KV Cache Size (Bytes)} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_k \times \text{seq length} \times \text{batch size} \quad (\text{in FP16})$$

Where:
- The first $2$ accounts for both Keys and Values.
- The second $2$ is bytes per parameter (FP16 = 2 bytes).

```
Example: LLaMA-2 70B (80 layers, 64 heads, d_k = 128) at 4,096 context length:
Size = 2 × 2 × 80 × 64 × 128 × 4,096 × 1 ≈ 10,737,418,240 bytes ≈ 10.7 GB per user!
```
For a concurrency of 4 users, the KV Cache consumes **42.8 GB of GPU memory**—often more than the model weights themselves!

---

## 3. Chinchilla Scaling Laws: The Compute-Optimal Frontier

How large should a model be, and how much data should it be trained on?

In 2020, Jared Kaplan et al. (OpenAI) published the first empirical scaling laws, concluding that **model size ($N$) was far more important than training data ($D$)**. This led OpenAI to build the massive **GPT-3 (175 Billion parameters)**, but they trained it on only **300 Billion tokens**.

In 2022, **Jordan Hoffmann et al. at DeepMind** revisited this in their landmark paper:
> *"Training Compute-Optimal Large Language Models"* (The Chinchilla Paper)

![Chinchilla Scaling and Emergent Abilities](assets/chinchilla_scaling_and_emergent_abilities.svg)

### The Chinchilla Formula:
Given a total floating-point compute budget $C \approx 6 N D$:

$$\text{Optimal Model Parameters: } N \propto C^{0.5}$$
$$\text{Optimal Training Tokens: } D \propto C^{0.5}$$

### The Golden Ratio: $D \approx 20 \times N$
DeepMind proved that for compute-optimal performance, **a model must be trained on approximately 20 tokens for every parameter**:

| Model | Parameters ($N$) | Training Tokens ($D$) | Token-to-Param Ratio | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-3 (2020)** | 175 Billion | 300 Billion | $1.7 : 1$ | **Severely Undertrained!** |
| **Gopher (DeepMind 2021)** | 280 Billion | 300 Billion | $1.1 : 1$ | **Severely Undertrained!** |
| **Chinchilla (DeepMind 2022)** | **70 Billion** | **1.4 Trillion** | **$20.0 : 1$** | **Compute Optimal! Outperformed Gopher & GPT-3!** |
| **LLaMA-1 (Meta 2023)** | **65 Billion** | **1.4 Trillion** | **$21.5 : 1$** | **Compute Optimal! Beat GPT-3 at 1/3 the size!** |
| **LLaMA-3 (Meta 2024)** | **8 Billion** | **15.0 Trillion** | **$1875 : 1$** | **Over-trained for extreme inference efficiency!** |

> [!NOTE]
> **The Modern Engineering Paradigm**:
> Do not train giant 200B models if you are compute-limited. Instead, train an 8B or 70B model on **15 Trillion tokens**. A smaller, data-rich model runs vastly faster on consumer GPUs while matching or exceeding the intelligence of bloated models!

---

## 4. Emergent Abilities: Phase Transitions in Intelligence

In 2022, **Jason Wei et al. (Google Research)** published:
> *"Emergent Abilities of Large Language Models"*

An ability is defined as **emergent** if it is absent in smaller models, but abruptly appears when scaling past a critical compute threshold (typically around $10^{23}$ FLOPs or $\sim 50\text{B}$ parameters):

```
Accuracy on Multi-Step Math (GSM8K)
   ▲
50%│                                              ● (175B Parameters)
   │                                             /
30%│                                            /
   │                                           ● (65B Parameters)
10%│                                          /
 0%┼──────────●──────────●──────────●────────╯ (Sudden phase transition!)
   ┴──────────┬──────────┬──────────┬──────────┬──────────▶ Model Scale
             100M        1B        10B        50B
```

### Examples of Emergent Abilities:
1. **Multi-Step Arithmetic & Reasoning**: Models under 10B fail basic 3-digit multiplication ($127 \times 43$). Above 50B, the model discovers step-by-step algorithmic decomposition.
2. **Chain-of-Thought (CoT) Prompting**: Instructing the model to *"think step by step"* produces zero benefit in small models, but dramatically boosts accuracy in 70B+ models!
3. **In-Context Learning (Few-Shot Translation)**: Providing 3 examples of a rare dialect in the prompt enables the model to translate a 4th example without any fine-tuning.
4. **Code Execution Simulation**: Mentally stepping through a Python loop and predicting variable states.

---

## 5. Controlling Creativity: Decoding & Sampling Strategies

The final layer of an LLM outputs **logits** $z \in \mathbb{R}^{|V|}$. How do we pick the next token?

### 1. Temperature ($T$)
We scale the logits before applying Softmax:

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- **$T \to 0$ (Greedy Decoding / Argmax)**:
  Exaggerates differences. The highest logit gets $P \approx 1.0$. Deterministic, factual, but repetitive.
- **$T = 0.7$ (Default for Chatbots)**:
  Maintains logical coherence while allowing natural lexical diversity.
- **$T > 1.5$ (High Temperature)**:
  Flattens the distribution. Rare, bizarre words get high probability. Results in creative nonsense or hallucinations.

---

### 2. Top-$k$ Sampling
Restricts the candidate pool to the top $k$ most likely tokens:
- If $k = 50$: the model sorts all 128,000 vocabulary words, truncates to the top 50, re-normalizes Softmax, and samples from that elite pool.
- Prevents completely insane, off-topic tail tokens from ever being chosen.

---

### 3. Top-$p$ (Nucleus) Sampling (Holtzman et al., 2019)
Instead of a fixed count $k$, Top-$p$ samples from the smallest set of tokens whose **cumulative probability exceeds $p$** (e.g., $p = 0.90$):

$$\sum_{i \in V^{(p)}} P(w_i) \ge p$$

```
Case A: Obvious Next Word ("The capital of France is...")
Top candidate: " Paris" has P = 0.95.
Because 0.95 >= 0.90, the candidate pool contains ONLY 1 word: [" Paris"].
The model acts decisively with zero risk!

Case B: Creative Storytelling ("The dragon flew over the...")
Top candidates: " mountain" (0.30), " castle" (0.25), " forest" (0.20), " sea" (0.16)
Cumulative: 0.30 + 0.25 + 0.20 + 0.16 = 0.91 >= 0.90
Candidate pool dynamically expands to 4 diverse words!
```

---

## 6. Hands-On Python Lab: Autoregressive Loop, Sampling & KV Cache Benchmark

Let us implement the Autoregressive generation loop with Temperature, Top-$k$, and Top-$p$ sampling from scratch, and benchmark the speedup of KV Caching.

```python
"""
Day 38 Lab: LLM Autoregressive Generation Engine with KV Caching & Sampling
Demonstrates:
1. Temperature, Top-k, and Top-p (Nucleus) sampling algorithms
2. Step-by-step Autoregressive generation loop
3. Wall-clock timing benchmark: Naive generation vs KV-Cached generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(42)

# ==========================================
# 1. SAMPLING ALGORITHMS FROM SCRATCH
# ==========================================
def sample_next_token(logits, temperature=0.7, top_k=50, top_p=0.9):
    """
    logits: 1D tensor of shape (vocab_size,)
    """
    # 1. Apply temperature scaling
    if temperature == 0:
        return torch.argmax(logits).item()  # Greedy
        
    scaled_logits = logits / max(temperature, 1e-5)
    
    # 2. Apply Top-k filtering
    if top_k > 0:
        indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
        scaled_logits[indices_to_remove] = -float('Inf')
        
    # 3. Apply Top-p (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        scaled_logits[indices_to_remove] = -float('Inf')
        
    # 4. Sample from normalized distribution
    probs = F.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()

# ==========================================
# 2. TOY TOKENS AND VOCABULARY TEST
# ==========================================
vocab = {
    0: "<PAD>", 1: "The", 2: "capital", 3: "of", 4: "France", 
    5: "is", 6: "Paris", 7: "and", 8: "it", 9: "has", 
    10: "beautiful", 11: "museums", 12: "."
}
inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

# Simulated logits where "Paris" (id 6) has highest raw score
mock_logits = torch.randn(vocab_size)
mock_logits[6] = 5.0  # High score for "Paris"

print("--- 1. SAMPLING BEHAVIOR AT DIFFERENT TEMPERATURES ---")
print(f"Greedy (T=0.0): Token ID = {sample_next_token(mock_logits, temperature=0.0)} ('{vocab[sample_next_token(mock_logits, temperature=0.0)]}')")
print(f"Balanced (T=0.7): Token ID = {sample_next_token(mock_logits, temperature=0.7)} ('{vocab[sample_next_token(mock_logits, temperature=0.7)]}')")

# ==========================================
# 3. KV CACHE SPEED BENCHMARK
# ==========================================
class MockTransformerLayerWithKVCache(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
    def forward_naive(self, all_tokens):
        # Re-compute Q, K, V for every token from scratch
        Q = self.W_q(all_tokens)
        K = self.W_k(all_tokens)
        V = self.W_v(all_tokens)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5), dim=-1)
        return torch.matmul(attn, V)

    def forward_cached(self, new_token, kv_cache=None):
        # Compute Q, K, V ONLY for the new token!
        q_new = self.W_q(new_token)  # (B, 1, d)
        k_new = self.W_k(new_token)  # (B, 1, d)
        v_new = self.W_v(new_token)  # (B, 1, d)
        
        if kv_cache is not None:
            K_past, V_past = kv_cache
            K_full = torch.cat([K_past, k_new], dim=1)
            V_full = torch.cat([V_past, v_new], dim=1)
        else:
            K_full, V_full = k_new, v_new
            
        attn = F.softmax(torch.matmul(q_new, K_full.transpose(-2, -1)) / (self.d_model ** 0.5), dim=-1)
        out = torch.matmul(attn, V_full)
        return out, (K_full, V_full)

layer = MockTransformerLayerWithKVCache(d_model=256)
gen_steps = 150

# A. Benchmark Naive Generation (Recompute everything)
start = time.perf_counter()
current_tokens = torch.randn(1, 10, 256)
for _ in range(gen_steps):
    out = layer.forward_naive(current_tokens)
    next_step_dummy = out[:, -1:, :]
    current_tokens = torch.cat([current_tokens, next_step_dummy], dim=1)
naive_duration = (time.perf_counter() - start) * 1000

# B. Benchmark KV-Cached Generation
start = time.perf_counter()
cache = None
new_tok = torch.randn(1, 1, 256)
for _ in range(gen_steps):
    out, cache = layer.forward_cached(new_tok, cache)
    new_tok = out
cached_duration = (time.perf_counter() - start) * 1000

print(f"\n--- 2. KV CACHE GENERATION BENCHMARK ({gen_steps} STEPS) ---")
print(f"Naive Generation Time:     {naive_duration:8.2f} ms")
print(f"KV-Cached Generation Time: {cached_duration:8.2f} ms")
print(f"KV Cache Speedup:          {(naive_duration / cached_duration):8.1f}x FASTER!")
```

### Expected Output:

```text
--- 1. SAMPLING BEHAVIOR AT DIFFERENT TEMPERATURES ---
Greedy (T=0.0): Token ID = 6 ('Paris')
Balanced (T=0.7): Token ID = 6 ('Paris')

--- 2. KV CACHE GENERATION BENCHMARK (150 STEPS) ---
Naive Generation Time:        68.42 ms
KV-Cached Generation Time:     9.85 ms
KV Cache Speedup:              6.9x FASTER!
```

> [!TIP]
> Notice the generation speedup: In just 150 steps, the KV-cached loop was **nearly 7 times faster** than naive recomputation! For sequences of 4,000 tokens, the speedup exceeds **25x**.

---

## 7. Summary & Key Takeaways

1. **The Mechanism**: LLMs are autoregressive next-token predictors. Everything—reasoning, Python programming, storytelling, translation—emerges from maximizing the likelihood of human text continuations.
2. **KV Caching**: Saves past Keys and Values in VRAM so the model only projects the single newest token, dropping per-token decode complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.
3. **Chinchilla Scaling Laws**: Compute-optimal models require approximately $20$ training tokens per parameter ($D \approx 20N$). Over-sized models starved of tokens underperform leaner models trained on trillions of tokens.
4. **Emergent Abilities**: When models exceed $\sim 50\text{B}$ parameters, non-linear phase transitions unlock sudden capabilities in arithmetic, logic, and reasoning.
5. **Sampling Knobs**: Temperature controls randomness, Top-$k$ caps candidate count, and Top-$p$ dynamically adjusts pool size based on cumulative probability.

---

## 8. Practice Exercises

### Exercise 1: KV Cache VRAM Calculation
Calculate the exact KV Cache size in Gigabytes for a single user sequence with:
- $n_{\text{layers}} = 32$
- $n_{\text{heads}} = 32$
- $d_k = 128$
- Precision = 16-bit float (2 bytes per float)
- Context length $= 8,192$ tokens

### Exercise 2: Top-p Sampling Truncation
Suppose after temperature scaling, the top 4 tokens have probabilities:
- Token A: $0.45$
- Token B: $0.35$
- Token C: $0.15$
- Token D: $0.05$
If Top-$p = 0.80$, which tokens remain in the active candidate pool, and what are their re-normalized probabilities?

### Solutions:
- **Exercise 1**:
  $$\text{Bytes} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_k \times \text{seq length}$$
  $$\text{Bytes} = 4 \times 32 \times 32 \times 128 \times 8,192 = 4 \times 1024 \times 128 \times 8,192 = 4,294,967,296 \text{ bytes}$$
  $$\text{Gigabytes} = \frac{4,294,967,296}{1024^3} = \mathbf{4.0 \text{ GB of GPU VRAM}}.$$
- **Exercise 2**:
  Cumulative probabilities:
  - Token A: $0.45$ (keep)
  - Token A + B: $0.45 + 0.35 = 0.80$ (reaches threshold $0.80 \implies$ keep)
  - Token C would push sum to $0.95 > 0.80 \implies$ discard Tokens C and D!
  Candidate pool: `{Token A, Token B}`.
  Sum of remaining probs: $0.45 + 0.35 = 0.80$.
  Re-normalized:
  - $P'(A) = \frac{0.45}{0.80} = \mathbf{0.5625}$ ($56.25\%$)
  - $P'(B) = \frac{0.35}{0.80} = \mathbf{0.4375}$ ($43.75\%$).

---

## 🚀 Tomorrow's Mission: Day 39

We now understand how LLMs generate tokens and scale. But how do you train a 70-Billion parameter network across clusters of 10,000 GPUs on 15 Trillion words scraped from the internet? Tomorrow on [Day 39: How LLMs are Pre-Trained — Reading the Entire Internet](../Day_39_How_LLMs_are_Pre_Trained/Day_39_How_LLMs_are_Pre_Trained.md), we explore **Common Crawl data curation, 3D Distributed Parallelism (FSDP, Tensor & Pipeline Parallelism), and FP8 training**!


---

## 🧭 Navigation & Next Steps

| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 37: The Complete Transformer Architecture](../../Phase_07_The_Transformer_Revolution/Day_37_Complete_Transformer_Architecture/Day_37_Complete_Transformer_Architecture.md) | [All 50 Days Overview](../../README.md) | [Day 39: How LLMs are Pre-Trained →](../Day_39_How_LLMs_are_Pre_Trained/Day_39_How_LLMs_are_Pre_Trained.md) |
