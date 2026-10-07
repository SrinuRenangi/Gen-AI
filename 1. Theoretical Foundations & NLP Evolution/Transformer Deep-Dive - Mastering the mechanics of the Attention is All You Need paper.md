# ⚡ Transformer Deep-Dive: Mastering the Mechanics of the "Attention is All You Need" Paper

> **Zero to Hero Gen AI Course — Module 01: Theoretical Foundations & NLP Evolution**
>
> 📅 Module 1 | ⏱️ Estimated Reading Time: 60 minutes | 🎯 Level: Intermediate to Advanced
>
> **Core Objective:** Deconstruct the seminal 2017 paper *"Attention Is All You Need"* by Vaswani et al. Master the foundational anatomy of the Transformer architecture: Scaled Dot-Product Attention, Multi-Head Attention mechanisms, the Bidirectional Encoder stack, the Autoregressive Masked Decoder stack, Cross-Attention (Encoder-Decoder Attention), Layer Normalization, Residual Connections, Sinusoidal Positional Encodings, and Position-wise Feed-Forward Networks.

---

## 📑 Table of Contents

1. [The Historical Context & Paradigm Shift (Vaswani et al., 2017)](#1-the-historical-context--paradigm-shift-vaswani-et-al-2017)
2. [Intuitive Mental Models & Analogies](#2-intuitive-mental-models--analogies)
   - [2.1 The Library Search Engine: Query, Key, and Value](#21-the-library-search-engine-query-key-and-value)
   - [2.2 The Specialist Advisory Board: Multi-Head Attention](#22-the-specialist-advisory-board-multi-head-attention)
   - [2.3 The One-Way Mirror: Encoder vs Causal Decoder](#23-the-one-way-mirror-encoder-vs-causal-decoder)
3. [Scaled Dot-Product Attention: The Mathematical Engine](#3-scaled-dot-product-attention-the-mathematical-engine)
   - [3.1 The Formal Formulation](#31-the-formal-formulation)
   - [3.2 Mathematical Proof: Why Scale by $1/\sqrt{d_k}$?](#32-mathematical-proof-why-scale-by-1sqrtd_k)
   - [3.3 Step-by-Step Matrix Computation Walkthrough](#33-step-by-step-matrix-computation-walkthrough)
4. [Multi-Head Attention (MHA): Learning Diverse Representation Subspaces](#4-multi-head-attention-mha-learning-diverse-representation-subspaces)
   - [4.1 Why Single-Head Attention Falls Short](#41-why-single-head-attention-falls-short)
   - [4.2 The Linear Projection & Splitting Mechanism](#42-the-linear-projection--splitting-mechanism)
   - [4.3 Concatenation and the Output Projection Matrix $W^O$](#43-concatenation-and-the-output-projection-matrix-wo)
   - [4.4 What Do Individual Heads Actually Learn?](#44-what-do-individual-heads-actually-learn)
5. [The Full Transformer Architecture Visualized](#5-the-full-transformer-architecture-visualized)
6. [The Encoder Stack: Contextualizing Input Sequences](#6-the-encoder-stack-contextualizing-input-sequences)
   - [6.1 Encoder Layer Composition ($N = 6$)](#61-encoder-layer-composition-n--6)
   - [6.2 Sub-Layer 1: Bidirectional Multi-Head Self-Attention](#62-sub-layer-1-bidirectional-multi-head-self-attention)
   - [6.3 Sub-Layer 2: Position-wise Feed-Forward Network (FFN)](#63-sub-layer-2-position-wise-feed-forward-network-ffn)
7. [The Decoder Stack: Autoregressive Generation with Guardrails](#7-the-decoder-stack-autoregressive-generation-with-guardrails)
   - [7.1 Decoder Layer Composition ($N = 6$)](#71-decoder-layer-composition-n--6)
   - [7.2 Sub-Layer 1: Masked Multi-Head Self-Attention (Causal Masking)](#72-sub-layer-1-masked-multi-head-self-attention-causal-masking)
   - [7.3 Sub-Layer 2: Multi-Head Cross-Attention (Encoder-Decoder)](#73-sub-layer-2-multi-head-cross-attention-encoder-decoder)
   - [7.4 The Final Linear Classifier & Softmax Projection](#74-the-final-linear-classifier--softmax-projection)
8. [Positional Encoding: Injecting Order into Permutation-Invariant Sets](#8-positional-encoding-injecting-order-into-permutation-invariant-sets)
   - [8.1 Why Transformers Have Zero Inherent Word Order](#81-why-transformers-have-zero-inherent-word-order)
   - [8.2 Sinusoidal Positional Encoding Equations](#82-sinusoidal-positional-encoding-equations)
   - [8.3 The Linear Geometric Transformation Property](#83-the-linear-geometric-transformation-property)
9. [Residual Connections & Layer Normalization (Add & Norm)](#9-residual-connections--layer-normalization-add--norm)
   - [9.1 Deep Gradient Preservation: Residual Pathways](#91-deep-gradient-preservation-residual-pathways)
   - [9.2 Layer Normalization vs Batch Normalization](#92-layer-normalization-vs-batch-normalization)
   - [9.3 Post-LN (Vaswani 2017) vs Pre-LN (Modern LLMs)](#93-post-ln-vaswani-2017-vs-pre-ln-modern-llms)
10. [Architectural Archetypes: Encoder-Only vs Decoder-Only vs Encoder-Decoder](#10-architectural-archetypes-encoder-only-vs-decoder-only-vs-encoder-decoder)
11. [Hands-On Python Lab: Transformer Mechanics from Scratch](#11-hands-on-python-lab-transformer-mechanics-from-scratch)
12. [Curated Video Walkthroughs & Visual Animations](#12-curated-video-walkthroughs--visual-animations)
13. [Self-Assessment & Review Questions](#13-self-assessment--review-questions)
14. [Summary & Key Takeaways](#14-summary--key-takeaways)

---

## 1. The Historical Context & Paradigm Shift (Vaswani et al., 2017)

In June 2017, a team of researchers at Google Brain and Google Research (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin) published a paper titled:

> **"Attention Is All You Need"** *(NeurIPS 2017)*

At the time, state-of-the-art machine translation and natural language processing were entirely dominated by **Recurrent Neural Networks (RNNs)**, specifically stacked bidirectional **LSTMs** and **GRUs** augmented with attention mechanisms.

While effective for short sequences, RNN architectures suffered from an insurmountable structural barrier: **temporal sequential execution**.

$$\text{Time Complexity to process step } T: \quad O(T) \text{ sequential operations}$$

To calculate hidden state $h_t$, an RNN *must* wait for hidden state $h_{t-1}$. This made training on massive web-scale corpora impossible:
- **GPUs sat idle:** Recurrent models could not utilize the massive thousands-of-cores parallel tensor hardware of modern GPUs.
- **Path Length was $O(T)$:** Information between distant words had to survive dozens or hundreds of sequential matrix multiplications, inevitably suffering from residual vanishing gradients or semantic degradation.

### The Radical Proposition

Vaswani et al. proposed an audacious hypothesis:
> **Discard recurrence and convolutions entirely. Rely solely on attention mechanisms to draw global dependencies between input and output.**

The result was the **Transformer**—an architecture that reduced the sequential distance between any two tokens in a sequence of length $T$ from $O(T)$ down to **$O(1)$**, allowing every single token to attend to every other token simultaneously across all GPU cores in a single parallel step.

```
+-----------------------------------------------------------------------------------------+
|                                THE PARADIGM SHIFT (2017)                               |
|                                                                                         |
|   Traditional NLP (RNN/LSTM):     Token 1  ──> Token 2  ──> Token 3  ──> Token 4       |
|                                   [Sequential Loop: Cannot Parallelize on GPUs]         |
|                                                                                         |
|   The Transformer (Vaswani 2017): Token 1 ◄────► Token 2 ◄────► Token 3 ◄────► Token 4  |
|                                   [All-to-All Parallel Attention: O(1) Sequential Path] |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Intuitive Mental Models & Analogies

Before inspecting the mathematical equations, let us build bulletproof conceptual intuition through real-world analogies.

### 2.1 The Library Search Engine: Query, Key, and Value

The core building block of the Transformer is the triplet: **Query ($Q$)**, **Key ($K$)**, and **Value ($V$)**.

This terminology originates from database retrieval and information search engines:

Imagine you walk into a vast university library:
1. **The Query ($Q$):** This is the **search term you type into the computer** (e.g., *"How do birds navigate using magnetic fields?"*). It represents **what the current token is looking for**.
2. **The Key ($K$):** These are the **catalog tags and book spine labels** on every book in the library (e.g., *"Ornithology / Avian Magnetoreception"*). It represents **what each token advertises about its contents**.
3. **The Value ($V$):** This is the **actual content written inside each book**. It represents **the substantive informational vector carried by the token**.

```
                ┌────────────────────────────────────────────────────────┐
                │          Query (Q): What am I looking for?             │
                └──────────────────────────┬─────────────────────────────┘
                                           │
                       Dot-Product Match against all Keys
                                           ▼
      ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
      │  Key 1: Tag "AI" │       │  Key 2: Tag "Fly"│       │ Key 3: Tag "Bird"│
      └────────┬─────────┘       └────────┬─────────┘       └────────┬─────────┘
               │                          │                          │
       Match: 0.05                Match: 0.15                Match: 0.80  (Softmax Weights)
               │                          │                          │
               ▼                          ▼                          ▼
      ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
      │ Value 1: Data    │       │ Value 2: Data    │       │ Value 3: Data    │
      └──────────────────┘       └──────────────────┘       └──────────────────┘
               │                          │                          │
               └──────────────────────────┼──────────────────────────┘
                                          ▼
                       Weighted Sum of Values = Context Vector
```

When token $A$ looks at token $B$, it computes the **dot product** between its Query vector $q_A$ and token $B$'s Key vector $k_B$. If their vectors point in the same direction in semantic space, the dot product is large. Softmax converts these scores into normalized attention weights that sum to $1.0$. Finally, the network takes a weighted sum of the **Values ($V$)**.

### 2.2 The Specialist Advisory Board: Multi-Head Attention

Imagine a CEO making a major strategic decision. If the CEO consults only a single generalist advisor, that advisor might focus entirely on legal risk and completely overlook financial profits, technical viability, and brand marketing.

Instead, the CEO forms a **board of 8 specialized advisors**:
- **Advisor 1 (Syntax Specialist):** Focuses on grammatical subject-verb agreement (*"Does 'they' match 'were'?"*).
- **Advisor 2 (Pronoun Specialist):** Focuses on coreference resolution (*"Who does 'it' refer to in the previous paragraph?"*).
- **Advisor 3 (Semantic Specialist):** Focuses on subject-action meaning (*"What action did the dog perform?"*).
- **Advisor 4 (Temporal Specialist):** Focuses on time and chronology (*"Did event X happen before event Y?"*).

Each advisor analyzes the exact same sentence, but from their own distinct mathematical projection subspace. At the end of the meeting, their insights are concatenated and synthesized into a single master recommendation. This is precisely what **Multi-Head Attention** achieves!

### 2.3 The One-Way Mirror: Encoder vs Causal Decoder

The original Transformer consists of two halves: an **Encoder** and a **Decoder**.
- **The Encoder (A Conference Room with Clear Glass):** Every token can look at every other token, past and future, simultaneously. The word at position 2 can see the word at position 50. It builds complete, bidirectional contextual representations of the input.
- **The Decoder (A One-Way Mirror / Exam Room with Dividers):** When generating text token-by-token (autoregressively), a student cannot be allowed to peek at the answer sheet ahead of time. The Decoder enforces **causal masking**: token $t$ is strictly forbidden from looking at tokens $t+1, t+2, \dots, T$. It can only look backward at past tokens it has already generated.

---

## 3. Scaled Dot-Product Attention: The Mathematical Engine

### 3.1 The Formal Formulation

The mathematical definition of Scaled Dot-Product Attention from Section 3.2.1 of the paper is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$ is the matrix of Queries ($n$ tokens, dimension $d_k$).
- $K \in \mathbb{R}^{m \times d_k}$ is the matrix of Keys ($m$ tokens, dimension $d_k$).
- $V \in \mathbb{R}^{m \times d_v}$ is the matrix of Values ($m$ tokens, dimension $d_v$).
- In self-attention, $n = m$ (the sequence attends to itself). In the original paper, $d_k = d_v = 64$.
- $QK^T \in \mathbb{R}^{n \times m}$ is the raw score compatibility matrix.
- $\sqrt{d_k}$ is the scaling factor.
- $\text{softmax}(\cdot)$ normalizes each row so the attention weights across all keys sum to $1$.

```
   Q Matrix             K^T Matrix                 Attention Weights                V Matrix             Output Matrix
  (n x d_k)             (d_k x m)                       (n x m)                     (m x d_v)              (n x d_v)

 ┌───────────┐         ┌───────────┐                 ┌───────────┐                ┌───────────┐          ┌───────────┐
 │   q_1     │         │           │                 │ a_11 a_12 │                │   v_1     │          │   out_1   │
 │   q_2     │    x    │ k_1   k_2 │    ==> Softmax  │ a_21 a_22 │      x         │   v_2     │    =     │   out_2   │
 │   ...     │         │           │     (QK^T/√d_k) │ ...   ... │                │   ...     │          │   ...     │
 │   q_n     │         │           │                 │ a_n1 a_nm │                │   v_m     │          │   out_n   │
 └───────────┘         └───────────┘                 └───────────┘                └───────────┘          └───────────┘
```

---

### 3.2 Mathematical Proof: Why Scale by $1/\sqrt{d_k}$?

A common interview question and theoretical puzzle: **Why is the division by $\sqrt{d_k}$ strictly necessary? Why not plain dot-product attention $QK^T$?**

Let us derive the variance of the dot product rigorously.

Assume the components of query vector $q \in \mathbb{R}^{d_k}$ and key vector $k \in \mathbb{R}^{d_k}$ are independent and identically distributed (i.i.d.) random variables with mean zero and variance one:

$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1$$
$$\mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$

The dot product between $q$ and $k$ is the sum of $d_k$ elementwise products:

$$S = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

Let us compute the expected value of $S$:

$$\mathbb{E}[S] = \mathbb{E}\left[\sum_{i=1}^{d_k} q_i k_i\right] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = \sum_{i=1}^{d_k} (0 \cdot 0) = 0$$

Now let us compute the variance of $S$:

$$\text{Var}(S) = \text{Var}\left(\sum_{i=1}^{d_k} q_i k_i\right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i)$$

Since $q_i$ and $k_i$ are independent:

$$\text{Var}(q_i k_i) = \mathbb{E}[(q_i k_i)^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] - 0$$

Since $\mathbb{E}[q_i^2] = \text{Var}(q_i) + (\mathbb{E}[q_i])^2 = 1 + 0 = 1$, and likewise $\mathbb{E}[k_i^2] = 1$:

$$\text{Var}(q_i k_i) = 1 \cdot 1 = 1$$

Therefore:

$$\text{Var}(S) = \sum_{i=1}^{d_k} 1 = d_k$$

$$\text{Standard Deviation of } S = \sqrt{d_k}$$

#### The Catastrophe Without Scaling:
As the dimensionality $d_k$ grows large (e.g., $d_k = 64$ or $128$):
- The magnitude of the dot products grows proportionally to $\sqrt{d_k}$. For $d_k = 64$, standard deviation is $\sqrt{64} = 8$.
- When numbers with values like $+16$ and $-16$ are fed into the **Softmax function**:
  $$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
- The largest value dominates overwhelmingly: $\text{softmax}([+16, -16]) \approx [1.0, 0.0]$.
- The softmax function is pushed into extreme saturation regions where its gradient is virtually **zero**:
  $$\frac{\partial \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j) \approx 1 \cdot (1 - 1) = 0$$
- **The gradients vanish catastrophically during backpropagation!**

#### The Scaling Solution:
By dividing the dot product by $\sqrt{d_k}$:

$$\text{Var}\left(\frac{S}{\sqrt{d_k}}\right) = \frac{1}{(\sqrt{d_k})^2} \text{Var}(S) = \frac{1}{d_k} \cdot d_k = 1$$

The variance is stabilized back to **$1.0$** regardless of how large $d_k$ is, keeping the softmax in its sensitive, gradient-rich linear regime!

---

### 3.3 Step-by-Step Matrix Computation Walkthrough

Let us trace numerical shapes through the full attention calculation:
1. **Input Matrix $X \in \mathbb{R}^{T \times d_{\text{model}}}$**: Sequence of $T = 4$ tokens, embedding dimension $d_{\text{model}} = 512$.
2. **Project to $Q, K, V$**:
   $$Q = X W^Q \quad (4 \times 64), \quad K = X W^K \quad (4 \times 64), \quad V = X W^V \quad (4 \times 64)$$
3. **Score Matrix**:
   $$A = Q K^T \quad (4 \times 64) \times (64 \times 4) = (4 \times 4)$$
4. **Scale**:
   $$A_{\text{scaled}} = \frac{A}{\sqrt{64}} = \frac{A}{8} \quad (4 \times 4)$$
5. **Apply Mask (if Decoder)**:
   Add $-\infty$ to illegal future token positions.
6. **Softmax Row-wise**:
   $$W = \text{softmax}(A_{\text{scaled}}) \quad (4 \times 4)$$
   Each row sums to $1.0000$.
7. **Context Output**:
   $$\text{Output} = W V \quad (4 \times 4) \times (4 \times 64) = (4 \times 64)$$

Every single token representation is now an information-dense blend of all relevant tokens in the context!

---

## 4. Multi-Head Attention (MHA): Learning Diverse Representation Subspaces

![Multi-Head Attention Deep Dive](assets/05_multihead_attention_deep_dive.jpg)

### 4.1 Why Single-Head Attention Falls Short

If we compute only a single attention matrix of dimension $d_{\text{model}} = 512$, the attention weights are forced to average together multiple competing linguistic signals:
- Token `"bank"` needs to attend to `"river"` to establish word sense.
- Token `"bank"` simultaneously needs to attend to `"the"` to establish definite noun phrase structure.
- Token `"bank"` simultaneously needs to attend to the verb `"overflowed"` to establish semantic agent-patient relations.

With only a single attention head, these diverse relationships collide and blur into a single compromise average.

### 4.2 The Linear Projection & Splitting Mechanism

Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, the authors linearly project the queries, keys, and values $h$ times with different, learned linear projections:

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

Where the projection parameter matrices are:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

In the standard Transformer base model:
- $d_{\text{model}} = 512$
- $h = 8$ parallel attention heads
- $d_k = d_v = d_{\text{model}} / h = 512 / 8 = 64$

> [!NOTE]
> **Zero Compute Penalty:** Because the dimension of each head is cut down to $d_k = 64$, the total computational cost of running all 8 heads in parallel is identical to running a single full-dimensional attention head with $d_{\text{model}} = 512$!

### 4.3 Concatenation and the Output Projection Matrix $W^O$

Once each of the 8 heads calculates its contextualized representation $\text{head}_i \in \mathbb{R}^{T \times 64}$, their outputs are **concatenated** along the feature dimension:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O$$

Where:
- $\text{Concat}(\dots) \in \mathbb{R}^{T \times (8 \times 64)} = \mathbb{R}^{T \times 512}$
- $W^O \in \mathbb{R}^{512 \times 512}$ is a learned linear projection matrix that mixes and synthesizes the signals collected across all 8 specialized heads.

### 4.4 What Do Individual Heads Actually Learn?

Empirical visualization of trained Transformer attention heads (e.g., Clark et al., *What Does BERT Look At?*) revealed that different heads naturally specialize into distinct linguistic detectors without any supervised labels:

| Head Index | Specialization Discovered | Real-World Linguistic Example |
|---|---|---|
| **Head 1** | **Direct Grammatical Objects** | Connects transitive verbs to their direct objects (`read` $\to$ `book`). |
| **Head 2** | **Coreference Resolution** | Connects third-person pronouns to antecedent nouns (`she` $\to$ `Marie Curie`). |
| **Head 3** | **Next Token / Local Syntax** | Attends strongly to the immediate subsequent token ($t \to t+1$). |
| **Head 4** | **Prepositional Modifiers** | Connects prepositional phrases to head nouns (`in the garden` $\to$ `flower`). |
| **Head 5** | **Delimiters & Punctuation** | Attends to periods, commas, and `[SEP]` tokens to gather broad sentence boundaries. |
| **Head 6** | **Passive Voice Inversion** | Detects passive voice agent phrases (`written by Shakespeare`). |
| **Head 7** | **Global Document Topic** | Distributes uniform soft weights across all topical nouns in the text. |
| **Head 8** | **Adjective-Noun Attribution** | Binds descriptive qualifiers directly to the subject (`red` $\to$ `car`). |

---

## 5. The Full Transformer Architecture Visualized

Below is the definitive high-resolution architecture diagram of the Transformer from the 2017 paper:

![Complete Transformer Architecture](assets/04_transformer_architecture_full.jpg)

### Architecture Highlights:
- **Left Side (Encoder Stack):** Repeated $N = 6$ times. Takes raw input tokens, embeds them, adds positional encodings, and processes them through Bidirectional Multi-Head Self-Attention + Feed-Forward networks.
- **Right Side (Decoder Stack):** Repeated $N = 6$ times. Takes target tokens shifted right, applies Causal Masking, receives Keys and Values from the top Encoder layer via Cross-Attention, and outputs next-token probability logits through a final Linear + Softmax layer.

---

## 6. The Encoder Stack: Contextualizing Input Sequences

### 6.1 Encoder Layer Composition ($N = 6$)

The Transformer Encoder is composed of a stack of **$N = 6$ identical layers**. Each individual layer contains **two sub-layers**:
1. **Sub-layer 1:** Multi-Head Self-Attention mechanism.
2. **Sub-layer 2:** Position-wise Feed-Forward Network (FFN).

Around each of the two sub-layers, there is a **residual connection** followed by **Layer Normalization**:

$$\text{Output}_{\text{sublayer}} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

All sub-layers in the model, as well as the embedding layers, produce outputs of fixed dimension $d_{\text{model}} = 512$.

```
                       INPUT EMBEDDINGS + POSITIONAL ENCODINGS
                                         │
                    ┌────────────────────┼───────────────────┐
                    │                    ▼                   │
                    │   ┌────────────────────────────────┐   │
                    │   │  Multi-Head Self-Attention     │   │
                    │   └────────────────┬───────────────┘   │
                    │                    ▼                   │
                    │         Residual Add & LayerNorm       │
                    │                    │                   │
                    │                    ▼                   │
                    │   ┌────────────────────────────────┐   │
                    │   │  Position-Wise Feed-Forward    │   │
                    │   └────────────────┬───────────────┘   │
                    │                    ▼                   │
                    │         Residual Add & LayerNorm       │
                    └────────────────────┬───────────────────┘
                                         ▼ (Repeated N = 6 Times)
                              ENCODER OUTPUT (K, V to Decoder)
```

### 6.2 Sub-Layer 1: Bidirectional Multi-Head Self-Attention

In the encoder, $Q, K,$ and $V$ all come from the exact same source: the output of the previous encoder layer (or the initial token embeddings in layer 1).

Because there is **no masking**, every token attends to all tokens in the entire sentence:
> *"The animal didn't cross the street because it was too tired."*

When encoding `"it"`, the self-attention mechanism computes dot products against all words. The highest attention score binds to `"animal"`, resolving the pronoun with zero human rules!

### 6.3 Sub-Layer 2: Position-wise Feed-Forward Network (FFN)

After the multi-head attention sub-layer gathers contextual relationships, the representation passes into a **Position-wise Feed-Forward Network**:

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

Where:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}} = \mathbb{R}^{512 \times 2048}$
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}} = \mathbb{R}^{2048 \times 512}$
- $\max(0, \cdot)$ is the ReLU activation function (modern architectures often use GELU or SwiGLU).

#### Intuition & Purpose of the FFN:
- **Attention routes and mixes information** between tokens across sequence length $T$.
- **The FFN processes and reasons** within each token independently across channel dimensions ($512 \to 2048 \to 512$).
- Researchers often describe the FFN as a **key-value memory bank** storing factual world knowledge learned during pre-training.

---

## 7. The Decoder Stack: Autoregressive Generation with Guardrails

### 7.1 Decoder Layer Composition ($N = 6$)

Like the encoder, the Decoder is composed of a stack of **$N = 6$ identical layers**. However, each decoder layer contains **three sub-layers** instead of two:
1. **Sub-layer 1:** Masked Multi-Head Self-Attention (prevents positions from attending to subsequent positions).
2. **Sub-layer 2:** Multi-Head Cross-Attention (attends over the final output of the Encoder stack).
3. **Sub-layer 3:** Position-wise Feed-Forward Network.

Just like the encoder, each sub-layer uses residual connections followed by Layer Normalization: $\text{LayerNorm}(x + \text{SubLayer}(x))$.

### 7.2 Sub-Layer 1: Masked Multi-Head Self-Attention (Causal Masking)

During training, we feed the entire target sentence into the decoder simultaneously for maximum GPU parallelism (called **Teacher Forcing**).

However, during real-world inference, language generation is strictly **autoregressive**: the model must predict token $y_t$ given only the previously generated tokens $y_1, y_2, \dots, y_{t-1}$.

If the decoder could see future tokens during training, it would simply memorize the answer key by looking ahead!

#### The Mathematical Causal Mask:
To prevent information leakage from future positions, we modify the attention score matrix prior to the softmax operation by adding a **causal mask matrix $M$**:

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Where mask matrix $M$ is defined as:

$$M_{ij} = \begin{cases} 0 & \text{if } j \le i \quad (\text{allowed: past or current token}) \\ -\infty & \text{if } j > i \quad (\text{forbidden: future token}) \end{cases}$$

For a sequence of length 4:

$$M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

When $e^{-\infty}$ is evaluated inside softmax:

$$e^{-\infty} = 0$$

The probability of attending to any future token becomes **identically zero**!

```
                      CAUSAL ATTENTION MASK MATRIX HEATMAP
                       Token 1   Token 2   Token 3   Token 4
            Token 1   [  0.85      0.00      0.00      0.00   ]  <-- Can only see Token 1
            Token 2   [  0.35      0.65      0.00      0.00   ]  <-- Can see Tokens 1 & 2
            Token 3   [  0.20      0.30      0.50      0.00   ]  <-- Can see Tokens 1, 2, 3
            Token 4   [  0.10      0.25      0.15      0.50   ]  <-- Can see Tokens 1, 2, 3, 4
```

### 7.3 Sub-Layer 2: Multi-Head Cross-Attention (Encoder-Decoder)

This is the bridge connecting the input language to the output language (e.g., English to German):
- **Queries ($Q$)** come from the **previous decoder sub-layer** (what the translation currently needs next).
- **Keys ($K$)** come from the **final output of the Encoder** (what the source sentence contains).
- **Values ($V$)** come from the **final output of the Encoder** (the actual representations of the source sentence).

This allows every position in the decoder to attend over all positions in the input sequence, perfectly mirroring traditional Seq2Seq attention mechanisms without any recurrent loops!

### 7.4 The Final Linear Classifier & Softmax Projection

The output of the top decoder layer is a tensor of shape $[B, T, d_{\text{model}}] = [B, T, 512]$.

To produce actual vocabulary words:
1. **Linear Layer:** A simple learned projection that projects the 512-dimensional vector into a giant vector of size $|V|$ (the vocabulary size, typically $32,000$ to $100,000$ tokens):
   $$\text{Logits} = x W_{\text{vocab}} + b_{\text{vocab}} \quad \in \mathbb{R}^{T \times |V|}$$
2. **Softmax:** Turns the raw logits into normalized probabilities over the dictionary:
   $$P(y_t = w) = \frac{\exp(\text{Logit}_w)}{\sum_{v \in V} \exp(\text{Logit}_v)}$$
3. **Sampling / Greedy Argmax:** The token with the highest probability is selected, appended to the prompt, and fed back into the decoder for the next step.

---

## 8. Positional Encoding: Injecting Order into Permutation-Invariant Sets

### 8.1 Why Transformers Have Zero Inherent Word Order

In an RNN, word order is built directly into the physical loop: token 1 enters at time $t=1$, token 2 at $t=2$.

In contrast, the Transformer computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Matrix multiplication is **permutation-equivariant**. If you shuffle the input sentence:
- *"Dog bites man"*
- *"Man bites dog"*

The mathematical dot product between `"dog"` and `"bites"` is 100% identical regardless of whether `"dog"` is at index 1 or index 3! Without an explicit intervention, **a Transformer treats sentences as an unordered bag of words**.

### 8.2 Sinusoidal Positional Encoding Equations

To give the model awareness of sequence order, Vaswani et al. added positional encoding vectors directly to the input embeddings:

$$X_{\text{input}} = \text{TokenEmbedding}(x) + PE$$

The authors chose fixed sine and cosine functions of different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Where:
- $pos$ is the token position in the sentence ($0, 1, 2, \dots, T-1$).
- $i$ is the dimension index ($0 \le i < d_{\text{model}} / 2$).
- $2i$ corresponds to even channel indices; $2i+1$ corresponds to odd channel indices.
- The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.

```
Dimension i ──► (High Frequency / Short Wavelength) ──► (Low Frequency / Long Wavelength)
pos = 0   [ sin(0)    cos(0)    sin(0)    cos(0)   ...   sin(0)    cos(0)   ]
pos = 1   [ sin(1/1)  cos(1/1)  sin(1/λ)  cos(1/λ) ...   sin(1/Λ)  cos(1/Λ) ]
pos = 2   [ sin(2/1)  cos(2/1)  sin(2/λ)  cos(2/λ) ...   sin(2/Λ)  cos(2/Λ) ]
```

### 8.3 The Linear Geometric Transformation Property

Why did the authors choose sinusoids over simple integer counts ($1, 2, 3, \dots$)?
1. **Normalized Range:** Simple integer counts grow unboundedly ($pos = 1000$ would overwhelm small embedding values). Sinusoids are strictly bounded within $[-1.0, +1.0]$.
2. **Relative Shift Linear Projection:** By trigonometric angle addition theorems:
   $$\sin(\alpha + \beta) = \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta)$$
   $$\cos(\alpha + \beta) = \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)$$

For any fixed offset $k$, the positional encoding at position $pos + k$ can be represented as a **linear transformation matrix $M_k$** of the positional encoding at position $pos$:

$$PE_{pos + k} = M_k \cdot PE_{pos}$$

This allows the self-attention mechanism to learn to attend to **relative positions** ($k$ tokens away) simply by applying a linear transformation!

---

## 9. Residual Connections & Layer Normalization (Add & Norm)

### 9.1 Deep Gradient Preservation: Residual Pathways

Deep neural networks suffer from the degradation problem: as depth increases, gradient signals vanish or explode.

The Transformer introduces **residual skip connections** (He et al., 2016) around every sub-layer:

$$y = x + \mathcal{F}(x)$$

When computing the backward derivative during backpropagation:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left( I + \frac{\partial \mathcal{F}(x)}{\partial x} \right) = \frac{\partial \mathcal{L}}{\partial y} + \frac{\partial \mathcal{L}}{\partial y}\frac{\partial \mathcal{F}(x)}{\partial x}$$

Notice the term **$\frac{\partial \mathcal{L}}{\partial y} \cdot I$**: The gradient can flow backward directly through the identity addition wire **without passing through any matrix multiplications**! This prevents gradient vanishing and allows Transformers to scale cleanly to hundreds of stacked layers.

### 9.2 Layer Normalization vs Batch Normalization

In computer vision (CNNs), **Batch Normalization (BN)** is standard. But in NLP, BN fails completely:
- BN computes statistics across the mini-batch for each feature dimension.
- In NLP, sentences have variable lengths. If a batch contains sentences of 5 words and 100 words, batch statistics for positions 6–100 are computed over only a tiny fraction of samples, resulting in massive variance and unstable training.

**Layer Normalization (LN)** (Ba et al., 2016) normalizes **across the feature dimensions for each individual sample independently**:

$$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$

$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

Where $\gamma$ and $\beta$ are learnable scale and shift parameters.

```
+-----------------------------------------------------------------------------------------+
|                  BATCH NORMALIZATION vs LAYER NORMALIZATION                            |
|                                                                                         |
|   Batch Normalization (BN):       Layer Normalization (LN):                             |
|   Normalize down across samples   Normalize across features                             |
|                                                                                         |
|         Feature 1  Feature 2            Feature 1  Feature 2                            |
|   S1  [    x          x    ]      S1  [────► normalize across this sample ────►]        |
|   S2  [    │          │    ]      S2  [────► normalize across this sample ────►]        |
|   S3  [    ▼          ▼    ]      S3  [────► normalize across this sample ────►]        |
|       (Dependent on batch size)       (Completely independent of batch size & length)   |
+-----------------------------------------------------------------------------------------+
```

### 9.3 Post-LN (Vaswani 2017) vs Pre-LN (Modern LLMs)

An important architectural evolution between 2017 and today:
- **Post-LN (Original 2017 Transformer):**
  $$x_{l+1} = \text{LayerNorm}(x_l + \text{SubLayer}(x_l))$$
  *Issue:* Gradients through the normalization layers can become unstable at initialization, requiring careful learning rate warmup and gradient clipping.
- **Pre-LN (Modern GPT-3, LLaMA, Mistral):**
  $$x_{l+1} = x_l + \text{SubLayer}(\text{LayerNorm}(x_l))$$
  *Benefit:* Keeps the residual highway completely unobstructed from layer 1 to layer $N$, allowing stable training from step 1 without fragile warmups.

---

## 10. Architectural Archetypes: Encoder-Only vs Decoder-Only vs Encoder-Decoder

Following the 2017 paper, the AI research community split the full Transformer into three distinct architectural families:

| Architecture Archetype | Pioneering Model | Attention Visibility | Typical Pre-Training Objective | Best Suited For |
|---|---|---|---|---|
| **Encoder-Only** | **BERT** (Devlin et al., 2018), RoBERTa | **Bidirectional** (All tokens see all tokens) | Masked Language Modeling (MLM): Predict `[MASK]` tokens | Classification, NER, Sentiment Analysis, Extractive QA |
| **Decoder-Only** | **GPT-1/2/3/4** (OpenAI), LLaMA, Mistral | **Causal Masked** (Tokens only see past tokens) | Autoregressive Next-Token Prediction: $p(w_t \mid w_{<t})$ | Open-ended text generation, coding assistants, reasoning chat |
| **Encoder-Decoder** | **Original Transformer** (2017), T5, BART | **Bidirectional** Encoder + **Causal** Decoder | Sequence-to-Sequence Denoising / Reconstruction | Language Translation, Abstractive Summarization |

---

## 11. Hands-On Python Lab: Transformer Mechanics from Scratch

This complete, runnable Python lab implements the foundational components from scratch using only `numpy`:
- Scaled Dot-Product Attention with and without causal masking.
- Multi-Head Attention splitting, concatenation, and output projection.
- Encoder-Decoder Cross-Attention.
- Sinusoidal Positional Encoding generator.

You can run the script directly from your terminal:
```bash
python "1. Theoretical Foundations & NLP Evolution/code/transformer_deep_dive_lab.py"
```

```python
"""
=============================================================================
Hands-On Lab: Complete Transformer Mechanics from Scratch (NumPy)
=============================================================================
Course: Zero to Hero Gen AI — Module 01: Theoretical Foundations & NLP Evolution
Paper: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import numpy as np

np.random.seed(42)

def softmax(z, axis=-1):
    """Numerically stable softmax."""
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)

# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention (with optional Causal Mask)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) + mask ) @ V
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask
        
    weights = softmax(scores, axis=-1)
    context = np.matmul(weights, V)
    return context, weights

# ---------------------------------------------------------------------------
# 2. Multi-Head Attention Class
# ---------------------------------------------------------------------------
class MultiHeadAttentionNumPy:
    def __init__(self, d_model=64, num_heads=4):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projection matrices
        self.W_Q = np.random.randn(d_model, d_model) * 0.05
        self.W_K = np.random.randn(d_model, d_model) * 0.05
        self.W_V = np.random.randn(d_model, d_model) * 0.05
        self.W_O = np.random.randn(d_model, d_model) * 0.05

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))  # [B, num_heads, seq_len, d_k]

    def combine_heads(self, x):
        # [B, num_heads, seq_len, d_k] -> [B, seq_len, d_model]
        x = np.transpose(x, (0, 2, 1, 3))
        batch_size, seq_len, _, _ = x.shape
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q_proj = np.matmul(Q, self.W_Q)
        K_proj = np.matmul(K, self.W_K)
        V_proj = np.matmul(V, self.W_V)

        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)

        context, weights = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        out = self.combine_heads(context)
        return np.matmul(out, self.W_O), weights

# ---------------------------------------------------------------------------
# 3. Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------
def get_positional_encoding(seq_len=6, d_model=16):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = np.power(10000, 2 * i / d_model)
            pe[pos, i] = np.sin(pos / div_term)
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / div_term)
    return pe
```

---

## 12. Curated Video Walkthroughs & Visual Animations

To master the nuances of the 2017 paper, watch these hand-curated, globally celebrated video lessons:

| # | Topic / Video Title | Recommended Video Link | Creator / Channel | Why Watch? (Visual & Technical Highlights) |
|---|---|---|---|---|
| 1 | **Attention Is All You Need (Paper Explained)** | [Attention Is All You Need](https://www.youtube.com/watch?v=iDulhoQ2pro) | **Yannic Kilcher** | Line-by-line breakdown of the original 2017 paper text, formulas, hyperparameters, and experimental BLEU results. |
| 2 | **Let's Build GPT from Scratch** | [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) | **Andrej Karpathy** | The undisputed gold standard: builds the full decoder Transformer layer-by-layer in raw Python/PyTorch with deep intuition. |
| 3 | **Attention in Transformers, Step-by-Step** | [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) | **3Blue1Brown** | Incredible 3D geometric animation showing how Query, Key, and Value vectors steer contextual information in high dimensions. |
| 4 | **Transformer Neural Networks, Clearly Explained!** | [Transformer Neural Networks, Clearly Explained!](https://www.youtube.com/watch?v=zxQyTK8quyY) | **StatQuest (Josh Starmer)** | Clear step-by-step breakdown of Encoders, Decoders, Positional Encodings, and Softmax scaling with hand-drawn clarity. |
| 5 | **Transformers, the Tech Behind LLMs** | [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) | **3Blue1Brown** | High-level synthesis of how stacking multi-head attention and feed-forward blocks gives rise to emergent LLM capabilities. |

---

### 🎬 Deep-Dive Video Breakdown

#### 1. [Yannic Kilcher — Attention Is All You Need (Paper Explained)](https://www.youtube.com/watch?v=iDulhoQ2pro)

[![Attention Is All You Need](https://img.youtube.com/vi/iDulhoQ2pro/hqdefault.jpg)](https://www.youtube.com/watch?v=iDulhoQ2pro)

- **Runtime:** ~40 mins | **Focus:** Academic paper walkthrough
- **Key Concepts Covered:**
  - Why Google Brain wanted to eliminate RNNs and sequential recurrences.
  - The exact architecture breakdown of the original base model vs big model.
  - Discussion of warm-up schedules and label smoothing.

---

#### 2. [Andrej Karpathy — Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

[![Let's build GPT from scratch](https://img.youtube.com/vi/kCc8FmEb1nY/hqdefault.jpg)](https://www.youtube.com/watch?v=kCc8FmEb1nY)

- **Runtime:** ~1 hr 56 mins | **Focus:** Coding full GPT Transformer from scratch
- **Key Concepts Covered:**
  - Coding Bigram language models up to self-attention blocks.
  - Building `Head`, `MultiHeadAttention`, `FeedForward`, and `Block` modules in PyTorch.
  - Explaining causal triangular masking with `torch.tril`.

---

#### 3. [3Blue1Brown — Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)

[![Attention in transformers, step-by-step](https://img.youtube.com/vi/eMlx5fFNoYc/hqdefault.jpg)](https://www.youtube.com/watch?v=eMlx5fFNoYc)

- **Runtime:** ~26 mins | **Focus:** 3D geometric visual intuition
- **Key Concepts Covered:**
  - How embedding spaces represent word semantics.
  - The geometric interpretation of Queries and Keys dot products.
  - Visualizing the dynamic update to token vectors as they absorb information from context.

---

#### 4. [StatQuest (Josh Starmer) — Transformer Neural Networks, Clearly Explained!](https://www.youtube.com/watch?v=zxQyTK8quyY)

[![Transformer Neural Networks](https://img.youtube.com/vi/zxQyTK8quyY/hqdefault.jpg)](https://www.youtube.com/watch?v=zxQyTK8quyY)

- **Runtime:** ~25 mins | **Focus:** Intuitive component-by-component cartoon walkthrough
- **Key Concepts Covered:**
  - How Positional Encodings are added to word embeddings.
  - Multi-Head Self-Attention calculation explained step by step.
  - How Encoder outputs pass to the Decoder via Cross-Attention.

---

#### 5. [3Blue1Brown — Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M)

[![Transformers, the tech behind LLMs](https://img.youtube.com/vi/wjZofJX0v4M/hqdefault.jpg)](https://www.youtube.com/watch?v=wjZofJX0v4M)

- **Runtime:** ~27 mins | **Focus:** The grand architectural synthesis
- **Key Concepts Covered:**
  - The full journey from tokenization to output probabilities.
  - How multilayer perceptrons (MLP / FFN blocks) store facts.
  - Why modern generative AI is fundamentally driven by attention.

---

## 13. Self-Assessment & Review Questions

Test your understanding of the mechanics behind the Transformer architecture.

### Part 1: Conceptual Questions

1. **Why does Scaled Dot-Product Attention divide the scores by $\sqrt{d_k}$? What mathematical failure happens if this scale factor is omitted?**
   <details>
   <summary><b>View Answer</b></summary>
   The dot product of two independent vectors with mean 0 and variance 1 has mean 0 and variance equal to $d_k$. For large dimensions (e.g., $d_k = 64$), the dot products grow very large in magnitude. When passed to the softmax function, large inputs cause the exponential values to polarize, pushing the softmax into regions with near-zero gradients (saturation). Dividing by $\sqrt{d_k}$ scales the variance back to $1.0$, preserving steady gradient flow.
   </details>

2. **In Decoder Cross-Attention, which sub-network provides the Queries ($Q$), and which sub-network provides the Keys ($K$) and Values ($V$)? Why?**
   <details>
   <summary><b>View Answer</b></summary>
   The <b>Queries ($Q$)</b> originate from the <b>Decoder's previous layer</b>, representing what the target output is currently seeking. The <b>Keys ($K$) and Values ($V$)</b> originate from the <b>final output of the Encoder</b>, representing the source context and information being translated or processed.
   </details>

3. **Why is Causal Masking required in the Decoder during training, but NOT in the Encoder?**
   <details>
   <summary><b>View Answer</b></summary>
   The Encoder's job is full understanding of the complete input, so looking both backward and forward (bidirectional attention) is advantageous. The Decoder generates output autoregressively ($p(y_t \mid y_{<t})$). During training, all target tokens are supplied simultaneously (teacher forcing). Without a causal mask setting future positions to $-\infty$, the model would cheat by attending to future tokens rather than learning to predict them.
   </details>

---

### Part 2: Mathematical Problems

4. **Given a Transformer model with $d_{\text{model}} = 768$ and $h = 12$ attention heads, calculate the dimension $d_k$ of each individual head and the total number of parameters in the four projection matrices ($W^Q, W^K, W^V, W^O$).**
   <details>
   <summary><b>View Answer</b></summary>
   - Individual head dimension: $d_k = d_{\text{model}} / h = 768 / 12 = 64$.<br>
   - Each projection matrix ($W^Q, W^K, W^V, W^O$) has dimension $d_{\text{model}} \times d_{\text{model}} = 768 \times 768 = 589,824$ parameters.<br>
   - Total parameters for all 4 matrices: $4 \times 589,824 = \mathbf{2,359,296}$ parameters (approx. 2.36M).
   </details>

5. **In an autoregressive decoder with sequence length $T = 4$, write down the exact 4x4 matrix added to the scaled dot-product scores prior to softmax.**
   <details>
   <summary><b>View Answer</b></summary>
   $$M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}$$
   The lower triangle and diagonal are $0$ (unmasked), while the strictly upper triangle consists of $-\infty$ (masked future positions).
   </details>

---

### Part 3: Fill-in-the-Blanks

6. Unlike Batch Normalization which computes statistics across samples, **Layer Normalization** computes statistics across the ____________________ dimension for each sample independently.
   <details>
   <summary><b>View Answer</b></summary>
   <b>hidden feature / channel</b> (or $d_{\text{model}}$)
   </details>

7. In the original 2017 Transformer paper, the dimension of the inner hidden layer in the position-wise Feed-Forward Network is $d_{\text{ff}} =$ __________.
   <details>
   <summary><b>View Answer</b></summary>
   <b>2048</b> ($4 \times d_{\text{model}} = 4 \times 512 = 2048$)
   </details>

8. The architectural design where the input is added directly to the sub-layer output ($x + \text{SubLayer}(x)$) is called a ____________________ connection.
   <details>
   <summary><b>View Answer</b></summary>
   <b>residual (or skip)</b>
   </details>

---

## 14. Summary & Key Takeaways

| Transformer Component | Mathematical Role | Primary Advantage |
|---|---|---|
| **Scaled Dot-Product** | $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | $O(1)$ sequential operations; variance scaling prevents vanishing gradients |
| **Multi-Head Attention** | $\text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$ | Captures multiple simultaneous representation subspaces (syntax, coreference, semantics) |
| **Encoder Stack** | $N = 6$ Bidirectional blocks | Fully bidirectional contextual representation of input tokens |
| **Decoder Stack** | $N = 6$ Causal Masked + Cross-Attention blocks | Preserves autoregressive factorization while conditioning on source encoder states |
| **Positional Encoding** | Sinusoidal frequency encoding added to embeddings | Injects token order into inherently permutation-invariant self-attention |
| **Residual + LayerNorm** | $\text{LN}(x + \text{SubLayer}(x))$ | Enables stable training across dozens of stacked layers without gradient degradation |
| **Feed-Forward Network** | $\max(0, xW_1 + b_1)W_2 + b_2$ | Applies non-linear transformations and serves as a factual key-value associative memory |
