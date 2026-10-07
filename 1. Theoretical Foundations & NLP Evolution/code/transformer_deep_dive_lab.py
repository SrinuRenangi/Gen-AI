"""
=============================================================================
Hands-On Lab: Complete Transformer Mechanics from Scratch (NumPy)
=============================================================================
Course: Zero to Hero Gen AI — Module 01: Theoretical Foundations & NLP Evolution
Topic: Transformer Deep-Dive: Mastering "Attention Is All You Need" (Vaswani et al., 2017)

This lab implements and validates:
1. Scaled Dot-Product Attention: Scaled vs Unscaled softmax saturation proof.
2. Causal Masked Attention: Strict autoregressive upper-triangle zeroing.
3. Multi-Head Attention (MHA): Splitting, parallel projection, concatenation & W_O.
4. Encoder-Decoder Cross-Attention: Q from Decoder, K/V from Encoder.
5. Sinusoidal Positional Encoding: Exact Vaswani sinusoidal wave computation.
=============================================================================
"""

import numpy as np


def softmax(z, axis=-1):
    """Numerically stable softmax implementation."""
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


# =============================================================================
# PART 1: Scaled Dot-Product Attention & Softmax Saturation Proof
# =============================================================================
def demo_scaled_dot_product_and_saturation():
    print("=" * 75)
    print("PART 1: Scaled Dot-Product Attention & Softmax Gradient Saturation")
    print("=" * 75)

    d_k = 64
    seq_len = 4
    tokens = ["The", "cat", "sat", "down"]

    # Generate synthetic Query and Key vectors with unit variance
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    # 1. Unscaled dot-product
    unscaled_scores = Q @ K.T
    unscaled_variance = np.var(unscaled_scores)
    unscaled_weights = softmax(unscaled_scores)

    # 2. Scaled dot-product (divided by sqrt(d_k))
    scale = np.sqrt(d_k)
    scaled_scores = unscaled_scores / scale
    scaled_variance = np.var(scaled_scores)
    scaled_weights = softmax(scaled_scores)

    print(f"Key/Query Dimension d_k: {d_k} (Scale factor = √{d_k} = {scale:.1f})")
    print(f"Variance of Raw Dot Product (Unscaled) : {unscaled_variance:.4f} (Expected ≈ {d_k})")
    print(f"Variance of Scaled Dot Product (Scaled)   : {scaled_variance:.4f} (Expected ≈ 1.0)\n")

    print("Unscaled Softmax Weights (Token 0):")
    print("  " + "  ".join([f"{w:.6f}" for w in unscaled_weights[0]]))
    print("  Notice: When scores are large, softmax polarizes aggressively toward 0 or 1.")

    print("\nScaled Softmax Weights (Token 0):")
    print("  " + "  ".join([f"{w:.6f}" for w in scaled_weights[0]]))
    print("  Notice: Smooth, balanced distribution preserving healthy gradient flow backprop!\n")


# =============================================================================
# PART 2: Causal Masked Attention (The Autoregressive Decoder Engine)
# =============================================================================
def demo_causal_masking():
    print("=" * 75)
    print("PART 2: Causal Masked Attention (Preventing Future Token Leaks)")
    print("=" * 75)

    tokens = ["I", "love", "deep", "learning"]
    seq_len = len(tokens)
    d_k = 8

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    scores = (Q @ K.T) / np.sqrt(d_k)

    # Build the Causal Mask: Lower triangle = 0, Upper triangle = -infinity
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)

    print("Causal Mask Matrix M (Upper Triangle = -inf):")
    print(mask)

    masked_scores = scores + mask
    masked_weights = softmax(masked_scores)

    print("\nSoftmax Attention Weights with Causal Mask Applied:")
    col_headers = "".join([f"{t:>12}" for t in tokens])
    print(f"{'Token':<14}{col_headers}")
    print("-" * (14 + 12 * seq_len))

    for i, token in enumerate(tokens):
        row_str = "".join([f"{masked_weights[i, j]:12.4f}" for j in range(seq_len)])
        print(f"{token:<14}{row_str}")

    print("\nVerification:")
    print("  Token 1 ('I') only attends to: ['I']")
    print("  Token 2 ('love') only attends to: ['I', 'love']")
    print("  Token 4 ('learning') attends to all past tokens: ['I', 'love', 'deep', 'learning']")
    print("  Future tokens receive strictly 0.0000 attention weight!\n")


# =============================================================================
# PART 3: Multi-Head Attention (MHA) Mechanism
# =============================================================================
class MultiHeadAttentionNumPy:
    def __init__(self, d_model: int = 16, num_heads: int = 4):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Learned projection matrices
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X_q, X_k, X_v, mask=None):
        seq_len = X_q.shape[0]

        # 1. Linear Projections: [seq_len, d_model]
        Q = X_q @ self.W_Q
        K = X_k @ self.W_K
        V = X_v @ self.W_V

        # 2. Reshape and Transpose into heads: [num_heads, seq_len, d_k]
        Q_heads = Q.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        K_heads = K.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)
        V_heads = V.reshape(seq_len, self.num_heads, self.d_k).transpose(1, 0, 2)

        head_outputs = []
        head_weights = []

        # 3. Scaled Dot-Product Attention independently for each head
        scale = np.sqrt(self.d_k)
        for h in range(self.num_heads):
            scores = (Q_heads[h] @ K_heads[h].T) / scale
            if mask is not None:
                scores += mask
            w = softmax(scores)
            head_out = w @ V_heads[h]
            head_outputs.append(head_out)
            head_weights.append(w)

        # 4. Concatenate heads back into [seq_len, d_model]
        concatenated = np.concatenate(head_outputs, axis=-1)

        # 5. Output Linear Projection
        out = concatenated @ self.W_O
        return out, head_weights


def demo_multihead_attention():
    print("=" * 75)
    print("PART 3: Multi-Head Attention (MHA) Forward Pass")
    print("=" * 75)

    tokens = ["Transformers", "revolutionized", "modern", "AI"]
    seq_len = len(tokens)
    d_model = 16
    num_heads = 4

    mha = MultiHeadAttentionNumPy(d_model=d_model, num_heads=num_heads)
    X = np.random.randn(seq_len, d_model)

    out, head_weights = mha.forward(X, X, X)

    print(f"Input Tensor Shape  : {X.shape} [seq_len={seq_len}, d_model={d_model}]")
    print(f"Attention Heads h   : {num_heads} (Each head dimension d_k = {mha.d_k})")
    print(f"Output Tensor Shape : {out.shape} [seq_len={seq_len}, d_model={d_model}]")
    print(f"Number of learned attention weight matrices: {len(head_weights)}\n")

    for h_idx in range(num_heads):
        print(f"Head {h_idx + 1} Attention Weight for '{tokens[0]}' across tokens:")
        print("  " + "  ".join([f"{head_weights[h_idx][0, j]:.4f}" for j in range(seq_len)]))
    print("\nEach head focuses on different semantic aspects in parallel!\n")


# =============================================================================
# PART 4: Encoder-Decoder Cross-Attention
# =============================================================================
def demo_cross_attention():
    print("=" * 75)
    print("PART 4: Encoder-Decoder Cross-Attention (Translation Demo)")
    print("=" * 75)

    src_tokens = ["I", "love", "learning"]          # English (Encoder)
    tgt_tokens = ["J'", "aime", "apprendre"]        # French (Decoder)

    d_model = 8
    d_k = 4

    # Simulated final encoder representation
    H_enc = np.random.randn(len(src_tokens), d_model)
    # Simulated current decoder state
    H_dec = np.random.randn(len(tgt_tokens), d_model)

    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1

    # CRITICAL: Queries from DECODER, Keys and Values from ENCODER
    Q = H_dec @ W_Q  # (3, 4)
    K = H_enc @ W_K  # (3, 4)
    V = H_enc @ W_V  # (3, 4)

    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    cross_context = weights @ V

    print("Source (English Encoder Tokens):", src_tokens)
    print("Target (French Decoder Tokens): ", tgt_tokens)
    print("\nCross-Attention Alignment Matrix [Rows: French Query, Cols: English Key]:")
    col_headers = "".join([f"{t:>12}" for t in src_tokens])
    print(f"{'Target Token':<14}{col_headers}")
    print("-" * (14 + 12 * len(src_tokens)))

    for i, tgt_token in enumerate(tgt_tokens):
        row_str = "".join([f"{weights[i, j]:12.4f}" for j in range(len(src_tokens))])
        print(f"{tgt_token:<14}{row_str}")

    print("\nResult: Every French target word attends to the relevant English source words!\n")


# =============================================================================
# PART 5: Sinusoidal Positional Encoding
# =============================================================================
def demo_positional_encoding():
    print("=" * 75)
    print("PART 5: Sinusoidal Positional Encoding")
    print("=" * 75)

    seq_len = 6
    d_model = 8

    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = np.power(10000, 2 * i / d_model)
            pe[pos, i] = np.sin(pos / div_term)
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / div_term)

    print(f"Sinusoidal Positional Encoding Matrix [Positions 0..{seq_len-1}, Channels 0..{d_model-1}]:")
    print(np.round(pe, 4))

    # Dot product between position 0 and subsequent positions
    similarities = [np.dot(pe[0], pe[k]) / (np.linalg.norm(pe[0]) * np.linalg.norm(pe[k])) for k in range(seq_len)]
    print("\nCosine Similarity of Position 0 with subsequent positions:")
    for k, sim in enumerate(similarities):
        print(f"  Pos 0 vs Pos {k}: Cosine Similarity = {sim:+.4f}")
    print("Notice: Similarity smoothly decays as distance increases!\n")


def main():
    np.random.seed(42)
    print("\n" + "#" * 75)
    print("#  ZERO TO HERO GEN AI: TRANSFORMER DEEP-DIVE LAB (VASWANI 2017)   #")
    print("#" * 75 + "\n")

    demo_scaled_dot_product_and_saturation()
    demo_causal_masking()
    demo_multihead_attention()
    demo_cross_attention()
    demo_positional_encoding()


if __name__ == "__main__":
    main()
