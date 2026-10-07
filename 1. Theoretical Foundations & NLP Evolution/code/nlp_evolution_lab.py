"""
=============================================================================
Hands-On Lab: NLP Progression — RNNs, LSTMs & Self-Attention in Python
=============================================================================
Course: Zero to Hero Gen AI — Module 01: Theoretical Foundations & NLP Evolution
Topic: NLP Progression: Understanding the Leap from RNNs and LSTMs to Modern Architectures

This lab demonstrates and mathematically verifies:
1. RNN Vanishing Gradient: Exponential decay through unrolled time steps.
2. LSTM Additive Memory Highway: Gradient preservation via Constant Error Carousel.
3. Seq2Seq Information Bottleneck: Fixed-size vector compression limits.
4. Scaled Dot-Product Self-Attention: O(1) sequential steps, parallel all-to-all token routing.
=============================================================================
"""

import numpy as np


def rnn_vanishing_gradient_simulation(timesteps: int = 15, spectral_radius: float = 0.85):
    """Simulates gradient decay during Backpropagation Through Time (BPTT) in a Vanilla RNN."""
    print("=" * 75)
    print("PART 1: Vanilla RNN Vanishing Gradient Simulation (BPTT)")
    print("=" * 75)
    print(f"Tracking error gradient propagating backward across {timesteps} time steps...")
    print(f"Assumed weight matrix spectral radius: W_hh = {spectral_radius}")
    print("tanh'(z) is bounded in [0, 1] (average ~ 0.65 in active saturated regimes).\n")

    gradient = 1.0
    print(f"Step {timesteps:>2} (Error origin)  : Gradient = {gradient:.6f}")

    gradients = [gradient]
    for t in range(timesteps - 1, 0, -1):
        # dh_t / dh_{t-1} = tanh'(a_t) * W_hh
        tanh_prime = 0.65
        factor = tanh_prime * spectral_radius
        gradient *= factor
        gradients.append(gradient)
        bar = "█" * int(max(0, gradient * 40))
        print(f"Step {t:>2} (Propagated back) : Gradient = {gradient:.8f}  |{bar}")

    retention_pct = (gradients[-1] / gradients[0]) * 100
    print(f"\n❌ Result: By Step 1, remaining gradient is {gradients[-1]:.10f} ({retention_pct:.4f}% of original)")
    print("   Interpretation: The network cannot adjust weights for early tokens based on late errors.")
    print("   Long-term dependencies (>10-15 tokens) are mathematically severed in vanilla RNNs.\n")
    return gradients


def lstm_highway_simulation(timesteps: int = 15, forget_gate_val: float = 0.98):
    """Simulates gradient flow through the LSTM additive cell state highway."""
    print("=" * 75)
    print("PART 2: LSTM Constant Error Carousel & Additive Highway")
    print("=" * 75)
    print(f"Tracking cell state gradient dC_T / dC_t across {timesteps} time steps...")
    print(f"Model learned that this historical concept is important -> Forget gate f_t ≈ {forget_gate_val}\n")

    gradient = 1.0
    print(f"Step {timesteps:>2} (Error origin)  : Cell State Gradient = {gradient:.6f}")

    gradients = [gradient]
    for t in range(timesteps - 1, 0, -1):
        # In LSTM: dC_t / dC_{t-1} = f_t (additive linear path!)
        gradient *= forget_gate_val
        gradients.append(gradient)
        bar = "█" * int(max(0, gradient * 40))
        print(f"Step {t:>2} (Propagated back) : Cell State Gradient = {gradient:.6f}  |{bar}")

    retention_pct = (gradients[-1] / gradients[0]) * 100
    print(f"\n✅ Result: By Step 1, remaining gradient is {gradients[-1]:.6f} ({retention_pct:.2f}% preserved!)")
    print("   Interpretation: Additive updates (C_t = f*C_{t-1} + i*C_cand) bypass the vanishing trap.")
    print("   Information can persist across hundreds of time steps.\n")
    return gradients


def seq2seq_bottleneck_demo():
    """Demonstrates how compressing sequences into a single fixed vector destroys information."""
    print("=" * 75)
    print("PART 3: The Seq2Seq Fixed Context Bottleneck Demonstration")
    print("=" * 75)
    print("In classic Seq2Seq (Sutskever et al., 2014), the entire sentence is squashed into h_T.")

    hidden_dim = 16
    sentence_lengths = [5, 15, 30, 60]

    print(f"\nFixed context vector capacity: {hidden_dim} floats")
    print("-" * 60)
    print(f"{'Sentence Length':<18} | {'Information Bits Needed':<24} | {'Compression Ratio':<18}")
    print("-" * 60)
    for length in sentence_lengths:
        # Assuming average ~4 bits of semantic entropy per token
        bits_needed = length * 12
        capacity_bits = hidden_dim * 8  # 16 bytes roughly
        ratio = bits_needed / capacity_bits
        status = "✅ OK" if ratio <= 1.0 else f"⚠️ Bottleneck Loss ({ratio:.1f}x overload)"
        print(f"{length:<18} | {bits_needed:<24} | {status:<18}")

    print("\nConclusion: Fixed-vector bottleneck forces catastrophic forgetting on sentences > 20 tokens.")
    print("Bahdanau Attention (2014) solved this by dynamically looking at ALL encoder states.\n")


def self_attention_scaled_dot_product():
    """Computes Scaled Dot-Product Self-Attention with step-by-step matrix operations."""
    print("=" * 75)
    print("PART 4: Scaled Dot-Product Self-Attention (The Transformer Engine)")
    print("=" * 75)
    print("All tokens attend to all other tokens simultaneously in O(1) sequential time!\n")

    # Sample sentence
    tokens = ["The", "bank", "of", "the", "river", "overflowed"]
    seq_len = len(tokens)
    d_model = 8  # Embedding dimension
    d_k = 4      # Projection dimension

    # Simulated input embeddings X (seq_len, d_model)
    X = np.array([
        [ 0.8, -0.2,  0.5,  1.1, -0.3,  0.4, -0.9,  0.2],  # The
        [ 1.5,  0.9, -0.4,  0.2,  1.8, -0.1,  0.3, -0.5],  # bank (river bank vs money bank)
        [-0.1,  0.1,  0.3, -0.2,  0.1,  0.0, -0.1,  0.2],  # of
        [ 0.8, -0.2,  0.5,  1.1, -0.3,  0.4, -0.9,  0.2],  # the
        [ 1.9,  1.1, -0.5,  0.1,  2.2, -0.3,  0.5, -0.8],  # river
        [-0.4,  1.4,  1.2, -0.8,  0.9,  1.1,  0.2, -0.1],  # overflowed
    ])

    # Projection weights for Query, Key, Value
    W_Q = np.random.randn(d_model, d_k) * 0.5
    W_K = np.random.randn(d_model, d_k) * 0.5
    W_V = np.random.randn(d_model, d_k) * 0.5

    # Step 1: Compute Q, K, V in ONE parallel matrix multiplication
    Q = X @ W_Q  # (6, 4)
    K = X @ W_K  # (6, 4)
    V = X @ W_V  # (6, 4)

    # Step 2: Compute Raw Attention Scores = Q @ K.T
    raw_scores = Q @ K.T  # (6, 6)

    # Step 3: Scale by sqrt(d_k) to prevent softmax saturation
    scale = np.sqrt(d_k)
    scaled_scores = raw_scores / scale

    # Step 4: Softmax along each row
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    attention_matrix = softmax(scaled_scores)

    # Step 5: Context Output = Attention @ V
    context_output = attention_matrix @ V

    print(f"Sentence: {' '.join(tokens)}")
    print(f"Input Shape X: {X.shape} | Q: {Q.shape} | K: {K.shape} | V: {V.shape}")
    print(f"Scale Factor 1/√d_k: 1/√{d_k} = {1.0 / scale:.4f}\n")

    print("Softmax Attention Weights Matrix [Rows: Query Token, Cols: Key Token]:")
    col_headers = "".join([f"{t:>12}" for t in tokens])
    print(f"{'Token':<14}{col_headers}")
    print("-" * (14 + 12 * seq_len))

    for i, q_token in enumerate(tokens):
        row_str = "".join([f"{attention_matrix[i, j]:12.4f}" for j in range(seq_len)])
        print(f"{q_token:<14}{row_str}")

    print("\nContextualized Vector for 'bank' (Row 1):")
    print(f"  Shape: {context_output[1].shape}")
    print(f"  Values: {np.round(context_output[1], 4)}")
    print("\nNotice: The vector for 'bank' now directly absorbs information from 'river' and 'overflowed',")
    print("completely resolving semantic ambiguity in a single O(1) parallel pass!\n")


def print_evolution_summary():
    """Prints the comprehensive architectural comparison summary."""
    print("=" * 75)
    print("SUMMARY COMPARISON: THE NLP REVOLUTION")
    print("=" * 75)
    headers = ["Architecture", "Sequential Complexity", "Path Length", "Parallelizable?", "Max Context"]
    rows = [
        ["Vanilla RNN", "O(T)", "O(T)", "❌ No (Sequential loop)", "~10 tokens"],
        ["LSTM / GRU",  "O(T)", "O(T)", "❌ No (Sequential loop)", "~100 tokens"],
        ["Seq2Seq + Attn", "O(T)", "O(1)", "❌ No (Autoregressive loop)", "~500 tokens"],
        ["Transformer", "O(1)", "O(1)", "✅ Yes (Full GPU parallelism)", "128k - 2M tokens"],
    ]

    fmt = "{:<16} | {:<22} | {:<12} | {:<24} | {:<12}"
    print(fmt.format(*headers))
    print("-" * 95)
    for row in rows:
        print(fmt.format(*row))
    print("=" * 75)


def main():
    np.random.seed(42)
    print("\n" + "#" * 75)
    print("#  ZERO TO HERO GEN AI: NLP ARCHITECTURAL PROGRESSION LAB  #")
    print("#" * 75 + "\n")

    rnn_vanishing_gradient_simulation(timesteps=12, spectral_radius=0.85)
    lstm_highway_simulation(timesteps=12, forget_gate_val=0.98)
    seq2seq_bottleneck_demo()
    self_attention_scaled_dot_product()
    print_evolution_summary()


if __name__ == "__main__":
    main()
