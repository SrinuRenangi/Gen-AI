"""
=============================================================================
Hands-On Lab: LLM Hyperparameters & Decoding Mechanics in Python (NumPy)
=============================================================================
Course: Zero to Hero Gen AI — Module 02: API Interaction & Prompt Engineering
Topic: Parameters: Tuning Hyperparameters (Temperature, Top-P, Penalties, Max Tokens)

This lab implements and validates:
1. Temperature Scaling: Transforming raw neural logits across T=0.0, 0.2, 0.7, 1.5.
2. Top-P (Nucleus) vs Top-K Filtering: Adaptive probability mass pruning.
3. Frequency & Presence Penalties: Suppressing degenerate repetitive loops.
4. Max Tokens Truncation Auditor: Detecting incomplete output cutoffs.
5. The Master Profile Configurator: Presets for Code, JSON, Chat, and Creative.
=============================================================================
"""

import numpy as np


# =============================================================================
# PART 1: Softmax with Temperature Scaling
# =============================================================================
def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Computes Softmax with temperature scaling."""
    if temperature <= 0.001:
        # T -> 0: Deterministic Greedy Argmax
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1.0
        return probs

    scaled_logits = logits / temperature
    exp_scaled = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_scaled / np.sum(exp_scaled)


def demo_temperature_scaling():
    print("=" * 75)
    print("PART 1: Softmax with Temperature Scaling (Entropy & Sharpness)")
    print("=" * 75)

    words = ["Paris", "London", "Rome", "Madrid", "Tokyo"]
    raw_logits = np.array([4.0, 2.2, 1.5, 0.8, -0.5])

    print("Candidate Words: " + ", ".join(words))
    print("Raw Logits (z) : " + str(raw_logits) + "\n")

    temperatures = [0.0, 0.2, 0.7, 1.5]

    for T in temperatures:
        probs = softmax_with_temperature(raw_logits, temperature=T)
        label = "Greedy Argmax" if T == 0.0 else f"T = {T}"
        print(f"--- Temperature: {label} ---")
        for w, p in zip(words, probs):
            bar = "█" * int(p * 40)
            print(f"  {w:<8}: {p:>6.2%} |{bar}")
        print()

    print("Insight: T=0 concentrates 100% on Paris; T=1.5 flattens distribution, amplifying lower choices!\n")


# =============================================================================
# PART 2: Top-P (Nucleus) vs Top-K Filtering
# =============================================================================
def apply_top_k(probs: np.ndarray, k: int = 3) -> np.ndarray:
    """Keeps strictly the top-K highest probability tokens."""
    indices = np.argsort(probs)[::-1]
    keep_indices = indices[:k]
    filtered = np.zeros_like(probs)
    filtered[keep_indices] = probs[keep_indices]
    return filtered / np.sum(filtered)


def apply_top_p(probs: np.ndarray, p: float = 0.9) -> tuple[np.ndarray, list]:
    """Dynamically keeps the smallest subset whose cumulative sum >= P."""
    indices = np.argsort(probs)[::-1]
    sorted_probs = probs[indices]
    cumulative = np.cumsum(sorted_probs)

    cutoff = np.searchsorted(cumulative, p)
    keep_indices = indices[:cutoff + 1]

    filtered = np.zeros_like(probs)
    filtered[keep_indices] = probs[keep_indices]
    return (filtered / np.sum(filtered)), list(keep_indices)


def demo_top_p_vs_top_k():
    print("=" * 75)
    print("PART 2: Top-P (Nucleus Sampling) vs Top-K Filtering")
    print("=" * 75)

    words = ["door", "window", "box", "letter", "map", "chest", "drawer", "safe"]
    # Scenario: Moderately uncertain prediction
    raw_logits = np.array([2.5, 2.2, 2.0, 1.8, 1.5, 1.2, 0.5, 0.1])
    probs = softmax_with_temperature(raw_logits, temperature=1.0)

    print("Original Distribution:")
    for w, p in zip(words, probs):
        print(f"  {w:<8}: {p:.4f}")

    # Top-K (K=3)
    top_k_probs = apply_top_k(probs, k=3)
    k_survivors = [words[i] for i in range(len(words)) if top_k_probs[i] > 0]
    print(f"\nTop-K (K=3) Filtered Candidates : {k_survivors} (Fixed headcount: 3)")

    # Top-P (P=0.80)
    top_p_probs, p_indices = apply_top_p(probs, p=0.80)
    p_survivors = [words[i] for i in p_indices]
    print(f"Top-P (P=0.80) Nucleus Candidates: {p_survivors} (Adaptive: {len(p_survivors)} tokens hit 80% mass)")

    print("\nTakeaway: Top-P adapts dynamically to cumulative certainty rather than forcing a rigid count.\n")


# =============================================================================
# PART 3: Frequency & Presence Penalties
# =============================================================================
def apply_penalties(logits: np.ndarray, token_counts: np.ndarray, freq_penalty: float, pres_penalty: float) -> np.ndarray:
    """Modifies logits using frequency and presence penalty subtractions."""
    presence_mask = (token_counts > 0).astype(float)
    return logits - (freq_penalty * token_counts) - (pres_penalty * presence_mask)


def demo_penalties():
    print("=" * 75)
    print("PART 3: Frequency & Presence Penalties (Loop Suppression)")
    print("=" * 75)

    words = ["very", "extremely", "really", "quite", "important"]
    base_logits = np.array([4.0, 2.5, 2.2, 1.8, 3.5])

    # Simulate token history where 'very' has already been emitted 4 times!
    token_counts = np.array([4, 0, 0, 0, 1])

    print("Repeated Token Count in Output History:")
    for w, c in zip(words, token_counts):
        print(f"  '{w}': appeared {c} times")

    # 1. Without Penalties
    probs_unpenalized = softmax_with_temperature(base_logits, temperature=0.7)

    # 2. With Frequency Penalty = 0.6, Presence Penalty = 0.4
    adjusted_logits = apply_penalties(base_logits, token_counts, freq_penalty=0.6, pres_penalty=0.4)
    probs_penalized = softmax_with_temperature(adjusted_logits, temperature=0.7)

    print("\nProbability Comparison for Next Token:")
    print(f"{'Word':<12} | {'Unpenalized Prob':<18} | {'Penalized Prob':<16} | {'Shift'}")
    print("-" * 65)

    for i, w in enumerate(words):
        p_un = probs_unpenalized[i]
        p_pen = probs_penalized[i]
        delta = p_pen - p_un
        print(f"{w:<12} | {p_un:<18.2%} | {p_pen:<16.2%} | {delta:+.2%}")

    print("\nResult: 'very' plummeted from 59% down to 3%, forcing the model to pick rich synonyms!\n")


# =============================================================================
# PART 4: Max Tokens Truncation & Finish Reason Auditor
# =============================================================================
def demo_max_tokens_auditor():
    print("=" * 75)
    print("PART 4: Max Tokens Truncation & Finish Reason Audit")
    print("=" * 75)

    full_response_text = (
        '{"status": "SUCCESS", "records": [{"id": 1, "value": "Customer A"}, '
        '{"id": 2, "value": "Customer B"}]}'
    )

    max_tokens_budget = 12  # Artificially low token budget

    # Simulate truncation at token limit
    truncated_snippet = full_response_text[:48]  # Cuts off midway
    finish_reason = "length"

    print(f"Configured max_tokens limit: {max_tokens_budget}")
    print(f"Model Output Received:\n\"{truncated_snippet}...\"")
    print(f"API finish_reason: '{finish_reason}'")

    if finish_reason == "length":
        print("\n❌ CRITICAL ERROR DETECTED: Response was cut off by max_tokens limit!")
        print("   Attempting to parse truncated string as JSON will fail:")
        try:
            import json
            json.loads(truncated_snippet)
        except Exception as e:
            print(f"   JSONDecodeError: {e}")
        print("   Defense: Increase max_tokens to 1500+ and check finish_reason == 'stop'.\n")


# =============================================================================
# PART 5: The Master Profile Configurator
# =============================================================================
PROFILES = {
    "SQL_AND_CODE": {
        "temperature": 0.0,
        "top_p": 1.0,
        "freq_penalty": 0.0,
        "pres_penalty": 0.0,
        "max_tokens": 1500,
        "desc": "Deterministic syntax with zero hallucination."
    },
    "STRUCTURED_JSON": {
        "temperature": 0.0,
        "top_p": 1.0,
        "freq_penalty": 0.0,
        "pres_penalty": 0.0,
        "max_tokens": 2000,
        "desc": "Strict schema adherence with plenty of output room."
    },
    "ENTERPRISE_RAG": {
        "temperature": 0.2,
        "top_p": 0.9,
        "freq_penalty": 0.1,
        "pres_penalty": 0.1,
        "max_tokens": 600,
        "desc": "Factual grounding with smooth natural phrasing."
    },
    "CREATIVE_WRITING": {
        "temperature": 0.9,
        "top_p": 0.95,
        "freq_penalty": 0.3,
        "pres_penalty": 0.3,
        "max_tokens": 2500,
        "desc": "Expressive vocabulary, novel analogies, and broad exploration."
    }
}


def demo_profiles():
    print("=" * 75)
    print("PART 5: Master Hyperparameter Profile Catalog")
    print("=" * 75)

    for name, config in PROFILES.items():
        print(f"📌 Profile: {name}")
        print(f"   Settings   : Temp={config['temperature']} | Top-P={config['top_p']} | Freq={config['freq_penalty']} | Pres={config['pres_penalty']}")
        print(f"   Max Tokens : {config['max_tokens']}")
        print(f"   Rationale  : {config['desc']}\n")


def main():
    print("\n" + "#" * 75)
    print("#   ZERO TO HERO GEN AI: HYPERPARAMETER TUNING & DECODING LAB      #")
    print("#" * 75 + "\n")

    demo_temperature_scaling()
    demo_top_p_vs_top_k()
    demo_penalties()
    demo_max_tokens_auditor()
    demo_profiles()


if __name__ == "__main__":
    main()
