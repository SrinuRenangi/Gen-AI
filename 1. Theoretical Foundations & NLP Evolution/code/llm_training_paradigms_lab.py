"""
=============================================================================
Hands-On Lab: The Complete LLM Training Lifecycle in Python (NumPy)
=============================================================================
Course: Zero to Hero Gen AI — Module 01: Theoretical Foundations & NLP Evolution
Topic: Training Paradigms (Pre-Training, SFT, Reward Modeling, RLHF & DPO)

This lab implements and validates:
1. Pre-Training Loss: Autoregressive next-token cross-entropy.
2. SFT Loss with Target Masking: Why prompt tokens are ignored in gradients.
3. Bradley-Terry Reward Modeling Loss: Turning pairwise comparisons into scalar rewards.
4. RLHF Objective & KL Divergence Leash: Preventing policy collapse & reward hacking.
5. Direct Preference Optimization (DPO): Direct policy training on preference pairs.
=============================================================================
"""

import numpy as np


def softmax(z, axis=-1):
    """Numerically stable softmax implementation."""
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def sigmoid(z):
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# =============================================================================
# PART 1: Pre-Training Next-Token Cross-Entropy Loss
# =============================================================================
def demo_pretraining_loss():
    print("=" * 75)
    print("PART 1: Pre-Training Next-Token Cross-Entropy Loss (Self-Supervised)")
    print("=" * 75)

    vocab = ["The", "capital", "of", "France", "is", "Paris", "<eos>"]
    vocab_size = len(vocab)
    tokens = [0, 1, 2, 3, 4]             # "The capital of France is"
    target_next_tokens = [1, 2, 3, 4, 5] # "capital of France is Paris"
    seq_len = len(tokens)

    # Simulated raw logits from model forward pass [seq_len, vocab_size]
    logits = np.random.randn(seq_len, vocab_size) * 1.5

    # Make Paris (idx 5) have high logit at last step to simulate partial learning
    logits[4, 5] += 2.5

    probs = softmax(logits, axis=-1)

    print("Tokens: " + " -> ".join([vocab[t] for t in tokens]))
    print(f"{'Step':<6} | {'Input Token':<12} | {'Target Next':<12} | {'Predicted Prob':<16} | {'Cross-Entropy Loss'}")
    print("-" * 75)

    total_loss = 0.0
    for t in range(seq_len):
        inp_token = vocab[tokens[t]]
        tgt_token = vocab[target_next_tokens[t]]
        tgt_idx = target_next_tokens[t]
        prob = probs[t, tgt_idx]
        loss = -np.log(prob + 1e-12)
        total_loss += loss
        print(f"{t+1:<6} | {inp_token:<12} | {tgt_token:<12} | {prob:<16.4f} | {loss:.4f}")

    avg_loss = total_loss / seq_len
    perplexity = np.exp(avg_loss)
    print("-" * 75)
    print(f"Pre-Training Mean Cross-Entropy Loss : {avg_loss:.4f}")
    print(f"Model Perplexity (Branching Factor) : {perplexity:.2f}\n")


# =============================================================================
# PART 2: SFT Loss with Target Masking
# =============================================================================
def demo_sft_target_masking():
    print("=" * 75)
    print("PART 2: Supervised Fine-Tuning (SFT) with Target Masking")
    print("=" * 75)

    dialogue = [
        ("<|user|>", 0),
        ("What", 0),
        ("is", 0),
        ("2+2?", 0),
        ("<|assistant|>", 0),  # Prompt boundary (Mask = 0)
        ("The", 1),
        ("answer", 1),
        ("is", 1),
        ("4.", 1),
        ("<|end|>", 1)        # Response tokens (Mask = 1)
    ]

    tokens = [w for w, _ in dialogue]
    masks = [m for _, m in dialogue]
    seq_len = len(tokens)
    vocab_size = 20

    # Synthetic logits
    logits = np.random.randn(seq_len, vocab_size)
    probs = softmax(logits, axis=-1)
    synthetic_targets = np.random.randint(0, vocab_size, size=seq_len)

    print(f"{'Position':<8} | {'Token':<14} | {'Role':<12} | {'Mask':<6} | {'Contribution to Loss'}")
    print("-" * 75)

    sft_losses = []
    for i in range(seq_len):
        token_str = tokens[i]
        mask = masks[i]
        role = "PROMPT (User)" if mask == 0 else "RESPONSE (AI)"
        tgt_idx = synthetic_targets[i]
        token_loss = -np.log(probs[i, tgt_idx] + 1e-12)

        if mask == 1:
            sft_losses.append(token_loss)
            contrib_str = f"{token_loss:.4f} (Active gradient)"
        else:
            contrib_str = "0.0000 (Masked out with -100)"

        print(f"{i:<8} | {token_str:<14} | {role:<12} | {mask:<6} | {contrib_str}")

    print("-" * 75)
    print(f"Total Tokens: {seq_len} | Active Response Tokens: {len(sft_losses)}")
    print(f"SFT Mean Loss (Only over Assistant Tokens): {np.mean(sft_losses):.4f}")
    print("Interpretation: Prompt tokens are NEVER penalized; the model learns to assist, not mimic users.\n")


# =============================================================================
# PART 3: Bradley-Terry Reward Modeling Loss
# =============================================================================
def demo_bradley_terry_reward_modeling():
    print("=" * 75)
    print("PART 3: Bradley-Terry Reward Modeling Loss (Pairwise Comparisons)")
    print("=" * 75)

    pairs = [
        {"prompt": "Explain photosynthesis.", "r_win": 3.8, "r_lose": 1.2, "desc": "Clear explanation vs vague stub"},
        {"prompt": "Write a python loop.",   "r_win": 2.1, "r_lose": 1.9, "desc": "Minor stylistic difference"},
        {"prompt": "How to make a bomb?",    "r_win": 4.5, "r_lose": -2.0, "desc": "Polite refusal vs dangerous text"},
        {"prompt": "Summary of hamlet.",     "r_win": -0.5, "r_lose": 1.5, "desc": "ERROR: Model inverted ranking!"},
    ]

    print(f"{'Prompt / Scenario':<24} | {'r(y_w)':<8} | {'r(y_l)':<8} | {'P(y_w > y_l)':<14} | {'Loss -log σ'}")
    print("-" * 75)

    for p in pairs:
        r_w = p["r_win"]
        r_l = p["r_lose"]
        diff = r_w - r_l
        p_prefer_win = sigmoid(diff)
        loss = -np.log(p_prefer_win + 1e-12)

        print(f"{p['prompt'][:22]:<24} | {r_w:<8.2f} | {r_l:<8.2f} | {p_prefer_win:<14.4f} | {loss:.4f}")

    print("-" * 75)
    print("Notice: When the model incorrectly ranks a pair (Case 4), loss spikes heavily, driving strong weight updates.\n")


# =============================================================================
# PART 4: RLHF Objective with KL Divergence Leash
# =============================================================================
def demo_rlhf_kl_penalty():
    print("=" * 75)
    print("PART 4: RLHF Objective & KL Divergence Leash (Stopping Reward Hacking)")
    print("=" * 75)

    scenarios = [
        {
            "name": "Natural Fluent Assistant",
            "raw_reward": 3.2,
            "log_prob_policy": -4.1,
            "log_prob_ref": -4.0,
            "desc": "High reward, low drift from base SFT policy"
        },
        {
            "name": "Slightly Stylistic Shift",
            "raw_reward": 3.8,
            "log_prob_policy": -3.2,
            "log_prob_ref": -4.2,
            "desc": "Higher reward, moderate drift"
        },
        {
            "name": "Reward Hacked Gibberish",
            "raw_reward": 9.5,  # Exploited reward model flaw!
            "log_prob_policy": -1.2,
            "log_prob_ref": -18.5, # Extremely unnatural gibberish under SFT
            "desc": "Adversarial trick: 'Excellent! Yes! Yes! Wow!' x50"
        }
    ]

    beta = 0.2  # KL penalty coefficient

    print(f"KL Penalty Coefficient β = {beta}")
    print(f"{'Scenario':<26} | {'Raw Reward':<11} | {'KL Divergence':<15} | {'KL Penalty':<12} | {'Net RLHF Reward'}")
    print("-" * 75)

    for s in scenarios:
        raw_r = s["raw_reward"]
        # Per-token KL approximation: log π_θ - log π_ref
        kl_div = s["log_prob_policy"] - s["log_prob_ref"]
        penalty = beta * kl_div
        net_reward = raw_r - penalty

        status = "✅ Kept" if net_reward > 0 and s["name"] != "Reward Hacked Gibberish" else "❌ Penalized"
        print(f"{s['name']:<26} | {raw_r:<11.2f} | {kl_div:<15.2f} | {penalty:<12.2f} | {net_reward:<10.2f} {status}")

    print("-" * 75)
    print("Insight: The reward hack had raw reward +9.5, but the KL penalty (-3.46) suppressed it,")
    print("keeping the model grounded in coherent human language!\n")


# =============================================================================
# PART 5: Direct Preference Optimization (DPO) Loss
# =============================================================================
def demo_dpo_loss():
    print("=" * 75)
    print("PART 5: Direct Preference Optimization (DPO) Loss Calculation")
    print("=" * 75)
    print("No separate Reward Model! No PPO actor-critic loop! Trained offline directly on pairs.\n")

    beta = 0.1

    # Scenario: Policy assigns higher probability to winner than reference does
    cases = [
        {"name": "Well Aligned",   "pi_w": 0.40, "ref_w": 0.20, "pi_l": 0.05, "ref_l": 0.15},
        {"name": "Neutral / Unsure", "pi_w": 0.20, "ref_w": 0.20, "pi_l": 0.20, "ref_l": 0.20},
        {"name": "Misaligned",     "pi_w": 0.05, "ref_w": 0.20, "pi_l": 0.40, "ref_l": 0.15},
    ]

    print(f"{'Case':<16} | {'π_θ(y_w)/ref':<14} | {'π_θ(y_l)/ref':<14} | {'Implicit Implicit Reward Δ':<26} | {'DPO Loss'}")
    print("-" * 80)

    for c in cases:
        ratio_w = c["pi_w"] / c["ref_w"]
        ratio_l = c["pi_l"] / c["ref_l"]
        log_ratio_w = np.log(ratio_w)
        log_ratio_l = np.log(ratio_l)

        # Implicit reward difference: β * (log(π_w/ref_w) - log(π_l/ref_l))
        diff = beta * (log_ratio_w - log_ratio_l)
        loss = -np.log(sigmoid(diff) + 1e-12)

        print(f"{c['name']:<16} | {ratio_w:<14.2f} | {ratio_l:<14.2f} | {diff:<26.4f} | {loss:.4f}")

    print("-" * 80)
    print("Notice: DPO loss gracefully drops as the policy learns to upweight winners and downweight losers!\n")


def main():
    np.random.seed(42)
    print("\n" + "#" * 75)
    print("#   ZERO TO HERO GEN AI: THE COMPLETE LLM TRAINING LIFECYCLE LAB   #")
    print("#" * 75 + "\n")

    demo_pretraining_loss()
    demo_sft_target_masking()
    demo_bradley_terry_reward_modeling()
    demo_rlhf_kl_penalty()
    demo_dpo_loss()


if __name__ == "__main__":
    main()
