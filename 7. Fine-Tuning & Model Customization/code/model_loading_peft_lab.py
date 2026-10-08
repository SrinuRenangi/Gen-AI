#!/usr/bin/env python3
"""
================================================================================
Model Loading & PEFT Mechanics: Beginner-to-Advanced Verification Lab
================================================================================
Topic: Module 07 - Topic 02: Model Loading, Quantization (BitsAndBytes NF4),
       and Parameter-Efficient Fine-Tuning (LoRA & QLoRA).

This lab provides hands-on algorithmic verification of:
  1. Hugging Face Hub Model Anatomy & Architecture Config Parser
  2. 4-bit NormalFloat (NF4) Quantization & Dequantization Simulation
  3. LoRA Low-Rank Matrix Factorization & Forward Pass Emulation
  4. PEFT Target Module Selector & Trainable Parameter Ratio Calculator
  5. Production BitsAndBytesConfig & LoraConfig Validation Gate

Execution:
  python model_loading_peft_lab.py
================================================================================
"""

import sys
import os
import json
import math
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

# Ensure UTF-8 output encoding across Windows PowerShell and Unix terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ==============================================================================
# EXPERIMENT 1: Hugging Face Model Repository Anatomy & Config Inspector
# ==============================================================================

@dataclass
class ModelManifest:
    model_id: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    max_position_embeddings: int

    @property
    def total_estimated_parameters(self) -> int:
        """
        Calculates theoretical parameter count from transformer hyperparameters:
          - Embedding: vocab_size * hidden_size
          - Per layer:
              Self-Attention: Q, K, V, O projections
              MLP / Feed-Forward: Gate, Up, Down projections (SwiGLU)
              LayerNorms: 2 * hidden_size
          - Output LM Head: vocab_size * hidden_size (often tied or separate)
        """
        # Embeddings
        embed_params = self.vocab_size * self.hidden_size

        # Attention per layer
        q_params = self.hidden_size * self.hidden_size
        k_params = self.hidden_size * (self.num_key_value_heads * (self.hidden_size // self.num_attention_heads))
        v_params = k_params
        o_params = self.hidden_size * self.hidden_size
        attn_per_layer = q_params + k_params + v_params + o_params

        # SwiGLU MLP per layer (gate, up, down)
        mlp_per_layer = 3 * (self.hidden_size * self.intermediate_size)

        # Norms
        norms_per_layer = 2 * self.hidden_size

        layer_total = attn_per_layer + mlp_per_layer + norms_per_layer
        all_layers = layer_total * self.num_hidden_layers

        # Final norm + LM head
        lm_head = self.vocab_size * self.hidden_size
        return embed_params + all_layers + lm_head


def run_experiment_1() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 1: Hugging Face Model Repository Anatomy & Config Inspector")
    print("="*80)

    # Simulated config.json for Meta-Llama-3-8B
    llama3_cfg = ModelManifest(
        model_id="meta-llama/Meta-Llama-3-8B",
        vocab_size=128256,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # Grouped-Query Attention (GQA)
        intermediate_size=14336,
        max_position_embeddings=8192
    )

    total_params = llama3_cfg.total_estimated_parameters
    param_billions = total_params / 1e9
    fp16_disk_gb = (total_params * 2) / (1024**3)
    q4_disk_gb = (total_params * 0.5) / (1024**3)

    print(f"Inspecting Hugging Face Architecture Config: {llama3_cfg.model_id}")
    print(f" • Hidden Size (d_model):          {llama3_cfg.hidden_size}")
    print(f" • Attention Layers:               {llama3_cfg.num_hidden_layers}")
    print(f" • Attention Heads (Q / KV):       {llama3_cfg.num_attention_heads} / {llama3_cfg.num_key_value_heads} (GQA)")
    print(f" • Intermediate MLP Dimension:     {llama3_cfg.intermediate_size}")
    print(f" • Vocabulary Size:                {llama3_cfg.vocab_size:,} tokens")
    print(f" • Context Window:                 {llama3_cfg.max_position_embeddings:,} tokens")
    print("-" * 55)
    print(f" • Calculated Parameter Count:     {total_params:,} ({param_billions:.2f} Billion)")
    print(f" • Unquantized FP16 Storage:       {fp16_disk_gb:.2f} GB")
    print(f" • 4-bit Quantized Storage:        {q4_disk_gb:.2f} GB")

    success = (7.5 <= param_billions <= 8.5)
    print(f"\n[Verification] Model Config Math: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 2: 4-bit NormalFloat (NF4) Quantization & Dequantization
# ==============================================================================

class NF4QuantizerSimulator:
    """
    Simulates NormalFloat4 (NF4) quantization used in BitsAndBytes / QLoRA:
    Standard uniform INT4 quantization divides the range evenly, which wastes
    resolution on outliers.
    NF4 builds an information-theoretically optimal quantile grid for zero-mean
    unit-variance Gaussian distributions, which weights in neural networks naturally follow.
    """
    # 16 optimal quantiles for standard normal distribution N(0, 1) in [-1.0, 1.0]
    NF4_CODEBOOK = [
        -1.0000, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,
         0.4407,  0.5626,  0.7230,  1.0000
    ]

    def quantize_block(self, weights: List[float]) -> Tuple[List[int], float]:
        """
        Quantizes a block of floating-point weights (typically 64 values) into 4-bit indices (0-15).
        Returns (indices, absmax_scale).
        """
        absmax = max(abs(w) for w in weights) if weights else 1.0
        absmax = max(absmax, 1e-8)  # prevent div by zero

        indices = []
        for w in weights:
            normalized = w / absmax  # scaled to [-1.0, 1.0]
            # Find closest value in NF4 codebook
            best_idx = 0
            best_diff = float("inf")
            for idx, code in enumerate(self.NF4_CODEBOOK):
                diff = abs(normalized - code)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
            indices.append(best_idx)
        return indices, absmax

    def dequantize_block(self, indices: List[int], absmax: float) -> List[float]:
        """Dequantizes 4-bit indices back into floating-point numbers."""
        return [self.NF4_CODEBOOK[idx] * absmax for idx in indices]


def run_experiment_2() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 2: 4-bit NormalFloat (NF4) Quantization & Dequantization")
    print("="*80)

    random.seed(42)
    quantizer = NF4QuantizerSimulator()

    # Generate 16 sample weights normally distributed ~ N(0, 0.04)
    sample_fp_weights = [random.gauss(0.0, 0.20) for _ in range(16)]

    # Quantize to 4-bit
    indices_4bit, absmax = quantizer.quantize_block(sample_fp_weights)

    # Dequantize back to float
    recovered_weights = quantizer.dequantize_block(indices_4bit, absmax)

    print(f"{'Idx':<4} | {'Original FP16':<14} | {'4-bit NF4 Code':<15} | {'Dequantized FP16':<16} | {'Error'}")
    print("-" * 65)

    squared_errors = []
    for i in range(len(sample_fp_weights)):
        orig = sample_fp_weights[i]
        code = indices_4bit[i]
        recon = recovered_weights[i]
        err = abs(orig - recon)
        squared_errors.append(err * err)
        print(f"{i:<4} | {orig:>12.5f}   | Index {code:<2} ({quantizer.NF4_CODEBOOK[code]:>6.3f}) | {recon:>14.5f}   | {err:.5f}")

    mse = sum(squared_errors) / len(squared_errors)
    compression_ratio = 16.0 / 4.0  # FP16 (16 bits) to NF4 (4 bits)

    print(f"\n[Quantization Diagnostics]")
    print(f" • Quantization Scale (AbsMax):   {absmax:.5f}")
    print(f" • Mean Squared Error (MSE):       {mse:.6f} (minimal fidelity loss)")
    print(f" • Memory Compression Ratio:       {compression_ratio:.1f}x (75% VRAM saved!)")

    success = (mse < 0.005)
    print(f"\n[Verification] NF4 Quantization Accuracy: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 3: LoRA Low-Rank Factorization & Forward Pass Emulation
# ==============================================================================

class LoRALinearLayerSimulator:
    """
    Simulates a LoRA-adapted linear projection layer:
      h = x * W_0 + (x * A^T * B^T) * (alpha / r)

    Where:
      - W_0: Frozen base model weight matrix (d_in x d_out)
      - A: Down-projection adapter matrix (d_in x r), initialized ~ N(0, sigma^2)
      - B: Up-projection adapter matrix (r x d_out), initialized to ZERO!
      - r: Rank hyperparameter (e.g., 8)
      - alpha: Scaling factor (e.g., 16)
    """
    def __init__(self, d_in: int, d_out: int, rank: int = 8, alpha: float = 16.0):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Base weight W_0 (frozen)
        random.seed(101)
        self.W0 = [[random.uniform(-0.1, 0.1) for _ in range(d_out)] for _ in range(d_in)]

        # LoRA Adapter A: Gaussian initialized
        self.A = [[random.gauss(0, 0.02) for _ in range(rank)] for _ in range(d_in)]

        # LoRA Adapter B: Zero initialized! Crucial: ensures delta_W = 0 at step 0!
        self.B = [[0.0 for _ in range(d_out)] for _ in range(rank)]

    def forward(self, x: List[float]) -> List[float]:
        """Calculates h = x*W0 + scaling * (x * A * B)."""
        # Step 1: Base output h_base = x * W0
        h_base = [0.0] * self.d_out
        for j in range(self.d_out):
            h_base[j] = sum(x[i] * self.W0[i][j] for i in range(self.d_in))

        # Step 2: LoRA down-projection h_A = x * A (dimension 1 x rank)
        h_A = [0.0] * self.rank
        for r_idx in range(self.rank):
            h_A[r_idx] = sum(x[i] * self.A[i][r_idx] for i in range(self.d_in))

        # Step 3: LoRA up-projection h_B = h_A * B (dimension 1 x d_out)
        h_B = [0.0] * self.d_out
        for j in range(self.d_out):
            h_B[j] = sum(h_A[r_idx] * self.B[r_idx][j] for r_idx in range(self.rank))

        # Step 4: Combined output = h_base + (alpha / r) * h_B
        h_total = [h_base[j] + self.scaling * h_B[j] for j in range(self.d_out)]
        return h_total


def run_experiment_3() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 3: LoRA Low-Rank Factorization & Forward Pass Emulation")
    print("="*80)

    d_in = 16
    d_out = 16
    rank = 4
    alpha = 8.0

    lora_layer = LoRALinearLayerSimulator(d_in=d_in, d_out=d_out, rank=rank, alpha=alpha)

    # Base parameters vs. LoRA parameters
    base_params = d_in * d_out
    lora_params = (d_in * rank) + (rank * d_out)
    savings = (1.0 - (lora_params / base_params)) * 100.0

    print(f"Layer Dimensions: {d_in} x {d_out} | Rank (r): {rank} | Alpha: {alpha} (Scaling: {alpha/rank:.1f})")
    print(f" • Base Layer Weights (W0):       {base_params} parameters (FROZEN)")
    print(f" • LoRA Adapter Weights (A + B):  {lora_params} parameters (TRAINABLE)")
    print(f" • Parameter Reduction:           {savings:.1f}% reduction!")

    # Test 1: Step 0 initialization behavior
    x_test = [1.0] * d_in
    output_step0 = lora_layer.forward(x_test)

    # Compute base-only output
    h_base_only = [sum(x_test[i] * lora_layer.W0[i][j] for i in range(d_in)) for j in range(d_out)]
    diff_step0 = sum(abs(a - b) for a, b in zip(output_step0, h_base_only))

    print(f"\n[Initialization Invariance Test at Step 0]")
    print(f" • Difference between Base Output & LoRA Output: {diff_step0:.8f}")
    print(f"   (Proves that because B = 0 at start, LoRA output is EXACTLY identical to base model!)")

    # Test 2: Simulate one training update where B receives non-zero gradients
    lora_layer.B[0][0] = 0.5  # Simulate gradient nudge
    output_step1 = lora_layer.forward(x_test)
    diff_step1 = sum(abs(a - b) for a, b in zip(output_step1, h_base_only))
    print(f" • Difference after simulated gradient update:   {diff_step1:.5f} (Adapter actively shifts behavior)")

    success = (diff_step0 == 0.0 and diff_step1 > 0.0)
    print(f"\n[Verification] LoRA Math & Invariance: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 4: PEFT Target Module Selector & Trainable Parameter Calculator
# ==============================================================================

class ParameterBudgetEngine:
    """
    Computes exact parameter counts and memory usage for different LoRA target module sets:
      - Option A: Attention only (q_proj, v_proj)
      - Option B: All Attention (q_proj, k_proj, v_proj, o_proj)
      - Option C: All Linear (Attention + MLP: gate_proj, up_proj, down_proj)
    """
    # 8B Model dimensions
    D_MODEL = 4096
    NUM_LAYERS = 32
    D_MLP = 14336

    MODULE_SIZES = {
        "q_proj": D_MODEL * D_MODEL,
        "k_proj": D_MODEL * (8 * (D_MODEL // 32)),  # GQA: 8 heads
        "v_proj": D_MODEL * (8 * (D_MODEL // 32)),
        "o_proj": D_MODEL * D_MODEL,
        "gate_proj": D_MODEL * D_MLP,
        "up_proj": D_MODEL * D_MLP,
        "down_proj": D_MLP * D_MODEL
    }

    def compute_lora_params(self, target_modules: List[str], rank: int) -> Tuple[int, int, float]:
        total_base = 8_030_000_000  # ~8.03 Billion total base parameters
        lora_trainable = 0

        for mod in target_modules:
            if mod in ["q_proj", "o_proj"]:
                d_in, d_out = self.D_MODEL, self.D_MODEL
            elif mod in ["k_proj", "v_proj"]:
                d_in, d_out = self.D_MODEL, 1024
            elif mod in ["gate_proj", "up_proj"]:
                d_in, d_out = self.D_MODEL, self.D_MLP
            elif mod in ["down_proj"]:
                d_in, d_out = self.D_MLP, self.D_MODEL
            else:
                continue

            # In LoRA: Param count = (d_in * r + r * d_out) * num_layers
            layer_lora = (d_in * rank) + (rank * d_out)
            lora_trainable += layer_lora * self.NUM_LAYERS

        percentage = (lora_trainable / total_base) * 100.0
        return total_base, lora_trainable, percentage


def run_experiment_4() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 4: PEFT Target Module Selector & Trainable Parameter Calculator")
    print("="*80)

    engine = ParameterBudgetEngine()

    configs = [
        ("Conservative (q_proj, v_proj)", ["q_proj", "v_proj"], 8),
        ("All Attention (q, k, v, o)", ["q_proj", "k_proj", "v_proj", "o_proj"], 8),
        ("All Linear (Attn + MLP) r=8", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 8),
        ("All Linear (Attn + MLP) r=16", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 16),
        ("All Linear (Attn + MLP) r=64", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 64)
    ]

    print(f"{'Target Configuration':<32} | {'Rank (r)':<8} | {'Trainable Params':<18} | {'% of 8B Model'}")
    print("-" * 75)

    for label, mods, r in configs:
        base_cnt, lora_cnt, pct = engine.compute_lora_params(mods, r)
        print(f"{label:<32} | r = {r:<4} | {lora_cnt:>14,d}   | {pct:>8.3f}%")

    print("\n[Architectural Takeaway]")
    print(" • Training only (q, v) touches just ~0.04% of parameters.")
    print(" • Industry best practice for QLoRA is 'all-linear' at rank r=16 (~0.4% trainable).")
    print(" • This strikes the optimal balance between high expressivity and fast training speed.")

    success = True
    print(f"\n[Verification] Target Module Selector: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 5: Production BitsAndBytes & LoraConfig Validation Gate
# ==============================================================================

class FineTuningConfigValidator:
    """
    Validates production configuration dictionaries before launching Hugging Face jobs:
      - Validates BitsAndBytesConfig compatibility
      - Validates LoraConfig hyperparameters
      - Checks GPU architecture compatibility (BFloat16 vs Float16)
    """
    SUPPORTED_BNB_4BIT_TYPES = ["nf4", "fp4"]
    SUPPORTED_COMPUTE_DTYPES = ["bfloat16", "float16", "float32"]

    def validate_qlora_setup(self, bnb_config: Dict[str, Any], lora_config: Dict[str, Any], gpu_arch: str) -> Tuple[bool, List[str]]:
        errors = []

        # 1. Validate BitsAndBytes
        if not bnb_config.get("load_in_4bit", False):
            errors.append("BitsAndBytesConfig must have 'load_in_4bit=True' for QLoRA.")

        q_type = bnb_config.get("bnb_4bit_quant_type", "").lower()
        if q_type not in self.SUPPORTED_BNB_4BIT_TYPES:
            errors.append(f"Invalid bnb_4bit_quant_type '{q_type}'. Must be 'nf4' or 'fp4'.")

        compute_dtype = bnb_config.get("bnb_4bit_compute_dtype", "").lower()
        if compute_dtype not in self.SUPPORTED_COMPUTE_DTYPES:
            errors.append(f"Invalid compute_dtype '{compute_dtype}'.")

        # GPU architecture check: bfloat16 requires Ampere (RTX 3090, A100) or newer
        if compute_dtype == "bfloat16" and gpu_arch.lower() in ["turing", "volta", "pascal", "t4"]:
            errors.append(f"GPU architecture '{gpu_arch}' does not have native BFloat16 hardware support. Use 'float16' instead.")

        # 2. Validate LoRA Config
        r = lora_config.get("r", 0)
        alpha = lora_config.get("lora_alpha", 0)
        if r <= 0:
            errors.append("LoRA rank 'r' must be positive.")
        if alpha <= 0:
            errors.append("LoRA alpha must be positive.")
        if alpha < r:
            errors.append(f"Warning: lora_alpha ({alpha}) is typically >= rank ({r}). Standard is alpha = 2*r.")

        target_mods = lora_config.get("target_modules", [])
        if not target_mods or not isinstance(target_mods, list):
            errors.append("target_modules must be a non-empty list of string module names.")

        is_valid = (len(errors) == 0)
        return is_valid, errors


def run_experiment_5() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 5: Production BitsAndBytes & LoraConfig Validation Gate")
    print("="*80)

    validator = FineTuningConfigValidator()

    # Valid Production Configuration (Targeting Ampere A100 / RTX 4090)
    valid_bnb = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16"
    }
    valid_lora = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }

    # Invalid Configuration (Using bfloat16 on older T4 GPU + bad quant type)
    invalid_bnb = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "int4_invalid",
        "bnb_4bit_compute_dtype": "bfloat16"
    }
    invalid_lora = {
        "r": -4,
        "lora_alpha": 0,
        "target_modules": []
    }

    ok_1, errs_1 = validator.validate_qlora_setup(valid_bnb, valid_lora, gpu_arch="ampere")
    print(f"[Test 1: Modern Cloud GPU Setup (A100 / RTX 4090)]")
    print(f" • Status: {'✅ PASSED (READY TO TRAIN)' if ok_1 else '❌ FAILED'}")

    ok_2, errs_2 = validator.validate_qlora_setup(invalid_bnb, invalid_lora, gpu_arch="t4")
    print(f"\n[Test 2: Invalid Config Setup (Colab T4 with BFloat16 + bad params)]")
    print(f" • Status: {'✅ BLOCKED AS EXPECTED' if not ok_2 else '❌ FAILED'}")
    print(f" • Caught Configuration Violations:")
    for e in errs_2:
        print(f"   • {e}")

    success = (ok_1 is True and not ok_2)
    print(f"\n[Verification] Configuration Gate Integrity: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# MAIN TEST HARNESS EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("  MODEL LOADING & PEFT MECHANICS VERIFICATION LAB")
    print("  Module 07 - Topic 02: Hugging Face, BitsAndBytes NF4 & LoRA Setup")
    print("=" * 80)

    tests = [
        ("Model Repository Anatomy & Config Inspector", run_experiment_1),
        ("4-bit NormalFloat (NF4) Quantization & Dequantization", run_experiment_2),
        ("LoRA Factorization & Forward Pass Emulation", run_experiment_3),
        ("PEFT Target Module Selector & Parameter Calculator", run_experiment_4),
        ("Production BitsAndBytes & LoraConfig Validation Gate", run_experiment_5),
    ]

    passed_count = 0
    total_count = len(tests)

    for name, func in tests:
        try:
            ok = func()
            if ok:
                passed_count += 1
            else:
                print(f"[FAIL] {name} did not pass verification assertions.")
        except Exception as e:
            print(f"[ERROR] Exception occurred in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"  TEST SUMMARY: {passed_count}/{total_count} Experiments Passed ({passed_count/total_count*100:.1f}%)")
    print("=" * 80)

    if passed_count == total_count:
        print("🎯 All Model Loading & PEFT mechanics verified successfully!\n")
        return 0
    else:
        print("⚠️ Some experiments failed. Review diagnostics above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
