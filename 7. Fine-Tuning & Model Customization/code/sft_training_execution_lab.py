#!/usr/bin/env python3
"""
================================================================================
End-to-End SFT Training Execution: Verification Lab
================================================================================
Topic: Module 07 - Topic 03: Preparing Instruction Datasets, Configuring
       SFTTrainer, and Monitoring Training Dynamics.

This lab provides hands-on algorithmic verification of:
  1. Chat Template Engine & Special Token Formatter (Jinja2 / ChatML)
  2. Sequence Packing Simulator (Eliminating Padding Waste for 3-4x Speedup)
  3. Micro-Batching & Gradient Accumulation Step Mathematical Equivalence
  4. Learning Rate Schedule with Warmup & Cosine Annealing Decay
  5. SFT Training Loop Simulator & Loss Curve Anomaly Detector (Overfitting / Divergence)

Execution:
  python sft_training_execution_lab.py
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
# EXPERIMENT 1: Chat Template Engine & Tokenizer Integration
# ==============================================================================

class ChatTemplateFormatter:
    """
    Simulates Hugging Face's `tokenizer.apply_chat_template()`:
    Formats structured message lists into standardized raw strings with special
    boundary tokens required for causal language models.
    """
    def format_chatml(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
        formatted_pieces = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            formatted_pieces.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            # Appends assistant header so the model knows it is its turn to speak
            formatted_pieces.append("<|im_start|>assistant\n")

        return "".join(formatted_pieces)


def run_experiment_1() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 1: Chat Template Engine & Tokenizer Integration")
    print("="*80)

    formatter = ChatTemplateFormatter()

    dialogue = [
        {"role": "system", "content": "You are a specialized SQL generation model."},
        {"role": "user", "content": "Find total orders placed in October 2024."},
        {"role": "assistant", "content": "SELECT COUNT(*) FROM orders WHERE order_date >= '2024-10-01' AND order_date < '2024-11-01';"}
    ]

    # Mode A: Training sequence (contains both prompt and assistant response)
    train_text = formatter.format_chatml(dialogue, add_generation_prompt=False)
    print("[Mode A: Training Prompt String (Full Conversation with EOS)]")
    print(train_text)

    # Mode B: Inference sequence (stops at assistant header to prompt generation)
    eval_text = formatter.format_chatml(dialogue[:2], add_generation_prompt=True)
    print("[Mode B: Inference Prompt String (add_generation_prompt=True)]")
    print(eval_text)

    success = ("<|im_start|>system" in train_text and
               "<|im_end|>" in train_text and
               eval_text.endswith("<|im_start|>assistant\n"))
    print(f"[Verification] Chat Template Formatting: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 2: Sequence Packing Simulator (Eliminating Padding Waste)
# ==============================================================================

@dataclass
class PackingStats:
    total_samples: int
    unpacked_padded_tokens: int
    packed_tokens: int
    wasted_pad_percentage: float
    throughput_speedup: float

class SequencePackingSimulator:
    """
    Demonstrates Sequence Packing (`packing=True` in TRL's SFTTrainer):
    Without packing: Every sequence in a batch is padded with [PAD] tokens to the
      longest sequence (or max_seq_length), wasting up to 70% of GPU compute!
    With packing: Multiple short sequences are packed back-to-back into a single
      contiguous sequence (separated by EOS tokens), maximizing token throughput.
    """
    def pack_sequences(self, sample_token_lengths: List[int], max_seq_len: int = 512) -> Tuple[List[List[int]], PackingStats]:
        # 1. Unpacked calculation (each sample padded to max_seq_len)
        unpacked_total = len(sample_token_lengths) * max_seq_len

        # 2. Packing algorithm (Bin Packing / First Fit Decreasing)
        packed_bins: List[List[int]] = []
        current_bin: List[int] = []
        current_len = 0

        for length in sample_token_lengths:
            if current_len + length <= max_seq_len:
                current_bin.append(length)
                current_len += length
            else:
                packed_bins.append(current_bin)
                current_bin = [length]
                current_len = length
        if current_bin:
            packed_bins.append(current_bin)

        packed_total = len(packed_bins) * max_seq_len
        actual_useful_tokens = sum(sample_token_lengths)
        wasted_pad_pct = (1.0 - (actual_useful_tokens / unpacked_total)) * 100.0
        speedup = unpacked_total / packed_total

        stats = PackingStats(
            total_samples=len(sample_token_lengths),
            unpacked_padded_tokens=unpacked_total,
            packed_tokens=packed_total,
            wasted_pad_percentage=round(wasted_pad_pct, 1),
            throughput_speedup=round(speedup, 2)
        )
        return packed_bins, stats


def run_experiment_2() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 2: Sequence Packing Simulator (Eliminating Padding Waste)")
    print("="*80)

    simulator = SequencePackingSimulator()

    # 12 sample dialogue lengths (varying from short 40-token QA to 220-token answers)
    sample_lengths = [65, 120, 45, 210, 80, 95, 180, 50, 110, 75, 140, 90]
    max_seq_len = 512

    bins, stats = simulator.pack_sequences(sample_lengths, max_seq_len=max_seq_len)

    print(f"Dataset Size: {stats.total_samples} dialogue samples | Target Context Window: {max_seq_len} tokens")
    print(f" • Unpacked Mode Batches:        {stats.total_samples} batches (Padded Tokens: {stats.unpacked_padded_tokens:,})")
    print(f" • Packed Mode Batches:          {len(bins)} batches (Packed Tokens: {stats.packed_tokens:,})")
    print(f" • Naive Padding Waste:          {stats.wasted_pad_percentage}% of compute wasted on [PAD] tokens!")
    print(f" • GPU Throughput Speedup:       {stats.throughput_speedup}x FASTER training with packing=True")

    print("\n[Packed Batch Distribution Details]")
    for i, b in enumerate(bins):
        print(f" • Packed Sequence #{i+1}: Contains {len(b)} dialogues -> Total tokens: {sum(b)}/{max_seq_len} ({sum(b)/max_seq_len*100:.1f}% full)")

    success = (stats.throughput_speedup >= 2.0 and len(bins) <= 4)
    print(f"\n[Verification] Sequence Packing Efficiency: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 3: Micro-Batching & Gradient Accumulation Math
# ==============================================================================

class GradientAccumulationSimulator:
    """
    Demonstrates Gradient Accumulation:
    When a GPU has limited VRAM, setting a large batch size (e.g., 16) causes OOM.
    Instead:
      per_device_train_batch_size = 2 (micro-batch)
      gradient_accumulation_steps = 8
      Effective Batch Size = 2 * 8 = 16!

    The gradients from 8 micro-batches are accumulated (summed/averaged) before
    calling optimizer.step(), mathematically mirroring the large batch update.
    """
    def simulate_accumulation(self, micro_batch_losses: List[float]) -> Tuple[float, float]:
        accum_steps = len(micro_batch_losses)
        # In PyTorch: loss = loss / accum_steps; loss.backward()
        scaled_losses = [l / accum_steps for l in micro_batch_losses]
        total_effective_loss = sum(scaled_losses)
        # Direct mean of raw losses
        direct_mean_loss = sum(micro_batch_losses) / accum_steps
        return total_effective_loss, direct_mean_loss


def run_experiment_3() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 3: Micro-Batching & Gradient Accumulation Math")
    print("="*80)

    sim = GradientAccumulationSimulator()

    # Simulate 8 micro-batch losses
    micro_losses = [2.45, 2.38, 2.51, 2.29, 2.40, 2.35, 2.48, 2.30]
    effective_batch_size = len(micro_losses) * 2  # 2 samples per micro-batch * 8 steps = 16

    accum_loss, direct_loss = sim.simulate_accumulation(micro_losses)

    print(f"Batch Architecture Configuration:")
    print(f" • per_device_train_batch_size:  2")
    print(f" • gradient_accumulation_steps:  {len(micro_losses)}")
    print(f" • Effective Batch Size:         {effective_batch_size} samples")
    print("-" * 55)
    print(f" • Accumulated Scaled Loss:      {accum_loss:.6f}")
    print(f" • Direct Full-Batch Loss:       {direct_loss:.6f}")
    print(f" • Numerical Difference:         {abs(accum_loss - direct_loss):.10f}")

    success = (abs(accum_loss - direct_loss) < 1e-6)
    print(f"\n[Verification] Gradient Accumulation Equivalence: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 4: Learning Rate Schedule with Warmup & Cosine Annealing
# ==============================================================================

class LRSchedulerSimulator:
    """
    Simulates Cosine Annealing with Linear Warmup:
      1. Warmup Phase (step <= warmup_steps):
         LR increases linearly from 0 to peak_lr:
           lr = peak_lr * (step / warmup_steps)
      2. Cosine Annealing Phase (step > warmup_steps):
         LR decays following a cosine curve down to min_lr:
           progress = (step - warmup_steps) / (total_steps - warmup_steps)
           lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))
    """
    def __init__(self, peak_lr: float = 2e-4, min_lr: float = 2e-5, total_steps: int = 100, warmup_ratio: float = 0.10):
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)

    def get_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            return self.peak_lr * (step / max(1, self.warmup_steps))
        else:
            progress = (step - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine_factor


def run_experiment_4() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 4: Learning Rate Schedule with Warmup & Cosine Annealing")
    print("="*80)

    scheduler = LRSchedulerSimulator(peak_lr=2e-4, min_lr=2e-5, total_steps=100, warmup_ratio=0.10)

    sample_steps = [0, 5, 10, 25, 50, 75, 100]
    print(f"{'Step':<8} | {'Phase':<18} | {'Calculated Learning Rate':<26} | {'Visual Progress'}")
    print("-" * 75)

    for s in sample_steps:
        lr = scheduler.get_lr(s)
        phase = "Linear Warmup" if s <= scheduler.warmup_steps else "Cosine Decay"
        bar_len = int((lr / scheduler.peak_lr) * 25)
        bar = "#" * bar_len
        print(f"Step {s:<3} | {phase:<18} | {lr:<26.6e} | {bar}")

    step_10_lr = scheduler.get_lr(10)
    step_100_lr = scheduler.get_lr(100)

    success = (math.isclose(step_10_lr, 2e-4, rel_tol=1e-3) and math.isclose(step_100_lr, 2e-5, rel_tol=1e-3))
    print(f"\n[Verification] LR Warmup & Cosine Curve: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 5: SFT Training Loop Simulator & Loss Curve Diagnostics
# ==============================================================================

@dataclass
class TrainingHistory:
    scenario_name: str
    train_losses: List[float]
    eval_losses: List[float]
    diagnosis: str
    early_stop_recommended: bool

class LossCurveDiagnosticsEngine:
    """
    Analyzes training trajectories to classify model health:
      1. Healthy Convergence: Train and Eval loss smoothly decrease together.
      2. Overfitting: Train loss drops, but Eval loss turns upward (> 10% increase).
      3. Divergence / Loss Explosion: Loss spikes to astronomical values or NaN.
    """
    def diagnose(self, scenario_name: str, train_losses: List[float], eval_losses: List[float]) -> TrainingHistory:
        # Check for divergence
        if any(math.isnan(l) or l > 10.0 for l in train_losses):
            return TrainingHistory(
                scenario_name=scenario_name,
                train_losses=train_losses,
                eval_losses=eval_losses,
                diagnosis="🚨 LOSS EXPLOSION / DIVERGENCE: Learning rate is too high or precision underflow occurred.",
                early_stop_recommended=True
            )

        # Check for overfitting
        min_eval = min(eval_losses)
        final_eval = eval_losses[-1]
        if final_eval > min_eval * 1.15:
            return TrainingHistory(
                scenario_name=scenario_name,
                train_losses=train_losses,
                eval_losses=eval_losses,
                diagnosis=f"⚠️ OVERFITTING DETECTED: Eval loss increased by {((final_eval/min_eval)-1)*100:.1f}% from minimum ({min_eval:.3f} -> {final_eval:.3f}).",
                early_stop_recommended=True
            )

        return TrainingHistory(
            scenario_name=scenario_name,
            train_losses=train_losses,
            eval_losses=eval_losses,
            diagnosis="✅ HEALTHY CONVERGENCE: Train and Eval losses decreased stably in tandem.",
            early_stop_recommended=False
        )


def run_experiment_5() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 5: SFT Training Loop Simulator & Loss Curve Diagnostics")
    print("="*80)

    engine = LossCurveDiagnosticsEngine()

    # Scenario 1: Healthy Run
    healthy_train = [2.50, 2.10, 1.75, 1.40, 1.15, 0.95, 0.82]
    healthy_eval  = [2.55, 2.15, 1.80, 1.45, 1.22, 1.05, 0.94]
    diag_1 = engine.diagnose("Run A: Optimal Parameters", healthy_train, healthy_eval)

    # Scenario 2: Overfitting Run (Trained for too many epochs)
    overfit_train = [2.50, 1.80, 1.20, 0.70, 0.40, 0.20, 0.10]
    overfit_eval  = [2.55, 1.90, 1.35, 1.15, 1.25, 1.55, 1.85]  # V-shaped turnaround
    diag_2 = engine.diagnose("Run B: Overfitting Run", overfit_train, overfit_eval)

    # Scenario 3: Loss Explosion (LR = 1e-2 instead of 2e-4)
    exploded_train = [2.50, 2.10, 4.80, 14.2, 85.0]
    exploded_eval  = [2.60, 2.30, 5.10, 15.0, 92.0]
    diag_3 = engine.diagnose("Run C: Divergent Run", exploded_train, exploded_eval)

    for diag in [diag_1, diag_2, diag_3]:
        print(f"\n[{diag.scenario_name}]")
        print(f" • Initial -> Final Train Loss: {diag.train_losses[0]:.2f} -> {diag.train_losses[-1]:.2f}")
        print(f" • Initial -> Final Eval Loss:  {diag.eval_losses[0]:.2f} -> {diag.eval_losses[-1]:.2f}")
        print(f" • Diagnostic Assessment:      {diag.diagnosis}")
        print(f" • Early Stopping Flag:         {'STOP & ROLLBACK' if diag.early_stop_recommended else 'PROCEED'}")

    success = (not diag_1.early_stop_recommended and diag_2.early_stop_recommended and diag_3.early_stop_recommended)
    print(f"\n[Verification] Loss Curve Diagnostics Integrity: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# MAIN TEST HARNESS EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("  END-TO-END SFT TRAINING EXECUTION VERIFICATION LAB")
    print("  Module 07 - Topic 03: SFTTrainer, Packing, Schedulers & Loss Diagnostics")
    print("=" * 80)

    tests = [
        ("Chat Template Engine & Tokenizer Integration", run_experiment_1),
        ("Sequence Packing Simulator (Padding Elimination)", run_experiment_2),
        ("Micro-Batching & Gradient Accumulation Equivalence", run_experiment_3),
        ("Learning Rate Schedule with Warmup & Cosine Decay", run_experiment_4),
        ("SFT Training Loop Simulator & Loss Curve Diagnostics", run_experiment_5),
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
        print("🎯 All SFT Training Execution systems verified successfully!\n")
        return 0
    else:
        print("⚠️ Some experiments failed. Review diagnostics above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
