#!/usr/bin/env python3
"""
================================================================================
Post-Training Lifecycle: Verification Lab
================================================================================
Topic: Module 07 - Topic 04: Merging Adapters, Multi-Format Export (HF & GGUF/Ollama),
       and Before-vs-After Performance Evaluation.

This verification suite walks through the complete post-training lifecycle:
  1. LoRA Adapter Saving & Manifest Serializer (adapter_config.json)
  2. Adapter Merging Simulator (merge_and_unload mathematical equivalence)
  3. Multi-Format Exporter & Ollama Modelfile Generator (GGUF pipeline)
  4. Before vs. After Quantitative Evaluation (Perplexity & Schema Compliance)
  5. Automated LLM-as-a-Judge Side-by-Side Scoring Engine

Execution:
  python post_training_eval_lab.py
================================================================================
"""

import sys
import os
import json
import math
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict

# Ensure UTF-8 output encoding across Windows PowerShell and Unix terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ==============================================================================
# EXPERIMENT 1: LoRA Adapter Saving & Manifest Serializer
# ==============================================================================

@dataclass
class AdapterConfig:
    base_model_name_or_path: str
    peft_type: str = "LORA"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

class AdapterArtifactManager:
    """
    Simulates saving the lightweight PEFT LoRA adapter:
      - adapter_config.json: Configuration parameters for reconstruction.
      - adapter_model.safetensors: Weights of matrices A and B only (~50MB - 150MB).
    """
    def serialize_adapter(self, output_dir: str, config: AdapterConfig, num_trainable_params: int) -> Dict[str, Any]:
        cfg_dict = asdict(config)
        adapter_disk_mb = (num_trainable_params * 2) / (1024 * 1024)  # FP16 precision
        manifest = {
            "config_file": f"{output_dir}/adapter_config.json",
            "weights_file": f"{output_dir}/adapter_model.safetensors",
            "config_data": cfg_dict,
            "trainable_parameters": num_trainable_params,
            "adapter_size_mb": round(adapter_disk_mb, 2)
        }
        return manifest


def run_experiment_1() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 1: LoRA Adapter Saving & Manifest Serializer")
    print("="*80)

    mgr = AdapterArtifactManager()
    cfg = AdapterConfig(
        base_model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
        r=16,
        lora_alpha=32
    )

    manifest = mgr.serialize_adapter("./my_sql_lora_adapter", cfg, num_trainable_params=41_943_040)

    print(f"Serialized Adapter Manifest: {manifest['config_file']}")
    print(f" • Base Model Identifier:    {manifest['config_data']['base_model_name_or_path']}")
    print(f" • LoRA Rank (r) / Alpha:    r={manifest['config_data']['r']} / alpha={manifest['config_data']['lora_alpha']}")
    print(f" • Target Modules:           {len(manifest['config_data']['target_modules'])} modules ({', '.join(manifest['config_data']['target_modules'][:4])}...)")
    print(f" • Adapter File Footprint:   {manifest['adapter_size_mb']} MB (vs 15,000 MB Base Model!)")
    print(f" • Storage Savings:          {(1 - (manifest['adapter_size_mb'] / 15000)) * 100:.2f}% lighter than base model")

    success = (manifest["adapter_size_mb"] < 150.0 and manifest["config_data"]["r"] == 16)
    print(f"\n[Verification] Adapter Serialization: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 2: Adapter Merging Simulator (`merge_and_unload`)
# ==============================================================================

class WeightMergingSimulator:
    """
    Demonstrates `model.merge_and_unload()`:
    During LoRA inference, computing:
      h = x * W_0 + (x * A * B) * (alpha / r)
    adds latency due to two separate matrix multiplications.

    By folding the adapter directly into the base weights:
      W_merged = W_0 + (alpha / r) * (B * A)
    we produce a single, unified 16-bit weight matrix with ZERO inference overhead!
    """
    def __init__(self, d_in: int = 4, d_out: int = 4, rank: int = 2, alpha: float = 4.0):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        random.seed(99)
        # Base matrix W_0 (d_in x d_out)
        self.W0 = [[random.uniform(0.1, 0.5) for _ in range(d_out)] for _ in range(d_in)]
        # LoRA down-projection A (d_in x rank)
        self.A = [[random.uniform(0.01, 0.1) for _ in range(rank)] for _ in range(d_in)]
        # LoRA up-projection B (rank x d_out)
        self.B = [[random.uniform(0.01, 0.1) for _ in range(d_out)] for _ in range(rank)]

    def compute_two_branch_forward(self, x: List[float]) -> List[float]:
        """Computes h = x*W0 + scaling * (x * A * B) dynamically."""
        # Branch 1: Base output
        h_base = [sum(x[i] * self.W0[i][j] for i in range(self.d_in)) for j in range(self.d_out)]

        # Branch 2: LoRA output
        h_A = [sum(x[i] * self.A[i][r] for i in range(self.d_in)) for r in range(self.rank)]
        h_B = [sum(h_A[r] * self.B[r][j] for r in range(self.rank)) for j in range(self.d_out)]

        return [h_base[j] + self.scaling * h_B[j] for j in range(self.d_out)]

    def merge_and_unload(self) -> List[List[float]]:
        """Computes W_merged = W0 + scaling * (A * B) permanently."""
        W_merged = [[0.0 for _ in range(self.d_out)] for _ in range(self.d_in)]

        for i in range(self.d_in):
            for j in range(self.d_out):
                delta_w_ij = sum(self.A[i][r] * self.B[r][j] for r in range(self.rank))
                W_merged[i][j] = self.W0[i][j] + self.scaling * delta_w_ij

        return W_merged

    def compute_merged_forward(self, x: List[float], W_merged: List[List[float]]) -> List[float]:
        """Computes h = x * W_merged (single matrix multiplication)."""
        return [sum(x[i] * W_merged[i][j] for i in range(self.d_in)) for j in range(self.d_out)]


def run_experiment_2() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 2: Adapter Merging Simulator (merge_and_unload)")
    print("="*80)

    sim = WeightMergingSimulator(d_in=4, d_out=4, rank=2, alpha=4.0)
    x_test = [1.2, -0.5, 0.8, 2.1]

    # Two-branch computation (Base + LoRA)
    h_two_branch = sim.compute_two_branch_forward(x_test)

    # Fold adapter weights into base weights
    W_merged = sim.merge_and_unload()

    # Single-branch computation (Merged matrix)
    h_merged = sim.compute_merged_forward(x_test, W_merged)

    print(f"Testing Mathematical Equivalence between Dynamic LoRA & Merged Model:")
    print(f"{'Dimension':<12} | {'Dynamic LoRA Output':<22} | {'Merged Model Output':<22} | {'Absolute Error'}")
    print("-" * 75)

    max_error = 0.0
    for idx, (dyn, mrg) in enumerate(zip(h_two_branch, h_merged)):
        err = abs(dyn - mrg)
        max_error = max(max_error, err)
        print(f"Output [{idx}]   | {dyn:>18.8f}     | {mrg:>18.8f}     | {err:.10f}")

    print(f"\n[Merging Benefits]")
    print(f" • Numerical Equivalence Error:  {max_error:.12f} (Perfect bitwise precision)")
    print(f" • Latency Impact:               Zero adapter overhead; runs at native base model speed!")
    print(f" • Deployment Compatibility:      Exportable as standard standalone Hugging Face model.")

    success = (max_error < 1e-7)
    print(f"\n[Verification] Adapter Merging Equivalence: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 3: Multi-Format Exporter & Ollama Modelfile Generator
# ==============================================================================

class ModelDeploymentExporter:
    """
    Generates configuration artifacts for:
      1. Standalone Hugging Face Hub repository upload
      2. Ollama / llama.cpp local GGUF execution
    """
    def generate_ollama_modelfile(self, base_gguf_path: str, system_prompt: str, temperature: float = 0.2) -> str:
        modelfile = (
            f"FROM {base_gguf_path}\n\n"
            f"# Set runtime hyperparameters\n"
            f"PARAMETER temperature {temperature}\n"
            f"PARAMETER top_p 0.9\n"
            f"PARAMETER stop \"<|im_end|>\"\n"
            f"PARAMETER stop \"<|endoftext|>\"\n\n"
            f"# Set system instruction\n"
            f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n\n"
            f"# Set standard ChatML prompt template\n"
            f"TEMPLATE \"\"\"\n"
            f"{{{{ if .System }}}}<|im_start|>system\n{{{{ .System }}}}<|im_end|>\n{{{{ end }}}}\n"
            f"{{{{ if .Prompt }}}}<|im_start|>user\n{{{{ .Prompt }}}}<|im_end|>\n{{{{ end }}}}\n"
            f"<|im_start|>assistant\n{{{{ .Response }}}}<|im_end|>\n"
            f"\"\"\"\n"
        )
        return modelfile


def run_experiment_3() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 3: Multi-Format Exporter & Ollama Modelfile Generator")
    print("="*80)

    exporter = ModelDeploymentExporter()
    sys_prompt = "You are an enterprise SQL generator. Output only valid PostgreSQL code."
    modelfile_content = exporter.generate_ollama_modelfile("./llama-3-8b-sql-q4_k_m.gguf", sys_prompt, temperature=0.1)

    print("[Generated Ollama Modelfile for Local Deployment]")
    print(modelfile_content)

    print("[Local Execution Instructions]")
    print(" $ ollama create my-sql-agent -f ./Modelfile")
    print(" $ ollama run my-sql-agent \"Show active subscribers in California.\"")

    success = ("FROM ./llama-3-8b-sql-q4_k_m.gguf" in modelfile_content and
               "PARAMETER stop \"<|im_end|>\"" in modelfile_content)
    print(f"\n[Verification] Ollama Modelfile Export: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 4: Before vs. After Quantitative Evaluation (Perplexity & Schema)
# ==============================================================================

@dataclass
class QuantitativeEvalResult:
    base_perplexity: float
    finetuned_perplexity: float
    base_json_compliance_pct: float
    finetuned_json_compliance_pct: float
    perplexity_improvement_pct: float
    compliance_improvement_pct: float

class PerformanceBenchmarkEngine:
    """
    Evaluates:
      1. Perplexity: PPL = exp(loss). Lower perplexity indicates the model is less
         surprised by the specialized domain language.
      2. JSON Schema Adherence: Does the model follow strict JSON formatting without
         chatty preamble or markdown syntax errors?
    """
    def benchmark(self) -> QuantitativeEvalResult:
        # Simulated test losses on a holdout set of 100 enterprise domain queries
        # Base model (generalist, not tuned for this specific domain)
        base_test_loss = 2.65
        # Fine-tuned model (specialized domain adapter)
        finetuned_test_loss = 1.33

        base_ppl = math.exp(base_test_loss)
        finetuned_ppl = math.exp(finetuned_test_loss)

        # Schema compliance test on 50 holdout structured extraction tasks
        # Base model often outputs conversational filler: "Here is your JSON: ```json..."
        base_valid_json = 14   # 14 / 50 = 28%
        finetuned_valid_json = 49 # 49 / 50 = 98%

        base_pct = (base_valid_json / 50) * 100.0
        finetuned_pct = (finetuned_valid_json / 50) * 100.0

        ppl_imp = ((base_ppl - finetuned_ppl) / base_ppl) * 100.0
        comp_imp = finetuned_pct - base_pct

        return QuantitativeEvalResult(
            base_perplexity=round(base_ppl, 2),
            finetuned_perplexity=round(finetuned_ppl, 2),
            base_json_compliance_pct=round(base_pct, 1),
            finetuned_json_compliance_pct=round(finetuned_pct, 1),
            perplexity_improvement_pct=round(ppl_imp, 1),
            compliance_improvement_pct=round(comp_imp, 1)
        )


def run_experiment_4() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 4: Before vs. After Quantitative Evaluation (Perplexity & Schema)")
    print("="*80)

    engine = PerformanceBenchmarkEngine()
    res = engine.benchmark()

    print(f"{'Evaluation Metric':<32} | {'Base Model (Before)':<20} | {'Fine-Tuned (After)':<20} | {'Net Improvement'}")
    print("-" * 88)
    print(f"{'Domain Perplexity (PPL)':<32} | {res.base_perplexity:<20.2f} | {res.finetuned_perplexity:<20.2f} | {res.perplexity_improvement_pct:>5.1f}% reduction")
    print(f"{'Strict JSON Compliance':<32} | {res.base_json_compliance_pct:<19.1f}% | {res.finetuned_json_compliance_pct:<19.1f}% | +{res.compliance_improvement_pct:>4.1f}% accuracy")

    print("\n[Key Quantitative Findings]")
    print(f" • Perplexity dropped from {res.base_perplexity} down to {res.finetuned_perplexity} (Model is 73% less surprised by domain jargon).")
    print(f" • JSON adherence jumped from {res.base_json_compliance_pct}% to {res.finetuned_json_compliance_pct}% (Zero preamble, parseable directly by APIs).")

    success = (res.finetuned_perplexity < res.base_perplexity and res.finetuned_json_compliance_pct >= 95.0)
    print(f"\n[Verification] Quantitative Benchmark Improvements: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# EXPERIMENT 5: Automated LLM-as-a-Judge Side-by-Side Scoring Engine
# ==============================================================================

@dataclass
class PromptEvaluationComparison:
    prompt_id: str
    user_query: str
    base_response: str
    finetuned_response: str
    base_score: int       # 1 - 5
    finetuned_score: int  # 1 - 5
    judge_critique: str

class LLMJudgeEvaluator:
    """
    Simulates LLM-as-a-Judge side-by-side evaluation:
    Grades responses on a 1-5 scale according to:
      1. Adherence to strict output formatting (No pleasantries/preambles)
      2. Syntax correctness and accuracy
      3. Absence of hallucinations
    """
    def evaluate_test_suite(self) -> List[PromptEvaluationComparison]:
        test_cases = [
            PromptEvaluationComparison(
                prompt_id="EVAL-01",
                user_query="Convert to SQL: Show active users who joined after June 2024.",
                base_response="Sure! Here is the SQL query you requested for active users:\n\nSELECT * FROM users WHERE status = 'active' AND date > 'June 2024';\n\nHope this helps!",
                finetuned_response="SELECT user_id, email FROM users WHERE is_active = TRUE AND created_at >= '2024-07-01';",
                base_score=2,
                finetuned_score=5,
                judge_critique="Base model included conversational fluff and used invalid date syntax ('June 2024'). Fine-tuned model output pure executable SQL with correct ISO timestamp."
            ),
            PromptEvaluationComparison(
                prompt_id="EVAL-02",
                user_query="Extract medical diagnosis: Patient presents with acute dyspnea, wheezing, and fever. Suspected bacterial pneumonia.",
                base_response="The patient might have bacterial pneumonia or asthma. You should consult a licensed physician immediately.",
                finetuned_response='{"primary_diagnosis": "Bacterial pneumonia", "icd10_code": "J15.9", "symptoms": ["acute dyspnea", "wheezing", "fever"]}',
                base_score=1,
                finetuned_score=5,
                judge_critique="Base model refused structured schema and gave generic disclaimer. Fine-tuned model strictly emitted valid JSON with ICD-10 mapping."
            ),
            PromptEvaluationComparison(
                prompt_id="EVAL-03",
                user_query="Convert to SQL: Count total products grouped by category in inventory.",
                base_response="SELECT COUNT(*) FROM products GROUP BY category;",
                finetuned_response="SELECT category, COUNT(product_id) AS total_products FROM inventory_products GROUP BY category;",
                base_score=3,
                finetuned_score=5,
                judge_critique="Base model omitted the group by column in SELECT and used generic table name. Fine-tuned model used exact internal database schema."
            )
        ]
        return test_cases


def run_experiment_5() -> bool:
    print("\n" + "="*80)
    print("▶ EXPERIMENT 5: Automated LLM-as-a-Judge Side-by-Side Scoring Engine")
    print("="*80)

    judge = LLMJudgeEvaluator()
    comparisons = judge.evaluate_test_suite()

    print(f"{'Test ID':<8} | {'Base Score':<11} | {'Fine-Tuned Score':<18} | {'Judgement Summary'}")
    print("-" * 88)

    base_total = 0
    finetuned_total = 0

    for c in comparisons:
        base_total += c.base_score
        finetuned_total += c.finetuned_score
        print(f"{c.prompt_id:<8} | {c.base_score}/5        | {c.finetuned_score}/5               | {c.judge_critique[:55]}...")

    avg_base = base_total / len(comparisons)
    avg_finetuned = finetuned_total / len(comparisons)

    print("\n[Judge Scorecard Aggregate]")
    print(f" • Average Base Model Score:       {avg_base:.1f} / 5.0")
    print(f" • Average Fine-Tuned Model Score:  {avg_finetuned:.1f} / 5.0")
    print(f" • Win Rate:                       100% of cases won by the fine-tuned adapter!")

    success = (avg_finetuned >= 4.5 and avg_base < 3.0)
    print(f"\n[Verification] LLM-as-a-Judge Evaluation: {'PASSED' if success else 'FAILED'}")
    return success


# ==============================================================================
# MAIN TEST HARNESS EXECUTION
# ==============================================================================

def main():
    print("=" * 80)
    print("  POST-TRAINING LIFECYCLE & EVALUATION VERIFICATION LAB")
    print("  Module 07 - Topic 04: Merging, GGUF/Ollama Export & Before-After Testing")
    print("=" * 80)

    tests = [
        ("LoRA Adapter Saving & Manifest Serializer", run_experiment_1),
        ("Adapter Merging Simulator (merge_and_unload)", run_experiment_2),
        ("Multi-Format Exporter & Ollama Modelfile Generator", run_experiment_3),
        ("Before vs. After Quantitative Evaluation", run_experiment_4),
        ("Automated LLM-as-a-Judge Side-by-Side Scoring Engine", run_experiment_5),
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
        print("🎯 All Post-Training Lifecycle and Evaluation systems verified successfully!\n")
        return 0
    else:
        print("⚠️ Some experiments failed. Review diagnostics above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
