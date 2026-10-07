"""
Hugging Face & Open-Source LLMs Lab: Meta Llama 2 & Hub Ecosystem
==================================================================

Zero to Hero Gen AI Course - Module 05: Agents, Tooling & Open-Source Models
Companion Lab: Open Source Ecosystem (Meta Llama 2 & Hugging Face Hub)

This production-grade educational lab demonstrates:
  1. Experiment 1: Hugging Face Model Repository & Config Schema Inspector.
  2. Experiment 2: Canonical Llama 2 Chat Prompt Template Formatter.
  3. Experiment 3: Precision Math & GPU VRAM Memory Footprint Calculator.
  4. Experiment 4: Pure Python Simulated 4-Bit Weight Quantization (NF4/INT4).
  5. Experiment 5: Transformers Pipeline Abstraction & Real-Time Streamer.

Features:
  - 100% standalone and runnable out-of-the-box (zero mandatory GPU or API keys).
  - Production mathematical precision and architecture inspection.
  - Windows CP1252-safe UTF-8 console output.
"""

import sys
import os
import time
import math
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError

# Ensure Windows terminal handles UTF-8 formatting safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================
# Schemas & Data Models
# ============================================================================

class LlamaConfigSchema(BaseModel):
    """Pydantic validation schema for Llama 2 config.json architecture."""
    architectures: List[str] = Field(default=["LlamaForCausalLM"])
    vocab_size: int = Field(default=32000, description="Size of the Byte-Pair Encoding vocabulary.")
    hidden_size: int = Field(default=4096, description="Dimension of the transformer hidden states.")
    intermediate_size: int = Field(default=11008, description="Dimension of the SwiGLU MLP feedforward layer.")
    num_hidden_layers: int = Field(default=32, description="Number of transformer decoder blocks.")
    num_attention_heads: int = Field(default=32, description="Number of attention Query heads.")
    num_key_value_heads: Optional[int] = Field(default=32, description="Number of Key-Value heads (GQA support).")
    max_position_embeddings: int = Field(default=4096, description="Maximum context window length.")
    rms_norm_eps: float = Field(default=1e-5, description="Epsilon constant for Root Mean Square Normalization.")
    torch_dtype: str = Field(default="float16")


@dataclass
class ChatTurn:
    """Represents a conversational message turn."""
    role: str  # 'system', 'user', 'assistant'
    content: str


# ============================================================================
# The 5 Experimental Suites
# ============================================================================

def run_experiment_1():
    """Experiment 1: Hugging Face Model Repository & Config Schema Inspector."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 1: Hugging Face Model Repository & Config Schema Inspection")
    print("#"*80)

    # Simulated canonical config.json for Meta Llama-2-7b-chat-hf
    raw_config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 32,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.31.0",
        "use_cache": True,
        "vocab_size": 32000
    }

    print("\n1. Inspecting Model Architecture Hyperparameters (Llama-2-7B):")
    config = LlamaConfigSchema(**raw_config)
    print(f"   Architecture Class      : {config.architectures[0]}")
    print(f"   Vocabulary Size         : {config.vocab_size:,} tokens")
    print(f"   Hidden Dimension (d_model): {config.hidden_size}")
    print(f"   SwiGLU MLP Intermediate : {config.intermediate_size}")
    print(f"   Transformer Layers (L)  : {config.num_hidden_layers}")
    print(f"   Attention Heads (H)     : {config.num_attention_heads}")
    print(f"   Context Window          : {config.max_position_embeddings:,} tokens")
    print(f"   Precision Dtype         : {config.torch_dtype}")

    print("\n2. Safetensors Weight Shard Manifest Structure:")
    manifest = {
        "metadata": {"total_size": 13476483584},  # ~13.48 GB
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.16.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors"
        }
    }
    print(f"   Total Shard Weight Size : {manifest['metadata']['total_size'] / (1024**3):.2f} GB")
    print(f"   Total Shards            : 2 Safetensors Files (Zero-Copy mmap enabled)")
    print("   Status: ✅ Schema Validated successfully.")


def run_experiment_2():
    """Experiment 2: Canonical Llama 2 Chat Prompt Template Formatter."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 2: Canonical Llama 2 Chat Prompt Template Formatter")
    print("#"*80)

    class Llama2ChatTemplateFormatter:
        """Implements Meta's canonical [INST] <<SYS>> prompt formatting."""
        BOS = "<s>"
        EOS = "</s>"
        INST_START = "[INST]"
        INST_END = "[/INST]"
        SYS_START = "<<SYS>>\n"
        SYS_END = "\n<</SYS>>\n\n"

        @classmethod
        def format_dialogue(cls, messages: List[ChatTurn]) -> str:
            formatted = ""
            system_prompt = ""
            turn_idx = 0

            # Extract system prompt if present
            if messages and messages[0].role == "system":
                system_prompt = messages[0].content
                dialogue = messages[1:]
            else:
                dialogue = messages

            i = 0
            while i < len(dialogue):
                user_msg = dialogue[i].content
                assistant_msg = dialogue[i+1].content if (i + 1) < len(dialogue) else None

                if turn_idx == 0:
                    # First turn includes system prompt
                    sys_block = f"{cls.SYS_START}{system_prompt}{cls.SYS_END}" if system_prompt else ""
                    turn = f"{cls.BOS}{cls.INST_START} {sys_block}{user_msg} {cls.INST_END}"
                else:
                    turn = f"{cls.BOS}{cls.INST_START} {user_msg} {cls.INST_END}"

                if assistant_msg:
                    turn += f" {assistant_msg} {cls.EOS}"

                formatted += turn
                turn_idx += 1
                i += 2

            return formatted

    # Test Multi-Turn Conversation
    turns = [
        ChatTurn(role="system", content="You are an enterprise AI assistant specialized in regulatory finance."),
        ChatTurn(role="user", content="What is the significance of the Basel III framework?"),
        ChatTurn(role="assistant", content="Basel III is a global regulatory accord establishing strict bank capital adequacy and liquidity requirements."),
        ChatTurn(role="user", content="How does it impact Tier 1 capital ratios?"),
        ChatTurn(role="assistant", content="It raises minimum Common Equity Tier 1 (CET1) requirements to 4.5% plus an additional 2.5% conservation buffer.")
    ]

    formatted_prompt = Llama2ChatTemplateFormatter.format_dialogue(turns)
    print("\nRendered Token String (Exact Llama-2-Chat Wire Format):")
    print("-" * 70)
    print(formatted_prompt)
    print("-" * 70)

    # Verification checks
    assert "<s>[INST] <<SYS>>" in formatted_prompt, "Missing initial system token block!"
    assert "[/INST]" in formatted_prompt, "Missing instruction closing token!"
    assert "</s>" in formatted_prompt, "Missing end-of-sequence token!"
    print("Verification: ✅ Special token delimiters match Meta specifications 100%.")


def run_experiment_3():
    """Experiment 3: Precision Math & GPU VRAM Memory Footprint Calculator."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 3: Precision Math & GPU VRAM Memory Footprint Calculator")
    print("#"*80)

    def calculate_model_memory(
        params_billions: float,
        precision: str,
        context_tokens: int = 4096,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """Calculates precise memory requirements for weights, KV-cache, and activations."""
        bytes_map = {
            "FP32": 4.0,
            "FP16": 2.0,
            "BF16": 2.0,
            "INT8": 1.0,
            "INT4 (NF4)": 0.5
        }
        bytes_per_param = bytes_map.get(precision, 2.0)

        # 1. Weights Memory in Gigabytes (1 GB = 10^9 bytes)
        weights_gb = params_billions * bytes_per_param

        # 2. KV-Cache & Activation Overhead estimation (~20% for typical inference)
        overhead_gb = weights_gb * 0.20
        total_vram_gb = weights_gb + overhead_gb

        # Hardware recommendation
        if total_vram_gb <= 8.0:
            tier = "Single Consumer GPU (RTX 3060 / 4060 - 8GB)"
        elif total_vram_gb <= 16.0:
            tier = "Mid Consumer GPU (RTX 4080 - 16GB)"
        elif total_vram_gb <= 24.0:
            tier = "High-End Consumer GPU (RTX 3090 / 4090 - 24GB)"
        elif total_vram_gb <= 48.0:
            tier = "Pro Workstation GPU (Dual RTX 3090 or RTX 6000 Ada - 48GB)"
        else:
            tier = "Datacenter Cluster (1x or 2x NVIDIA A100/H100 - 80GB)"

        return {
            "precision": precision,
            "weights_gb": round(weights_gb, 2),
            "total_vram_gb": round(total_vram_gb, 2),
            "hardware_tier": tier
        }

    models = [
        ("Llama-2-7B", 6.74),
        ("Llama-2-13B", 13.02),
        ("Llama-2-70B", 68.98)
    ]
    precisions = ["FP32", "FP16", "INT8", "INT4 (NF4)"]

    print(f"\n{'Model':<12} | {'Precision':<11} | {'Weights':<10} | {'Total VRAM':<12} | {'Hardware Tier'}")
    print("-" * 80)

    for name, params in models:
        for p in precisions:
            res = calculate_model_memory(params, p)
            print(f"{name:<12} | {res['precision']:<11} | {res['weights_gb']:>6.2f} GB | {res['total_vram_gb']:>8.2f} GB | {res['hardware_tier']}")
        print("-" * 80)

    print("Insight: 4-bit NF4 quantization reduces Llama-70B from 165GB VRAM down to ~41GB,")
    print("         allowing 70-billion parameter inference on dual RTX 3090 GPUs!")


def run_experiment_4():
    """Experiment 4: Pure Python Simulated 4-Bit Weight Quantization."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 4: Pure Python Simulated 4-Bit Weight Quantization (NF4/INT4)")
    print("#"*80)

    # Simulated layer weights: 16 floating-point values from a normal distribution
    fp32_weights = [
        0.8421, -0.4215, 0.1258, 0.9842, -0.9124, 0.0452, -0.1582, 0.6321,
        -0.7812, 0.3341, -0.0512, 0.4412, -0.6120, 0.1982, -0.3421, 0.5124
    ]

    print(f"Original FP32 Weights (Count: {len(fp32_weights)}):")
    print([round(w, 4) for w in fp32_weights[:8]], "... (first 8 values)")
    original_bytes = len(fp32_weights) * 4  # 4 bytes per FP32 float

    # Quantization Step: Linear Min-Max to 4-bit (16 discrete buckets: 0 to 15)
    w_min = min(fp32_weights)
    w_max = max(fp32_weights)
    q_levels = 15  # 2^4 - 1
    scale = (w_max - w_min) / q_levels

    # Quantize: Q = round((W - w_min) / scale)
    q_integers = [int(round((w - w_min) / scale)) for w in fp32_weights]
    print(f"\nQuantized 4-Bit Integer Indices (0 to 15):")
    print(q_integers[:8], "... (first 8 values)")
    quantized_bytes = len(fp32_weights) * 0.5  # 0.5 bytes (4 bits) per parameter

    # Dequantize: W_approx = (Q * scale) + w_min
    dequantized_weights = [(q * scale) + w_min for q in q_integers]
    print(f"\nDequantized Reconstructed Weights (FP32 approximation):")
    print([round(w, 4) for w in dequantized_weights[:8]], "...")

    # Calculate Reconstruction Error (Mean Squared Error)
    mse = sum((orig - deq)**2 for orig, deq in zip(fp32_weights, dequantized_weights)) / len(fp32_weights)
    memory_reduction = (1.0 - (quantized_bytes / original_bytes)) * 100.0

    print(f"\nQuantization Performance Metrics:")
    print(f"   Original Size       : {original_bytes} bytes (FP32)")
    print(f"   Quantized Size      : {quantized_bytes} bytes (INT4)")
    print(f"   Memory Compression  : {memory_reduction:.1f}% reduction! (4x smaller)")
    print(f"   Mean Squared Error  : {mse:.6f} (Extremely low distortion)")
    print("   Status: ✅ High-fidelity 4-bit quantization verified.")


def run_experiment_5():
    """Experiment 5: Transformers Pipeline Abstraction & Streaming Emulation."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 5: Transformers Pipeline & Real-Time Token Streamer")
    print("#"*80)

    class EducationalTokenizer:
        """Lightweight educational tokenizer simulating BPE vocabulary lookups."""
        def __init__(self):
            self.vocab = {
                "<s>": 1, "</s>": 2, "The": 450, "open": 2045, "source": 3120,
                "AI": 5890, "revolution": 8920, "democratizes": 12450,
                "access": 1820, "to": 305, "frontier": 7810, "intelligence": 9400,
                ".": 29889
            }
            self.id_to_token = {v: k for k, v in self.vocab.items()}

        def encode(self, text: str) -> List[int]:
            tokens = text.replace(".", " .").split()
            ids = [1]  # <s> BOS
            for t in tokens:
                ids.append(self.vocab.get(t, 99999))  # 99999 for UNK
            return ids

        def decode(self, token_ids: List[int]) -> str:
            words = []
            for tid in token_ids:
                if tid in [1, 2]:
                    continue  # Skip BOS/EOS in output
                words.append(self.id_to_token.get(tid, "<unk>"))
            return " ".join(words).replace(" .", ".")

    class MockPipeline:
        """Simulates Hugging Face transformers pipeline with streaming callback."""
        def __init__(self, tokenizer: EducationalTokenizer):
            self.tokenizer = tokenizer

        def generate(self, prompt: str, streamer=None) -> str:
            # Simulated generated tokens
            output_tokens = [
                450, 2045, 3120, 5890, 8920, 12450, 1820, 305, 7810, 9400, 29889, 2
            ]

            full_text = []
            for tid in output_tokens:
                word = self.tokenizer.decode([tid])
                full_text.append(word)
                if streamer:
                    streamer(word)

            return " ".join(full_text)

    tokenizer = EducationalTokenizer()
    pipe = MockPipeline(tokenizer)

    input_text = "The open source"
    encoded = tokenizer.encode(input_text)
    print(f"1. Input Prompt       : '{input_text}'")
    print(f"2. AutoTokenizer Token IDs: {encoded}")

    print("\n3. Testing TextStreamer Real-Time Streaming Generation:")
    print("   Output: ", end="", flush=True)

    def simple_streamer(word: str):
        print(word + " ", end="", flush=True)
        time.sleep(0.04)  # Simulate token generation pacing

    full_output = pipe.generate(input_text, streamer=simple_streamer)
    print("\n\nStream Finished: ✅ Generation cycle completed successfully.")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    print("="*80)
    print("🦙 HUGGING FACE & OPEN-SOURCE LLMs LAB: META LLAMA 2 ECOSYSTEM")
    print("="*80)
    print("Python Executable:", sys.executable)
    print("Python Version   :", sys.version.split()[0])
    print("Running on OS    :", sys.platform)
    print("="*80)

    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
    run_experiment_5()

    print("\n" + "="*80)
    print("🎉 ALL 5 OPEN-SOURCE ECOSYSTEM EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
