"""
Sequential Chaining Lab: SimpleSequentialChain, SequentialChain, and Modern LCEL
================================================================================
Zero to Hero Gen AI Course — Module 03: The LangChain Framework & Chaining

This standalone educational lab demonstrates the mechanics of connecting
multiple LLM calls sequentially:
1. SimpleSequentialChain: Single-variable linear passing & information bottleneck
2. SequentialChain: Multi-variable state accumulation across stages
3. Modern LCEL Equivalent: Replicating state accumulation via RunnablePassthrough.assign()
4. Guardrail Interceptors: Validating & self-healing intermediate stage outputs
5. Full State Audit & Latency Profiling

Usage:
    py sequential_chains_lab.py
"""

import sys
import os
import json
import time
from typing import Any, Callable, Dict, List, Optional

# Ensure UTF-8 output on Windows consoles to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =====================================================================
# SECTION 1: CORE SIMULATION ENGINE (STANDALONE + LIVE SUPPORT)
# =====================================================================

class SimulatedLLM:
    """
    Intelligent simulated LLM engine that returns realistic domain responses
    for educational purposes without requiring paid API keys, while also
    supporting live OpenAI API execution if OPENAI_API_KEY is configured.
    """
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.5):
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.live_client = None

        if self.api_key and not self.api_key.startswith("sk-mock"):
            try:
                from openai import OpenAI
                self.live_client = OpenAI(api_key=self.api_key)
            except Exception:
                self.live_client = None

    def generate(self, prompt: str) -> str:
        if self.live_client:
            try:
                resp = self.live_client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"   [Notice: Falling back to simulated engine due to: {e}]")

        # Intelligent heuristic simulator
        p_lower = prompt.lower()
        if "catchy name" in p_lower:
            return "AeroPulse Drone Dynamics"
        elif "tagline" in p_lower or "slogan" in p_lower:
            return "Lifesaving Logistics on Electric Wings."
        elif "summarize this customer review" in p_lower or "summary:" in p_lower:
            return "Left earbud stopped charging after 3 weeks, and the charging case LED blinks red continuously."
        elif "classify sentiment" in p_lower:
            if "great" in p_lower and "stopped" in p_lower:
                return "NEGATIVE"
            return "NEGATIVE"
        elif "defect" in p_lower:
            return "Charging Case Power Management Failure / Earbud Pin Corrosion"
        elif "resolution email" in p_lower:
            return (
                "Dear Valued Customer,\n\n"
                "Thank you for reaching out regarding your AirPulse Noise-Cancelling Headphones. "
                "We sincerely apologize for the frustration caused by the left earbud charging failure and case LED alert.\n\n"
                "Because your purchase is within our 30-day warranty window, we have dispatched a prepaid return label and "
                "initiated an immediate expedited replacement of the complete earbud and charging case kit.\n\n"
                "Sincerely,\nAirPulse Customer Support"
            )
        else:
            return f"Processed output for prompt ({len(prompt)} chars)."


# =====================================================================
# SECTION 2: LEGACY CHAIN IMPLEMENTATIONS (FROM SCRATCH)
# =====================================================================

class LegacyLLMChain:
    """Simulates LangChain's legacy LLMChain."""
    def __init__(self, llm: SimulatedLLM, template: str, input_keys: List[str], output_key: str):
        self.llm = llm
        self.template = template
        self.input_keys = input_keys
        self.output_key = output_key

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Format template with input keys
        prompt_text = self.template.format(**{k: inputs[k] for k in self.input_keys})
        result = self.llm.generate(prompt_text)
        return {self.output_key: result}


class SimpleSequentialChain:
    """
    Simulates LangChain's legacy SimpleSequentialChain:
    Strictly single-variable in, single-variable out.
    """
    def __init__(self, chains: List[LegacyLLMChain], verbose: bool = True):
        self.chains = chains
        self.verbose = verbose

    def run(self, initial_input: str) -> str:
        current_val = initial_input
        if self.verbose:
            print("\n--- [SimpleSequentialChain Execution Start] ---")
            print(f"📥 Initial Input: '{current_val}'")

        for idx, chain in enumerate(self.chains, 1):
            if self.verbose:
                print(f"\n🔗 Executing Stage {idx} (Input Key: {chain.input_keys[0]} -> Output: {chain.output_key})...")
            step_input = {chain.input_keys[0]: current_val}
            step_output = chain.run(step_input)
            current_val = step_output[chain.output_key]
            if self.verbose:
                print(f"📤 Stage {idx} Output: '{current_val}'")

        if self.verbose:
            print("\n--- [SimpleSequentialChain Finished] ---")
        return current_val


class SequentialChain:
    """
    Simulates LangChain's legacy SequentialChain:
    Multi-input, multi-output with accumulated state dictionary.
    """
    def __init__(
        self,
        chains: List[LegacyLLMChain],
        input_variables: List[str],
        output_variables: List[str],
        verbose: bool = True
    ):
        self.chains = chains
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.verbose = verbose

    def __call__(self, initial_inputs: Dict[str, Any], return_all: bool = False) -> Dict[str, Any]:
        state = dict(initial_inputs)
        if self.verbose:
            print("\n--- [SequentialChain Execution Start] ---")
            print(f"📥 Initial State Keys: {list(state.keys())}")

        for idx, chain in enumerate(self.chains, 1):
            if self.verbose:
                print(f"\n🔗 Executing Stage {idx} (Required Inputs: {chain.input_keys} -> Producing: {chain.output_key})...")
            # Verify input prerequisites
            for k in chain.input_keys:
                if k not in state:
                    raise KeyError(f"Stage {idx} requires missing key '{k}' from accumulated state!")
            
            step_output = chain.run(state)
            state[chain.output_key] = step_output[chain.output_key]
            if self.verbose:
                print(f"📤 Added to State: '{chain.output_key}' = {step_output[chain.output_key][:65]}...")

        if self.verbose:
            print("\n--- [SequentialChain Execution Finished] ---")

        if return_all:
            return state
        return {k: state[k] for k in self.output_variables if k in state}


# =====================================================================
# SECTION 3: MODERN LCEL PIPELINE ENGINE
# =====================================================================

class ModernLCELPipeline:
    """
    Simulates modern LCEL execution using RunnablePassthrough.assign()
    semantics without legacy Chain abstractions.
    """
    def __init__(self):
        self.steps: List[tuple[str, Callable[[Dict[str, Any]], Any]]] = []

    def assign(self, key: str, func: Callable[[Dict[str, Any]], Any]) -> "ModernLCELPipeline":
        self.steps.append((key, func))
        return self

    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        state = dict(initial_state)
        for key, func in self.steps:
            state[key] = func(state)
        return state


# =====================================================================
# LAB EXPERIMENTS & DEMONSTRATION SUITE
# =====================================================================

def banner(title: str):
    print("\n" + "#" * 72)
    print(f"##  {title}")
    print("#" * 72)


def experiment_1_simple_sequential():
    banner("EXPERIMENT 1: SimpleSequentialChain & The Information Bottleneck")
    llm = SimulatedLLM()

    # Stage 1: Generate company name
    chain_name = LegacyLLMChain(
        llm=llm,
        template="Suggest a catchy name for a startup that: {description}.",
        input_keys=["description"],
        output_key="company_name"
    )

    # Stage 2: Generate tagline
    chain_tagline = LegacyLLMChain(
        llm=llm,
        template="Write a punchy 5-word tagline for the company: {company_name}.",
        input_keys=["company_name"],
        output_key="tagline"
    )

    simple_pipeline = SimpleSequentialChain(chains=[chain_name, chain_tagline], verbose=True)

    idea = "builds autonomous electric cargo drones for emergency medical deliveries"
    final_output = simple_pipeline.run(idea)
    
    print("\n⚠️  [Information Bottleneck Analysis]:")
    print("   Notice that Stage 2 ONLY received 'AeroPulse Drone Dynamics'.")
    print("   The fact that it was specifically for 'emergency medical deliveries' was LOST")
    print("   because SimpleSequentialChain cannot pass prior inputs forward!")


def experiment_2_multi_variable_sequential():
    banner("EXPERIMENT 2: Multi-Variable SequentialChain with State Accumulation")
    llm = SimulatedLLM()

    # Stage 1: Summarize review
    chain_summary = LegacyLLMChain(
        llm=llm,
        template="Summarize this customer review for product '{product}':\n{review}",
        input_keys=["product", "review"],
        output_key="summary"
    )

    # Stage 2: Classify sentiment
    chain_sentiment = LegacyLLMChain(
        llm=llm,
        template="Classify sentiment of this issue summary as POSITIVE, NEUTRAL, or NEGATIVE:\n{summary}",
        input_keys=["summary"],
        output_key="sentiment"
    )

    # Stage 3: Classify defect category
    chain_defect = LegacyLLMChain(
        llm=llm,
        template="Given product '{product}' and issue '{summary}', identify the hardware defect.",
        input_keys=["product", "summary"],
        output_key="defect"
    )

    # Stage 4: Draft resolution email
    chain_email = LegacyLLMChain(
        llm=llm,
        template="Draft a resolution email for product '{product}'.\nIssue: {summary}\nDefect: {defect}\nSentiment: {sentiment}",
        input_keys=["product", "summary", "defect", "sentiment"],
        output_key="reply_email"
    )

    support_pipeline = SequentialChain(
        chains=[chain_summary, chain_sentiment, chain_defect, chain_email],
        input_variables=["product", "review"],
        output_variables=["summary", "sentiment", "defect", "reply_email"],
        verbose=True
    )

    sample_input = {
        "product": "AirPulse Noise-Cancelling Headphones",
        "review": "I bought these 3 weeks ago. The sound was incredible, but yesterday the left earbud stopped charging completely. The case LED blinks red and won't reset."
    }

    final_state = support_pipeline(sample_input, return_all=True)

    print("\n📊 Final Accumulated State Dictionary:")
    for k, v in final_state.items():
        if k != "review":
            print(f"   🔑 [{k}]:\n      {v}\n")


def experiment_3_modern_lcel_assign():
    banner("EXPERIMENT 3: Modern LCEL Pipeline via RunnablePassthrough.assign()")
    llm = SimulatedLLM()

    # Building modern LCEL pipeline using .assign() pattern
    pipeline = ModernLCELPipeline()
    pipeline.assign(
        "summary",
        lambda state: llm.generate(f"Summarize this review: {state['review']}")
    ).assign(
        "sentiment",
        lambda state: llm.generate(f"Classify sentiment of '{state['summary']}' as POSITIVE, NEUTRAL, or NEGATIVE")
    ).assign(
        "reply_email",
        lambda state: llm.generate(
            f"Write a resolution email for {state['product']}. Summary: {state['summary']}, Sentiment: {state['sentiment']}"
        )
    )

    input_payload = {
        "product": "AirPulse Noise-Cancelling Headphones",
        "review": "Left earbud stopped charging after 3 weeks."
    }

    print(f"Input: {input_payload}")
    lcel_result = pipeline.invoke(input_payload)
    print("\nModern LCEL Output State:")
    for key in ["summary", "sentiment", "reply_email"]:
        print(f"   [{key}]: {lcel_result[key]}")


def experiment_4_guardrail_interceptors():
    banner("EXPERIMENT 4: Intermediate Guardrail Interceptor & Self-Healing")

    def raw_unreliable_sentiment(state: Dict[str, Any]) -> str:
        # Simulate an LLM emitting slightly malformed formatting
        return "  sentiment_label: NEGATIVE (confidence: 98%)  "

    def guardrail_sanitizer(raw_val: str) -> str:
        # Enforce canonical uppercase enum: POSITIVE, NEUTRAL, NEGATIVE
        raw_clean = raw_val.strip().upper()
        for valid in ["POSITIVE", "NEUTRAL", "NEGATIVE"]:
            if valid in raw_clean:
                return valid
        return "NEUTRAL"  # Self-healing fallback

    raw_sentiment = raw_unreliable_sentiment({})
    clean_sentiment = guardrail_sanitizer(raw_sentiment)

    print(f"1. Raw LLM Intermediate Output:  '{raw_sentiment}'")
    print(f"2. Sanitized Interceptor Output:  '{clean_sentiment}'")
    print("   ✅ Downstream stage receives clean enum without crashing downstream parsers!")


def experiment_5_performance_profiling():
    banner("EXPERIMENT 5: Latency Profiling (Serial Stages vs Single Prompt)")

    # Simulate timings
    t0 = time.time()
    time.sleep(0.04)  # Stage 1: Summarize
    time.sleep(0.03)  # Stage 2: Sentiment
    time.sleep(0.05)  # Stage 3: Email
    sequential_time = time.time() - t0

    t1 = time.time()
    time.sleep(0.07)  # Monolith: Single Prompt doing all 3
    monolith_time = time.time() - t1

    print(f"⏱️  3-Stage Sequential Pipeline Total Latency: {sequential_time * 1000:.1f} ms")
    print(f"⏱️  Single-Prompt Monolith Total Latency:       {monolith_time * 1000:.1f} ms")
    print("\n💡 Architectural Insight:")
    print("   Sequential pipelines execute serially, resulting in higher overall latency,")
    print("   but yield significantly higher accuracy, intermediate testability, and modularity!")


def main():
    print("""
========================================================================
   SEQUENTIAL CHAINING LAB: SimpleSequentialChain & SequentialChain
========================================================================
    """)
    experiment_1_simple_sequential()
    experiment_2_multi_variable_sequential()
    experiment_3_modern_lcel_assign()
    experiment_4_guardrail_interceptors()
    experiment_5_performance_profiling()
    print("\n✅ All 5 Sequential Chaining experiments completed successfully!\n")


if __name__ == "__main__":
    main()
