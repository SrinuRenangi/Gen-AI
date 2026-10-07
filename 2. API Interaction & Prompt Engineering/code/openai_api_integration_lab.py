"""
=============================================================================
Hands-On Lab: Production OpenAI API Integration & Resilience in Python
=============================================================================
Course: Zero to Hero Gen AI — Module 02: API Interaction & Prompt Engineering
Topic: OpenAI API Integration: Environments, Keys & Request-Response Cycles

This lab implements and validates:
1. Environment Variable Validation: Secure loading and secret masking.
2. Chat Completion Request-Response Lifecycle: Payload assembly & token cost auditing.
3. Streaming Token Generator (SSE): Simulating real-time token rendering & TTFT.
4. Resilient Exponential Backoff with Jitter: Handling rate limits (HTTP 429).
5. Structured JSON Output Extraction: Machine-readable data generation.
=============================================================================
"""

import os
import sys
import time
import random
import json


# =============================================================================
# PART 1: Environment & Secret Hygiene
# =============================================================================
def check_environment():
    print("=" * 75)
    print("PART 1: Environment Variable Check & Secret Hygiene")
    print("=" * 75)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ℹ️  OPENAI_API_KEY is not set in environment.")
        print("   Running in SIMULATED PRODUCTION MODE (Realistic Mock Engine).")
        print("   All request-response cycles, tokens, and payloads will be verified.")
        is_live = False
    else:
        # Never log plain keys!
        masked = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 12 else "sk-***"
        print(f"✅ Found active OPENAI_API_KEY: {masked}")
        is_live = True

    print("Secret Hygiene Rules:")
    print("  • Never commit .env files to Git repositories.")
    print("  • Never hardcode api_key='sk-...' in source code.")
    print("  • Use project-scoped keys with soft/hard monthly spend caps.\n")
    return is_live


# =============================================================================
# PART 2: Chat Completion Request-Response & Cost Accounting
# =============================================================================
def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Calculates exact dollar cost for an OpenAI API call."""
    pricing_per_million = {
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    }
    rates = pricing_per_million.get(model, pricing_per_million["gpt-4o-mini"])
    cost_in = (prompt_tokens / 1_000_000) * rates["prompt"]
    cost_out = (completion_tokens / 1_000_000) * rates["completion"]
    return cost_in + cost_out


def demo_chat_completion(is_live: bool):
    print("=" * 75)
    print("PART 2: Chat Completion Request-Response Lifecycle & Token Accounting")
    print("=" * 75)

    messages = [
        {"role": "system", "content": "You are a concise data science mentor. Explain in 2 sentences."},
        {"role": "user", "content": "What is the difference between supervised and unsupervised learning?"}
    ]

    model = "gpt-4o-mini"
    print(f"Model: {model} | Temperature: 0.3 | Max Tokens: 150")
    print("Payload Messages Array:")
    for m in messages:
        print(f"  [{m['role'].upper()}]: {m['content']}")

    if is_live:
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=150
            )
            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            finish_reason = response.choices[0].finish_reason
        except Exception as e:
            print(f"⚠️ Live API call failed ({e}). Falling back to simulation.")
            is_live = False

    if not is_live:
        # Realistic Mock Response
        content = (
            "Supervised learning trains on labeled input-output pairs to predict known outcomes, "
            "whereas unsupervised learning discovers hidden patterns and clusters within unlabeled data."
        )
        prompt_tokens = 34
        completion_tokens = 28
        total_tokens = 62
        finish_reason = "stop"

    cost = calculate_cost(prompt_tokens, completion_tokens, model)

    print("\n--- API RESPONSE ---")
    print(f"Assistant Content:\n\"{content}\"\n")
    print(f"Finish Reason    : {finish_reason} (Completed successfully)")
    print(f"Prompt Tokens    : {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Total Tokens     : {total_tokens}")
    print(f"Estimated Cost   : ${cost:.6f} USD\n")


# =============================================================================
# PART 3: Streaming Tokens via Server-Sent Events (SSE)
# =============================================================================
def demo_streaming(is_live: bool):
    print("=" * 75)
    print("PART 3: Streaming Response (Server-Sent Events / SSE)")
    print("=" * 75)
    print("Demonstrating Time to First Token (TTFT) optimization...\n")

    prompt = "Write a short haiku about neural networks."
    print(f"Prompt: \"{prompt}\"")
    print("Streaming Tokens: ", end="", flush=True)

    if is_live:
        try:
            from openai import OpenAI
            client = OpenAI()
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            start_time = time.time()
            ttft = None
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if ttft is None:
                        ttft = (time.time() - start_time) * 1000
                    print(delta, end="", flush=True)
            print(f"\n\n⚡ Time to First Token (TTFT): {ttft:.1f}ms\n")
            return
        except Exception:
            pass

    # High-fidelity streaming generator simulation
    simulated_tokens = [
        "Layers ", "of ", "neurons ", "\n",
        "Learning ", "patterns ", "in ", "the ", "dark ", "\n",
        "Insights ", "spark ", "awake."
    ]

    start_time = time.time()
    time.sleep(0.18)  # 180ms TTFT
    ttft = (time.time() - start_time) * 1000

    for token in simulated_tokens:
        print(token, end="", flush=True)
        time.sleep(0.04)  # 40ms per token cadence

    print(f"\n\n⚡ Time to First Token (TTFT): {ttft:.1f}ms (Instant perceived latency!)")
    print("Notice: Words appear in real-time rather than freezing the UI for seconds.\n")


# =============================================================================
# PART 4: Resilient Exponential Backoff with Jitter
# =============================================================================
def demo_exponential_backoff():
    print("=" * 75)
    print("PART 4: Production Resilience — Exponential Backoff with Jitter (HTTP 429)")
    print("=" * 75)

    def flaky_api_call(attempt_state={"calls": 0}):
        attempt_state["calls"] += 1
        if attempt_state["calls"] < 3:
            # Simulate RateLimitError (HTTP 429)
            raise ConnectionResetError(f"HTTP 429: Rate limit exceeded (TPM quota reached).")
        return {"status": 200, "result": "Success on attempt 3!"}

    max_retries = 4
    base_delay = 0.5
    max_delay = 8.0

    print("Initiating API call that triggers rate limits on initial attempts...")
    success = False

    for attempt in range(max_retries):
        try:
            res = flaky_api_call()
            print(f"✅ Success on Attempt {attempt + 1}: {res['result']}")
            success = True
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Permanent Failure after {max_retries} attempts.")
                raise e

            # Formula: Uniform(0, min(max_delay, base * 2^attempt))
            backoff_cap = min(max_delay, base_delay * (2 ** attempt))
            jittered_sleep = random.uniform(0.1, backoff_cap)
            print(f"⚠️ Attempt {attempt + 1} Failed: {e}")
            print(f"   Applying Exponential Backoff + Jitter: sleeping {jittered_sleep:.3f}s...")
            time.sleep(jittered_sleep)

    print("\nResult: Exponential backoff with jitter prevented crashing and bypassed the rate limit!\n")


# =============================================================================
# PART 5: Structured JSON Outputs
# =============================================================================
def demo_structured_json():
    print("=" * 75)
    print("PART 5: Structured Outputs with response_format={'type': 'json_object'}")
    print("=" * 75)

    mock_raw_output = """
    {
        "sentiment": "positive",
        "urgency_score": 2,
        "category": "feature_request",
        "summary": "User loves the speed of the API and requests support for streaming tool calls."
    }
    """

    print("Parsing raw model output string into Python dictionary...")
    parsed_json = json.loads(mock_raw_output.strip())

    print("Parsed JSON Object:")
    print(json.dumps(parsed_json, indent=2))
    print(f"\nExtracted Sentiment: {parsed_json['sentiment'].upper()}")
    print(f"Extracted Category : {parsed_json['category'].upper()}")
    print("Safe for programmatic consumption in database pipelines!\n")


def main():
    print("\n" + "#" * 75)
    print("#   ZERO TO HERO GEN AI: OPENAI API INTEGRATION & RESILIENCE LAB   #")
    print("#" * 75 + "\n")

    is_live = check_environment()
    demo_chat_completion(is_live)
    demo_streaming(is_live)
    demo_exponential_backoff()
    demo_structured_json()


if __name__ == "__main__":
    main()
