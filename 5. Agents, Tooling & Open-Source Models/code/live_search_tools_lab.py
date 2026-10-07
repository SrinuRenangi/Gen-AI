"""
Live Search Tools Lab: Connecting Models to Live Data via Search APIs
=====================================================================

Zero to Hero Gen AI Course - Module 05: Agents, Tooling & Open-Source Models
Companion Lab: External Integration (Search APIs & Web Grounding)

This production-grade educational lab demonstrates:
  1. Experiment 1: Parametric Knowledge Hallucination vs Live Search Grounding.
  2. Experiment 2: Multi-Provider Search Gateway (SerpAPI, Tavily & DuckDuckGo).
  3. Experiment 3: Production In-Memory TTL Cache & Quota Optimizer.
  4. Experiment 4: Multi-Tier Search Provider Fallback Cascade & Error Shielding.
  5. Experiment 5: Verifiable In-Line Citation Grounding & Negative Constraint Prompting.

Features:
  - 100% standalone and runnable out-of-the-box (zero mandatory external API keys).
  - High-fidelity realistic SERP simulator for local execution + live API readiness.
  - Windows CP1252-safe UTF-8 console output.
"""

import sys
import os
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError

# Ensure Windows terminal handles UTF-8 formatting safely
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================================
# Data Models & Schemas
# ============================================================================

class WebSearchInput(BaseModel):
    """Type-safe input schema for live web search queries."""
    query: str = Field(
        ...,
        description="The optimized keyword search query (e.g. 'SpaceX Starship Flight 5 booster catch date')."
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of search results to retrieve (1-10)."
    )
    time_frame: Optional[Literal["d", "w", "m", "y"]] = Field(
        default=None,
        description="Time filter: 'd' (past 24h), 'w' (past week), 'm' (past month), 'y' (past year)."
    )


@dataclass
class SearchResultItem:
    """Standardized representation of a single search result across providers."""
    title: str
    snippet: str
    url: str
    source_type: str = "organic"  # 'answer_box', 'knowledge_graph', 'organic'
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "snippet": self.snippet,
            "url": self.url,
            "source_type": self.source_type
        }


# ============================================================================
# Search Engines & Providers
# ============================================================================

class MockSearchProvider:
    """
    Realistic multi-provider search engine simulator.
    Returns rich SERP payloads for 2024 real-world events and historical queries.
    """
    def __init__(self, provider_name: str, fail_rate: float = 0.0):
        self.provider_name = provider_name
        self.fail_rate = fail_rate

    def search(self, query: str, max_results: int = 3) -> List[SearchResultItem]:
        # Simulate network latency
        time.sleep(0.04)

        # Simulate provider failure if requested
        if self.fail_rate >= 1.0:
            raise ConnectionError(f"HTTP 503 Service Unavailable from {self.provider_name} gateway.")

        q_lower = query.lower()

        # Database of realistic real-time events post-2023
        if "starship" in q_lower or "spacex" in q_lower:
            return [
                SearchResultItem(
                    title="SpaceX Catches Super Heavy Booster in Historic Starship Flight 5",
                    snippet="On October 13, 2024, SpaceX achieved an engineering milestone by catching the Starship Super Heavy booster using the mechanical 'chopstick' arms of the Mechazilla launch tower at Starbase, Texas.",
                    url="https://spacex.com/updates/flight-5",
                    source_type="answer_box"
                ),
                SearchResultItem(
                    title="Reuters: SpaceX Mechazilla Tower Catches Rocket Booster",
                    snippet="SpaceX launched its fifth Starship test flight from South Texas and successfully recovered the massive Super Heavy booster mid-air as it returned to the launch pad.",
                    url="https://reuters.com/technology/space/spacex-starship-flight-5-catch-2024",
                    source_type="organic"
                ),
                SearchResultItem(
                    title="BBC News - SpaceX Starship test flight 5 details",
                    snippet="The flight demonstrated the feasibility of rapid rocket reusability, with FAA approval granted for the orbital test trajectory.",
                    url="https://bbc.com/news/science-environment-starship-5",
                    source_type="organic"
                )
            ][:max_results]

        elif "olympic" in q_lower or "100m" in q_lower or "gold" in q_lower:
            return [
                SearchResultItem(
                    title="Noah Lyles Wins Men's 100m Gold in Paris 2024 Olympics",
                    snippet="American sprinter Noah Lyles won the Olympic gold medal in the men's 100 meters at the Paris 2024 Games on August 4, 2024, finishing in a personal best 9.79 seconds in a photo finish.",
                    url="https://olympics.com/en/paris-2024/results/athletics/mens-100m",
                    source_type="answer_box"
                ),
                SearchResultItem(
                    title="Kishane Thompson Takes Silver in 100m Final",
                    snippet="Jamaica's Kishane Thompson took the silver medal just five thousandths of a second behind Lyles with 9.79 seconds.",
                    url="https://espn.com/olympics/story/_/id/paris-2024-mens-100m",
                    source_type="organic"
                )
            ][:max_results]

        elif "nvidia" in q_lower or "blackwell" in q_lower:
            return [
                SearchResultItem(
                    title="Nvidia Blackwell Ultra Architecture Overview",
                    snippet="Nvidia announced its next-generation Blackwell B200 GPU architecture featuring 208 billion transistors, delivering up to 30x faster inference for trillion-parameter LLMs.",
                    url="https://nvidianews.nvidia.com/news/nvidia-blackwell-platform",
                    source_type="knowledge_graph"
                ),
                SearchResultItem(
                    title="Tom's Hardware: Nvidia Blackwell Shipping Timeline",
                    snippet="CEO Jensen Huang confirmed Blackwell GPUs entered full mass production in Q4 2024, with major hyperscalers deploying clusters in early 2025.",
                    url="https://tomshardware.com/pc-components/gpus/nvidia-blackwell-status",
                    source_type="organic"
                )
            ][:max_results]

        else:
            return [
                SearchResultItem(
                    title=f"General Web Search Results for: {query}",
                    snippet=f"General informative overview regarding {query}. Contains overview data, background analysis, and industry references.",
                    url=f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    source_type="organic"
                )
            ][:max_results]


# ============================================================================
# Caching & Multi-Tier Resilience Gateway
# ============================================================================

class CachedSearchGateway:
    """
    Production-ready Search Gateway with SHA-256 in-memory caching and TTL eviction.
    """
    def __init__(self, primary_provider, secondary_provider=None, default_ttl_seconds: int = 300):
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.default_ttl = default_ttl_seconds
        self.cache: Dict[str, Tuple[float, List[SearchResultItem]]] = {}
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _hash_key(self, query: str, max_results: int) -> str:
        raw = f"{query.strip().lower()}:{max_results}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def search(self, query: str, max_results: int = 3) -> Tuple[List[SearchResultItem], str]:
        """
        Executes search with cache check and fallback cascade.
        Returns: (results, source_indicator)
        """
        self.total_requests += 1
        key = self._hash_key(query, max_results)
        now = time.time()

        # 1. Check Cache
        if key in self.cache:
            expiry, cached_results = self.cache[key]
            if now < expiry:
                self.cache_hits += 1
                return cached_results, "CACHE_HIT"
            else:
                del self.cache[key]  # Expired

        self.cache_misses += 1

        # 2. Try Primary Provider
        try:
            results = self.primary_provider.search(query, max_results=max_results)
            self.cache[key] = (now + self.default_ttl, results)
            return results, f"PRIMARY ({self.primary_provider.provider_name})"
        except Exception as primary_err:
            # 3. Fallback to Secondary Provider
            if self.secondary_provider:
                try:
                    fallback_results = self.secondary_provider.search(query, max_results=max_results)
                    self.cache[key] = (now + self.default_ttl, fallback_results)
                    return fallback_results, f"FALLBACK ({self.secondary_provider.provider_name})"
                except Exception as fallback_err:
                    raise RuntimeError(f"All search providers failed! Primary: {primary_err} | Fallback: {fallback_err}")
            else:
                raise primary_err


# ============================================================================
# Context Formatter & Grounded Synthesizer
# ============================================================================

class GroundedCitationSynthesizer:
    """
    Assembles search snippets with explicit numeric anchors [1], [2]
    and simulates LLM grounded synthesis with strict citation constraints.
    """

    @staticmethod
    def format_search_context(items: List[SearchResultItem]) -> str:
        """Formats search items into indexed context blocks."""
        lines = []
        for i, item in enumerate(items, start=1):
            source_tag = f"[{item.source_type.upper()}]" if item.source_type != "organic" else ""
            lines.append(f"[{i}] {source_tag} Title: {item.title}")
            lines.append(f"    URL: {item.url}")
            lines.append(f"    Snippet: {item.snippet}")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def synthesize_answer(question: str, items: List[SearchResultItem], enforce_negative: bool = True) -> Dict[str, Any]:
        """Synthesizes grounded output with audit citations."""
        if not items:
            return {
                "answer": "The search engine returned zero results. Unable to provide a verified answer.",
                "citations": [],
                "grounded": False
            }

        # Check relevance
        q_tokens = set(question.lower().split())
        relevant_items = []
        for item in items:
            text = (item.title + " " + item.snippet).lower()
            if any(t in text for t in q_tokens if len(t) > 3):
                relevant_items.append(item)

        if not relevant_items and enforce_negative:
            return {
                "answer": (
                    "Based strictly on the provided search results, there is no verified information "
                    "answering your question. The system refuses to speculate without grounding."
                ),
                "citations": [],
                "grounded": False
            }

        # Build grounded synthesis for known topics
        primary = relevant_items[0] if relevant_items else items[0]
        
        if "starship" in question.lower() or "spacex" in question.lower():
            answer = (
                "SpaceX conducted the historic Starship Flight 5 test on October 13, 2024, "
                "successfully catching the Super Heavy booster mid-air using the 'chopstick' mechanical arms "
                "of the Mechazilla launch tower at Starbase, Texas [1]. This milestone validated rapid "
                "rocket reusability for future orbital flights [2]."
            )
            citations = [items[0].url, items[1].url]
        elif "100m" in question.lower() or "olympic" in question.lower():
            answer = (
                "American sprinter Noah Lyles won the men's 100-meter gold medal at the Paris 2024 Olympics "
                "on August 4, 2024, finishing in 9.79 seconds [1]. He edged out Jamaica's Kishane Thompson "
                "by five thousandths of a second in a dramatic photo finish [2]."
            )
            citations = [items[0].url, items[1].url]
        else:
            answer = f"According to verified search records, {primary.snippet} [1]."
            citations = [primary.url]

        return {
            "answer": answer,
            "citations": citations,
            "grounded": True,
            "source_count": len(items)
        }


# ============================================================================
# The 5 Experimental Suites
# ============================================================================

def run_experiment_1():
    """Experiment 1: Parametric Knowledge Hallucination vs Live Search Grounding."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 1: Parametric Hallucination vs Live Search Grounding")
    print("#"*80)

    question = "Who caught the SpaceX Starship booster during Flight 5 in October 2024 and how?"
    print(f"\nUser Query: '{question}'\n")

    # Part A: Static Model (Cutoff 2023)
    print("--- [Part A: Static Pre-Trained LLM (Knowledge Cutoff: 2023)] ---")
    static_response = (
        "As of my last update, SpaceX has not attempted to catch a Starship booster with mechanical arms. "
        "Historically, Falcon 9 boosters land on autonomous drone ships in the ocean using landing legs."
    )
    print(f"Static Model Output:\n\"{static_response}\"")
    print("Evaluation: ❌ OUTDATED / FACTUALLY FALSE (Pre-training cutoff lacks 2024 events).\n")

    # Part B: Search-Grounded Agent
    print("--- [Part B: Live Search Grounded Pipeline] ---")
    search_provider = MockSearchProvider(provider_name="Tavily-Live")
    results = search_provider.search(question, max_results=2)
    synthesizer = GroundedCitationSynthesizer()
    grounded = synthesizer.synthesize_answer(question, results)

    print(f"Retrieved Search Context:\n{synthesizer.format_search_context(results)}\n")
    print(f"Grounded Model Output:\n\"{grounded['answer']}\"")
    print(f"Verified Citations: {grounded['citations']}")
    print("Evaluation: ✅ 100% ACCURATE & GROUNDED IN LIVE FACTS.")


def run_experiment_2():
    """Experiment 2: Multi-Provider Search Gateway (SerpAPI, Tavily & DuckDuckGo)."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 2: Multi-Provider Search Gateway & Schema Validation")
    print("#"*80)

    # 1. Pydantic validation
    print("\n1. Testing Pydantic Input Schema Validation:")
    try:
        valid_input = WebSearchInput(query="Nvidia Blackwell GPU shipping schedule", max_results=3, time_frame="w")
        print(f"   ✅ Valid Input Accepted: query='{valid_input.query}', max_results={valid_input.max_results}, time='{valid_input.time_frame}'")
    except ValidationError as e:
        print(f"   Validation Error: {e}")

    try:
        invalid_input = WebSearchInput(query="test", max_results=99)  # Exceeds max 10
        print(f"   Failed to catch invalid max_results: {invalid_input}")
    except ValidationError as e:
        print(f"   ✅ Pydantic Caught Out-of-Bounds max_results: {e.errors()[0]['msg']}")

    # 2. Testing Search Result Normalization
    print("\n2. Querying Providers with Normalized SERP Decomposition:")
    providers = [
        MockSearchProvider(provider_name="Google-SerpAPI"),
        MockSearchProvider(provider_name="Tavily-Search"),
        MockSearchProvider(provider_name="DuckDuckGo-Open")
    ]

    for p in providers:
        items = p.search("Noah Lyles Paris 2024 Olympics 100m", max_results=2)
        print(f"\n   [{p.provider_name}] returned {len(items)} normalized items:")
        for item in items:
            print(f"     - Type: {item.source_type:<14} | Title: {item.title[:45]}...")


def run_experiment_3():
    """Experiment 3: Production In-Memory TTL Cache & Quota Optimizer."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 3: Production In-Memory TTL Cache & Quota Optimizer")
    print("#"*80)

    provider = MockSearchProvider(provider_name="SerpAPI-Live")
    gateway = CachedSearchGateway(primary_provider=provider, default_ttl_seconds=60)

    queries = [
        "SpaceX Starship flight 5 booster catch",
        "SpaceX Starship flight 5 booster catch",  # Repeat (Cache Hit)
        "Nvidia Blackwell B200 transistor count",
        "SpaceX Starship flight 5 booster catch",  # Repeat (Cache Hit)
        "Noah Lyles Paris 2024 100m gold",
        "Nvidia Blackwell B200 transistor count",  # Repeat (Cache Hit)
    ]

    print(f"Simulating {len(queries)} user queries against Search Gateway:\n")
    for idx, q in enumerate(queries, start=1):
        t0 = time.time()
        results, source = gateway.search(q, max_results=2)
        latency_ms = (time.time() - t0) * 1000.0
        print(f"   Query #{idx}: '{q[:35]}...' -> [{source:<20}] Latency: {latency_ms:.2f}ms")

    print("\nCache Performance Audit:")
    print(f"   Total Requests: {gateway.total_requests}")
    print(f"   Cache Hits    : {gateway.cache_hits} ({(gateway.cache_hits/gateway.total_requests)*100:.1f}%)")
    print(f"   Cache Misses  : {gateway.cache_misses} ({(gateway.cache_misses/gateway.total_requests)*100:.1f}%)")
    print("   API Cost Savings: 50.0% reduction in external billable API hits!")


def run_experiment_4():
    """Experiment 4: Provider Fallback Cascade & Error Shielding."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 4: Provider Fallback Cascade & Error Shielding")
    print("#"*80)

    # Primary provider configured to FAIL (HTTP 503)
    failing_primary = MockSearchProvider(provider_name="Tavily-Primary", fail_rate=1.0)
    healthy_secondary = MockSearchProvider(provider_name="SerpAPI-Secondary", fail_rate=0.0)

    resilient_gateway = CachedSearchGateway(
        primary_provider=failing_primary,
        secondary_provider=healthy_secondary
    )

    query = "SpaceX Starship booster catch"
    print(f"Executing Query: '{query}'")
    print("Configured: Primary=Tavily (FAILING) | Secondary=SerpAPI (HEALTHY)\n")

    results, source = resilient_gateway.search(query, max_results=2)
    print(f"Outcome: Gateway safely intercepted error and routed to: {source}")
    print(f"Retrieved {len(results)} items successfully from fallback.")
    assert "FALLBACK" in source, "Gateway failed to invoke fallback provider!"
    print("✅ High-Availability Test Passed: Process did not crash; user received verified data.")


def run_experiment_5():
    """Experiment 5: Verifiable Citation Grounding & Negative Constraint Enforcement."""
    print("\n" + "#"*80)
    print("🧪 EXPERIMENT 5: Verifiable Citations & Negative Constraint Prompting")
    print("#"*80)

    synthesizer = GroundedCitationSynthesizer()

    # Part A: Grounded Synthesis with Citations
    print("--- [Part A: Query with Verified Search Evidence] ---")
    items = [
        SearchResultItem(
            title="Noah Lyles Wins Men's 100m Gold in Paris 2024",
            snippet="Noah Lyles won the 100m gold in 9.79s in Paris 2024.",
            url="https://olympics.com/100m-paris",
            source_type="answer_box"
        ),
        SearchResultItem(
            title="Kishane Thompson Takes Silver",
            snippet="Jamaica's Kishane Thompson took silver with 9.79s.",
            url="https://espn.com/athletics/silver",
            source_type="organic"
        )
    ]
    res_a = synthesizer.synthesize_answer("Who won the 100m in Paris 2024?", items)
    print(f"Answer:\n{res_a['answer']}\n")
    print("Citations Footer:")
    for idx, c in enumerate(res_a['citations'], start=1):
        print(f"   [{idx}] {c}")

    # Part B: Adversarial Unverifiable Query (Negative Constraint Enforcement)
    print("\n--- [Part B: Negative Constraint Prompting (Information Not in SERP)] ---")
    adversarial_query = "What is the secret recipe of Coca-Cola locked in the Atlanta vault?"
    print(f"Adversarial Query: '{adversarial_query}'")
    empty_items: List[SearchResultItem] = []
    res_b = synthesizer.synthesize_answer(adversarial_query, empty_items, enforce_negative=True)
    print(f"Model Output:\n\"{res_b['answer']}\"")
    print(f"Grounded: {res_b['grounded']} (Hallucination Prevented: ✅)")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    print("="*80)
    print("🌐 LIVE SEARCH APIS & EXTERNAL MODEL INTEGRATION LAB")
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
    print("🎉 ALL 5 SEARCH INTEGRATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
