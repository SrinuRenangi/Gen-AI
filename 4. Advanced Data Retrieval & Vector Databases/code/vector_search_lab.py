"""
Vector Search & High-Dimensional Distance Metrics Lab
=====================================================
Zero to Hero Gen AI Course — Module 04: Advanced Data Retrieval & Vector Databases

This standalone educational lab demonstrates the mathematical and algorithmic
mechanics of vector search:
1. The 4 Distance Metrics Compared (Cosine, Dot Product, Euclidean L2, Manhattan L1)
2. The Curse of Dimensionality: Hypersphere Shell Concentration Simulation
3. The Distance Concentration Phenomenon Across Dimensions (d = 2 to 1536)
4. Exact k-NN Brute-Force Search Engine Benchmarking
5. Single-Stage Filtered Vector Search vs Post-Filtering Failure

Usage:
    py vector_search_lab.py
"""

import sys
import os
import math
import time
import random
import heapq
from typing import Dict, List, Tuple, Any, Optional

# Ensure UTF-8 output on Windows consoles to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =====================================================================
# SECTION 1: CORE VECTOR MATHEMATICS ENGINE
# =====================================================================

def dot_product(u: List[float], v: List[float]) -> float:
    """Computes Dot Product: sum(u_i * v_i)"""
    return sum(a * b for a, b in zip(u, v))


def vector_norm(u: List[float]) -> float:
    """Computes Euclidean Length: sqrt(sum(u_i^2))"""
    return math.sqrt(sum(a * a for a in u))


def cosine_similarity(u: List[float], v: List[float]) -> float:
    """Computes Cosine Similarity: (u . v) / (||u|| * ||v||)"""
    norm_u = vector_norm(u)
    norm_v = vector_norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot_product(u, v) / (norm_u * norm_v)


def euclidean_distance(u: List[float], v: List[float]) -> float:
    """Computes Euclidean L2 Distance: sqrt(sum((u_i - v_i)^2))"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def manhattan_distance(u: List[float], v: List[float]) -> float:
    """Computes Manhattan L1 Distance: sum(|u_i - v_i|)"""
    return sum(abs(a - b) for a, b in zip(u, v))


def l2_normalize(u: List[float]) -> List[float]:
    """Scales vector to unit Euclidean length (||u|| = 1.0)"""
    norm = vector_norm(u)
    if norm == 0:
        return u
    return [x / norm for x in u]


def generate_random_vector(dim: int, normalize: bool = True) -> List[float]:
    """Generates a random Gaussian vector on a d-dimensional hypersphere."""
    vec = [random.gauss(0.0, 1.0) for _ in range(dim)]
    return l2_normalize(vec) if normalize else vec


# =====================================================================
# LAB EXPERIMENTS & DEMONSTRATION SUITE
# =====================================================================

def banner(title: str):
    print("\n" + "#" * 72)
    print(f"##  {title}")
    print("#" * 72)


def experiment_1_distance_metrics_compared():
    banner("EXPERIMENT 1: The Four Pillar Distance Metrics Compared")

    # Three vectors: Query, Highly Relevant Document, Completely Irrelevant Document
    random.seed(42)
    dim = 64
    base_signal = [random.gauss(0.0, 1.0) for _ in range(dim)]

    # Query
    q = l2_normalize(base_signal)
    # Doc 1: Aligned topic with slight noise
    d_rel = l2_normalize([x + random.gauss(0.0, 0.2) for x in base_signal])
    # Doc 2: Independent random direction
    d_irr = generate_random_vector(dim, normalize=True)

    # Compute all 4 metrics
    metrics = {
        "Cosine Similarity (Higher = Closer)": (cosine_similarity(q, d_rel), cosine_similarity(q, d_irr)),
        "Dot Product (Higher = Closer)":       (dot_product(q, d_rel),       dot_product(q, d_irr)),
        "Euclidean L2 (Lower = Closer)":       (euclidean_distance(q, d_rel), euclidean_distance(q, d_irr)),
        "Manhattan L1 (Lower = Closer)":       (manhattan_distance(q, d_rel), manhattan_distance(q, d_irr))
    }

    print(f"{'Metric':<38} | {'Relevant Doc':<15} | {'Irrelevant Doc':<15} | {'Ranking Valid?':<12}")
    print("-" * 88)
    for name, (rel_val, irr_val) in metrics.items():
        is_valid = (rel_val > irr_val) if ("Cosine" in name or "Dot" in name) else (rel_val < irr_val)
        valid_badge = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"{name:<38} | {rel_val:<15.4f} | {irr_val:<15.4f} | {valid_badge:<12}")

    print("-" * 88)
    print("💡 Notice: All 4 metrics agree on the relative rank ordering for normalized vectors!")


def experiment_2_hypersphere_shell_concentration():
    banner("EXPERIMENT 2: Hypersphere Outer Shell Concentration Simulation")

    # Fraction of volume in the outer 1% skin: 1 - (1 - epsilon)^d
    epsilon = 0.01  # Thin 1% surface boundary
    dimensions = [2, 3, 5, 10, 50, 100, 512, 1536, 3072]

    print(f"Testing a thin 1% surface layer (epsilon = {epsilon}) across dimensions:\n")
    print(f"{'Dimensions (d)':<16} | {'Inner Core Volume %':<24} | {'Outer 1% Skin Volume %':<24}")
    print("-" * 70)

    for d in dimensions:
        inner_ratio = (1.0 - epsilon) ** d
        outer_ratio = 1.0 - inner_ratio
        inner_pct = inner_ratio * 100.0
        outer_pct = outer_ratio * 100.0
        print(f"{d:<16} | {inner_pct:<24.6f}% | {outer_pct:<24.6f}%")

    print("-" * 70)
    print("⚠️  Geometric Reality:")
    print("   At d = 1,536 (OpenAI embeddings), 99.99998% of the hypersphere's volume")
    print("   resides within the outermost 1% skin! The interior is effectively hollow vacuum.")


def experiment_3_distance_concentration_phenomenon():
    banner("EXPERIMENT 3: Distance Concentration Phenomenon Across Dimensions")

    # For varying dimensions, generate 200 random unit vectors and measure pairwise Euclidean distances
    test_dims = [2, 10, 50, 256, 1536]
    n_samples = 200

    print(f"Sampling {n_samples} vectors per dimension to measure distance contrast:\n")
    print(f"{'Dim (d)':<10} | {'Min Dist':<12} | {'Max Dist':<12} | {'Mean Dist':<12} | {'Contrast Ratio (Max-Min)/Min':<28}")
    print("-" * 82)

    for d in test_dims:
        vectors = [generate_random_vector(d) for _ in range(n_samples)]
        distances = []
        for i in range(len(vectors)):
            for j in range(i + 1, min(i + 20, len(vectors))):
                distances.append(euclidean_distance(vectors[i], vectors[j]))

        min_d = min(distances)
        max_d = max(distances)
        mean_d = sum(distances) / len(distances)
        contrast = (max_d - min_d) / min_d if min_d > 0 else 0.0

        print(f"{d:<10} | {min_d:<12.4f} | {max_d:<12.4f} | {mean_d:<12.4f} | {contrast:<28.4f}")

    print("-" * 82)
    print("💡 Insight:")
    print("   As dimensions explode (d=1536), the contrast ratio drops drastically!")
    print("   Random vectors all converge to an average distance of sqrt(2) ≈ 1.414,")
    print("   proving they are almost universally near-orthogonal (90 degrees)!")


def experiment_4_exact_knn_benchmark():
    banner("EXPERIMENT 4: Exact k-NN Brute-Force Search Engine Benchmark")

    dim = 256
    num_docs = 5000
    top_k = 5

    print(f"Building in-memory vector collection: {num_docs} documents, dimension = {dim}...")
    random.seed(101)
    db = [
        {"id": idx, "vector": generate_random_vector(dim), "title": f"Document_{idx:05d}"}
        for idx in range(num_docs)
    ]

    query_vec = generate_random_vector(dim)

    # Brute-force scan
    t_start = time.perf_counter()
    scores = []
    for item in db:
        sim = dot_product(query_vec, item["vector"])
        scores.append((sim, item["id"], item["title"]))

    # Sort descending
    top_results = sorted(scores, key=lambda x: x[0], reverse=True)[:top_k]
    t_elapsed = (time.perf_counter() - t_start) * 1000.0

    print(f"\n⏱️  Exact k-NN Scan Latency: {t_elapsed:.2f} ms ({num_docs} vectors evaluated)")
    print(f"\nTop-{top_k} Most Semantically Similar Documents:")
    for rank, (score, doc_id, title) in enumerate(top_results, 1):
        print(f"   #{rank} [{title}]: Cosine Score = {score:.4f}")

    print(f"\n📊 Complexity Projection:")
    print(f"   At 5,000 vectors:  {t_elapsed:.2f} ms")
    print(f"   At 1,000,000 vectors:  {(t_elapsed * (1_000_000 / num_docs)):.1f} ms  (Too slow for interactive chat!)")
    print(f"   This demonstrates why ANN indexes (HNSW) are essential for enterprise retrieval.")


def experiment_5_filtered_vector_search():
    banner("EXPERIMENT 5: Single-Stage Filtered Search vs Post-Filtering Failure")

    dim = 64
    num_docs = 1000
    random.seed(77)

    # Database where only 2% of documents belong to tenant "Enterprise-Acme"
    db = []
    for i in range(num_docs):
        tenant = "Enterprise-Acme" if i % 50 == 0 else "Public-Standard"
        db.append({
            "id": i,
            "vector": generate_random_vector(dim),
            "tenant": tenant,
            "title": f"Doc_{i} (Tenant: {tenant})"
        })

    query_vec = generate_random_vector(dim)
    target_tenant = "Enterprise-Acme"
    top_k = 5

    print(f"Database Size: {num_docs} docs. Target Filter: tenant == '{target_tenant}' (Only 2% of database)\n")

    # 1. NAIVE POST-FILTERING: Search Top-10 globally, then apply filter
    global_top10 = sorted(
        [(dot_product(query_vec, doc["vector"]), doc) for doc in db],
        key=lambda x: x[0],
        reverse=True
    )[:10]

    post_filtered = [item for item in global_top10 if item[1]["tenant"] == target_tenant]
    print(f"1. Naive Post-Filtering Results (Search Top-10 -> Filter by Tenant):")
    print(f"   Returned Candidates: {len(post_filtered)} / {top_k} requested")
    for sim, doc in post_filtered:
        print(f"   - {doc['title']} (Score: {sim:.4f})")
    if len(post_filtered) < top_k:
        print(f"   ❌ TRUNCATION FAILURE: Returned fewer than {top_k} results because Acme was rare in global Top-10!\n")

    # 2. SINGLE-STAGE FILTERED SEARCH: Evaluate filter during candidate selection
    single_stage_candidates = []
    for doc in db:
        if doc["tenant"] == target_tenant:  # Metadata filter check
            sim = dot_product(query_vec, doc["vector"])
            single_stage_candidates.append((sim, doc))

    single_stage_top_k = sorted(single_stage_candidates, key=lambda x: x[0], reverse=True)[:top_k]
    print(f"2. Single-Stage Filtered Search Results (Filter evaluated during traversal):")
    print(f"   Returned Candidates: {len(single_stage_top_k)} / {top_k} requested")
    for rank, (sim, doc) in enumerate(single_stage_top_k, 1):
        print(f"   #{rank} {doc['title']} (Score: {sim:.4f})")
    print("   ✅ SUCCESS: Delivered full Top-K relevant documents without truncation!")


def main():
    print("""
========================================================================
   VECTOR SEARCH & HIGH-DIMENSIONAL DISTANCE METRICS LAB
========================================================================
    """)
    experiment_1_distance_metrics_compared()
    experiment_2_hypersphere_shell_concentration()
    experiment_3_distance_concentration_phenomenon()
    experiment_4_exact_knn_benchmark()
    experiment_5_filtered_vector_search()
    print("\n✅ All 5 Vector Search experiments completed successfully!\n")


if __name__ == "__main__":
    main()
