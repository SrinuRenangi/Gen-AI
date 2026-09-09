# Day 48: RAG Part 2 — Advanced RAG, Re-Ranking & Evaluation


| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 47: RAG Part 1](../Day_47_RAG_Part_1/Day_47_RAG_Part_1.md) | [All 50 Days Overview](../../README.md) | [Day 49: AI Agents & Function Calling →](../Day_49_AI_Agents/Day_49_AI_Agents.md) |

Welcome to **Day 48 of our 50-Day Generative AI Masterclass**! In [Day 47](../Day_47_RAG_Part_1/Day_47_RAG_Part_1.md), you mastered the foundational RAG pipeline: document parsing, chunking with sliding overlap, dense embeddings, and HNSW vector indexing.

However, anyone who has deployed a basic "Naive RAG" system into production knows that it often fails in real-world scenarios:
- A user queries an exact error code like `"ERR_881"`, but the dense embedding retrieves generic articles about error handling.
- The retriever returns 20 chunks, but the model ignores the gold chunk because it was buried in the middle of the prompt (**"Lost in the Middle"**).
- The system confidently generates answers that sound convincing but are completely unsupported by the retrieved documents.

Today, we level up to **Advanced Production RAG**:
1. **Query Transformation & HyDE (Hypothetical Document Embeddings)**.
2. **Hybrid Search & Reciprocal Rank Fusion (RRF)**: Merging BM25 keywords with dense vectors.
3. **Cross-Encoder Re-Ranking**: Transforming weak Top-50 candidate pools into high-precision Top-3 gold nuggets.
4. **The RAG Evaluation Triad**: Quantifying Context Relevance, Faithfulness, and Answer Relevance.

---

## 1. The Core Mental Model: The Legal Research Clerk

Think of Advanced RAG as an experienced paralegal preparing a case dossier for a senior judge:

```
+-----------------------------------------------------------------------------------+
|                        THE PARALEGAL'S RETRIEVAL WORKFLOW                         |
+-----------------------------------------------------------------------------------+
|  1. The Lawyer's Ambiguous Question:                                              |
|     "Did that tech company breach fiduciary duties during the 2023 merger?"       |
|                                                                                   |
|  2. Naive RAG (The Intern):                                                       |
|     Grabs the first 10 folders with the word "merger" on the cover and dumps all  |
|     3,000 pages directly onto the judge's desk. The judge is overwhelmed.        |
|                                                                                   |
|  3. Advanced RAG (The Master Paralegal):                                          |
|     • Expands the question into specific legal statutes (HyDE & Query Expansion). |
|     • Runs both exact docket searches (BM25) and conceptual searches (Vectors).   |
|     • Merges the results via Reciprocal Rank Fusion (RRF).                        |
|     • Thoroughly reads the Top 50 pages with deep scrutiny (Cross-Encoder Re-Rank)|
|     • Hands the judge ONLY the 2 exact paragraphs with highlighted citations.     |
|     • The judge delivers a flawless, 100% grounded verdict!                       |
+-----------------------------------------------------------------------------------+
```

---

## 2. Why "Naive RAG" Fails in Production

To fix RAG, we must first understand the two primary modes of failure in naive systems:

### 1. The Vocabulary Mismatch Problem
- Dense embedding models excel at semantic themes (`"feline"` $\approx$ `"kitten"`), but struggle with **exact alphanumeric tokens**: model numbers (`"RTX-4090-Ti"`), error codes (`"ERR_9012"`), medical drugs (`"Atorvastatin 20mg"`), or database table names (`"usr_acct_v2"`).
- Keyword search (BM25) nails exact tokens, but completely misses synonyms and contextual concepts.

### 2. The "Lost in the Middle" Phenomenon
In 2023, Nelson Liu et al. (Stanford / UC Berkeley) published a landmark study:
> **Language models do not treat context uniformly.**

```
                     ATTENTION WEIGHT ACROSS PROMPT TOKENS
                     
    High Attention                                          High Attention
        |                                                        |
       \ /                                                      \ /
      +------------------------------------------------------------+
      | [START OF PROMPT]  ...   [MIDDLE TOKENS]   ...  [END/QUERY]|
      +------------------------------------------------------------+
                                       / \
                                        |
                          "The Valley of Forgetfulness"
                             (Massive Attention Drop)
```

When you stuff 15 or 20 retrieved chunks into an LLM prompt, the model pays intense attention to the first chunk and the last chunk, while **consistently overlooking facts placed in the middle**! 

To guarantee accuracy, you must **drastically reduce the number of chunks injected** by filtering through a high-precision re-ranker.

---

## 3. The Advanced RAG Architecture

Modern production RAG decouples the pipeline into three distinct phases: **Pre-Retrieval, Hybrid Retrieval, and Post-Retrieval Re-Ranking**:

![Advanced RAG Architecture](assets/advanced_rag_retrieval_and_reranking.svg)

Let's dissect each component.

---

### Technique 1: Hypothetical Document Embeddings (HyDE)

When a user asks a question, there is an **asymmetry** between the query and the document:
- Query: Short, interrogative, sparse (`"What causes database connection pool exhaustion?"`).
- Document: Long, declarative, detailed (`"When asynchronous worker threads fail to release client sockets back to HikariCP..."`).

Because questions and answers occupy slightly different neighborhoods in embedding space, searching directly with the question can miss the best documents.

In 2022, Luyu Gao et al. introduced **HyDE**:
1. Take the user's question and prompt an LLM: *"Write a hypothetical, plausible paragraph that answers this question."*
2. Even if the hypothetical answer contains slight factual inaccuracies, **its vocabulary, style, and syntax match real documents**!
3. Embed this hypothetical answer and use its vector to query the database.

HyDE dramatically shifts the search vector into the "document manifold", yielding significantly higher recall on dense corpora.

---

### Technique 2: Hybrid Search via Reciprocal Rank Fusion (RRF)

Instead of choosing between BM25 keyword search and dense vector search, **production systems always run both in parallel**.

```
User Query 
     |
     +---> [ BM25 Search ]  --------> Ranked List A: [Doc 19, Doc 42, Doc 03, ...]
     |
     +---> [ Vector Search ] -------> Ranked List B: [Doc 42, Doc 88, Doc 19, ...]
                                               |
                                     [ Reciprocal Rank Fusion ]
                                               |
                                      Merged Unified Ranking
```

#### The Reciprocal Rank Fusion (RRF) Formula
How do you merge a BM25 score (which ranges from $0$ to $+50$) with a Cosine Similarity score (which ranges from $0$ to $1$)? You cannot simply add raw scores together without complex, brittle calibration!

**Reciprocal Rank Fusion (RRF)** solves this by ignoring raw score magnitudes entirely and operating strictly on **rank positions**:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where:
- $M$: The set of retrieval systems (e.g., $M = \{\text{BM25}, \text{Vector}\}$).
- $r_m(d)$: The 1-based rank position of document $d$ in system $m$.
- $k$: A constant smoothing parameter, standardly set to **$k = 60$** (Cormack et al., 2009).

#### Step-by-Step Hand Calculation: RRF in Action

Suppose we retrieve top candidates across BM25 and Vector Search:
- **Document 42**: Rank #2 in BM25 ($r_1 = 2$); Rank #1 in Vector ($r_2 = 1$).
- **Document 19**: Rank #1 in BM25 ($r_1 = 1$); Rank #3 in Vector ($r_2 = 3$).
- **Document 88**: Not in BM25 Top 10 ($r_1 = \infty$); Rank #2 in Vector ($r_2 = 2$).

Let's compute the RRF scores with $k = 60$:

| Document | BM25 Term $\frac{1}{60 + r_{\text{bm25}}}$ | Vector Term $\frac{1}{60 + r_{\text{vec}}}$ | Total RRF Score | Final Blended Rank |
| :---: | :---: | :---: | :---: | :---: |
| **Doc 42** | $\frac{1}{60 + 2} = \frac{1}{62} \approx 0.01613$ | $\frac{1}{60 + 1} = \frac{1}{61} \approx 0.01639$ | $0.01613 + 0.01639 = \mathbf{0.03252}$ | **Rank #1 (Winner!)** |
| **Doc 19** | $\frac{1}{60 + 1} = \frac{1}{61} \approx 0.01639$ | $\frac{1}{60 + 3} = \frac{1}{63} \approx 0.01587$ | $0.01639 + 0.01587 = \mathbf{0.03226}$ | **Rank #2** |
| **Doc 88** | $0.00000$ (Missing) | $\frac{1}{60 + 2} = \frac{1}{62} \approx 0.01613$ | $0.00000 + 0.01613 = \mathbf{0.01613}$ | **Rank #3** |

RRF effortlessly rewards documents that appear consistently near the top across both retrieval modalities without requiring score normalization!

---

### Technique 3: Cross-Encoder Re-Ranking

Once RRF produces the Top 50 candidate chunks, we do NOT feed all 50 chunks to the LLM. Instead, we pass them through a **Cross-Encoder Re-Ranker** (such as **Cohere Re-Rank** or **BGE-Reranker-Large**).

#### Bi-Encoder vs. Cross-Encoder

```
BI-ENCODER (Vector Search / Fast):
[ Query ]    ---> [ Transformer Encoder ] ---> Vector Q \
                                                          Dot Product (No cross-token attention!)
[ Document ] ---> [ Transformer Encoder ] ---> Vector D /

CROSS-ENCODER (Re-Ranker / Deep & Precise):
[CLS] Query Tokens [SEP] Document Tokens [EOS] ---> [ Deep Transformer (24 Layers) ] ---> Scalar Score [0, 1]
                                                    ^
                                                    |-- Every single word in the query attends 
                                                        directly to every single word in the document!
```

- **Bi-Encoders** compress each text independently into a single vector. They are blazing fast ($\mathcal{O}(1)$ via HNSW), but lose fine-grained token-level cross-interactions.
- **Cross-Encoders** feed the query and document together into full bidirectional self-attention layers. This allows the model to spot exact negations, logical contradictions, and nuanced qualifiers.

A Cross-Encoder takes the Top 50 noisy candidates and ruthlessly filters them down to the **Top 3 pristine gold chunks**, eliminating 90%+ of irrelevant noise before generation!

---

## 4. The RAG Evaluation Triad

How do you know if your RAG pipeline is improving or degrading as you tweak chunk sizes and prompts?

The industry-standard framework (formalized by **Ragas** and **TruLens**) evaluates RAG along three orthogonal axes:

![The RAG Evaluation Triad](assets/rag_evaluation_triad_matrix.svg)

### The 3 Core Metrics

| Metric | Target Relation | Question it Answers | How It is Evaluated |
| :--- | :--- | :--- | :--- |
| **1. Context Precision / Relevance** | $\text{Query} \longleftrightarrow \text{Context}$ | *"Did the retriever fetch only relevant signal, or did it bring back useless noise?"* | An evaluator LLM checks the percentage of retrieved sentences that directly contribute to answering the query. |
| **2. Faithfulness / Groundedness** | $\text{Context} \longleftrightarrow \text{Answer}$ | *"Is every single claim in the response supported by the retrieved text (zero hallucinations)?"* | Decomposes the answer into atomic statements and verifies if each statement can be logically deduced from the context. |
| **3. Answer Relevance** | $\text{Query} \longleftrightarrow \text{Answer}$ | *"Did the response directly resolve the user's specific prompt without rambling?"* | Generates hypothetical questions from the answer and calculates semantic similarity against the original query. |

---

## 5. Production Hands-On Lab: Implementing Hybrid Search & RRF in Python

Let's build a runnable Python script that implements:
1. Lexical BM25 ranking.
2. Dense vector similarity ranking.
3. Reciprocal Rank Fusion (RRF) blending from scratch.
4. Cross-Encoder re-ranking simulation and citation-grounded answer generation.

### Python Script: `advanced_rag_pipeline.py`

```python
"""
advanced_rag_pipeline.py
Production-grade implementation of:
1. Sparse BM25 Keyword Scoring
2. Dense Vector Scoring
3. Reciprocal Rank Fusion (RRF) Algorithm
4. Cross-Encoder Re-Ranking & Grounded Answer Synthesis
Author: GenAI 50-Day Masterclass
"""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple

# =====================================================================
# 1. CORPUS DATA
# =====================================================================
DOCUMENTS = [
    {
        "id": "doc_01",
        "title": "Database Connection Management",
        "text": "Error code ERR_881 indicates a gateway timeout when microservices attempt to acquire pooled PostgreSQL connections during high concurrency spikes."
    },
    {
        "id": "doc_02",
        "title": "General Error Codes",
        "text": "System errors range from ERR_100 to ERR_900. Contact system administrator if repeated unhandled exceptions occur in backend worker nodes."
    },
    {
        "id": "doc_03",
        "title": "Payment Settlement Workflows",
        "text": "Payment gateway processing fails with error code ERR_881 when the downstream payment acquirer API does not respond within the 5,000ms SLA window."
    },
    {
        "id": "doc_04",
        "title": "Holiday Leave Policy",
        "text": "All staff receive 20 days paid leave per calendar year. Approvals must be submitted 14 days in advance through the HR portal."
    }
]

# =====================================================================
# 2. TOY BM25 KEYWORD RETRIEVER
# =====================================================================
class SimpleBM25Retriever:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs

    def search(self, query: str) -> List[Tuple[str, float]]:
        """Scores documents based on exact keyword overlap."""
        query_terms = set(query.lower().split())
        results = []
        for doc in self.docs:
            doc_terms = doc["text"].lower().split()
            # Count term overlap
            overlap = sum(1 for term in query_terms if term in doc_terms)
            score = float(overlap)
            results.append((doc["id"], score))
        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# =====================================================================
# 3. TOY DENSE VECTOR RETRIEVER
# =====================================================================
class SimpleDenseRetriever:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs

    def search(self, query: str) -> List[Tuple[str, float]]:
        """Simulates semantic concept proximity scores."""
        # Simulated similarity scores
        simulated_scores = {
            "doc_01": 0.82,  # Mentions timeout & microservices
            "doc_03": 0.91,  # Mentions payment gateway & 5000ms SLA
            "doc_02": 0.45,  # Mentions errors
            "doc_04": 0.05   # Irrelevant HR document
        }
        results = [(doc["id"], simulated_scores.get(doc["id"], 0.0)) for doc in self.docs]
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# =====================================================================
# 4. RECIPROCAL RANK FUSION (RRF) ALGORITHM
# =====================================================================
def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Computes RRF score for all documents across multiple ranked lists:
    RRF(d) = sum_m ( 1 / (k + rank_m(d)) )
    """
    rrf_scores = {}

    for ranked_list in ranked_lists:
        for rank_idx, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank_idx)

    # Sort descending by RRF score
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_rrf


# =====================================================================
# 5. CROSS-ENCODER RE-RANKING SIMULATION & SYNTHESIS
# =====================================================================
def cross_encoder_rerank(
    query: str,
    candidate_ids: List[str],
    doc_lookup: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, float]]:
    """
    Simulates cross-encoder evaluating query-document token interactions.
    Scores range from 0.0 to 1.0.
    """
    print("\n[Cross-Encoder] Scoring Candidate Chunks with Full Attention...")
    cross_scores = {
        "doc_03": 0.965, # Perfect match for payment capture ERR_881
        "doc_01": 0.620, # Mentions ERR_881, but for PostgreSQL, not payments
        "doc_02": 0.110, # Generic error code range
        "doc_04": 0.001  # Noise
    }
    ranked = [(doc_id, cross_scores.get(doc_id, 0.0)) for doc_id in candidate_ids]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# =====================================================================
# 6. RUNNABLE VERIFICATION WORKFLOW
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("ADVANCED RAG: HYBRID SEARCH & CROSS-ENCODER RE-RANKING")
    print("=" * 65)

    doc_map = {d["id"]: d for d in DOCUMENTS}
    user_query = "What does error ERR_881 mean for payment gateway transactions?"
    print(f"User Query: '{user_query}'\n")

    # 1. Run BM25 and Dense Retrieval
    bm25_engine = SimpleBM25Retriever(DOCUMENTS)
    dense_engine = SimpleDenseRetriever(DOCUMENTS)

    bm25_ranked = bm25_engine.search(user_query)
    dense_ranked = dense_engine.search(user_query)

    print("1. Individual Retrieval Rankings:")
    print(f"   BM25 Top Order  : {[doc_id for doc_id, _ in bm25_ranked]}")
    print(f"   Dense Top Order : {[doc_id for doc_id, _ in dense_ranked]}")

    # 2. Reciprocal Rank Fusion
    rrf_blended = reciprocal_rank_fusion([bm25_ranked, dense_ranked], k=60)
    print("\n2. Blended Reciprocal Rank Fusion (RRF) Scores:")
    for rank, (doc_id, score) in enumerate(rrf_blended, 1):
        print(f"   Rank #{rank}: {doc_id} -> RRF Score: {score:.5f}")

    # 3. Cross-Encoder Re-Ranking on Top Candidates
    top_candidates = [doc_id for doc_id, _ in rrf_blended[:3]]
    reranked = cross_encoder_rerank(user_query, top_candidates, doc_map)

    print("\n3. Cross-Encoder Re-Ranked Results:")
    for rank, (doc_id, score) in enumerate(reranked, 1):
        print(f"   Gold Rank #{rank}: {doc_id} (Relevance Score: {score:.3f})")

    # 4. Generate Final Grounded Synthesis
    gold_doc = doc_map[reranked[0][0]]
    print("\n" + "=" * 65)
    print("4. FINAL GROUNDED RESPONSE WITH CITATIONS:")
    print("=" * 65)
    print(f"Based on our payment documentation [1]:")
    print(f"Error code ERR_881 occurs when the payment gateway experiences a timeout because")
    print(f"the downstream payment acquirer API does not respond within the 5,000ms SLA window.")
    print(f"\nSources:")
    print(f"  [1] Document: '{gold_doc['title']}' (ID: {gold_doc['id']})")
    print("=" * 65)
```

---

## 6. Advanced RAG Comparison Matrix

| Component | Naive RAG | Advanced Production RAG |
| :--- | :--- | :--- |
| **Retrieval Method** | Dense Vector Search Only | **Hybrid (Sparse BM25 + Dense Vectors via RRF)** |
| **Query Handling** | Raw user string verbatim | **Query Rewriting, Sub-Query Decomposition, HyDE** |
| **Context Filtering** | Dumps Top 10–20 chunks into prompt | **Cross-Encoder Re-Ranking filters down to Top 2–3 gold chunks** |
| **Lost-in-the-Middle** | High vulnerability; models overlook facts | **Immune; context is compressed and ordered by relevance** |
| **Evaluation** | Human qualitative spot checks | **Automated RAG Triad (Context Precision, Faithfulness, Relevance)** |
| **Production Latency** | $\sim 50\text{ms}$ | $\sim 150\text{ms}$ (Trade-off for 99%+ accuracy) |

---

## 7. Self-Check Exercises & Solutions

### Question 1: Reciprocal Rank Fusion Hand Calculation
A company uses BM25 and Vector Search. For a user query, document `"spec_v3"` is ranked:
- Rank #3 in BM25
- Rank #2 in Vector Search
Using standard RRF with smoothing constant $k = 60$, calculate the exact numerical RRF score for `"spec_v3"`.

**Solution**:
$$\text{RRF}(\text{spec-v3}) = \frac{1}{60 + r_{\text{bm25}}} + \frac{1}{60 + r_{\text{vector}}}$$
$$\text{RRF}(\text{spec-v3}) = \frac{1}{60 + 3} + \frac{1}{60 + 2} = \frac{1}{63} + \frac{1}{62}$$
$$\frac{1}{63} \approx 0.015873, \quad \frac{1}{62} \approx 0.016129$$
$$\text{RRF}(\text{spec-v3}) = 0.015873 + 0.016129 = \mathbf{0.032002}$$

---

### Question 2: Why Cross-Encoders Cannot Replace Vector Databases
If Cross-Encoders are significantly more accurate than Bi-Encoders (Vector search), why don't we simply use a Cross-Encoder to evaluate all 10,000,000 documents in our database directly?

**Solution**:
Cross-Encoders require concatenating the query with each document and passing both through all 24+ Transformer layers with full cross-attention ($\mathcal{O}(L^2)$ complexity). Evaluating 10,000,000 documents sequentially for a single query would take hours of GPU compute and cost thousands of dollars per search.
Bi-Encoders (vector search) precompute document embeddings offline; at query time, finding the Top 50 candidates via HNSW takes only **5 milliseconds**. The Cross-Encoder is then run only on those 50 candidates ($\sim 50\text{ms}$ total), giving us **the best of both worlds: sub-100ms latency with maximum semantic precision**.

---

### Question 3: Diagnosing RAG Triad Failures
An evaluation run on a customer support RAG system reveals:
- **Context Relevance = 0.92 (High)**
- **Answer Relevance = 0.88 (High)**
- **Faithfulness = 0.35 (Critically Low)**

What is happening in this system, and what exact changes must be made to fix it?

**Solution**:
The retriever is doing its job well (fetching the correct documents), and the answer appears to address the user's question. However, the critically low Faithfulness score ($0.35$) means that **the generator LLM is hallucinating unsupported claims rather than sticking to the retrieved context**.
**Fixes**:
1. Tighten the generation system prompt: *"Answer ONLY based on the facts explicitly stated inside `<context>`. If the context does not explicitly state the answer, reply 'I do not have that information'."*
2. Lower the generation `temperature` to $0.0$.
3. Force the model to generate inline citation markers (e.g., `[Doc 1, Section 2]`) for every claim.

---

## 8. Summary & Next Steps

Today, you mastered Advanced Enterprise RAG:
- **Pre-Retrieval**: Query Expansion and Hypothetical Document Embeddings (HyDE).
- **Hybrid Retrieval**: Unifying BM25 lexical precision with dense vector semantics via Reciprocal Rank Fusion (RRF).
- **Post-Retrieval**: Cross-Encoder Re-Ranking to eliminate "Lost in the Middle" attention degradation.
- **The RAG Triad**: Quantifying Context Precision, Grounded Faithfulness, and Answer Relevance.

Tomorrow in **Day 49: AI Agents & Function Calling — LLMs That Take Action**, we graduate from passive question-answering systems to **autonomous AI agents** that plan, invoke external tools, query SQL databases, and browse the web using the **ReAct loop**!


---

## 🧭 Navigation & Next Steps

| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 47: RAG Part 1](../Day_47_RAG_Part_1/Day_47_RAG_Part_1.md) | [All 50 Days Overview](../../README.md) | [Day 49: AI Agents & Function Calling →](../Day_49_AI_Agents/Day_49_AI_Agents.md) |
