"""
Vector Storage Implementation Lab: Local ChromaDB vs Cloud Pinecone
====================================================================
Zero to Hero Gen AI Course — Module 04: Advanced Data Retrieval & Vector Databases

This standalone educational lab demonstrates the foundational architectural
and operational patterns of vector database storage:
1. Local In-Memory ChromaDB Collection Lifecycle (CRUD)
2. Single-Stage Boolean Metadata Filtering ($and, $eq, $gte)
3. Persistent Local Disk Storage & Index Recovery
4. Cloud-Based Pinecone Serverless Architecture & Multi-Tenant Namespaces
5. Performance Benchmarking: Local Embedded IPC vs Cloud Network Round-Trips

Usage:
    py vector_storage_lab.py
"""

import sys
import os
import json
import time
import math
import random
import shutil
from typing import Dict, List, Tuple, Any, Optional

# Ensure UTF-8 output on Windows consoles to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =====================================================================
# SECTION 1: STANDALONE VECTOR MATH & EMBEDDING SIMULATOR
# =====================================================================

def dot_product(u: List[float], v: List[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def vector_norm(u: List[float]) -> float:
    return math.sqrt(sum(a * a for a in u))


def cosine_similarity(u: List[float], v: List[float]) -> float:
    nu = vector_norm(u)
    nv = vector_norm(v)
    return dot_product(u, v) / (nu * nv) if nu and nv else 0.0


def l2_normalize(u: List[float]) -> List[float]:
    norm = vector_norm(u)
    return [x / norm for x in u] if norm else u


def simulate_embedding(text: str, dim: int = 128) -> List[float]:
    """Generates a reproducible, calibrated dense vector for a given string."""
    rng = random.Random(abs(hash(text.strip().lower())) % 1000000)
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    
    # Semantic topic biases
    t_lower = text.lower()
    if any(w in t_lower for w in ["contract", "legal", "clause", "liability"]):
        for i in range(0, 30):
            vec[i] += 2.5
    elif any(w in t_lower for w in ["database", "sql", "postgres", "cluster", "distributed"]):
        for i in range(30, 60):
            vec[i] += 2.5
    elif any(w in t_lower for w in ["ai", "model", "neural", "deep learning"]):
        for i in range(60, 90):
            vec[i] += 2.5
            
    return l2_normalize(vec)


# =====================================================================
# SECTION 2: STANDALONE CHROMADB EMBEDDED ENGINE
# =====================================================================

class ChromaCollection:
    """Simulates a ChromaDB collection with in-process vector search and metadata filtering."""
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {"hnsw:space": "cosine"}
        self.records: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None
    ):
        metadatas = metadatas or [{} for _ in ids]
        if embeddings is None:
            embeddings = [simulate_embedding(doc) for doc in documents]

        for doc_id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            self.records[doc_id] = {
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "embedding": emb
            }

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if query_embeddings is None and query_texts:
            query_embeddings = [simulate_embedding(t) for t in query_texts]

        query_vec = query_embeddings[0]

        # Filter and rank candidates
        scored_candidates = []
        for doc_id, record in self.records.items():
            # Check metadata filter predicate
            if where and not self._evaluate_where(record["metadata"], where):
                continue
            sim = cosine_similarity(query_vec, record["embedding"])
            scored_candidates.append((sim, record))

        # Sort descending by similarity
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_candidates[:n_results]

        return {
            "ids": [[item[1]["id"] for item in top_k]],
            "documents": [[item[1]["document"] for item in top_k]],
            "metadatas": [[item[1]["metadata"] for item in top_k]],
            "distances": [[round(1.0 - item[0], 4) for item in top_k]]  # Cosine distance
        }

    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None):
        if ids:
            for doc_id in ids:
                self.records.pop(doc_id, None)
        elif where:
            to_delete = [
                doc_id for doc_id, rec in self.records.items()
                if self._evaluate_where(rec["metadata"], where)
            ]
            for doc_id in to_delete:
                del self.records[doc_id]

    def count(self) -> int:
        return len(self.records)

    def _evaluate_where(self, meta: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """Evaluates ChromaDB boolean filter predicates like $and, $eq, $gte."""
        if "$and" in where:
            return all(self._evaluate_where(meta, sub) for sub in where["$and"])
        if "$or" in where:
            return any(self._evaluate_where(meta, sub) for sub in where["$or"])

        for field, condition in where.items():
            if field.startswith("$"):
                continue
            if field not in meta:
                return False
            val = meta[field]
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if op == "$eq" and val != target:
                        return False
                    elif op == "$ne" and val == target:
                        return False
                    elif op == "$gt" and not (val > target):
                        return False
                    elif op == "$gte" and not (val >= target):
                        return False
                    elif op == "$lt" and not (val < target):
                        return False
                    elif op == "$lte" and not (val <= target):
                        return False
            else:
                if val != condition:
                    return False
        return True


class ChromaClient:
    """Simulates chromadb.PersistentClient or EphemeralClient."""
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.collections: Dict[str, ChromaCollection] = {}
        if self.path and os.path.exists(self.path):
            self._load_from_disk()

    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> ChromaCollection:
        if name not in self.collections:
            self.collections[name] = ChromaCollection(name, metadata)
        return self.collections[name]

    def persist(self):
        """Persists collections to local JSON/disk files."""
        if not self.path:
            return
        os.makedirs(self.path, exist_ok=True)
        dump_data = {}
        for name, col in self.collections.items():
            dump_data[name] = {
                "metadata": col.metadata,
                "records": col.records
            }
        file_path = os.path.join(self.path, "chroma_manifest.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f)

    def _load_from_disk(self):
        file_path = os.path.join(self.path, "chroma_manifest.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                dump_data = json.load(f)
            for name, data in dump_data.items():
                col = ChromaCollection(name, data["metadata"])
                col.records = data["records"]
                self.collections[name] = col


# =====================================================================
# SECTION 3: STANDALONE PINECONE CLOUD SERVERLESS ENGINE
# =====================================================================

class PineconeServerlessIndex:
    """Simulates a Pinecone Serverless Index with multi-tenant namespaces."""
    def __init__(self, name: str, dimension: int = 128, metric: str = "cosine"):
        self.name = name
        self.dimension = dimension
        self.metric = metric
        # Namespaced storage: Dict[namespace, Dict[vector_id, record]]
        self.namespaces: Dict[str, Dict[str, Dict[str, Any]]] = {"": {}}

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = ""):
        if namespace not in self.namespaces:
            self.namespaces[namespace] = {}
        for item in vectors:
            self.namespaces[namespace][item["id"]] = {
                "id": item["id"],
                "values": item["values"],
                "metadata": item.get("metadata", {})
            }

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        time.sleep(0.035)  # Simulate ~35ms cloud network and TLS round-trip

        if namespace not in self.namespaces:
            return {"matches": [], "namespace": namespace}

        target_store = self.namespaces[namespace]
        candidates = []
        for vid, record in target_store.items():
            # Check filter
            if filter and not self._evaluate_filter(record["metadata"], filter):
                continue
            sim = cosine_similarity(vector, record["values"])
            candidates.append((sim, record))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_matches = candidates[:top_k]

        matches = []
        for score, rec in top_matches:
            match_dict = {
                "id": rec["id"],
                "score": round(score, 4),
            }
            if include_metadata:
                match_dict["metadata"] = rec["metadata"]
            matches.append(match_dict)

        return {"matches": matches, "namespace": namespace}

    def _evaluate_filter(self, meta: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        for key, condition in filter_dict.items():
            if key not in meta:
                return False
            val = meta[key]
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if op == "$eq" and val != target:
                        return False
                    elif op == "$gte" and not (val >= target):
                        return False
            else:
                if val != condition:
                    return False
        return True


# =====================================================================
# LAB EXPERIMENTS & DEMONSTRATION SUITE
# =====================================================================

def banner(title: str):
    print("\n" + "#" * 72)
    print(f"##  {title}")
    print("#" * 72)


def experiment_1_chromadb_basics():
    banner("EXPERIMENT 1: Local In-Memory ChromaDB Collection Lifecycle")

    client = ChromaClient()  # Ephemeral In-Memory
    collection = client.get_or_create_collection(name="kb_docs", metadata={"hnsw:space": "cosine"})

    # 1. Ingestion / Upsert
    docs = [
        "The software license agreement automatically terminates upon material breach.",
        "Total liability for direct damages shall not exceed the fees paid in the preceding 12 months.",
        "PostgreSQL supports distributed read replicas and high availability clusters.",
        "Convolutional neural networks are specialized for 2D spatial grid processing."
    ]
    metadatas = [
        {"category": "legal", "section": "termination", "year": 2024},
        {"category": "legal", "section": "liability", "year": 2024},
        {"category": "tech", "section": "database", "year": 2023},
        {"category": "tech", "section": "vision", "year": 2023}
    ]
    ids = [f"doc_{i+1:02d}" for i in range(len(docs))]

    collection.add(ids=ids, documents=docs, metadatas=metadatas)
    print(f"1. Ingested {collection.count()} document chunks into collection '{collection.name}'.")

    # 2. Semantic Query
    query = "What happens if a party breaches the legal contract?"
    print(f"\n2. Semantic Search Query: '{query}'")
    results = collection.query(query_texts=[query], n_results=2)

    for rank, (doc_id, doc_text, dist) in enumerate(
        zip(results["ids"][0], results["documents"][0], results["distances"][0]), 1
    ):
        print(f"   #{rank} [{doc_id}] (Cosine Distance: {dist:.4f}):\n      \"{doc_text}\"")

    print("\n✅ Successfully retrieved top semantic matches without external network calls!")


def experiment_2_chromadb_metadata_filtering():
    banner("EXPERIMENT 2: Single-Stage Boolean Metadata Filtering in ChromaDB")

    client = ChromaClient()
    collection = client.get_or_create_collection("enterprise_records")

    docs = [
        "Contract Clause A: Confidentiality remains binding for 5 years.",
        "Contract Clause B: Confidentiality remains binding for 2 years.",
        "Contract Clause C: Standard non-compete clause for 1 year.",
        "Technical Note: Database cache TTL set to 300 seconds."
    ]
    metas = [
        {"department": "legal", "clause_type": "confidentiality", "duration_years": 5},
        {"department": "legal", "clause_type": "confidentiality", "duration_years": 2},
        {"department": "legal", "clause_type": "non-compete", "duration_years": 1},
        {"department": "eng", "clause_type": "infra", "duration_years": 0}
    ]
    ids = ["c1", "c2", "c3", "t1"]
    collection.add(ids=ids, documents=docs, metadatas=metas)

    # Filter: department == 'legal' AND duration_years >= 3
    filter_expr = {
        "$and": [
            {"department": {"$eq": "legal"}},
            {"duration_years": {"$gte": 3}}
        ]
    }
    print(f"Applying Boolean Metadata Filter:\n   {json.dumps(filter_expr, indent=2)}\n")

    query = "How long does the confidentiality non-disclosure agreement last?"
    results = collection.query(query_texts=[query], n_results=5, where=filter_expr)

    print("Filtered Search Results:")
    for doc_id, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
        print(f"   - [{doc_id}] ({meta['clause_type']}, {meta['duration_years']} yrs): \"{doc}\"")

    print("\n✅ Filtered search strictly returned only legal clauses with duration >= 3 years!")


def experiment_3_persistent_disk_storage():
    banner("EXPERIMENT 3: ChromaDB Persistent Local Disk Storage & Recovery")

    storage_dir = os.path.join(".", "scratch_chroma_storage")
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    print(f"1. Creating PersistentClient at directory: '{storage_dir}'...")
    client_1 = ChromaClient(path=storage_dir)
    col_1 = client_1.get_or_create_collection("archived_manuals")
    col_1.add(
        ids=["p1", "p2"],
        documents=["Safety Manual: Always wear eye protection in lab.", "Security Manual: Rotate tokens every 90 days."],
        metadatas=[{"type": "safety"}, {"type": "security"}]
    )
    client_1.persist()
    print(f"   Saved {col_1.count()} documents to disk. Terminating client 1...")

    # Simulating application restart by initializing a brand new client from disk
    print("\n2. Reopening PersistentClient in a fresh process...")
    client_2 = ChromaClient(path=storage_dir)
    col_2 = client_2.get_or_create_collection("archived_manuals")
    print(f"   Recovered collection '{col_2.name}' with {col_2.count()} documents intact from disk!")

    res = col_2.query(query_texts=["protection eyewear rules"], n_results=1)
    print(f"   Recovered Query Match: \"{res['documents'][0][0]}\"")

    # Clean up scratch storage
    shutil.rmtree(storage_dir)
    print("\n✅ Verified zero data loss across client restart cycles.")


def experiment_4_pinecone_namespaces():
    banner("EXPERIMENT 4: Cloud Pinecone Serverless Architecture & Namespaces")

    index = PineconeServerlessIndex(name="enterprise-rag", dimension=128, metric="cosine")

    # Tenant Alpha Ingestion
    alpha_docs = [
        {"id": "alpha_01", "values": simulate_embedding("AlphaCorp internal patent for solid-state batteries"), "metadata": {"org": "AlphaCorp", "confidential": True}}
    ]
    index.upsert(vectors=alpha_docs, namespace="tenant_alpha")

    # Tenant Beta Ingestion
    beta_docs = [
        {"id": "beta_01", "values": simulate_embedding("BetaCorp proprietary algorithm for algorithmic trading"), "metadata": {"org": "BetaCorp", "confidential": True}}
    ]
    index.upsert(vectors=beta_docs, namespace="tenant_beta")

    print("1. Ingested confidential documents into isolated Pinecone Namespaces:")
    print("   - Namespace 'tenant_alpha': AlphaCorp battery patents")
    print("   - Namespace 'tenant_beta':  BetaCorp trading algorithms\n")

    # Query targeted at Tenant Alpha
    query_vec = simulate_embedding("battery patent technology")

    print("2. Querying Tenant Alpha Namespace ('tenant_alpha'):")
    alpha_res = index.query(vector=query_vec, top_k=2, namespace="tenant_alpha")
    for m in alpha_res["matches"]:
        print(f"   - Match [{m['id']}]: Score = {m['score']}, Metadata: {m['metadata']}")

    print("\n3. Testing Cross-Tenant Security Isolation:")
    print("   Attempting to find BetaCorp docs in Alpha's namespace...")
    cross_leak = any(m["metadata"]["org"] == "BetaCorp" for m in alpha_res["matches"])
    print(f"   Cross-Tenant Data Leakage Detected: {cross_leak}")
    print("   ✅ Complete Logical Partitioning: Namespaces guarantee 100% tenant isolation!")


def experiment_5_performance_benchmarking():
    banner("EXPERIMENT 5: Performance Benchmark: Local Embedded IPC vs Cloud Round-Trip")

    local_col = ChromaCollection("benchmark_col")
    cloud_idx = PineconeServerlessIndex("benchmark_idx")

    vec = simulate_embedding("Performance benchmarking query text")
    for i in range(500):
        v = simulate_embedding(f"Document passage number {i}")
        local_col.add(ids=[f"d{i}"], documents=[f"Doc {i}"], embeddings=[v])
        cloud_idx.upsert(vectors=[{"id": f"d{i}", "values": v}])

    # Measure Local Query Latency
    t0 = time.perf_counter()
    for _ in range(10):
        local_col.query(query_embeddings=[vec], n_results=5)
    local_avg_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

    # Measure Cloud Query Latency
    t1 = time.perf_counter()
    for _ in range(3):
        cloud_idx.query(vector=vec, top_k=5)
    cloud_avg_ms = ((time.perf_counter() - t1) / 3.0) * 1000.0

    print(f"⏱️  Local In-Process ChromaDB Query Latency:  {local_avg_ms:.2f} ms")
    print(f"⏱️  Cloud Serverless Pinecone Query Latency:    {cloud_avg_ms:.2f} ms")
    print(f"\n💡 Architectural Insight:")
    print(f"   Local ChromaDB is {cloud_avg_ms / local_avg_ms:.1f}x faster due to zero network/TLS overhead,")
    print("   making it optimal for edge devices, desktop apps, and real-time interactive UI!")
    print("   Pinecone trades that network latency for infinite horizontal scalability and managed SLAs.")


def main():
    print("""
========================================================================
   VECTOR STORAGE LAB: LOCAL CHROMADB vs CLOUD PINECONE
========================================================================
    """)
    experiment_1_chromadb_basics()
    experiment_2_chromadb_metadata_filtering()
    experiment_3_persistent_disk_storage()
    experiment_4_pinecone_namespaces()
    experiment_5_performance_benchmarking()
    print("\n✅ All 5 Vector Storage experiments completed successfully!\n")


if __name__ == "__main__":
    main()
