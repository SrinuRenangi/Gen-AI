"""
Pinecone Vector Database Manager
=================================
Enterprise-grade Pinecone client wrapper supporting:
- Serverless index provisioning (AWS / us-east-1)
- Batch upserts with dense vectors and rich JSON metadata
- Cosine similarity search with Top-K and metadata filtering
- Automatic fallback to an Offline In-Memory Vector Store when no API key is provided
"""

import os
import time
from typing import List, Dict, Any, Optional
import numpy as np


class OfflinePineconeIndex:
    """
    In-memory vector index replicating Pinecone's exact client interface
    for offline development, unit tests, and demonstrations without API keys.
    """

    def __init__(self, name: str, dimension: int, metric: str = "cosine"):
        self.name = name
        self.dimension = dimension
        self.metric = metric.lower()
        # Storage format: {namespace: {id: {"values": np.ndarray, "metadata": dict}}}
        self.namespaces: Dict[str, Dict[str, Dict[str, Any]]] = {"": {}}

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = "") -> Dict[str, Any]:
        """
        Upsert a list of vector dicts.
        Format: [{"id": "doc1", "values": [...], "metadata": {...}}]
        """
        if namespace not in self.namespaces:
            self.namespaces[namespace] = {}

        count = 0
        for item in vectors:
            v_id = str(item["id"])
            vals = np.array(item["values"], dtype=np.float32)
            meta = item.get("metadata", {})
            self.namespaces[namespace][v_id] = {
                "values": vals,
                "metadata": meta,
            }
            count += 1

        return {"upserted_count": count}

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        namespace: str = "",
    ) -> Dict[str, Any]:
        """
        Query the index for nearest neighbors.
        """
        ns_data = self.namespaces.get(namespace, {})
        if not ns_data:
            return {"matches": [], "namespace": namespace}

        q_vec = np.array(vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            q_norm = 1.0

        candidates = []
        for v_id, record in ns_data.items():
            meta = record["metadata"]

            # Evaluate metadata filter if provided
            if filter:
                match_filter = True
                for field, condition in filter.items():
                    val = meta.get(field)
                    if isinstance(condition, dict):
                        if "$eq" in condition and val != condition["$eq"]:
                            match_filter = False
                        elif "$ne" in condition and val == condition["$ne"]:
                            match_filter = False
                        elif "$in" in condition and val not in condition["$in"]:
                            match_filter = False
                    else:
                        if val != condition:
                            match_filter = False
                if not match_filter:
                    continue

            # Compute similarity score
            doc_vec = record["values"]
            doc_norm = np.linalg.norm(doc_vec)
            if doc_norm == 0:
                doc_norm = 1.0

            if self.metric == "cosine":
                score = float(np.dot(q_vec, doc_vec) / (q_norm * doc_norm))
            elif self.metric == "dotproduct":
                score = float(np.dot(q_vec, doc_vec))
            elif self.metric == "euclidean":
                dist = float(np.linalg.norm(q_vec - doc_vec))
                score = 1.0 / (1.0 + dist)
            else:
                score = float(np.dot(q_vec, doc_vec) / (q_norm * doc_norm))

            candidates.append({
                "id": v_id,
                "score": score,
                "metadata": meta if include_metadata else None,
            })

        # Sort descending by similarity score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return {"matches": candidates[:top_k], "namespace": namespace}

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        namespace: str = "",
    ) -> Dict[str, Any]:
        """Delete records from the index."""
        if namespace not in self.namespaces:
            return {}

        if delete_all:
            self.namespaces[namespace] = {}
        elif ids:
            for v_id in ids:
                self.namespaces[namespace].pop(v_id, None)

        return {}

    def describe_index_stats(self) -> Dict[str, Any]:
        """Return total vector count per namespace."""
        total = sum(len(v) for v in self.namespaces.values())
        ns_stats = {ns: {"vector_count": len(v)} for ns, v in self.namespaces.items()}
        return {
            "total_vector_count": total,
            "dimension": self.dimension,
            "metric": self.metric,
            "namespaces": ns_stats,
        }


class PineconeVectorStore:
    """
    High-level Pinecone Vector Database Manager.
    Automatically manages connections, index lifecycle, and graceful offline fallback.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.is_offline = not bool(self.api_key)
        self.pc = None
        self.active_index = None
        self.active_index_name = ""

        if not self.is_offline:
            try:
                from pinecone import Pinecone

                self.pc = Pinecone(api_key=self.api_key)
                print("[PineconeStore] Connected successfully to Pinecone Cloud API.")
            except Exception as e:
                print(f"[PineconeStore] Pinecone Cloud init failed ({e}). Using Offline Mock.")
                self.is_offline = True

        if self.is_offline:
            print("[PineconeStore] Initialized in Offline Mock Mode (Zero API Key Required).")

    def get_or_create_index(
        self,
        index_name: str = "genai-knowledge-base",
        dimension: int = 384,
        metric: str = "cosine",
    ) -> Any:
        """
        Creates a Pinecone serverless index if it does not exist, then binds it.
        """
        self.active_index_name = index_name

        if self.is_offline:
            self.active_index = OfflinePineconeIndex(
                name=index_name,
                dimension=dimension,
                metric=metric,
            )
            return self.active_index

        try:
            from pinecone import ServerlessSpec

            existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

            if index_name not in existing_indexes:
                print(f"[PineconeStore] Creating serverless index '{index_name}' (dim={dimension}, metric={metric})...")
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                while not self.pc.describe_index(index_name).status["ready"]:
                    time.sleep(1)
                print(f"[PineconeStore] Index '{index_name}' is ready!")

            self.active_index = self.pc.Index(index_name)
            return self.active_index

        except Exception as e:
            print(f"[PineconeStore] Cloud index operation failed ({e}). Falling back to Offline.")
            self.is_offline = True
            self.active_index = OfflinePineconeIndex(
                name=index_name,
                dimension=dimension,
                metric=metric,
            )
            return self.active_index

    def upsert_documents(
        self,
        records: List[Dict[str, Any]],
        namespace: str = "",
    ) -> Dict[str, Any]:
        """
        Batch upsert vectors and metadata.
        Each record should contain:
        {"id": "doc_1", "values": [0.1, -0.2, ...], "metadata": {"title": ..., "text": ...}}
        """
        if not self.active_index:
            raise RuntimeError("No active index selected. Call get_or_create_index() first.")

        return self.active_index.upsert(vectors=records, namespace=namespace)

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Search for top_k most similar vectors.
        Returns list of match dictionaries with id, score, and metadata.
        """
        if not self.active_index:
            raise RuntimeError("No active index selected. Call get_or_create_index() first.")

        results = self.active_index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
            namespace=namespace,
        )
        return results.get("matches", [])

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve vector count and namespace information."""
        if not self.active_index:
            return {}
        return self.active_index.describe_index_stats()
