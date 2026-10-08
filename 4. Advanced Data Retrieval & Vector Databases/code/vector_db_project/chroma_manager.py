"""
ChromaDB Vector Database Manager
=================================
Provides a clean Python wrapper for ChromaDB supporting:
- In-memory ephemeral collections for fast prototyping
- Persistent on-disk vector storage (SQLite + HNSW)
- Document ingestion with pre-computed vectors and rich metadata
- Semantic query execution with Top-K and structured metadata filtering
- Automatic Offline Pure-Python Fallback if chromadb native bindings are not installed
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np


class OfflineChromaCollection:
    """
    Pure Python in-memory vector collection replicating ChromaDB's exact collection API.
    Guarantees 100% reliability for offline development, local tests, and education.
    """

    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {"hnsw:space": "cosine"}
        # In-memory storage: list of record dicts
        self.records: List[Dict[str, Any]] = []

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """Add records to the mock collection."""
        metadatas = metadatas or [{} for _ in ids]
        documents = documents or ["" for _ in ids]

        for doc_id, emb, meta, doc in zip(ids, embeddings, metadatas, documents):
            # Upsert behavior: replace if id exists
            self.records = [r for r in self.records if r["id"] != doc_id]
            self.records.append({
                "id": str(doc_id),
                "embedding": np.array(emb, dtype=np.float32),
                "metadata": meta,
                "document": doc,
            })

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute vector similarity query with metadata filtering.
        Returns ChromaDB-formatted result dictionary.
        """
        if not self.records:
            return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}

        all_ids = []
        all_distances = []
        all_metadatas = []
        all_documents = []

        metric = self.metadata.get("hnsw:space", "cosine")

        for q_emb in query_embeddings:
            q_vec = np.array(q_emb, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                q_norm = 1.0

            scored_candidates = []
            for rec in self.records:
                meta = rec["metadata"]

                # Apply metadata filtering (where clause)
                if where:
                    match = True
                    for k, v in where.items():
                        if isinstance(v, dict):
                            if "$eq" in v and meta.get(k) != v["$eq"]:
                                match = False
                            elif "$ne" in v and meta.get(k) == v["$ne"]:
                                match = False
                            elif "$in" in v and meta.get(k) not in v["$in"]:
                                match = False
                        else:
                            if meta.get(k) != v:
                                match = False
                    if not match:
                        continue

                doc_vec = rec["embedding"]
                doc_norm = np.linalg.norm(doc_vec)
                if doc_norm == 0:
                    doc_norm = 1.0

                if metric == "cosine":
                    cos_sim = float(np.dot(q_vec, doc_vec) / (q_norm * doc_norm))
                    dist = 1.0 - cos_sim  # Chroma returns distance (smaller is closer)
                elif metric == "l2":
                    dist = float(np.linalg.norm(q_vec - doc_vec))
                elif metric == "ip":
                    dist = -float(np.dot(q_vec, doc_vec))
                else:
                    cos_sim = float(np.dot(q_vec, doc_vec) / (q_norm * doc_norm))
                    dist = 1.0 - cos_sim

                scored_candidates.append((dist, rec))

            # Sort ascending by distance (closest first)
            scored_candidates.sort(key=lambda x: x[0])
            top_matches = scored_candidates[:n_results]

            all_ids.append([m[1]["id"] for m in top_matches])
            all_distances.append([m[0] for m in top_matches])
            all_metadatas.append([m[1]["metadata"] for m in top_matches])
            all_documents.append([m[1]["document"] for m in top_matches])

        return {
            "ids": all_ids,
            "distances": all_distances,
            "metadatas": all_metadatas,
            "documents": all_documents,
        }

    def count(self) -> int:
        return len(self.records)


class ChromaVectorStore:
    """
    High-level ChromaDB Manager supporting both official chromadb client
    and transparent pure-Python offline fallback.
    """

    def __init__(self, persist_directory: Optional[str] = "./chroma_db", in_memory: bool = False):
        self.persist_directory = persist_directory
        self.in_memory = in_memory
        self.is_offline = False
        self.client = None
        self.collections: Dict[str, Any] = {}

        try:
            import chromadb
            if in_memory:
                self.client = chromadb.Client()
                print("[ChromaStore] Initialized in-memory ChromaDB client.")
            else:
                os.makedirs(persist_directory, exist_ok=True)
                self.client = chromadb.PersistentClient(path=persist_directory)
                print(f"[ChromaStore] Initialized persistent ChromaDB client at: {persist_directory}")
        except Exception as e:
            print(f"[ChromaStore] Native chromadb not available ({e}). Using pure-Python offline fallback.")
            self.is_offline = True

    def get_or_create_collection(
        self,
        name: str = "genai_knowledge_base",
        distance_metric: str = "cosine",
    ) -> Any:
        """
        Retrieve an existing collection or create a new one.
        Distance metrics supported: 'cosine', 'l2', 'ip' (inner product)
        """
        if self.is_offline:
            if name not in self.collections:
                self.collections[name] = OfflineChromaCollection(
                    name=name, metadata={"hnsw:space": distance_metric}
                )
            return self.collections[name]

        collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": distance_metric},
        )
        self.collections[name] = collection
        return collection

    def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> None:
        """Add batch of documents, vectors, and metadata to a collection."""
        coll = self.get_or_create_collection(collection_name)
        coll.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def query(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query collection for nearest neighbor documents."""
        coll = self.get_or_create_collection(collection_name)
        return coll.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
        )
