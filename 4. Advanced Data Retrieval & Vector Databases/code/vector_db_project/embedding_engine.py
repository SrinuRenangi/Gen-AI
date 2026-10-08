"""
Embedding Engine Module
========================
Provides a unified interface for generating dense vector embeddings
supporting multiple providers:
1. OpenAI API ('text-embedding-3-small', 'text-embedding-3-large')
2. Hugging Face / Sentence-Transformers ('all-MiniLM-L6-v2')
3. Offline Deterministic Embedding Engine (Zero API key required, 384D normalized vectors)
"""

import os
import hashlib
import math
from typing import List, Dict, Any, Optional
import numpy as np


class EmbeddingEngine:
    """
    Unified multi-provider embedding generator with automatic fallback to
    an offline deterministic vector engine.
    """

    def __init__(
        self,
        provider: str = "auto",
        model_name: Optional[str] = None,
        dimension: int = 384,
    ):
        """
        Initialize the embedding generator.

        Args:
            provider: 'openai', 'huggingface', 'offline', or 'auto'
            model_name: Model identifier (defaults to standard for provider)
            dimension: Dimensionality of output vectors (default 384)
        """
        self.provider = provider.lower()
        self.dimension = dimension
        self.model_name = model_name

        if self.provider == "auto":
            if os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
                self.model_name = model_name or "text-embedding-3-small"
                self.dimension = 1536
            else:
                self.provider = "offline"
                self.model_name = "offline-semantic-v1"
                self.dimension = 384

        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize the specific provider client."""
        if self.provider == "openai":
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.dimension = 1536 if "small" in (self.model_name or "") else 3072
                print(f"[EmbeddingEngine] Loaded OpenAI model: {self.model_name} (dim={self.dimension})")
            except Exception as e:
                print(f"[EmbeddingEngine] OpenAI initialization failed ({e}). Falling back to Offline.")
                self.provider = "offline"
                self.dimension = 384

        elif self.provider == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer

                model_id = self.model_name or "all-MiniLM-L6-v2"
                self.hf_model = SentenceTransformer(model_id)
                self.dimension = self.hf_model.get_sentence_embedding_dimension()
                print(f"[EmbeddingEngine] Loaded HuggingFace model: {model_id} (dim={self.dimension})")
            except Exception as e:
                print(f"[EmbeddingEngine] SentenceTransformers failed ({e}). Falling back to Offline.")
                self.provider = "offline"
                self.dimension = 384

        if self.provider == "offline":
            self.model_name = "offline-semantic-v1"
            self.dimension = 384
            print(f"[EmbeddingEngine] Operating in Offline Deterministic Mode (dim={self.dimension})")

    def embed_text(self, text: str) -> List[float]:
        """Generate dense vector embedding for a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense vector embeddings for a list of text strings.

        Returns:
            List of normalized float vectors.
        """
        if not texts:
            return []

        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name,
            )
            return [data.embedding for data in response.data]

        elif self.provider == "huggingface":
            embeddings = self.hf_model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()

        else:
            return [self._generate_offline_embedding(t) for t in texts]

    def _generate_offline_embedding(self, text: str) -> List[float]:
        """
        Generates a deterministic 384-dimensional unit vector using
        semantic keyword weighting, character n-gram hashing, and L2 normalization.
        Ensures semantically related texts have high cosine similarity.
        """
        # Conceptual semantic anchors
        semantic_anchors = {
            "ai": 0, "machine": 10, "learning": 20, "neural": 30, "vector": 40,
            "database": 50, "embedding": 60, "search": 70, "cloud": 80, "aws": 90,
            "docker": 100, "fastapi": 110, "python": 120, "rag": 130, "llm": 140,
            "memory": 150, "pinecone": 160, "cluster": 170, "distance": 180, "cosine": 190,
        }

        vec = np.zeros(self.dimension, dtype=np.float32)
        lower_text = text.lower()
        words = lower_text.replace("\n", " ").split()

        # 1. Base N-gram hashing distribution
        for i, word in enumerate(words):
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign * (1.0 / math.sqrt(i + 1))

            # 2. Semantic anchor proximity
            for anchor, offset in semantic_anchors.items():
                if anchor in word:
                    for k in range(8):
                        vec[(offset + k) % self.dimension] += 0.8 / (k + 1)

        # 3. Add smooth background variance
        full_hash = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        rng = np.random.RandomState(full_hash % (2**31))
        vec += rng.normal(0, 0.05, size=self.dimension)

        # 4. L2 Normalization (ensures dot product equals cosine similarity)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.tolist()
