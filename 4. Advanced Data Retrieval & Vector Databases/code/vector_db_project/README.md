# 🌲 Hands-On Project: Pinecone Vector Database & Semantic RAG Pipeline

A production-grade Python implementation demonstrating how to build a semantic search and Retrieval-Augmented Generation (RAG) system using **Pinecone Vector Database** and **dense vector embeddings**.

---

## 🌟 Key Features

1. **Multi-Provider Embedding Generator (`embedding_engine.py`)**:
   - Supports **OpenAI API** (`text-embedding-3-small`, `text-embedding-3-large`).
   - Supports **Hugging Face** / `sentence-transformers` (`all-MiniLM-L6-v2`).
   - **Offline Deterministic Mode:** Generates normalized 384D dense vectors without needing any API keys, so the project runs immediately out-of-the-box!

2. **Enterprise Pinecone Manager (`pinecone_manager.py`)**:
   - Manages Pinecone Serverless indexes (AWS `us-east-1`).
   - Supports batch upserts with dense vectors and rich JSON metadata.
   - Executes Cosine Similarity queries with Top-K and metadata filtering.
   - **Offline Mock Engine:** Replicates Pinecone's exact cloud API in-memory for offline testing.

3. **Complete Semantic Pipeline (`demo.py`)**:
   - Ingests unstructured technical documents across AI, Cloud, and Database domains.
   - Performs natural language semantic search.
   - Applies structured metadata filters (`{"category": {"$eq": "Cloud"}}`).
   - Assembles grounded context into an LLM-ready RAG prompt.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
If you have a Pinecone or OpenAI API key, configure them in your environment or a `.env` file:
```bash
# Optional: If omitted, the system automatically uses the Offline Engine
export PINECONE_API_KEY="your-pinecone-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run the Demonstration
```bash
python demo.py
```
