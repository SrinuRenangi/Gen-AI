"""
Hands-On Pinecone Vector Database Demo
======================================
Demonstrates the full vector database lifecycle:
1. Ingestion: Convert raw texts into dense embeddings
2. Upsert: Store vectors + rich metadata into Pinecone
3. Semantic Search: Query by meaning rather than exact keywords
4. Metadata Filtering: Filter results by category/attribute
5. RAG Retrieval: Assemble retrieved context into an augmented prompt
"""

import os
import json
from embedding_engine import EmbeddingEngine
from pinecone_manager import PineconeVectorStore

# 1. Sample Unstructured Document Corpus
DOCUMENTS = [
    {
        "id": "doc_01",
        "title": "Transformers and Self-Attention",
        "category": "AI",
        "text": (
            "The Transformer architecture replaces recurrent loops with self-attention mechanisms. "
            "It computes Queries, Keys, and Values (Q, K, V) across all tokens in parallel, "
            "allowing language models to capture long-range contextual dependencies effortlessly."
        ),
    },
    {
        "id": "doc_02",
        "title": "Vector Databases and HNSW Graph Indexing",
        "category": "Database",
        "text": (
            "Vector databases store high-dimensional embeddings and use Approximate Nearest Neighbor (ANN) "
            "algorithms like Hierarchical Navigable Small World (HNSW). Instead of scanning millions of vectors, "
            "HNSW navigates multi-layer geometric graphs with logarithmic search complexity."
        ),
    },
    {
        "id": "doc_03",
        "title": "Amazon Bedrock Serverless Foundation Models",
        "category": "Cloud",
        "text": (
            "Amazon Bedrock provides managed serverless access to high-performing foundation models like "
            "Claude 3.5 Sonnet and Llama 3 via the Converse API. It eliminates GPU cluster management and integrates "
            "with AWS PrivateLink for secure enterprise data isolation."
        ),
    },
    {
        "id": "doc_04",
        "title": "LangChain ReAct Autonomous Agents",
        "category": "AI",
        "text": (
            "Autonomous AI agents use the ReAct framework: alternating between Thought, Action, and Observation. "
            "By pairing an LLM with external tools (calculators, web search, APIs), agents solve multi-step problems "
            "and retain conversational state in scratchpad memory."
        ),
    },
    {
        "id": "doc_05",
        "title": "PostgreSQL Relational ACID Guarantees",
        "category": "Database",
        "text": (
            "Relational databases excel at structured tables, strict schemas, and ACID transaction guarantees. "
            "However, traditional B-Tree indexes fail when searching unstructured high-dimensional embeddings, "
            "which is why extensions like pgvector or dedicated vector databases are needed."
        ),
    },
    {
        "id": "doc_06",
        "title": "Docker Multi-Stage Production Builds",
        "category": "Cloud",
        "text": (
            "Production Dockerfiles use multi-stage builds to compile dependencies in a temporary stage and copy "
            "only the final runtime artifacts into a slim image. Non-root user permissions and health check probes "
            "ensure secure deployment on AWS ECS and App Runner."
        ),
    },
]


def run_pinecone_demo():
    print("=" * 70)
    print("🚀 DAY 06: PINECONE VECTOR DATABASE & SEMANTIC RETRIEVAL DEMO")
    print("=" * 70)

    # Step 1: Initialize Embedding Engine & Vector Store
    embedder = EmbeddingEngine(provider="auto")
    vector_store = PineconeVectorStore()

    index_name = "course-knowledge-base"
    vector_store.get_or_create_index(
        index_name=index_name,
        dimension=embedder.dimension,
        metric="cosine",
    )

    # Step 2: Vectorize Documents and Upsert to Pinecone
    print(f"\n[Step 1] Ingesting & Vectorizing {len(DOCUMENTS)} documents...")
    records_to_upsert = []
    for doc in DOCUMENTS:
        vec = embedder.embed_text(doc["text"])
        records_to_upsert.append({
            "id": doc["id"],
            "values": vec,
            "metadata": {
                "title": doc["title"],
                "category": doc["category"],
                "text": doc["text"],
            },
        })

    upsert_res = vector_store.upsert_documents(records_to_upsert)
    print(f"✅ Successfully upserted vectors into Pinecone! Response: {upsert_res}")

    stats = vector_store.get_stats()
    print(f"📊 Index Stats: Total Vectors = {stats.get('total_vector_count')}")

    # Step 3: Semantic Search Demonstration
    query_1 = "How do AI systems store embeddings and search by meaning instead of exact words?"
    print("\n" + "-" * 70)
    print(f"🔍 [Query 1 - Semantic Search]: \"{query_1}\"")
    print("-" * 70)

    q1_vec = embedder.embed_text(query_1)
    results_1 = vector_store.similarity_search(q1_vec, top_k=2)

    for rank, match in enumerate(results_1, start=1):
        print(f"\n  Rank #{rank} | Similarity Score: {match['score']:.4f}")
        print(f"  Title: {match['metadata']['title']} (Category: {match['metadata']['category']})")
        print(f"  Snippet: {match['metadata']['text'][:120]}...")

    # Step 4: Metadata Filtering Demonstration
    query_2 = "What are the best serverless options for running scalable AI in the cloud?"
    print("\n" + "-" * 70)
    print(f"🎯 [Query 2 - Filtered Search (category='Cloud')]: \"{query_2}\"")
    print("-" * 70)

    q2_vec = embedder.embed_text(query_2)
    results_2 = vector_store.similarity_search(
        q2_vec,
        top_k=2,
        filter={"category": {"$eq": "Cloud"}},
    )

    for rank, match in enumerate(results_2, start=1):
        print(f"\n  Rank #{rank} | Similarity Score: {match['score']:.4f}")
        print(f"  Title: {match['metadata']['title']} (Category: {match['metadata']['category']})")
        print(f"  Snippet: {match['metadata']['text']}")

    # Step 5: RAG Prompt Assembly
    print("\n" + "-" * 70)
    print("🧠 [Step 5] Retrieval-Augmented Generation (RAG) Prompt Constructor")
    print("-" * 70)

    user_question = "Why do relational databases struggle with high-dimensional embeddings?"
    q_rag_vec = embedder.embed_text(user_question)
    rag_matches = vector_store.similarity_search(q_rag_vec, top_k=2)

    context_str = "\n\n".join([
        f"--- Source: {m['metadata']['title']} (Score: {m['score']:.2f}) ---\n{m['metadata']['text']}"
        for m in rag_matches
    ])

    rag_prompt = f"""You are a Principal Cloud & AI Architect. Answer the user question based strictly on the retrieved context below.

### RETRIEVED CONTEXT:
{context_str}

### USER QUESTION:
{user_question}

### GROUNDED ANSWER:"""

    print("Synthesized RAG Prompt for LLM:\n")
    print(rag_prompt)
    print("\n" + "=" * 70)
    print("🎉 Pinecone Vector DB Pipeline execution completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_pinecone_demo()
