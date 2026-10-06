"""
Hands-On Vector Database Demo: Pinecone & ChromaDB
==================================================
Demonstrates the full vector database lifecycle across BOTH industry standards:
1. Ingestion: Convert raw texts into dense embeddings
2. Pinecone: Enterprise cloud-native serverless vector storage & retrieval
3. ChromaDB: Lightweight, open-source local embedded vector storage & retrieval
4. Semantic Search: Query by meaning rather than exact keywords
5. Metadata Filtering: Filter results by category/attribute
6. RAG Retrieval: Assemble retrieved context into an augmented prompt
"""

import os
import json
from embedding_engine import EmbeddingEngine
from pinecone_manager import PineconeVectorStore
from chroma_manager import ChromaVectorStore

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


def run_demo():
    print("=" * 80)
    print("🚀 DAY 06: VECTOR DATABASE MASTERY — PINECONE vs CHROMADB DEMO")
    print("=" * 80)

    # Common Embedding Engine
    embedder = EmbeddingEngine(provider="auto")

    # Pre-compute document embeddings once
    print(f"\n[Step 1] Ingesting & Embedding {len(DOCUMENTS)} documents...")
    doc_vectors = [embedder.embed_text(d["text"]) for d in DOCUMENTS]

    # =========================================================================
    # PART 1: PINECONE VECTOR DATABASE (Cloud SaaS / Serverless)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🌲 PART 1: PINECONE VECTOR DATABASE (Enterprise Cloud Serverless)")
    print("=" * 80)

    pinecone_store = PineconeVectorStore()
    pinecone_store.get_or_create_index(
        index_name="course-knowledge-base",
        dimension=embedder.dimension,
        metric="cosine",
    )

    pinecone_records = [
        {
            "id": doc["id"],
            "values": vec,
            "metadata": {
                "title": doc["title"],
                "category": doc["category"],
                "text": doc["text"],
            },
        }
        for doc, vec in zip(DOCUMENTS, doc_vectors)
    ]

    upsert_res = pinecone_store.upsert_documents(pinecone_records)
    print(f"✅ Upserted {len(DOCUMENTS)} records into Pinecone! Response: {upsert_res}")

    # Query 1 in Pinecone: Semantic Search
    query_text = "How do AI systems search by meaning instead of exact words?"
    print(f"\n🔍 [Pinecone Query 1 - Semantic Search]: \"{query_text}\"")
    q_vec = embedder.embed_text(query_text)
    p_results = pinecone_store.similarity_search(q_vec, top_k=2)

    for rank, match in enumerate(p_results, start=1):
        print(f"  Rank #{rank} | Similarity Score: {match['score']:.4f} | Title: {match['metadata']['title']}")

    # =========================================================================
    # PART 2: CHROMADB (Open-Source Local / Embedded)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🧪 PART 2: CHROMADB (Open-Source Local / Persistent Embedded DB)")
    print("=" * 80)

    chroma_store = ChromaVectorStore(persist_directory="./chroma_db", in_memory=True)
    collection_name = "course_knowledge_base"
    chroma_store.get_or_create_collection(name=collection_name, distance_metric="cosine")

    chroma_store.add_documents(
        collection_name=collection_name,
        ids=[d["id"] for d in DOCUMENTS],
        embeddings=doc_vectors,
        metadatas=[{"title": d["title"], "category": d["category"]} for d in DOCUMENTS],
        documents=[d["text"] for d in DOCUMENTS],
    )
    print(f"✅ Ingested {len(DOCUMENTS)} records into ChromaDB collection '{collection_name}'!")

    # Query 1 in ChromaDB: Semantic Search
    print(f"\n🔍 [ChromaDB Query 1 - Semantic Search]: \"{query_text}\"")
    c_res = chroma_store.query(
        collection_name=collection_name,
        query_embeddings=[q_vec],
        n_results=2,
    )

    for rank in range(len(c_res["ids"][0])):
        doc_id = c_res["ids"][0][rank]
        dist = c_res["distances"][0][rank]
        meta = c_res["metadatas"][0][rank]
        # In Chroma with cosine distance: distance = 1 - cosine_similarity
        similarity = 1.0 - dist
        print(f"  Rank #{rank+1} | Distance: {dist:.4f} (Sim: {similarity:.4f}) | Title: {meta.get('title')}")

    # Metadata Filtered Search in ChromaDB
    filtered_query = "What are the best serverless options for running scalable AI in the cloud?"
    print(f"\n🎯 [ChromaDB Query 2 - Filtered Search (category='Cloud')]: \"{filtered_query}\"")
    f_vec = embedder.embed_text(filtered_query)
    c_filtered = chroma_store.query(
        collection_name=collection_name,
        query_embeddings=[f_vec],
        n_results=2,
        where={"category": "Cloud"},
    )

    for rank in range(len(c_filtered["ids"][0])):
        meta = c_filtered["metadatas"][0][rank]
        doc_text = c_filtered["documents"][0][rank]
        print(f"  Rank #{rank+1} | Title: {meta.get('title')} | Category: {meta.get('category')}")
        print(f"  Snippet: {doc_text[:110]}...")

    # =========================================================================
    # PART 3: RAG PROMPT ASSEMBLY
    # =========================================================================
    print("\n" + "=" * 80)
    print("🧠 PART 3: RAG PROMPT ASSEMBLY (RETRIEVAL -> CONTEXT INJECTION)")
    print("=" * 80)

    user_question = "Why do relational databases struggle with high-dimensional embeddings?"
    rag_q_vec = embedder.embed_text(user_question)
    rag_matches = pinecone_store.similarity_search(rag_q_vec, top_k=2)

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
    print("\n" + "=" * 80)
    print("🎉 Pinecone & ChromaDB comparison pipeline finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
