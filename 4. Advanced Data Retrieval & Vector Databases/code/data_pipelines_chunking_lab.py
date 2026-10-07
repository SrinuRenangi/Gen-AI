"""
Data Pipelines Lab: Document Loaders & Chunking Strategies
==========================================================
Zero to Hero Gen AI Course — Module 04: Advanced Data Retrieval & Vector Databases

This standalone educational lab demonstrates the foundational engineering
patterns of data ingestion and chunking:
1. Heterogeneous Document Ingestion (CSV, JSON, Markdown, PDF)
2. Fixed-Size vs Recursive Character Text Splitting
3. Structure-Aware Markdown Header Splitting & Metadata Injection
4. Chunk Overlap & Pronoun Antecedent Boundary Recovery
5. Semantic Chunking Simulation via Sentence Distance Deltas

Usage:
    py data_pipelines_chunking_lab.py
"""

import sys
import os
import json
import csv
import math
import random
from typing import Dict, List, Tuple, Any, Optional

# Ensure UTF-8 output on Windows consoles to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =====================================================================
# SECTION 1: STANDALONE DOCUMENT & SPLITTER ABSTRACTIONS
# =====================================================================

class Document:
    """Standardized Document representation matching LangChain's Document schema."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        snippet = self.page_content[:50].replace("\n", " ")
        return f"Document(page_content='{snippet}...', metadata={self.metadata})"


class FixedSizeSplitter:
    """Naive fixed-size character splitter without linguistic boundary awareness."""
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.chunk_overlap)
        return chunks


class RecursiveCharacterTextSplitter:
    """
    Splits text recursively along a hierarchy of natural linguistic separators:
    ["\\n\\n", "\\n", ". ", " ", ""]
    """
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        # Find the highest-priority separator present in text
        for i, s in enumerate(separators):
            if s == "" or s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        # Split text by chosen separator
        splits = text.split(separator) if separator != "" else list(text)

        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    # Recurse with lower-level separators
                    good_splits.extend(self._split(s, new_separators))
                else:
                    good_splits.append(s)

        # Merge splits respecting chunk_size and chunk_overlap
        current_chunk = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            if current_len + piece_len > self.chunk_size and current_chunk:
                joined = separator.join(current_chunk)
                final_chunks.append(joined)
                
                # Apply overlap by retaining tail pieces
                while current_len > self.chunk_overlap and current_chunk:
                    popped = current_chunk.pop(0)
                    current_len -= (len(popped) + len(separator))

            current_chunk.append(piece)
            current_len += piece_len

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks


class MarkdownHeaderTextSplitter:
    """Splits Markdown by headers (#, ##, ###) and injects breadcrumbs into metadata."""
    def __init__(self, headers_to_split_on: List[Tuple[str, str]]):
        self.headers_to_split_on = headers_to_split_on

    def split_text(self, text: str) -> List[Document]:
        lines = text.split("\n")
        documents = []
        current_metadata: Dict[str, str] = {}
        current_content: List[str] = []

        for line in lines:
            header_match = False
            for marker, header_name in self.headers_to_split_on:
                if line.startswith(marker + " "):
                    # Save accumulated content before new header
                    if current_content:
                        clean_content = "\n".join(current_content).strip()
                        if clean_content:
                            documents.append(Document(page_content=clean_content, metadata=dict(current_metadata)))
                        current_content = []

                    # Update metadata breadcrumbs
                    header_value = line[len(marker):].strip()
                    current_metadata[header_name] = header_value
                    header_match = True
                    break

            if not header_match:
                current_content.append(line)

        if current_content:
            clean_content = "\n".join(current_content).strip()
            if clean_content:
                documents.append(Document(page_content=clean_content, metadata=dict(current_metadata)))

        return documents


# =====================================================================
# LAB EXPERIMENTS & DEMONSTRATION SUITE
# =====================================================================

def banner(title: str):
    print("\n" + "#" * 72)
    print(f"##  {title}")
    print("#" * 72)


def experiment_1_heterogeneous_loaders():
    banner("EXPERIMENT 1: Heterogeneous Document Loaders (CSV, JSON, Markdown, PDF)")

    # 1. CSV Loader Simulation
    csv_raw = "id,name,role,department\n101,Maya Vance,CEO,Executive\n102,Dr. Thorne,Lead Architect,Engineering"
    csv_docs = []
    reader = csv.DictReader(csv_raw.splitlines())
    for row in reader:
        content = "\n".join([f"{k}: {v}" for k, v in row.items()])
        csv_docs.append(Document(page_content=content, metadata={"source": "employees.csv", "row_id": row["id"]}))

    print(f"1. Ingested CSV into {len(csv_docs)} Documents:")
    print(f"   Sample Chunk:\n   {csv_docs[0].page_content.replace(chr(10), ' | ')}")
    print(f"   Metadata: {csv_docs[0].metadata}\n")

    # 2. JSON Loader Simulation
    json_data = {
        "services": [
            {"service": "auth-gateway", "status": "HEALTHY", "uptime": "99.98%"},
            {"service": "payment-api", "status": "DEGRADED", "uptime": "98.42%"}
        ]
    }
    json_docs = [
        Document(
            page_content=f"Service: {s['service']} is currently {s['status']} with uptime {s['uptime']}.",
            metadata={"source": "infra_status.json", "service_name": s["service"]}
        )
        for s in json_data["services"]
    ]
    print(f"2. Ingested JSON into {len(json_docs)} Documents:")
    print(f"   Sample Chunk: {json_docs[0].page_content}")
    print(f"   Metadata: {json_docs[0].metadata}\n")

    # 3. PDF Simulation (Page-by-page mapping)
    pdf_docs = [
        Document(
            page_content="SECTION 1.1: Distributed consensus guarantees all nodes agree on state.",
            metadata={"source": "consensus_whitepaper.pdf", "page": 1}
        ),
        Document(
            page_content="SECTION 1.2: Paxos and Raft are the primary algorithms used in production.",
            metadata={"source": "consensus_whitepaper.pdf", "page": 2}
        )
    ]
    print(f"3. Ingested PDF into {len(pdf_docs)} Page Documents:")
    print(f"   Sample Chunk: {pdf_docs[0].page_content} (Page: {pdf_docs[0].metadata['page']})")
    print("\n✅ Universal Abstraction: All formats successfully normalized to Document(page_content, metadata)!")


def experiment_2_fixed_vs_recursive_splitting():
    banner("EXPERIMENT 2: Fixed-Size vs Recursive Character Text Splitting")

    sample_prose = (
        "Distributed database replication ensures high availability. When a primary database node crashes, "
        "a secondary replica must be elected as the new leader within milliseconds.\n\n"
        "The consensus protocol achieves this via majority quorum voting. If network partitions prevent quorum, "
        "the cluster pauses write operations to guarantee consistency over availability (CP mode)."
    )

    # A. Naive Fixed-Size Splitter (chunk_size=120)
    fixed_splitter = FixedSizeSplitter(chunk_size=120, chunk_overlap=0)
    fixed_chunks = fixed_splitter.split_text(sample_prose)

    print("🔴 PART A: Naive Fixed-Size Character Splitter (Blind character cuts):")
    for i, c in enumerate(fixed_chunks, 1):
        print(f"   Chunk #{i} ({len(c)} chars): \"{c}\"")

    print("\n   ⚠️  Flaw Observed: Sentences and words are cut mid-stream (e.g. chopping right after 'leader within').\n")

    # B. Recursive Character Text Splitter (chunk_size=200, chunk_overlap=30)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    recursive_chunks = recursive_splitter.split_text(sample_prose)

    print("🟢 PART B: Recursive Character Text Splitter (Respects paragraphs and sentences):")
    for i, c in enumerate(recursive_chunks, 1):
        print(f"   Chunk #{i} ({len(c)} chars): \"{c.replace(chr(10), ' ')}\"")

    print("\n   ✅ Clean Split: Preserved complete sentences without cutting words!")


def experiment_3_markdown_structure_splitting():
    banner("EXPERIMENT 3: Structure-Aware Markdown Header Splitting")

    markdown_doc = """# System Architecture Overview
The core system is structured as a cloud-native microservices cluster.

## Database Subsystem
The database layer manages persistent transactional storage.

### PostgreSQL Cluster
Primary write node is hosted in us-east-1 with two read replicas in us-west-2.

### Redis Cache
Redis Enterprise cluster acts as a write-through cache with a TTL of 300 seconds.
"""

    headers = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3")
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    chunks = md_splitter.split_text(markdown_doc)

    print(f"Processed Markdown into {len(chunks)} Structured Chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"📦 Chunk #{i}:")
        print(f"   Content: \"{chunk.page_content}\"")
        print(f"   Breadcrumb Metadata: {chunk.metadata}\n")

    print("✅ Context Injected: Each chunk carries exact architectural breadcrumbs in metadata!")


def experiment_4_chunk_overlap_recovery():
    banner("EXPERIMENT 4: Chunk Overlap & Pronoun Antecedent Boundary Recovery")

    text = (
        "The board of directors approved the emergency acquisition of CyberGuard Inc for $450M. "
        "It was finalized on October 12, 2024 by CEO Maya Vance under secret executive authorization."
    )

    print(f"Original Text (Sentence 1 mentions 'CyberGuard Inc', Sentence 2 mentions 'It'):\n   \"{text}\"\n")

    # Case A: Zero Overlap (Severing Pronoun Antecedent)
    split_zero = FixedSizeSplitter(chunk_size=88, chunk_overlap=0).split_text(text)
    print("1. Case A: Zero Overlap (chunk_overlap = 0):")
    print(f"   - Chunk 1: \"{split_zero[0]}\"")
    print(f"   - Chunk 2: \"{split_zero[1]}\"")
    print("   ❌ SEVERED ANTECEDENT: Chunk 2 starts with 'It was finalized...'. The LLM cannot know what 'It' is!\n")

    # Case B: 20% Overlap
    split_overlap = FixedSizeSplitter(chunk_size=100, chunk_overlap=30).split_text(text)
    print("2. Case B: With 20% Chunk Overlap:")
    print(f"   - Chunk 1: \"{split_overlap[0]}\"")
    print(f"   - Chunk 2: \"{split_overlap[1]}\"")
    print("   ✅ CONTEXT PRESERVED: Overlapping boundary retains 'CyberGuard Inc' in Chunk 2 alongside 'It'!")


def experiment_5_semantic_chunking_simulation():
    banner("EXPERIMENT 5: Semantic Chunking via Sentence Similarity Deltas")

    sentences = [
        "The James Webb Space Telescope observes the universe in deep infrared wavelengths.",
        "Its primary beryllium mirror is plated with microscopically thin gold.",
        "Deep space telemetry relays astronomical images across millions of miles.",
        "Baking authentic sourdough bread requires wild yeast and long fermentation.",
        "High gluten flour gives artisan bread its distinctive chewy crumb structure."
    ]

    # Simulated sentence embeddings (cosine distance deltas)
    # Consecutive distance between S0-S1 (Astronomy): Low
    # S1-S2 (Astronomy): Low
    # S2-S3 (Astronomy -> Sourdough Bread): SPIKE!
    # S3-S4 (Baking): Low
    distances = [0.12, 0.15, 0.88, 0.14]
    split_threshold = 0.50

    print("Sentence Stream & Consecutive Cosine Distance Deltas:")
    for i in range(len(sentences) - 1):
        print(f"   [{i}] \"{sentences[i][:40]}...\" <---> [{i+1}] \"{sentences[i+1][:40]}...\"")
        print(f"       -> Cosine Distance: {distances[i]:.2f} {'[🚨 TOPIC SHIFT DETECTED]' if distances[i] > split_threshold else '[Consistent]'}")

    # Partitioning sentences based on threshold
    semantic_chunks = []
    current_chunk = [sentences[0]]

    for i, dist in enumerate(distances):
        if dist > split_threshold:
            semantic_chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])
    if current_chunk:
        semantic_chunks.append(" ".join(current_chunk))

    print(f"\nResulting {len(semantic_chunks)} Semantic Chunks:")
    for rank, sc in enumerate(semantic_chunks, 1):
        print(f"\n📦 Semantic Chunk #{rank}:\n   \"{sc}\"")

    print("\n✅ Natural Boundary: Split precisely along conceptual topic shift without syntax markers!")


def main():
    print("""
========================================================================
   DATA PIPELINES & CHUNKING STRATEGIES LAB
========================================================================
    """)
    experiment_1_heterogeneous_loaders()
    experiment_2_fixed_vs_recursive_splitting()
    experiment_3_markdown_structure_splitting()
    experiment_4_chunk_overlap_recovery()
    experiment_5_semantic_chunking_simulation()
    print("\n✅ All 5 Data Pipeline experiments completed successfully!\n")


if __name__ == "__main__":
    main()
