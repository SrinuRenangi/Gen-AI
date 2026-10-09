"""
Clinical Medical Chatbot - Hybrid Knowledge Retrieval Engine
=============================================================
Module 08: Capstone Projects - Project 02: Clinical Medical Chatbot
File: hybrid_retriever.py

Implements:
1. Curated Clinical Knowledge Base (PubMed/WHO Guidelines).
2. BM25 Lexical Keyword Ranker (Medical Subject Headings / MeSH).
3. Dense Semantic Cosine Similarity Ranker.
4. Reciprocal Rank Fusion (RRF, k=60) Hybrid Merging.
"""

from __future__ import annotations
import math
import re
from typing import List, Dict, Any, Tuple
from models import ClinicalDocument


# Curated Evidence-Based Clinical Guideline Corpus
CLINICAL_CORPUS: List[ClinicalDocument] = [
    ClinicalDocument(
        id="GUIDE-001",
        title="American Diabetes Association (ADA) 2024 Standards of Care: Type 2 Diabetes",
        mesh_terms=["Diabetes Mellitus, Type 2", "Metformin", "HbA1c", "Hyperglycemia", "Microvascular Complications"],
        content=(
            "First-line pharmacotherapy for Type 2 Diabetes is metformin alongside comprehensive lifestyle modification. "
            "Target HbA1c is generally < 7.0% for non-pregnant adults without significant hypoglycemia. "
            "In patients with established atherosclerotic cardiovascular disease (ASCVD), heart failure, or CKD, "
            "SGLT2 inhibitors or GLP-1 receptor agonists with proven cardiovascular benefit are indicated irrespective of baseline HbA1c."
        ),
        citation="Diabetes Care 2024;47(Suppl. 1):S1-S343"
    ),
    ClinicalDocument(
        id="GUIDE-002",
        title="IDSA/ATS Consensus Guidelines on the Management of Community-Acquired Pneumonia in Adults",
        mesh_terms=["Pneumonia", "Community-Acquired", "Streptococcus pneumoniae", "Amoxicillin", "Azithromycin", "CURB-65"],
        content=(
            "Diagnosis of Community-Acquired Pneumonia (CAP) requires clinical signs (fever, cough, dyspnea, pleuritic chest pain) "
            "plus a demonstrable infiltrate on chest radiography. Outpatient empiric therapy in healthy adults without comorbidities: "
            "amoxicillin 1g TID or doxycycline 100mg BID. In outpatients with comorbidities (COPD, diabetes, heart disease): "
            "combination therapy with amoxicillin/clavulanate plus a macrolide, or respiratory fluoroquinolone monotherapy."
        ),
        citation="Am J Respir Crit Care Med. 2019;200(7):e45-e67"
    ),
    ClinicalDocument(
        id="GUIDE-003",
        title="AHA/ACC/HFSA Heart Failure Management Guidelines: Heart Failure with Reduced Ejection Fraction (HFrEF)",
        mesh_terms=["Heart Failure", "HFrEF", "Ejection Fraction", "Beta Blockers", "ARNI", "SGLT2i", "Spironolactone"],
        content=(
            "Guideline-Directed Medical Therapy (GDMT) for HFrEF (LVEF <= 40%) consists of 4 foundational drug pillars: "
            "1. ARNI (Sacubitril/Valsartan) or ACEi/ARB; 2. Evidence-based Beta-blocker (Carvedilol, Metoprolol succinate, Bisoprolol); "
            "3. Mineralocorticoid Receptor Antagonist (MRA: Spironolactone or Eplerenone); 4. SGLT2 inhibitor (Dapagliflozin or Empagliflozin). "
            "Loop diuretics (Furosemide) are titrated for fluid overload and congestive symptom management."
        ),
        citation="Circulation. 2022;145(18):e895-e1032"
    ),
    ClinicalDocument(
        id="GUIDE-004",
        title="ACR Guideline for the Treatment of Rheumatoid Arthritis",
        mesh_terms=["Arthritis, Rheumatoid", "Methotrexate", "Synovitis", "Joint Pain", "Morning Stiffness", "DMARDs"],
        content=(
            "Rheumatoid Arthritis (RA) is characterized by chronic, symmetrical polyarthritis predominantly targeting small joints "
            "of hands and feet, accompanied by prominent morning stiffness lasting > 30 minutes. "
            "Methotrexate monotherapy is strongly recommended as first-line conventional synthetic DMARD (csDMARD) "
            "over other csDMARDs in DMARD-naive patients with moderate-to-high disease activity. Target is treat-to-target remission."
        ),
        citation="Arthritis Care Res. 2021;73(7):924-939"
    ),
    ClinicalDocument(
        id="GUIDE-005",
        title="GINA Global Strategy for Asthma Management and Prevention",
        mesh_terms=["Asthma", "Inhaled Corticosteroid", "Formoterol", "Bronchospasm", "Wheezing", "SABA"],
        content=(
            "GINA no longer recommends SABA (albuterol) alone without inhaled corticosteroid (ICS) due to risk of severe exacerbations. "
            "Track 1 preferred reliever is low-dose ICS-formoterol across all severity steps. "
            "Symptoms include variable expiratory airflow limitation, episodic wheezing, shortness of breath, and nocturnal chest tightness."
        ),
        citation="Global Initiative for Asthma. Global Strategy 2023 Update."
    )
]


class ClinicalHybridRetriever:
    """
    Hybrid Retrieval combining BM25 Lexical Keyword matching with
    dense semantic embeddings via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, documents: List[ClinicalDocument] = CLINICAL_CORPUS, rrf_k: int = 60):
        self.documents = documents
        self.rrf_k = rrf_k
        self._build_lexical_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple biomedical tokenization."""
        return re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", text.lower())

    def _build_lexical_index(self):
        """Constructs inverted index and BM25 term frequency mappings."""
        self.doc_tokens = [self._tokenize(d.title + " " + " ".join(d.mesh_terms) + " " + d.content) for d in self.documents]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)

        # Document frequencies
        self.df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Computes Okapi BM25 score for a document."""
        k1 = 1.5
        b = 0.75
        N = len(self.documents)
        doc_len = self.doc_lens[doc_idx]
        tokens = self.doc_tokens[doc_idx]

        term_freqs: Dict[str, int] = {}
        for t in tokens:
            term_freqs[t] = term_freqs.get(t, 0) + 1

        score = 0.0
        for q in query_tokens:
            if q not in self.df:
                continue
            df_val = self.df[q]
            idf = math.log((N - df_val + 0.5) / (df_val + 0.5) + 1.0)
            tf = term_freqs.get(q, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / max(denominator, 1e-6))

        return score

    def _mock_dense_score(self, query: str, doc: ClinicalDocument) -> float:
        """
        Calculates semantic overlap based on character n-grams and clinical terms,
        providing a fast, zero-dependency proxy for a dense embedding model.
        """
        q_lower = query.lower()
        title_lower = doc.title.lower()
        content_lower = doc.content.lower()

        overlap = 0.0
        # Check MeSH term matches
        for mesh in doc.mesh_terms:
            if mesh.lower() in q_lower:
                overlap += 3.0

        # Check title word matches
        for word in doc.title.split():
            if len(word) > 4 and word.lower() in q_lower:
                overlap += 1.5

        # Content relevance
        for q_word in q_lower.split():
            if len(q_word) > 4 and q_word in content_lower:
                overlap += 0.5

        return overlap

    def retrieve_hybrid(self, query: str, top_k: int = 3) -> List[Tuple[ClinicalDocument, float]]:
        """
        Executes both BM25 lexical ranking and dense semantic ranking,
        then fuses them using Reciprocal Rank Fusion (RRF):
        RRF(d) = sum(1 / (k + rank_m(d)))
        """
        query_tokens = self._tokenize(query)

        # 1. Lexical BM25 Ranking
        bm25_scored = [(self.documents[i], self._bm25_score(query_tokens, i)) for i in range(len(self.documents))]
        bm25_sorted = sorted(bm25_scored, key=lambda x: x[1], reverse=True)
        bm25_ranks = {doc.id: rank + 1 for rank, (doc, _) in enumerate(bm25_sorted)}

        # 2. Dense Semantic Ranking
        dense_scored = [(doc, self._mock_dense_score(query, doc)) for doc in self.documents]
        dense_sorted = sorted(dense_scored, key=lambda x: x[1], reverse=True)
        dense_ranks = {doc.id: rank + 1 for rank, (doc, _) in enumerate(dense_sorted)}

        # 3. Reciprocal Rank Fusion
        fused_scores: Dict[str, float] = {}
        for doc in self.documents:
            r_bm25 = bm25_ranks.get(doc.id, 999)
            r_dense = dense_ranks.get(doc.id, 999)
            rrf_score = (1.0 / (self.rrf_k + r_bm25)) + (1.0 / (self.rrf_k + r_dense))
            fused_scores[doc.id] = rrf_score

        # 4. Sort and return top_k
        fused_ranked = sorted(
            [(doc, fused_scores[doc.id]) for doc in self.documents],
            key=lambda x: x[1],
            reverse=True
        )

        return fused_ranked[:top_k]
