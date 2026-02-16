"""Hybrid search engine combining BM25 keyword search with dense vector retrieval.

This module implements Reciprocal Rank Fusion (RRF) to merge results
from a BM25 index and a FAISS vector index, providing both lexical
precision and semantic recall.

Default weighting: 30% BM25, 70% semantic (configurable).
"""

import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config.settings import get_settings
from src.processing.embeddings import EmbeddingService


class HybridSearchEngine:
    """Hybrid search combining BM25 (sparse) and FAISS (dense) retrieval.

    Parameters
    ----------
    embedding_service : EmbeddingService | None
        Pre-configured embedding service. Created automatically if not
        provided.
    bm25_weight : float
        Weight for BM25 scores in the fused ranking.
    semantic_weight : float
        Weight for semantic similarity scores in the fused ranking.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        bm25_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
    ):
        settings = get_settings()
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight
        self.semantic_weight = semantic_weight if semantic_weight is not None else settings.semantic_weight

        self.embedding_service = embedding_service or EmbeddingService()
        self.documents: list[Document] = []
        self._bm25_index: Optional[BM25Okapi] = None
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._tokenized_corpus: list[list[str]] = []

    def add_documents(self, documents: list[Document]) -> None:
        """Index a batch of documents into both BM25 and FAISS stores."""
        self.documents.extend(documents)

        # Build BM25 index
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in self.documents
        ]
        self._bm25_index = BM25Okapi(self._tokenized_corpus)

        # Build FAISS index
        texts = [doc.page_content for doc in self.documents]
        embeddings = self.embedding_service.embed_texts(texts)
        embedding_matrix = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embedding_matrix)

        dimension = embedding_matrix.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dimension)
        self._faiss_index.add(embedding_matrix)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """Run hybrid search and return ranked (document, score) pairs.

        Uses Reciprocal Rank Fusion to combine BM25 and semantic scores.
        """
        settings = get_settings()
        k = top_k or settings.top_k_results

        if not self.documents:
            return []

        bm25_results = self._bm25_search(query, k)
        semantic_results = self._semantic_search(query, k)

        fused = self._reciprocal_rank_fusion(
            bm25_results, semantic_results, k
        )
        return fused

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs from BM25."""
        if self._bm25_index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def _semantic_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs from FAISS vector search."""
        if self._faiss_index is None:
            return []

        query_embedding = np.array(
            [self.embedding_service.embed_query(query)], dtype=np.float32
        )
        faiss.normalize_L2(query_embedding)

        scores, indices = self._faiss_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[int, float]],
        semantic_results: list[tuple[int, float]],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[tuple[Document, float]]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = w_bm25 * (1 / (rrf_k + rank_bm25)) + w_sem * (1 / (rrf_k + rank_sem))
        """
        fused_scores: dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(bm25_results, start=1):
            rrf_score = self.bm25_weight * (1.0 / (rrf_k + rank))
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + rrf_score

        for rank, (doc_idx, _) in enumerate(semantic_results, start=1):
            rrf_score = self.semantic_weight * (1.0 / (rrf_k + rank))
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + rrf_score

        # Sort by fused score descending
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            (self.documents[doc_idx], score)
            for doc_idx, score in sorted_results[:top_k]
        ]

    def save(self, directory: str | Path) -> None:
        """Persist the search index to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(path / "faiss.index"))

        # Save BM25 + documents
        with open(path / "bm25_data.pkl", "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )

    def load(self, directory: str | Path) -> None:
        """Load a previously saved search index."""
        path = Path(directory)

        # Load FAISS index
        faiss_path = path / "faiss.index"
        if faiss_path.exists():
            self._faiss_index = faiss.read_index(str(faiss_path))

        # Load BM25 + documents
        bm25_path = path / "bm25_data.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self._tokenized_corpus = data["tokenized_corpus"]
                self._bm25_index = BM25Okapi(self._tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing."""
        return text.lower().split()
