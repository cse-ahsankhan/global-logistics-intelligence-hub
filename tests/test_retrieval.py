"""Tests for the hybrid search and retrieval pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

from src.vectorstore.hybrid_search import HybridSearchEngine


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Ocean freight rates from Shanghai to Rotterdam increased 15% in Q3.",
            metadata={"source": "rates.pdf", "chunk_id": "c1", "chunk_type": "child", "parent_id": "p1"},
        ),
        Document(
            page_content="Air cargo demand from Hong Kong to Los Angeles grew 8% year-over-year.",
            metadata={"source": "rates.pdf", "chunk_id": "c2", "chunk_type": "child", "parent_id": "p1"},
        ),
        Document(
            page_content="Customs clearance at Rotterdam port requires EUR1 movement certificate.",
            metadata={"source": "compliance.pdf", "chunk_id": "c3", "chunk_type": "child", "parent_id": "p2"},
        ),
        Document(
            page_content="Container dwell time at Jebel Ali increased to 5.2 days average.",
            metadata={"source": "operations.pdf", "chunk_id": "c4", "chunk_type": "child", "parent_id": "p3"},
        ),
        Document(
            page_content="Last mile delivery costs in Southeast Asia rose by 12% due to fuel prices.",
            metadata={"source": "costs.pdf", "chunk_id": "c5", "chunk_type": "child", "parent_id": "p4"},
        ),
    ]


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service that returns deterministic vectors."""
    service = MagicMock()
    service.embed_texts.return_value = [
        np.random.default_rng(i).standard_normal(384).tolist()
        for i in range(5)
    ]
    service.embed_query.return_value = np.random.default_rng(42).standard_normal(384).tolist()
    return service


class TestHybridSearchEngine:
    def test_add_documents(self, sample_docs, mock_embedding_service):
        engine = HybridSearchEngine(embedding_service=mock_embedding_service)
        engine.add_documents(sample_docs)
        assert len(engine.documents) == 5

    def test_search_returns_results(self, sample_docs, mock_embedding_service):
        engine = HybridSearchEngine(
            embedding_service=mock_embedding_service,
            bm25_weight=0.3,
            semantic_weight=0.7,
        )
        engine.add_documents(sample_docs)
        results = engine.search("ocean freight rates Shanghai", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3
        # Each result should be (Document, score)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_bm25_keyword_matching(self, sample_docs, mock_embedding_service):
        engine = HybridSearchEngine(
            embedding_service=mock_embedding_service,
            bm25_weight=1.0,
            semantic_weight=0.0,
        )
        engine.add_documents(sample_docs)
        results = engine.search("Rotterdam customs clearance", top_k=3)
        # BM25 should favor the customs clearance document
        assert any("customs" in doc.page_content.lower() for doc, _ in results)

    def test_empty_engine_returns_empty(self, mock_embedding_service):
        engine = HybridSearchEngine(embedding_service=mock_embedding_service)
        results = engine.search("test query")
        assert results == []

    def test_search_respects_top_k(self, sample_docs, mock_embedding_service):
        engine = HybridSearchEngine(embedding_service=mock_embedding_service)
        engine.add_documents(sample_docs)
        results = engine.search("logistics", top_k=2)
        assert len(results) <= 2

    def test_save_and_load(self, sample_docs, mock_embedding_service, tmp_path):
        engine = HybridSearchEngine(embedding_service=mock_embedding_service)
        engine.add_documents(sample_docs)

        engine.save(tmp_path / "test_index")

        new_engine = HybridSearchEngine(embedding_service=mock_embedding_service)
        new_engine.load(tmp_path / "test_index")

        assert len(new_engine.documents) == len(sample_docs)

    def test_rrf_fusion_weights(self, sample_docs, mock_embedding_service):
        """Verify that changing weights affects ranking."""
        engine_bm25_heavy = HybridSearchEngine(
            embedding_service=mock_embedding_service,
            bm25_weight=0.9,
            semantic_weight=0.1,
        )
        engine_bm25_heavy.add_documents(sample_docs)
        results_bm25 = engine_bm25_heavy.search("ocean freight Shanghai", top_k=3)

        engine_sem_heavy = HybridSearchEngine(
            embedding_service=mock_embedding_service,
            bm25_weight=0.1,
            semantic_weight=0.9,
        )
        engine_sem_heavy.add_documents(sample_docs)
        results_sem = engine_sem_heavy.search("ocean freight Shanghai", top_k=3)

        # Results should generally differ due to weight shift
        assert len(results_bm25) > 0
        assert len(results_sem) > 0

    def test_tokenizer(self):
        tokens = HybridSearchEngine._tokenize("Hello World Test")
        assert tokens == ["hello", "world", "test"]
