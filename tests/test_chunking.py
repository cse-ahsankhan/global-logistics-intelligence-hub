"""Tests for the semantic chunking module."""

import pytest
from langchain_core.documents import Document

from src.processing.chunking import SemanticChunker, ChunkResult


@pytest.fixture
def chunker():
    return SemanticChunker(
        parent_chunk_size=500,
        child_chunk_size=200,
        child_chunk_overlap=20,
    )


@pytest.fixture
def sample_document():
    return Document(
        page_content=(
            "Global Logistics Report Q3 2024\n\n"
            "Ocean freight rates from Shanghai to Rotterdam have increased by 15% "
            "compared to Q2, driven by Red Sea diversions and increased demand. "
            "Average transit time is now 35 days via the Cape of Good Hope route. "
            "Container availability remains tight with equipment imbalances "
            "persisting across major trade lanes.\n\n"
            "Air cargo volumes grew 8% year-over-year with strong demand from "
            "e-commerce and pharmaceutical sectors. Rate per kg from Hong Kong "
            "to Los Angeles averaged $4.85, up from $4.20 in Q2.\n\n"
            "| Route | Mode | Transit Days | Rate Change |\n"
            "| --- | --- | --- | --- |\n"
            "| Shanghai-Rotterdam | Ocean | 35 | +15% |\n"
            "| Shenzhen-LA | Ocean | 18 | +12% |\n"
            "| HKG-LAX | Air | 2 | +15.5% |\n"
        ),
        metadata={"source": "logistics_report_q3.pdf", "page": 1},
    )


class TestSemanticChunker:
    def test_chunk_documents_returns_chunk_result(self, chunker, sample_document):
        result = chunker.chunk_documents([sample_document])
        assert isinstance(result, ChunkResult)
        assert len(result.parent_chunks) > 0
        assert len(result.child_chunks) > 0

    def test_child_chunks_have_parent_ids(self, chunker, sample_document):
        result = chunker.chunk_documents([sample_document])
        for child in result.child_chunks:
            assert "parent_id" in child.metadata
            assert child.metadata["chunk_type"] == "child"

    def test_parent_chunks_have_correct_type(self, chunker, sample_document):
        result = chunker.chunk_documents([sample_document])
        for parent in result.parent_chunks:
            assert parent.metadata["chunk_type"] == "parent"
            assert "chunk_id" in parent.metadata

    def test_table_separation(self, chunker, sample_document):
        result = chunker.chunk_documents([sample_document])
        table_chunks = [
            c for c in result.parent_chunks
            if c.metadata.get("content_format") == "table"
        ]
        assert len(table_chunks) > 0
        # Table should contain pipe characters (Markdown table)
        assert "|" in table_chunks[0].page_content

    def test_metadata_preserved(self, chunker, sample_document):
        result = chunker.chunk_documents([sample_document])
        for chunk in result.child_chunks:
            assert chunk.metadata["source"] == "logistics_report_q3.pdf"

    def test_empty_document(self, chunker):
        doc = Document(page_content="", metadata={"source": "empty.pdf"})
        result = chunker.chunk_documents([doc])
        assert isinstance(result, ChunkResult)

    def test_multiple_documents(self, chunker):
        docs = [
            Document(page_content=f"Document {i} content about logistics.", metadata={"source": f"doc{i}.pdf"})
            for i in range(3)
        ]
        result = chunker.chunk_documents(docs)
        sources = {c.metadata["source"] for c in result.child_chunks}
        assert len(sources) == 3

    def test_parent_child_linkage(self, chunker, sample_document):
        """Verify every child's parent_id maps to an existing parent."""
        result = chunker.chunk_documents([sample_document])
        parent_ids = {p.metadata["chunk_id"] for p in result.parent_chunks}
        for child in result.child_chunks:
            assert child.metadata["parent_id"] in parent_ids
