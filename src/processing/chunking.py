"""Semantic chunking with parent-child indexing for supply chain documents.

Implements a two-tier chunking strategy:
  1. Parent chunks — larger contextual windows for grounding answers
  2. Child chunks  — smaller, semantically coherent units for retrieval

Child chunks carry a reference back to their parent so the retriever
can fetch surrounding context when needed.
"""

import re
import uuid
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class ChunkResult:
    """Container for parent and child chunks produced by the chunker."""

    parent_chunks: list[Document] = field(default_factory=list)
    child_chunks: list[Document] = field(default_factory=list)


class SemanticChunker:
    """Two-tier semantic chunker with table-aware splitting.

    Parameters
    ----------
    parent_chunk_size : int
        Token-approximate size for parent (context) chunks.
    child_chunk_size : int
        Token-approximate size for child (retrieval) chunks.
    child_chunk_overlap : int
        Overlap between consecutive child chunks.
    """

    TABLE_PATTERN = re.compile(
        r"(\|[^\n]+\|\n\|[-| :]+\|\n(?:\|[^\n]+\|\n?)*)", re.MULTILINE
    )

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size: int = 512,
        child_chunk_overlap: int = 50,
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_documents(self, documents: list[Document]) -> ChunkResult:
        """Split documents into parent and child chunks.

        Each child chunk's metadata includes a ``parent_id`` field
        that links it back to the parent chunk it was derived from.
        """
        result = ChunkResult()

        for doc in documents:
            text_segments, table_segments = self._separate_tables(doc.page_content)

            # Process text through parent → child hierarchy
            parent_docs = self._parent_splitter.create_documents(
                texts=[text_segments],
                metadatas=[doc.metadata],
            )

            for parent in parent_docs:
                parent_id = str(uuid.uuid4())
                parent.metadata = {
                    **parent.metadata,
                    "chunk_id": parent_id,
                    "chunk_type": "parent",
                }
                result.parent_chunks.append(parent)

                children = self._child_splitter.create_documents(
                    texts=[parent.page_content],
                    metadatas=[parent.metadata],
                )
                for child in children:
                    child.metadata = {
                        **child.metadata,
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_type": "child",
                        "parent_id": parent_id,
                    }
                    result.child_chunks.append(child)

            # Tables become their own parent chunks (kept intact)
            for table_text in table_segments:
                table_id = str(uuid.uuid4())
                table_doc = Document(
                    page_content=table_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": table_id,
                        "chunk_type": "parent",
                        "content_format": "table",
                    },
                )
                result.parent_chunks.append(table_doc)

                # Table also acts as its own child for direct retrieval
                child_table = Document(
                    page_content=table_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_type": "child",
                        "parent_id": table_id,
                        "content_format": "table",
                    },
                )
                result.child_chunks.append(child_table)

        return result

    def _separate_tables(self, text: str) -> tuple[str, list[str]]:
        """Separate Markdown tables from prose text.

        Returns
        -------
        tuple
            (prose_text, list_of_table_strings)
        """
        tables = self.TABLE_PATTERN.findall(text)
        prose = self.TABLE_PATTERN.sub("", text).strip()
        return prose, [t.strip() for t in tables if t.strip()]
