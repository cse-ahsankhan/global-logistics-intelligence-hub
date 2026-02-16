"""Retriever with parent-child context expansion.

Retrieves child chunks via hybrid search, then expands each result
to include its parent chunk for richer context. Implements
LangChain's BaseRetriever interface for seamless chain integration.
"""

from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

from src.vectorstore.hybrid_search import HybridSearchEngine


class HybridRetriever(BaseRetriever):
    """LangChain-compatible retriever backed by hybrid search.

    Performs retrieval on child chunks, then expands context by
    fetching the corresponding parent chunk for each result.

    Parameters
    ----------
    search_engine : HybridSearchEngine
        The hybrid search engine containing indexed documents.
    top_k : int
        Number of results to retrieve.
    expand_to_parent : bool
        Whether to replace child chunks with their parent content.
    """

    top_k: int = 5
    expand_to_parent: bool = True
    _search_engine: HybridSearchEngine = PrivateAttr()
    _parent_lookup: dict[str, Document] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        top_k: int = 5,
        expand_to_parent: bool = True,
        **kwargs,
    ):
        super().__init__(top_k=top_k, expand_to_parent=expand_to_parent, **kwargs)
        self._search_engine = search_engine
        self._parent_lookup = self._build_parent_lookup()

    def _build_parent_lookup(self) -> dict[str, Document]:
        """Build a mapping from chunk_id → Document for parent chunks."""
        lookup = {}
        for doc in self._search_engine.documents:
            if doc.metadata.get("chunk_type") == "parent":
                chunk_id = doc.metadata.get("chunk_id", "")
                if chunk_id:
                    lookup[chunk_id] = doc
        return lookup

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        """Retrieve documents relevant to the query."""
        results = self._search_engine.search(query, top_k=self.top_k)

        if not self.expand_to_parent:
            return [doc for doc, _ in results]

        # Expand child chunks to include parent context
        expanded = []
        seen_parents: set[str] = set()

        for doc, score in results:
            parent_id = doc.metadata.get("parent_id", "")

            if parent_id and parent_id in self._parent_lookup:
                if parent_id not in seen_parents:
                    parent_doc = self._parent_lookup[parent_id]
                    expanded_doc = Document(
                        page_content=parent_doc.page_content,
                        metadata={
                            **parent_doc.metadata,
                            "retrieval_score": score,
                            "expanded_from_child": doc.metadata.get("chunk_id", ""),
                        },
                    )
                    expanded.append(expanded_doc)
                    seen_parents.add(parent_id)
            else:
                expanded.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "retrieval_score": score},
                    )
                )

        return expanded
