"""Azure AI Search integration for production vector store.

Provides index management and search operations against Azure AI Search
(formerly Azure Cognitive Search). Used when Azure credentials are
configured; otherwise the system falls back to local FAISS.
"""

from typing import Any, Optional

from langchain_core.documents import Document

from config.settings import get_settings


class AzureSearchStore:
    """Wrapper around Azure AI Search for document indexing and retrieval.

    This class manages index creation, document upsert, and hybrid
    search (vector + keyword) against an Azure AI Search service.
    """

    def __init__(self, index_name: Optional[str] = None):
        self.settings = get_settings()

        if not self.settings.use_azure_search:
            raise RuntimeError(
                "Azure AI Search credentials not configured. "
                "Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY."
            )

        self.index_name = index_name or self.settings.azure_search_index_name
        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Initialize Azure Search client."""
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential

        credential = AzureKeyCredential(self.settings.azure_search_api_key)
        return SearchClient(
            endpoint=self.settings.azure_search_endpoint,
            index_name=self.index_name,
            credential=credential,
        )

    def upsert_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> int:
        """Upload or update documents with their embeddings.

        Returns the number of documents successfully indexed.
        """
        batch = []
        for doc, embedding in zip(documents, embeddings):
            record = {
                "id": doc.metadata.get("chunk_id", ""),
                "content": doc.page_content,
                "content_vector": embedding,
                "source": doc.metadata.get("source", ""),
                "chunk_type": doc.metadata.get("chunk_type", ""),
                "parent_id": doc.metadata.get("parent_id", ""),
                "metadata": str(doc.metadata),
            }
            batch.append(record)

        result = self._client.upload_documents(documents=batch)
        return sum(1 for r in result if r.succeeded)

    def search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[Document]:
        """Perform hybrid search combining keyword and vector similarity."""
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector",
        )

        results = self._client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k,
            select=["id", "content", "source", "chunk_type", "parent_id", "metadata"],
        )

        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    "chunk_id": result["id"],
                    "source": result["source"],
                    "chunk_type": result["chunk_type"],
                    "parent_id": result.get("parent_id", ""),
                    "search_score": result.get("@search.score", 0.0),
                },
            )
            documents.append(doc)

        return documents
