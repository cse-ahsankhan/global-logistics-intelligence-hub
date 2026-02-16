"""Embedding service supporting Azure OpenAI and local sentence-transformers.

Provides a unified interface for generating embeddings, automatically
selecting the backend based on available configuration.
"""

from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import get_settings


class EmbeddingService:
    """Unified embedding service with automatic backend selection.

    Falls back from Azure OpenAI → OpenAI → local sentence-transformers
    depending on which credentials are available.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self._embeddings: Optional[Embeddings] = None
        self._model_name = model_name
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Select and initialize the embedding backend."""
        if self.settings.use_azure_openai:
            from langchain_openai import AzureOpenAIEmbeddings

            self._embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self.settings.azure_openai_embedding_deployment,
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
            )
            self._backend = "azure_openai"

        elif self.settings.openai_api_key:
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=self._model_name or "text-embedding-3-small",
                api_key=self.settings.openai_api_key,
            )
            self._backend = "openai"

        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._model_name or "all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._backend = "sentence_transformers"

    @property
    def backend(self) -> str:
        """Return which embedding backend is active."""
        return self._backend

    @property
    def underlying(self) -> Embeddings:
        """Return the underlying LangChain Embeddings instance."""
        if self._embeddings is None:
            raise RuntimeError("Embedding backend not initialized")
        return self._embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        return self.underlying.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query string."""
        return self.underlying.embed_query(query)

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Generate embeddings for a list of Document objects."""
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
