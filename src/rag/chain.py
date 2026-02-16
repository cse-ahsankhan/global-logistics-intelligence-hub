"""End-to-end RAG chain using LangChain Expression Language (LCEL).

Orchestrates retrieval → context formatting → generation with
full source attribution. Supports both streaming and non-streaming
invocation.
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.rag.generator import ResponseGenerator, SYSTEM_PROMPT
from src.rag.retriever import HybridRetriever


@dataclass
class RAGResponse:
    """Structured response from the RAG chain."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    context_documents: list[Document] = field(default_factory=list)


class RAGChain:
    """Composable RAG chain with retrieval, generation, and attribution.

    Combines hybrid retrieval with LLM generation using LCEL,
    ensuring every answer includes traceable source references.

    Parameters
    ----------
    retriever : HybridRetriever
        The retriever for fetching relevant context.
    generator : ResponseGenerator | None
        The generator for producing answers. Created automatically
        if not provided.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: Optional[ResponseGenerator] = None,
    ):
        self.retriever = retriever
        self.generator = generator or ResponseGenerator()
        self._chain = self._build_chain()

    def _build_chain(self):
        """Build the LCEL chain: retrieve → format → generate."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])

        chain = (
            {
                "context": self.retriever | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.generator.llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, query: str) -> RAGResponse:
        """Run the full RAG pipeline and return a structured response."""
        # Retrieve context
        context_docs = self.retriever.invoke(query)

        # Generate answer
        answer = self._chain.invoke(query)

        # Extract source attribution
        sources = self._extract_sources(context_docs)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query,
            context_documents=context_docs,
        )

    async def ainvoke(self, query: str) -> RAGResponse:
        """Async version of invoke."""
        context_docs = await self.retriever.ainvoke(query)
        answer = await self._chain.ainvoke(query)
        sources = self._extract_sources(context_docs)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query,
            context_documents=context_docs,
        )

    @staticmethod
    def _format_docs(documents: list[Document]) -> str:
        """Format a list of Documents into a single context string."""
        if not documents:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")

            header = f"[Document {i}] Source: {source}"
            if page:
                header += f", Page {page}"

            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_sources(documents: list[Document]) -> list[dict]:
        """Extract source metadata from retrieved documents."""
        sources = []
        seen = set()

        for doc in documents:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            key = f"{source}:{page}"

            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": source,
                    "page": page,
                    "chunk_type": doc.metadata.get("chunk_type", ""),
                    "relevance_score": doc.metadata.get("retrieval_score", 0.0),
                })

        return sources
