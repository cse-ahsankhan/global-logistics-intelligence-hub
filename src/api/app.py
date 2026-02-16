"""FastAPI application exposing the RAG pipeline as a REST API.

Endpoints
---------
POST /query  — Submit a question and receive an answer with sources.
GET  /health — Health check for monitoring and load balancers.
POST /ingest — Upload and index documents into the knowledge base.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from config.settings import get_settings
from src.processing.chunking import SemanticChunker
from src.processing.embeddings import EmbeddingService
from src.processing.pii_masking import PIIMasker
from src.rag.chain import RAGChain, RAGResponse
from src.rag.generator import ResponseGenerator
from src.rag.retriever import HybridRetriever
from src.vectorstore.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)

# Global state for the running application
_search_engine: Optional[HybridSearchEngine] = None
_rag_chain: Optional[RAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and clean up on shutdown."""
    global _search_engine, _rag_chain

    settings = get_settings()
    logger.info("Initializing services (env=%s)", settings.environment)

    embedding_service = EmbeddingService()
    _search_engine = HybridSearchEngine(embedding_service=embedding_service)

    # Load persisted index if available
    index_path = Path("data/index")
    if index_path.exists():
        logger.info("Loading persisted index from %s", index_path)
        _search_engine.load(index_path)

    retriever = HybridRetriever(
        search_engine=_search_engine,
        top_k=settings.top_k_results,
    )
    generator = ResponseGenerator()
    _rag_chain = RAGChain(retriever=retriever, generator=generator)

    logger.info("Services initialized (embedding backend: %s)", embedding_service.backend)
    yield

    logger.info("Shutting down services")


app = FastAPI(
    title="Global Logistics Intelligence Hub",
    description="RAG-powered API for supply chain document intelligence",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Request / Response models ---

class QueryRequest(BaseModel):
    """Incoming question payload."""

    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class SourceInfo(BaseModel):
    """Metadata about a source document used in the answer."""

    source: str
    page: Optional[int | str] = None
    chunk_type: str = ""
    relevance_score: float = 0.0


class QueryResponse(BaseModel):
    """Structured answer returned to the caller."""

    answer: str
    sources: list[SourceInfo]
    query: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    environment: str
    documents_indexed: int
    embedding_backend: str


# --- Endpoints ---

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Submit a supply chain question and receive a grounded answer."""
    if _rag_chain is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    start = time.perf_counter()

    try:
        result: RAGResponse = _rag_chain.invoke(request.question)
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to process query")

    elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryResponse(
        answer=result.answer,
        sources=[SourceInfo(**s) for s in result.sources],
        query=result.query,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health status."""
    settings = get_settings()
    doc_count = len(_search_engine.documents) if _search_engine else 0

    embedding_service = EmbeddingService()
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        documents_indexed=doc_count,
        embedding_backend=embedding_service.backend,
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Upload and index a document into the knowledge base."""
    if _search_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".xlsx", ".csv", ".txt"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save uploaded file temporarily
    temp_path = Path("data/sample") / file.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    temp_path.write_bytes(content)

    try:
        # Load document based on type
        if suffix == ".pdf":
            from src.ingestion.pdf_loader import PDFLoader
            docs = PDFLoader(temp_path).load()
        elif suffix in {".xlsx", ".csv"}:
            from src.ingestion.excel_loader import ExcelLoader
            docs = ExcelLoader(temp_path).load()
        else:
            from langchain_core.documents import Document
            docs = [Document(page_content=content.decode("utf-8"), metadata={"source": file.filename})]

        # PII masking
        masker = PIIMasker()
        for doc in docs:
            result = masker.mask(doc.page_content)
            doc.page_content = result.masked_text

        # Chunk
        chunker = SemanticChunker()
        chunk_result = chunker.chunk_documents(docs)
        all_chunks = chunk_result.parent_chunks + chunk_result.child_chunks

        # Index
        _search_engine.add_documents(all_chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "pages_loaded": len(docs),
            "chunks_created": len(all_chunks),
            "total_documents_indexed": len(_search_engine.documents),
        }

    except Exception as e:
        logger.exception("Ingestion failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
