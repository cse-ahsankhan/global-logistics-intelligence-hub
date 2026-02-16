# Architecture

## System Overview

The Global Logistics Intelligence Hub follows a layered architecture designed for modularity, testability, and seamless migration between local development and Azure cloud deployment.

## Data Flow

```
Documents (PDF/Excel/API)
    │
    ▼
┌──────────────┐
│  Ingestion   │  Parse documents, extract tables as Markdown
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  PII Masking │  Presidio + custom logistics recognizers
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Chunking    │  Parent chunks (1500 tokens) → Child chunks (512 tokens)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embedding   │  Azure OpenAI / sentence-transformers
└──────┬───────┘
       │
       ├──► BM25 Index (keyword)
       │
       └──► FAISS Index (vector)
               │
               ▼
       ┌──────────────┐
       │ Hybrid Search │  Reciprocal Rank Fusion (30% BM25 / 70% semantic)
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Retriever   │  Context expansion: child → parent lookup
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  Generator   │  Azure OpenAI GPT-4o with domain-specific prompt
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │  RAG Chain   │  LCEL pipeline with source attribution
       └──────┬───────┘
              │
              ▼
         FastAPI Response
```

## Design Decisions

### Parent-Child Indexing

The chunking strategy uses a two-tier approach:

- **Parent chunks (1500 tokens)**: Provide broader context for grounding LLM answers. These are not directly retrieved but are referenced when a child chunk matches.
- **Child chunks (512 tokens)**: Optimized for retrieval precision. Smaller chunks produce better embedding representations for semantic search.

When a child chunk is retrieved, the system looks up its parent to provide the LLM with surrounding context, reducing hallucination.

### Hybrid Search with RRF

Pure vector search misses exact keyword matches critical in logistics (container IDs, HS codes, route names). Pure BM25 misses semantic similarity. The hybrid approach combines both:

1. **BM25** scores documents by keyword overlap (TF-IDF based)
2. **FAISS** scores documents by embedding cosine similarity
3. **Reciprocal Rank Fusion** merges the two ranked lists:
   ```
   score(d) = w_bm25 × 1/(k + rank_bm25(d)) + w_sem × 1/(k + rank_sem(d))
   ```
   Default weights: 30% BM25, 70% semantic (k=60).

### PII Masking Strategy

Supply chain documents contain both standard PII (names, emails) and domain-specific identifiers (container numbers, customs references). The masking pipeline:

1. Runs Presidio's built-in recognizers for standard PII
2. Applies custom regex recognizers for logistics identifiers
3. Returns a reversible mapping (`[PERSON_1] → "John Smith"`) for authorized audit access
4. Masked text is stored in the vector index; the mapping is stored separately with access controls

### Backend Flexibility

Every component supports multiple backends:

| Component | Production | Development |
|-----------|-----------|-------------|
| LLM | Azure OpenAI GPT-4o | OpenAI API |
| Embeddings | Azure OpenAI text-embedding-3-large | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | Azure AI Search | FAISS (in-memory) |
| PII Detection | Presidio + spaCy en_core_web_lg | Presidio + spaCy en_core_web_lg |

Backend selection is automatic based on environment variables — no code changes required.

## Scalability Considerations

- **Document ingestion**: Batch processing with async support for large document sets
- **Vector index**: FAISS for local development; Azure AI Search for production with built-in sharding
- **API**: FastAPI with async endpoints and configurable concurrency
- **Embeddings**: Batched embedding generation to minimize API calls
