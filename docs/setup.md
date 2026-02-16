# Setup Guide

## Prerequisites

- Python 3.11 or higher
- pip or conda for package management
- (Optional) Azure subscription for Azure OpenAI and Azure AI Search
- (Optional) OpenAI API key for local development

## Installation

### 1. Clone and set up the environment

```bash
git clone https://github.com/ahsanrazakhan/global-logistics-intelligence-hub.git
cd global-logistics-intelligence-hub

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Install spaCy language model

The PII masking module requires a spaCy language model:

```bash
python -m spacy download en_core_web_lg
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

#### Option A: Azure OpenAI (Production)

```env
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

#### Option B: OpenAI API (Development)

```env
OPENAI_API_KEY=your-openai-key
```

#### Option C: Fully Local (No API keys)

Leave all API keys empty. The system will use:
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- FAISS for vector search
- BM25 for keyword search

Note: LLM generation (the `/query` endpoint) requires at least an OpenAI API key. Retrieval and search work without any API keys.

## Running the Application

### API Server

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API docs are available at `http://localhost:8000/docs`.

### Interactive Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

### Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_chunking.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Azure AI Search Setup (Optional)

If using Azure AI Search instead of local FAISS:

1. Create an Azure AI Search resource in the Azure Portal
2. Create an index with the following fields:
   - `id` (string, key)
   - `content` (string, searchable)
   - `content_vector` (Collection(Edm.Single), dimensions: 3072 for text-embedding-3-large)
   - `source` (string, filterable)
   - `chunk_type` (string, filterable)
   - `parent_id` (string, filterable)
   - `metadata` (string)

3. Configure the environment:
   ```env
   AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
   AZURE_SEARCH_API_KEY=your-admin-key
   AZURE_SEARCH_INDEX_NAME=logistics-knowledge-base
   ```

## Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_lg

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t logistics-intelligence-hub .
docker run -p 8000:8000 --env-file .env logistics-intelligence-hub
```

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `ModuleNotFoundError: spacy` | Run `pip install spacy && python -m spacy download en_core_web_lg` |
| FAISS installation fails | Use `pip install faiss-cpu` (not `faiss-gpu` unless CUDA is configured) |
| Presidio slow on first run | Normal — spaCy model loads on first call, subsequent calls are fast |
| `OPENAI_API_KEY not set` | Set the key in `.env` or use fully-local mode (search only) |
