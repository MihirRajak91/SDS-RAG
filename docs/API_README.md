# SDS-RAG API

## Overview

The SDS-RAG API provides RESTful endpoints for document processing, retrieval-augmented generation (RAG), and conversational AI capabilities for financial document analysis.

## Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Start the API Server

```bash
python src/api_server.py
```

The API will be available at: `http://localhost:8000`

### 3. View API Documentation

- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc

### 4. Test the API

```bash
python test_api.py
```

## API Endpoints

### Health Checks

- `GET /api/v1/health` - Overall system health
- `GET /api/v1/health/rag` - RAG service health
- `GET /api/v1/health/chat` - Chat service health
- `GET /api/v1/vectors/health` - Vector database health

### Document Processing

- `POST /api/v1/documents/upload` - Upload and process PDF
- `POST /api/v1/documents/process` - Process PDF from file path
- `POST /api/v1/documents/batch` - Batch process multiple PDFs
- `GET /api/v1/documents/{file_name}/summary` - Get document summary
- `DELETE /api/v1/documents/{file_name}` - Remove document
- `GET /api/v1/documents` - List all documents

### RAG & Search

- `POST /api/v1/search` - Semantic search with filters
- `GET /api/v1/search/tables` - Search table content
- `GET /api/v1/search/text` - Search narrative text
- `GET /api/v1/embeddings/status` - Embeddings status

### Chat & Conversational AI

- `POST /api/v1/chat` - Chat query with RAG
- `POST /api/v1/chat/suggestions` - Chat with follow-up suggestions
- `GET /api/v1/chat/document/{name}/overview` - Document overview
- `POST /api/v1/chat/tables` - Table-focused queries
- `POST /api/v1/chat/compare` - Compare across documents
- `GET /api/v1/chat/suggestions/financial` - Financial question suggestions

### Vector Storage

- `GET /api/v1/vectors/collection/info` - Collection information
- `GET /api/v1/vectors/documents/by-source/{file}` - Documents by source
- `DELETE /api/v1/vectors/documents/by-source/{file}` - Delete by source
- `GET /api/v1/vectors/search/raw` - Raw similarity search
- `GET /api/v1/vectors/stats` - Database statistics
- `POST /api/v1/vectors/recreate-collection` - Recreate collection
- `GET /api/v1/vectors/collections` - List collections

## Configuration

Set environment variables:

- `API_HOST` - Host address (default: 0.0.0.0)
- `API_PORT` - Port number (default: 8000)
- `API_RELOAD` - Auto-reload on changes (default: true)
- `LOG_LEVEL` - Logging level (default: info)

## Example Usage

### Upload and Process Document

```python
import requests

# Upload PDF file
with open("financial_report.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/documents/upload",
        files={"file": f}
    )

print(response.json())
```

### Ask Questions About Documents

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={
        "query": "What was the revenue growth in Q3?",
        "num_results": 5,
        "min_confidence": 0.7
    }
)

result = response.json()
print(f"Answer: {result['response']}")
print(f"Sources: {result['sources_found']}")
```

### Search Documents

```python
import requests

# Search for specific content
response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query": "operating cash flow",
        "limit": 10,
        "content_type": "table_summary",
        "min_confidence": 0.8
    }
)

results = response.json()
for result in results["results"]:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
    print("---")
```

## Error Handling

The API provides consistent error responses with:

- `error` - Error type
- `message` - Human-readable error message
- `timestamp` - ISO timestamp
- `path` - Request path that caused the error
- `details` - Additional error details (for validation errors)

### Common HTTP Status Codes

- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error
- `503` - Service Unavailable

## Requirements

### Services

- **Qdrant**: Vector database for embeddings
- **Google AI**: For LLM responses (Gemini)

### Dependencies

- FastAPI
- Uvicorn
- Qdrant Client
- LangChain
- Sentence Transformers
- PDFPlumber
- Pydantic

## Development

### Adding New Endpoints

1. Create new router in `src/sds_rag/api/routers/`
2. Add request/response models in `src/sds_rag/api/models.py`
3. Include router in `src/sds_rag/api/main.py`
4. Add tests to verify functionality

### Running in Development

```bash
# Start with auto-reload
python src/api_server.py

# Or with uvicorn directly
uvicorn sds_rag.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment

```bash
# Install production dependencies
poetry install --only main

# Run with production settings
API_RELOAD=false LOG_LEVEL=warning python src/api_server.py

# Or with gunicorn (recommended)
gunicorn sds_rag.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```