# SDS-RAG Project Structure

## Overview

The SDS-RAG project has been reorganized into a clean, professional structure that follows Python best practices and separates concerns appropriately.

## Directory Structure

```
SDS-RAG/
├── 📁 src/                          # Application source code
│   ├── 📄 app.py                    # Main Streamlit application entry point
│   └── 📁 sds_rag/                  # Main package
│       ├── __init__.py
│       ├── 📁 api/                  # API endpoints (future)
│       ├── 📁 config/               # Configuration management
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── 📁 core/                 # Core business logic
│       │   ├── __init__.py
│       │   └── document_processor.py
│       ├── 📁 models/               # Data models and schemas
│       │   ├── __init__.py
│       │   └── schemas.py
│       ├── 📁 services/             # Business services
│       │   ├── __init__.py
│       │   ├── chat_service.py
│       │   ├── classification_service.py
│       │   ├── embedding_service.py
│       │   ├── extraction_service.py
│       │   ├── llm_service.py
│       │   ├── parsing_service.py
│       │   ├── rag_service.py
│       │   ├── validation_service.py
│       │   └── vector_storage_service.py
│       ├── 📁 ui/                   # Streamlit UI components
│       │   ├── __init__.py
│       │   ├── chat_interface.py
│       │   ├── document_viewer.py
│       │   ├── file_upload.py
│       │   └── sidebar.py
│       └── 📁 utils/                # Utility functions
│           └── __init__.py
├── 📁 scripts/                      # Development and utility scripts
│   ├── health_check.py              # Comprehensive system health check
│   ├── setup_services.bat          # Windows service setup script
│   └── simple_health_check.py      # Basic connectivity check
├── 📁 examples/                     # Usage examples and demos
│   ├── chatbot_usage.py             # Chatbot usage example
│   ├── pdf_extraction.py           # PDF processing example
│   └── rag_usage.py                 # RAG pipeline example
├── 📁 tests/                        # Test files
│   ├── __init__.py
│   ├── test_embeddings.py
│   ├── test_rag_chain.py
│   └── test_vector_store.py
├── 📁 docs/                         # Documentation
│   ├── project_structure.md        # This file
│   └── setup_services.md           # Service setup guide
├── 📁 docker/                       # Container configuration
│   └── docker-compose.yml          # Docker services definition
├── 📁 data/                         # Data storage
│   ├── processed/                   # Processed documents
│   └── uploads/                     # Upload staging area
├── 📁 config/                       # Environment configuration
├── 📄 README.md                     # Main project documentation
├── 📄 pyproject.toml               # Python project configuration
├── 📄 requirements.txt             # Python dependencies
└── 📄 proposed_structure.md        # Original reorganization plan
```

## Module Descriptions

### Core Application (`src/sds_rag/`)

- **`config/`**: Application configuration and settings management
- **`core/`**: Core business logic including document processing orchestration
- **`models/`**: Pydantic models and data schemas
- **`services/`**: Business service layer with specialized functionality:
  - `chat_service.py`: Complete RAG chatbot orchestration
  - `rag_service.py`: RAG pipeline management
  - `embedding_service.py`: Text embedding generation
  - `llm_service.py`: Language model integration
  - `vector_storage_service.py`: Qdrant vector database operations
  - Other specialized services for extraction, parsing, etc.
- **`ui/`**: Streamlit user interface components
- **`utils/`**: Shared utility functions

### Supporting Structure

- **`scripts/`**: Development tools and maintenance scripts
- **`examples/`**: Demonstration and usage examples
- **`tests/`**: Unit and integration tests
- **`docs/`**: Project documentation
- **`docker/`**: Containerization configuration
- **`data/`**: Data storage directories

## Import Guidelines

### Within the Package
Use relative imports within the same package:
```python
# In services/chat_service.py
from .rag_service import RAGService
from .llm_service import LLMService
```

### From External Scripts
Scripts and examples should import using the full package name:
```python
# In scripts/health_check.py
from sds_rag.services.chat_service import ChatService
from sds_rag.services.rag_service import RAGService
```

### Path Setup for Scripts
Scripts need to add the src directory to Python path:
```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))
```

## Benefits

✅ **Clear separation of concerns**  
✅ **Professional Python package structure**  
✅ **Easy navigation and maintenance**  
✅ **Scalable architecture**  
✅ **Better IDE support**  
✅ **Follows Python best practices**  

## Running the Application

```bash
# From project root
cd src
python app.py

# Or run health checks
python scripts/health_check.py

# Run examples
python examples/chatbot_usage.py
```