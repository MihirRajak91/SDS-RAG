# SDS-RAG: Simple Document Search with RAG
poetry run streamlit run src/app.py
A streamlined RAG (Retrieval Augmented Generation) application built with:
- **Frontend**: Streamlit
- **LLM**: Google Gemini API
- **Embeddings**: MiniLM (sentence-transformers)
- **Vector Store**: Qdrant
- **Framework**: LangChain
- **Package Manager**: Poetry

## Quick Start

### Prerequisites
- Python 3.9+
- Poetry ([Installation guide](https://python-poetry.org/docs/#installation))
- Docker (for Qdrant)

### Setup

1. **Clone and navigate to project**
   ```bash
   cd SDS-RAG
   ```

2. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start Qdrant vector database**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

5. **Run the application**
   ```bash
   poetry run streamlit run app.py
   ```

## Development Commands

```bash
# Install dependencies
poetry install

# Add a new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Activate virtual environment
poetry shell

# Run tests
poetry run pytest

# Format code
poetry run black .

# Sort imports
poetry run isort .

# Type checking
poetry run mypy src/

# Run linting
poetry run flake8 src/

# Update dependencies
poetry update
```

## File Structure

```
SDS-RAG/
├── pyproject.toml                 # Poetry configuration and dependencies
├── .gitignore                     # Git ignore rules
├── app.py                         # Main Streamlit application
├── .env.example                   # Environment variables template
├── README.md                      # This file
├── data/                          # Data storage
│   ├── uploads/                   # Uploaded documents
│   └── processed/                 # Processed documents
├── src/                          # Source code
│   ├── components/               # Streamlit UI components
│   │   ├── sidebar.py           # Configuration sidebar
│   │   ├── file_upload.py       # File upload interface
│   │   ├── chat_interface.py    # Chat UI
│   │   └── document_viewer.py   # Document display
│   ├── utils/                   # Core functionality
│   │   ├── document_processor.py # Document text extraction
│   │   ├── embeddings.py        # MiniLM embeddings
│   │   ├── vector_store.py      # Qdrant operations
│   │   ├── llm.py              # Gemini API integration
│   │   ├── rag_chain.py        # RAG implementation
│   │   └── helpers.py          # Utility functions
│   └── config/                 # Configuration
│       └── settings.py         # App settings
└── tests/                      # Test files
    ├── test_rag_chain.py
    ├── test_embeddings.py
    └── test_vector_store.py
```

## Features

- Upload PDF and text documents
- Automatic text extraction and chunking
- Vector embeddings with MiniLM
- Semantic search with Qdrant
- Question answering with Gemini
- Clean Streamlit interface

## Technology Stack

- **Poetry**: Dependency management
- **Streamlit**: Web interface
- **LangChain**: RAG framework
- **Qdrant**: Vector database
- **MiniLM**: Text embeddings
- **Gemini**: Language model
- **PyPDF2**: PDF processing

## Development

This project uses Poetry for dependency management and includes:
- Code formatting with Black
- Import sorting with isort
- Type checking with MyPy
- Linting with Flake8
- Testing with pytest

Run `poetry install` to set up the development environment with all tools.
