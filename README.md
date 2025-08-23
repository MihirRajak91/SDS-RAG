# SDS-RAG: Simple Document Search with RAG

A streamlined RAG (Retrieval Augmented Generation) application built with:
- **Frontend**: Streamlit
- **LLM**: Google Gemini API
- **Embeddings**: MiniLM (sentence-transformers)
- **Vector Store**: Qdrant
- **Framework**: LangChain

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run Qdrant (Docker)**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Start Application**
   ```bash
   streamlit run app.py
   ```

## File Structure

```
SDS-RAG/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Project dependencies
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

- **Streamlit**: Web interface
- **LangChain**: RAG framework
- **Qdrant**: Vector database
- **MiniLM**: Text embeddings
- **Gemini**: Language model
- **PyPDF2**: PDF processing
