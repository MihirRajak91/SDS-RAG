# 📊 SDS-RAG Streamlit Frontend

A comprehensive web interface for the Simple Document Search with Retrieval Augmented Generation (SDS-RAG) system. This Streamlit application provides an intuitive interface for uploading, processing, searching, and chatting with financial documents using advanced AI.

## 🌟 Features

### 📄 Document Management
- **Single File Upload**: Upload individual PDF documents with real-time processing
- **Batch Processing**: Process multiple PDFs from a directory
- **Document Analysis**: Get detailed insights about processed documents
- **Document Removal**: Remove documents from the vector database

### 🔍 Intelligent Search
- **Semantic Search**: Find relevant information using natural language queries
- **Advanced Filtering**: Filter by content type, table type, source file, and confidence
- **Real-time Results**: Get instant search results with relevance scores

### 💬 AI Chat Assistant
- **Natural Language Q&A**: Ask questions about your documents in plain English
- **Context-Aware Responses**: Get answers based on relevant document content
- **Follow-up Suggestions**: Receive intelligent follow-up question suggestions
- **Chat History**: Track your conversation history

### 🔧 System Management
- **Health Monitoring**: Real-time system health checks
- **Performance Metrics**: View processing statistics and system status
- **Configuration Management**: Centralized configuration with environment variables

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+** installed
2. **Poetry** for dependency management
3. **Qdrant** vector database running
4. **Google AI API Key** for LLM services

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd SDS-RAG
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your configuration
   # Most importantly, set your GOOGLE_API_KEY
   ```

4. **Start Qdrant (using Docker):**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

5. **Launch the application:**
   ```bash
   # Using the launcher script (recommended)
   python run_app.py
   
   # Or directly with Streamlit
   streamlit run app.py
   ```

6. **Access the application:**
   Open your browser and go to `http://localhost:8501`

## 📋 Configuration

### Environment Variables

Create a `.env` file based on `.env.example` with your configuration:

```bash
# Required: Google AI API Key
GOOGLE_API_KEY=your_api_key_here

# Optional: Customize other settings
QDRANT_HOST=localhost
QDRANT_PORT=6333
STREAMLIT_PORT=8501
```

### Configuration Options

| Category | Variable | Default | Description |
|----------|----------|---------|-------------|
| **LLM** | `GOOGLE_API_KEY` | - | Google AI API key (required) |
| | `LLM_MODEL` | `gemini-1.5-flash` | LLM model to use |
| **Qdrant** | `QDRANT_HOST` | `localhost` | Qdrant server host |
| | `QDRANT_PORT` | `6333` | Qdrant server port |
| **Streamlit** | `STREAMLIT_PORT` | `8501` | Application port |
| | `MAX_UPLOAD_SIZE_MB` | `100` | Maximum file upload size |
| **Processing** | `MIN_CONFIDENCE` | `0.6` | Minimum confidence threshold |
| | `BATCH_SIZE` | `10` | Batch processing size |

## 🎯 Using the Application

### 1. Upload Documents

**Single Upload:**
1. Go to "📄 Document Upload" page
2. Choose a PDF file
3. Click "🚀 Process Document"
4. View processing results

**Batch Upload:**
1. Enter directory path containing PDFs
2. Click "🔍 Scan Directory" to preview files
3. Click "🚀 Process All Files"
4. Monitor batch processing progress

### 2. Search Documents

**Basic Search:**
1. Go to "🔍 Search Documents" page
2. Enter your search query (e.g., "revenue for Q4")
3. Click "🔍 Search"
4. Browse results with relevance scores

**Advanced Search:**
1. Expand "🔧 Advanced Filters"
2. Set content type, table type, source file, or confidence filters
3. Adjust maximum results
4. Perform filtered search

### 3. Chat with AI

**Start Conversation:**
1. Go to "💬 Chat Assistant" page
2. Type your question in natural language
3. Review AI response with source citations
4. Continue the conversation

**Use Quick Questions:**
1. Click on suggested questions in the sidebar
2. Adjust chat settings (context documents, confidence)
3. View conversation history

### 4. Manage Documents

**View Analysis:**
1. Go to "📋 Document Management"
2. Enter filename to analyze
3. View content breakdown and statistics

**Remove Documents:**
1. Enter filename to remove
2. Confirm removal
3. View removal results

### 5. Monitor System

**Check Health:**
1. Go to "🔧 System Status"
2. View component status (Qdrant, LLM, etc.)
3. Monitor system metrics
4. Use auto-refresh for real-time monitoring

## 🛠️ Advanced Usage

### Custom Launcher Options

The `run_app.py` script provides advanced options:

```bash
# Run with custom port
python run_app.py --port 8502

# Skip health checks (for development)
python run_app.py --skip-health-checks

# Debug mode
python run_app.py --debug

# Setup environment only
python run_app.py --setup-only
```

### Configuration Management

```python
from config import config, print_config_summary

# View current configuration
print_config_summary()

# Access specific configuration
qdrant_host = config.qdrant.host
streamlit_port = config.streamlit.port
```

## 🔍 Troubleshooting

### Common Issues

**1. "Services not available" error:**
- Check if Qdrant is running: `docker ps`
- Verify Qdrant connection: `telnet localhost 6333`
- Check environment variables in `.env`

**2. "Google API Key not set" error:**
- Ensure `GOOGLE_API_KEY` is set in `.env`
- Verify the API key is valid
- Check Google AI Studio for key status

**3. File upload fails:**
- Check file size (default max: 100MB)
- Ensure file is a valid PDF
- Verify temp directory exists and is writable

**4. Search returns no results:**
- Check if documents are processed
- Lower confidence threshold
- Try broader search terms
- Verify Qdrant has data

### Debug Mode

Run with debug logging:

```bash
python run_app.py --debug
```

### Health Checks

Run health checks manually:

```python
from src.sds_rag.services.chat_service import ChatService

chat_service = ChatService()
health = chat_service.health_check()
print(health)
```

## 📊 Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
├─────────────────────────────────────────────────────────────┤
│  📄 Upload    🔍 Search    💬 Chat    📋 Manage    🔧 Status │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     RAG Services                            │
├──────────────────┬──────────────────┬─────────────────────────┤
│  Document        │  Vector Storage  │  Chat Service           │
│  Processing      │  (Qdrant)        │  (Google AI)            │
└──────────────────┴──────────────────┴─────────────────────────┘
```

### Data Flow

1. **Upload**: PDF → Document Processor → Embeddings → Vector DB
2. **Search**: Query → Embeddings → Vector Search → Results
3. **Chat**: Question → Context Retrieval → LLM → Response

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or contributions:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the configuration documentation

---

**Happy Document Analysis! 📊✨**