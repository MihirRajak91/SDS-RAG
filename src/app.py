"""
SDS-RAG Streamlit Frontend - Home Page

A comprehensive web interface for the Simple Document Search with 
Retrieval Augmented Generation system. This is the main home page.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the RAG services and configuration
from src.sds_rag.services.rag_service import RAGService
from src.sds_rag.services.chat_service import ChatService
from sds_rag.config import config

# Import page functions
try:
    from pages.upload import main as show_upload_page_main
    from pages.search import main as show_search_page_main  
    from pages.chat import main as show_chat_page_main
    
    def show_upload_page(rag_service):
        show_upload_page_main()
    
    def show_search_page(rag_service):
        show_search_page_main()
        
    def show_chat_page(chat_service):
        show_chat_page_main()
        
except ImportError as e:
    st.error(f"Could not import page modules: {e}")
    
    # Fallback functions
    def show_upload_page(rag_service):
        st.error("Upload page not available. Please run: poetry run streamlit run src/pages/upload.py")
    
    def show_search_page(rag_service):
        st.error("Search page not available. Please run: poetry run streamlit run src/pages/search.py")
        
    def show_chat_page(chat_service):
        st.error("Chat page not available. Please run: poetry run streamlit run src/pages/chat.py")

# Page configuration
st.set_page_config(
    page_title=config.streamlit.title,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_services():
    """Initialize RAG and Chat services with caching."""
    try:
        rag_service = RAGService(
            qdrant_host=config.qdrant.host,
            qdrant_port=config.qdrant.port,
            embedding_model=config.embedding.embedding_model_name
        )
        chat_service = ChatService(
            google_api_key=config.llm.google_api_key,
            qdrant_host=config.qdrant.host,
            qdrant_port=config.qdrant.port,
            embedding_model=config.embedding.embedding_model_name
        )
        return rag_service, chat_service, True
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return None, None, False

def show_home_page(rag_service: RAGService, chat_service: ChatService):
    """Display the home page with overview and quick actions."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome to SDS-RAG")
        st.markdown("""
        **Simple Document Search with Retrieval Augmented Generation**
        
        This system helps you analyze financial documents using advanced AI:
        
        - 📄 **Upload PDF Documents**: Process financial reports, statements, and documents
        - 🔍 **Intelligent Search**: Find relevant information using semantic search
        - 💬 **AI Chat Assistant**: Ask questions about your documents in natural language
        - 📊 **Data Extraction**: Automatically extract tables and structured data
        - 🏷️ **Smart Classification**: Identify income statements, balance sheets, cash flows
        
        ### Quick Start Guide:
        1. **Upload** your financial PDF documents
        2. **Search** for specific information or data points
        3. **Chat** with the AI assistant about your documents
        4. **Manage** your document library
        """)
    
    with col2:
        st.header("📈 System Overview")
        
        # Get system health
        try:
            health = chat_service.health_check()
            collection_info = health["rag_service"]["collection_info"]
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Documents Stored", collection_info.get("points_count", 0))
            st.metric("Vector Dimension", collection_info.get("vector_size", 0))
            st.metric("Distance Metric", collection_info.get("distance", "Unknown"))
            st.markdown('</div>', unsafe_allow_html=True)
            
            if health["chat_service_healthy"]:
                st.markdown('<div class="success-box">✅ All systems operational</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">❌ System issues detected</div>', 
                           unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Could not fetch system status: {e}")
        
        # Quick actions
        st.header("🚀 Quick Actions")
        st.info("💡 **Navigation Instructions:**\n\nTo use other features:\n- Run `poetry run streamlit run src/pages/upload.py` for document upload\n- Run `poetry run streamlit run src/pages/search.py` for search\n- Run `poetry run streamlit run src/pages/chat.py` for AI chat")
        
        with st.expander("🔧 Alternative: Use Session State Navigation"):
            page = st.selectbox("Choose Page:", ["Home", "Upload", "Search", "Chat"])
            if page != "Home":
                st.session_state.current_page = page
                st.rerun()

def main():
    """Main application function - Home page."""
    
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Header
    st.markdown('<h1 class="main-header">📊 SDS-RAG: Financial Document Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Initialize services
    rag_service, chat_service, services_ready = initialize_services()
    
    if not services_ready:
        st.error("❌ Services not available. Please check your configuration.")
        return
    
    # Handle different pages based on session state
    if st.session_state.current_page == "Upload":
        show_upload_page(rag_service)
    elif st.session_state.current_page == "Search":
        show_search_page(rag_service) 
    elif st.session_state.current_page == "Chat":
        show_chat_page(chat_service)
    else:
        # Default to home page
        show_home_page(rag_service, chat_service)

if __name__ == "__main__":
    main()