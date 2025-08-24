"""
SDS-RAG Streamlit Frontend

A comprehensive web interface for the Simple Document Search with 
Retrieval Augmented Generation system. Provides document upload, 
search, chat, and management capabilities.
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Import the RAG services and configuration
from src.sds_rag.services.rag_service import RAGService
from src.sds_rag.services.chat_service import ChatService
from src.sds_rag.utils import validate_pdf_file, get_file_size_human, find_pdf_files
from sds_rag.config import config

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
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
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

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 SDS-RAG: Financial Document Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Initialize services
    rag_service, chat_service, services_ready = initialize_services()
    
    if not services_ready:
        st.error("❌ Services not available. Please check your configuration.")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📄 Document Upload", "🔍 Search Documents", 
             "💬 Chat Assistant", "📋 Document Management", "🔧 System Status"]
        )
        
        # System health check in sidebar
        if st.button("🔄 Check System Health"):
            with st.spinner("Checking system health..."):
                health = chat_service.health_check()
                if health["chat_service_healthy"]:
                    st.success("✅ System is healthy")
                else:
                    st.error("❌ System issues detected")
    
    # Route to selected page
    if page == "🏠 Home":
        show_home_page(rag_service, chat_service)
    elif page == "📄 Document Upload":
        show_upload_page(rag_service)
    elif page == "🔍 Search Documents":
        show_search_page(rag_service)
    elif page == "💬 Chat Assistant":
        show_chat_page(chat_service)
    elif page == "📋 Document Management":
        show_management_page(rag_service)
    elif page == "🔧 System Status":
        show_status_page(chat_service)

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
        if st.button("📤 Upload Document", use_container_width=True):
            st.switch_page("pages/upload.py")
        if st.button("🔍 Search Now", use_container_width=True):
            st.switch_page("pages/search.py")
        if st.button("💬 Start Chat", use_container_width=True):
            st.switch_page("pages/chat.py")

def show_upload_page(rag_service: RAGService):
    """Display document upload page."""
    
    st.header("📄 Document Upload")
    
    # Upload methods tabs
    tab1, tab2 = st.tabs(["Single File Upload", "Batch Upload"])
    
    with tab1:
        st.subheader("Upload Single PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload financial documents like annual reports, 10-K filings, etc."
        )
        
        if uploaded_file is not None:
            # File information
            file_size = len(uploaded_file.getvalue())
            st.info(f"📁 **File:** {uploaded_file.name} ({get_file_size_human(file_size)})")
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                process_single_file(uploaded_file, rag_service)
    
    with tab2:
        st.subheader("Batch Upload from Directory")
        
        directory_path = st.text_input(
            "Directory Path",
            placeholder="Enter full path to directory containing PDFs",
            help="Process multiple PDF files from a directory"
        )
        
        if directory_path and st.button("🔍 Scan Directory"):
            if os.path.exists(directory_path):
                pdf_files = find_pdf_files(directory_path)
                if pdf_files:
                    st.success(f"Found {len(pdf_files)} PDF files:")
                    for pdf_file in pdf_files[:10]:  # Show first 10
                        st.write(f"• {pdf_file.name}")
                    if len(pdf_files) > 10:
                        st.write(f"... and {len(pdf_files) - 10} more files")
                    
                    if st.button("🚀 Process All Files", type="primary"):
                        process_batch_files(directory_path, rag_service)
                else:
                    st.warning("No PDF files found in the directory")
            else:
                st.error("Directory does not exist")

def process_single_file(uploaded_file, rag_service: RAGService):
    """Process a single uploaded file."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        progress_bar.progress(25)
        status_text.text("Validating PDF file...")
        
        # Validate file
        is_valid, errors = validate_pdf_file(temp_path)
        if not is_valid:
            st.error(f"Invalid PDF file: {', '.join(errors)}")
            return
        
        progress_bar.progress(50)
        status_text.text("Processing document...")
        
        # Process document
        start_time = time.time()
        result = rag_service.process_and_store_document(temp_path)
        processing_time = time.time() - start_time
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        # Cleanup
        os.remove(temp_path)
        
        # Display results
        if result.get("processing_successful"):
            display_processing_results(result, processing_time)
        else:
            st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
        # Cleanup on error
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_batch_files(directory_path: str, rag_service: RAGService):
    """Process multiple files from a directory."""
    
    try:
        pdf_files = find_pdf_files(directory_path)
        
        if not pdf_files:
            st.warning("No PDF files found in the directory")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        # Process files
        results = []
        successful = 0
        
        for i, pdf_file in enumerate(pdf_files):
            progress = (i + 1) / len(pdf_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {pdf_file.name}...")
            
            try:
                result = rag_service.process_and_store_document(str(pdf_file))
                results.append(result)
                
                if result.get("processing_successful"):
                    successful += 1
                    
            except Exception as e:
                st.error(f"Failed to process {pdf_file.name}: {e}")
        
        # Display batch results
        with results_container:
            st.success(f"Batch processing complete: {successful}/{len(pdf_files)} files processed successfully")
            
            # Summary table
            if results:
                display_batch_results(results)
                
    except Exception as e:
        st.error(f"Error in batch processing: {e}")

def display_processing_results(result: Dict[str, Any], processing_time: float):
    """Display results of document processing."""
    
    st.markdown('<div class="success-box">✅ Document processed successfully!</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tables Processed", result.get("tables_processed", 0))
    with col2:
        st.metric("Text Chunks", result.get("text_chunks_processed", 0))
    with col3:
        st.metric("Vector Points", result.get("vector_points_stored", 0))
    with col4:
        st.metric("Processing Time", f"{processing_time:.1f}s")
    
    # Detailed information
    with st.expander("📊 Detailed Results"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Processing Summary:**")
            st.write(f"• File: {result.get('file_name')}")
            st.write(f"• Total Pages: {result.get('total_pages')}")
            st.write(f"• High Confidence Tables: {result.get('high_confidence_tables')}")
            st.write(f"• Average Confidence: {result.get('average_confidence', 0):.2f}")
        
        with col2:
            st.write("**Quality Metrics:**")
            st.write(f"• Success Rate: {result.get('success_rate', 0):.1f}%")
            st.write(f"• Documents Created: {result.get('embedded_documents_created')}")
            st.write(f"• Processing Time: {result.get('processing_timestamp')}")

def display_batch_results(results: List[Dict[str, Any]]):
    """Display results of batch processing."""
    
    # Create summary DataFrame
    df_data = []
    for result in results:
        df_data.append({
            "File Name": result.get("file_name", "Unknown"),
            "Status": "✅ Success" if result.get("processing_successful") else "❌ Failed",
            "Tables": result.get("tables_processed", 0),
            "Text Chunks": result.get("text_chunks_processed", 0),
            "Pages": result.get("total_pages", 0),
            "Confidence": f"{result.get('average_confidence', 0):.2f}",
            "Success Rate": f"{result.get('success_rate', 0):.1f}%"
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)

def show_search_page(rag_service: RAGService):
    """Display document search page."""
    
    st.header("🔍 Search Financial Documents")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="e.g., 'revenue for Q4', 'total assets', 'cash flow from operations'",
            help="Enter your search query in natural language"
        )
    
    with col2:
        num_results = st.selectbox("Max Results", [5, 10, 15, 20], index=1)
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            content_type = st.selectbox(
                "Content Type",
                ["Any", "table_summary", "table_row", "narrative_text"],
                help="Filter by type of content"
            )
        
        with col2:
            table_type = st.selectbox(
                "Table Type", 
                ["Any", "income_statement", "balance_sheet", "cash_flow", "notes", "other"],
                help="Filter by financial statement type"
            )
        
        with col3:
            source_file = st.text_input(
                "Source File",
                placeholder="filename.pdf",
                help="Filter by specific document"
            )
        
        with col4:
            min_confidence = st.slider(
                "Min Confidence",
                0.0, 1.0, 0.6, 0.1,
                help="Minimum confidence score for results"
            )
    
    # Search button
    if st.button("🔍 Search", type="primary") and query:
        perform_search(query, rag_service, num_results, content_type, table_type, source_file, min_confidence)

def perform_search(query: str, rag_service: RAGService, num_results: int, 
                  content_type: str, table_type: str, source_file: str, min_confidence: float):
    """Perform the search and display results."""
    
    with st.spinner("Searching..."):
        try:
            # Prepare filters
            filters = {
                "limit": num_results,
                "min_confidence": min_confidence
            }
            
            if content_type != "Any":
                filters["content_type"] = content_type
            if table_type != "Any":
                filters["table_type"] = table_type
            if source_file.strip():
                filters["source_file"] = source_file.strip()
            
            # Perform search
            results = rag_service.search_financial_data(query, **filters)
            
            # Display results
            if results:
                st.success(f"Found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} - Score: {result['score']:.3f}", expanded=(i <= 3)):
                        display_search_result(result)
            else:
                st.warning("No results found. Try adjusting your search terms or filters.")
                
        except Exception as e:
            st.error(f"Search failed: {e}")

def display_search_result(result: Dict[str, Any]):
    """Display a single search result."""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Content:**")
        st.write(result["content"])
    
    with col2:
        st.markdown("**Metadata:**")
        metadata = result["metadata"]
        st.write(f"**Source:** {metadata.get('source', 'Unknown')}")
        st.write(f"**Page:** {metadata.get('page', 'Unknown')}")
        st.write(f"**Type:** {metadata.get('content_type', 'Unknown')}")
        
        if metadata.get("table_type"):
            st.write(f"**Table Type:** {metadata['table_type']}")
        
        if metadata.get("confidence_score"):
            st.write(f"**Confidence:** {metadata['confidence_score']:.2f}")

def show_chat_page(chat_service: ChatService):
    """Display the chat interface."""
    
    st.header("💬 AI Chat Assistant")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🔧 Chat Settings")
        
        num_results = st.selectbox("Context Documents", [3, 5, 8, 10], index=1)
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.1)
        
        # Quick question suggestions
        st.subheader("💡 Quick Questions")
        quick_questions = [
            "What was the total revenue?",
            "Show me the operating expenses",
            "What are the key financial highlights?",
            "How much cash was generated?",
            "What are the main assets?",
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                st.session_state.current_question = question
    
    with col1:
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer.get("response", ""))
                    
                    # Show sources if available
                    if answer.get("sources_found", 0) > 0:
                        with st.expander(f"📄 Sources ({answer['sources_found']})"):
                            for j, doc in enumerate(answer.get("context_documents", [])[:3]):
                                st.write(f"**Source {j+1}:** {doc.get('metadata', {}).get('source', 'Unknown')}")
                                st.write(f"*{doc.get('content', '')[:200]}...*")
        
        # Chat input
        user_question = st.chat_input("Ask a question about your documents...")
        
        # Handle quick questions
        if "current_question" in st.session_state:
            user_question = st.session_state.current_question
            del st.session_state.current_question
        
        if user_question:
            process_chat_message(user_question, chat_service, num_results, min_confidence)

def process_chat_message(question: str, chat_service: ChatService, num_results: int, min_confidence: float):
    """Process a chat message and get AI response."""
    
    with st.spinner("Thinking..."):
        try:
            response = chat_service.chat_with_suggestions(
                user_query=question,
                num_results=num_results,
                min_confidence=min_confidence
            )
            
            # Add to chat history
            st.session_state.chat_history.append((question, response))
            
            # Rerun to update display
            st.rerun()
            
        except Exception as e:
            st.error(f"Chat failed: {e}")

def show_management_page(rag_service: RAGService):
    """Display document management page."""
    
    st.header("📋 Document Management")
    
    # Document list and management
    tab1, tab2 = st.tabs(["Document Library", "Document Analysis"])
    
    with tab1:
        st.subheader("📚 Your Document Library")
        
        # This would require implementing a list documents method
        st.info("Document listing feature requires additional implementation in the vector storage service.")
        
        # Document removal
        st.subheader("🗑️ Remove Document")
        
        file_to_remove = st.text_input(
            "File Name to Remove",
            placeholder="Enter filename (e.g., annual_report.pdf)",
            help="Remove all content from a specific document"
        )
        
        if file_to_remove and st.button("🗑️ Remove Document", type="secondary"):
            try:
                removed_count = rag_service.remove_document(file_to_remove)
                if removed_count > 0:
                    st.success(f"Removed {removed_count} document chunks from '{file_to_remove}'")
                else:
                    st.warning(f"No documents found for '{file_to_remove}'")
            except Exception as e:
                st.error(f"Error removing document: {e}")
    
    with tab2:
        st.subheader("📊 Document Analysis")
        
        analyze_file = st.text_input(
            "Analyze Document",
            placeholder="Enter filename to analyze",
            help="Get detailed analysis of a processed document"
        )
        
        if analyze_file and st.button("📊 Analyze"):
            try:
                summary = rag_service.get_document_summary(analyze_file)
                
                if summary.get("found"):
                    display_document_analysis(summary)
                else:
                    st.warning(f"Document '{analyze_file}' not found in the database")
                    
            except Exception as e:
                st.error(f"Error analyzing document: {e}")

def display_document_analysis(summary: Dict[str, Any]):
    """Display detailed document analysis."""
    
    st.success(f"Analysis for: {summary['file_name']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", summary.get("total_documents", 0))
    with col2:
        st.metric("Pages Covered", summary.get("pages_covered", 0))
    with col3:
        st.metric("Content Types", len(summary.get("content_types", {})))
    with col4:
        st.metric("Table Types", len(summary.get("table_types", {})))
    
    # Content breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Content Types")
        content_types = summary.get("content_types", {})
        if content_types:
            df_content = pd.DataFrame(list(content_types.items()), 
                                    columns=["Type", "Count"])
            st.bar_chart(df_content.set_index("Type"))
        else:
            st.write("No content type data available")
    
    with col2:
        st.subheader("Table Types")
        table_types = summary.get("table_types", {})
        if table_types:
            df_tables = pd.DataFrame(list(table_types.items()), 
                                   columns=["Type", "Count"])
            st.bar_chart(df_tables.set_index("Type"))
        else:
            st.write("No table type data available")
    
    # Page coverage
    pages = summary.get("page_numbers", [])
    if pages:
        st.subheader("Page Coverage")
        st.write(f"Pages with content: {', '.join(map(str, sorted(pages)))}")

def show_status_page(chat_service: ChatService):
    """Display system status and health information."""
    
    st.header("🔧 System Status")
    
    # System health
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Status", type="primary"):
            st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    try:
        health = chat_service.health_check()
        
        # Overall status
        if health["chat_service_healthy"]:
            st.markdown('<div class="success-box">✅ System Status: All services operational</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ System Status: Issues detected</div>', 
                       unsafe_allow_html=True)
        
        # Component status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🗄️ Vector Database (Qdrant)")
            rag_health = health["rag_service"]
            
            if rag_health["qdrant_healthy"]:
                st.success("✅ Qdrant is healthy")
            else:
                st.error("❌ Qdrant connection issues")
            
            # Collection info
            collection_info = rag_health.get("collection_info", {})
            if collection_info:
                st.write("**Collection Information:**")
                st.write(f"• Name: {collection_info.get('name', 'Unknown')}")
                st.write(f"• Documents: {collection_info.get('points_count', 0)}")
                st.write(f"• Vector Size: {collection_info.get('vector_size', 0)}")
                st.write(f"• Distance: {collection_info.get('distance', 'Unknown')}")
        
        with col2:
            st.subheader("🤖 LLM Service")
            llm_health = health["llm_service"]
            
            if llm_health["service_healthy"]:
                st.success("✅ LLM service is healthy")
            else:
                st.error("❌ LLM service issues")
            
            st.write("**Service Information:**")
            st.write(f"• Model: {llm_health.get('model_name', 'Unknown')}")
            st.write(f"• Provider: {llm_health.get('provider', 'Unknown')}")
            
            # Embedding info
            embedding_model = rag_health.get("embedding_model", "Unknown")
            st.write(f"• Embedding Model: {embedding_model}")
    
    except Exception as e:
        st.error(f"Could not fetch system status: {e}")
    
    # Additional system information
    with st.expander("📋 System Information"):
        st.write("**Environment Information:**")
        st.write(f"• Python: {os.sys.version}")
        st.write(f"• Streamlit: {st.__version__}")
        st.write(f"• Working Directory: {os.getcwd()}")
        st.write(f"• Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()