
"""
Upload Page for SDS-RAG Streamlit Application
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the RAG services and configuration
from src.sds_rag.services.rag_service import RAGService
from src.sds_rag.utils import validate_pdf_file, find_pdf_files
from sds_rag.config import config

def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

# Page configuration
st.set_page_config(
    page_title="Upload - SDS-RAG",
    page_icon="📄",
    layout="wide"
)

# Initialize services
@st.cache_resource
def get_rag_service():
    """Initialize and cache RAG service."""
    return RAGService()

def process_single_file(uploaded_file, rag_service: RAGService):
    """Process a single uploaded file."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save temporary file
        status_text.text("💾 Saving file...")
        progress_bar.progress(10)
        
        temp_path = Path(f"temp/{uploaded_file.name}")
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Validate file
        status_text.text("✅ Validating file...")
        progress_bar.progress(30)
        
        if not validate_pdf_file(temp_path):
            st.error("❌ Invalid PDF file")
            return
        
        # Process document
        status_text.text("🔄 Processing document...")
        progress_bar.progress(50)
        
        result = rag_service.process_and_store_document(str(temp_path))
        progress_bar.progress(100)
        
        # Cleanup
        temp_path.unlink()
        status_text.text("✅ Processing complete!")
        
        # Show results
        st.success(f"""
        ✅ **Document processed successfully!**
        
        📊 **Statistics:**
        - **Tables processed:** {result.get('tables_processed', 0)}
        - **Text chunks processed:** {result.get('text_chunks_processed', 0)}
        - **Embedded documents created:** {result.get('embedded_documents_created', 0)}
        - **Vector points stored:** {result.get('vector_points_stored', 0)}
        - **High confidence tables:** {result.get('high_confidence_tables', 0)}
        - **Average confidence:** {result.get('average_confidence', 0):.2f}
        - **Success rate:** {result.get('success_rate', 0):.2f}%
        - **Total pages:** {result.get('total_pages', 0)}
        - **File size:** {format_file_size(len(uploaded_file.getvalue()))}
        """)
        
        # Show processing details
        if result.get('processing_successful', False):
            st.success(f"Document '{result.get('file_name', 'Unknown')}' has been successfully processed and stored in the vector database.")
        else:
            st.warning("Document processing completed but some issues were encountered.")
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        if temp_path.exists():
            temp_path.unlink()
    finally:
        progress_bar.empty()
        status_text.empty()

def process_batch_files(directory_path: str, rag_service: RAGService):
    """Process multiple files from a directory."""
    
    pdf_files = find_pdf_files(directory_path)
    if not pdf_files:
        st.warning("No PDF files found in directory")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    processed_files = 0
    total_chunks = 0
    failed_files = []
    
    try:
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"🔄 Processing: {pdf_file.name} ({i+1}/{len(pdf_files)})")
            progress_bar.progress((i + 1) / len(pdf_files))
            
            try:
                result = rag_service.process_and_store_document(str(pdf_file))
                processed_files += 1
                total_chunks += result.get('vector_points_stored', 0)
                
                with results_container:
                    st.success(f"✅ {pdf_file.name} - {result.get('vector_points_stored', 0)} vectors stored")
                    
            except Exception as e:
                failed_files.append((pdf_file.name, str(e)))
                with results_container:
                    st.error(f"❌ {pdf_file.name} - Failed: {str(e)}")
        
        # Final summary
        status_text.text("✅ Batch processing complete!")
        
        st.success(f"""
        🎉 **Batch processing summary:**
        - **Files processed:** {processed_files}/{len(pdf_files)}
        - **Total vectors stored:** {total_chunks}
        - **Failed files:** {len(failed_files)}
        """)
        
        if failed_files:
            with st.expander("❌ Failed files details"):
                for filename, error in failed_files:
                    st.write(f"**{filename}:** {error}")
                    
    except Exception as e:
        st.error(f"❌ Batch processing error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def main():
    """Main upload page function."""
    st.header("📄 Document Upload")
    
    # Initialize RAG service
    rag_service = get_rag_service()
    
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
            st.info(f"📁 **File:** {uploaded_file.name} ({format_file_size(file_size)})")
            
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

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home"):
            st.switch_page("app.py")
    with col2:
        if st.button("🔍 Search"):
            st.switch_page("pages/search.py")
    with col3:
        if st.button("💬 Chat"):
            st.switch_page("pages/chat.py")

if __name__ == "__main__":
    main()