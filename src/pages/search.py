"""
Search Page for SDS-RAG Streamlit Application
"""

import streamlit as st
from typing import Dict, Any

# Import the RAG services and configuration
from src.sds_rag.services.rag_service import RAGService
from sds_rag.config import config

# Initialize services
@st.cache_resource
def get_rag_service():
    """Initialize and cache RAG service."""
    return RAGService()

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

def main():
    """Main search page function."""
    st.header("🔍 Search Financial Documents")
    
    # Initialize RAG service
    rag_service = get_rag_service()
    
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

    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home"):
            st.switch_page("app.py")
    with col2:
        if st.button("📄 Upload"):
            st.switch_page("pages/upload.py")
    with col3:
        if st.button("💬 Chat"):
            st.switch_page("pages/chat.py")

if __name__ == "__main__":
    main()