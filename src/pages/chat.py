"""
Chat Page for SDS-RAG Streamlit Application
"""

import streamlit as st
from typing import Dict, Any

# Import the chat service and configuration
from src.sds_rag.services.chat_service import ChatService
from sds_rag.config import config

# Initialize services
@st.cache_resource
def get_chat_service():
    """Initialize and cache Chat service."""
    return ChatService()

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

def main():
    """Main chat page function."""
    st.header("💬 AI Chat Assistant")
    
    # Initialize chat service
    chat_service = get_chat_service()
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🔧 Chat Settings")
        
        num_results = st.selectbox("Context Documents", [3, 5, 8, 10], index=1)
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.6, 0.1)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
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
        if st.button("🔍 Search"):
            st.switch_page("pages/search.py")

if __name__ == "__main__":
    main()