"""
Chat Service - Complete RAG chatbot combining retrieval and generation.

Orchestrates the complete RAG pipeline: query → vector search → LLM generation
to provide intelligent responses about financial documents.
"""

import logging
from typing import List, Dict, Any, Optional

from src.sds_rag.services.rag_service import RAGService
from src.sds_rag.services.llm_service import LLMService
from src.sds_rag.utils import StructuredLogger, Timer, log_performance, validate_search_query

logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(__name__)


class ChatService:
    """Complete RAG chatbot for financial document Q&A."""
    
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize chat service with RAG and LLM components.
        
        Args:
            google_api_key (str, optional): Google AI API key for Gemini
            qdrant_host (str): Qdrant server host
            qdrant_port (int): Qdrant server port
            embedding_model (str): HuggingFace embedding model
        """
        with Timer("Initializing chat service components") as timer:
            self.rag_service = RAGService(
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                embedding_model=embedding_model
            )
            self.llm_service = LLMService(api_key=google_api_key)
        
        structured_logger.log_component_init(
            component="ChatService",
            config={
                "qdrant_host": qdrant_host,
                "qdrant_port": qdrant_port,
                "embedding_model": embedding_model,
                "init_time": timer.elapsed_human
            }
        )
        logger.info("Chat service initialized with RAG and LLM components")
    
    @log_performance
    def chat(
        self,
        user_query: str,
        num_results: int = 5,
        content_type: Optional[str] = None,
        table_type: Optional[str] = None,
        source_file: Optional[str] = None,
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """
        Process user query and generate response using RAG pipeline.
        
        Args:
            user_query (str): User's question about financial data
            num_results (int): Number of documents to retrieve for context
            content_type (str, optional): Filter by content type
            table_type (str, optional): Filter by table type
            source_file (str, optional): Filter by source document
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            Dict[str, Any]: Complete response with answer and context
        """
        # Validate query
        is_valid, validation_errors = validate_search_query(user_query)
        if not is_valid:
            return {
                "query": user_query,
                "response": f"Invalid query: {', '.join(validation_errors)}",
                "sources_found": 0,
                "context_documents": [],
                "success": False,
                "error": "Query validation failed"
            }
        
        try:
            with Timer(f"Processing chat query") as timer:
                structured_logger.log_chat_query(
                    query=user_query[:100],
                    filters={
                        "content_type": content_type,
                        "table_type": table_type,
                        "source_file": source_file,
                        "min_confidence": min_confidence
                    }
                )
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.rag_service.search_financial_data(
                query=user_query,
                limit=num_results,
                content_type=content_type,
                table_type=table_type,
                source_file=source_file,
                min_confidence=min_confidence
            )
            
            # Step 2: Generate response using LLM
            if retrieved_docs:
                response_text = self.llm_service.generate_financial_response(
                    user_query=user_query,
                    context_documents=retrieved_docs,
                    document_name=source_file
                )
            else:
                response_text = self._generate_no_context_response(user_query)
            
                # Step 3: Format complete response
                response = {
                    "query": user_query,
                    "response": response_text,
                    "sources_found": len(retrieved_docs),
                    "context_documents": retrieved_docs,
                    "filters_applied": {
                        "content_type": content_type,
                        "table_type": table_type,
                        "source_file": source_file,
                        "min_confidence": min_confidence
                    },
                    "processing_time": timer.elapsed_human,
                    "success": True
                }
                
                structured_logger.log_chat_response(
                    query=user_query[:100],
                    sources_found=len(retrieved_docs),
                    response_length=len(response_text),
                    processing_time=timer.elapsed_human,
                    success=True
                )
                
                return response
            
        except Exception as e:
            structured_logger.log_chat_response(
                query=user_query[:100],
                sources_found=0,
                response_length=0,
                processing_time="0s",
                success=False,
                error=str(e)
            )
            logger.error(f"Error in chat processing: {e}")
            return {
                "query": user_query,
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources_found": 0,
                "context_documents": [],
                "success": False,
                "error": str(e)
            }
    
    def chat_with_suggestions(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """
        Process query and provide follow-up suggestions.
        
        Args:
            user_query (str): User's question
            **kwargs: Additional arguments for chat method
            
        Returns:
            Dict[str, Any]: Response with follow-up suggestions
        """
        response = self.chat(user_query, **kwargs)
        
        if response["success"] and response["sources_found"] > 0:
            # Generate follow-up suggestions based on context
            suggestions = self._generate_follow_up_suggestions(
                user_query, 
                response["context_documents"]
            )
            response["follow_up_suggestions"] = suggestions
        
        return response
    
    def get_document_overview(self, document_name: str) -> Dict[str, Any]:
        """
        Get AI-generated overview of a financial document.
        
        Args:
            document_name (str): Name of the document to summarize
            
        Returns:
            Dict[str, Any]: Document overview and summary
        """
        try:
            # Get all documents from the file
            documents = self.rag_service.get_document_summary(document_name)
            
            if not documents["found"]:
                return {
                    "document_name": document_name,
                    "found": False,
                    "message": "Document not found in the database."
                }
            
            # Get sample content for summary
            all_docs = self.rag_service.vector_storage.get_documents_by_source(document_name)
            
            # Generate AI summary
            summary = self.llm_service.generate_summary_response(all_docs, document_name)
            
            return {
                "document_name": document_name,
                "found": True,
                "ai_summary": summary,
                "document_stats": documents,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating document overview: {e}")
            return {
                "document_name": document_name,
                "found": False,
                "success": False,
                "error": str(e)
            }
    
    def ask_about_tables(self, user_query: str, table_type: str = None) -> Dict[str, Any]:
        """
        Specialized method for table-focused queries.
        
        Args:
            user_query (str): Question about tables
            table_type (str, optional): Specific table type to focus on
            
        Returns:
            Dict[str, Any]: Table-focused response
        """
        return self.chat(
            user_query=user_query,
            num_results=8,
            content_type="table_summary",  # Focus on table summaries
            table_type=table_type,
            min_confidence=0.7
        )
    
    def compare_periods(self, user_query: str, source_files: List[str]) -> Dict[str, Any]:
        """
        Compare data across multiple time periods/documents.
        
        Args:
            user_query (str): Comparison question
            source_files (List[str]): Documents to compare
            
        Returns:
            Dict[str, Any]: Comparative analysis response
        """
        try:
            all_results = []
            
            # Search each document
            for source_file in source_files:
                results = self.rag_service.search_financial_data(
                    query=user_query,
                    limit=3,
                    source_file=source_file,
                    min_confidence=0.6
                )
                
                # Add source identifier to each result
                for result in results:
                    result["source_document"] = source_file
                
                all_results.extend(results)
            
            if all_results:
                # Create comparative prompt
                comparative_query = f"Compare the following data across different periods: {user_query}"
                response_text = self.llm_service.generate_financial_response(
                    user_query=comparative_query,
                    context_documents=all_results
                )
            else:
                response_text = "I couldn't find comparable data across the specified documents for your query."
            
            return {
                "query": user_query,
                "response": response_text,
                "sources_compared": source_files,
                "total_sources_found": len(all_results),
                "context_documents": all_results,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in period comparison: {e}")
            return {
                "query": user_query,
                "response": f"Error performing comparison: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _generate_no_context_response(self, user_query: str) -> str:
        """Generate response when no context documents are found."""
        return f"""I couldn't find specific information in the financial documents to answer your question about "{user_query}". 

This could be because:
- The information isn't available in the processed documents
- Your query might need to be more specific
- The confidence threshold filtered out potentially relevant results

Try:
- Rephrasing your question with more specific financial terms
- Checking if the relevant documents have been uploaded and processed
- Using broader search terms initially"""
    
    def _generate_follow_up_suggestions(
        self, 
        original_query: str, 
        context_docs: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate contextual follow-up question suggestions."""
        suggestions = []
        
        # Analyze available content types and table types
        content_types = set()
        table_types = set()
        
        for doc in context_docs:
            metadata = doc.get("metadata", {})
            if metadata.get("content_type"):
                content_types.add(metadata["content_type"])
            if metadata.get("table_type"):
                table_types.add(metadata["table_type"])
        
        # Generate suggestions based on available data
        if "table_summary" in content_types:
            if "income_statement" in table_types:
                suggestions.append("What was the revenue growth trend?")
                suggestions.append("Show me the expense breakdown")
            if "balance_sheet" in table_types:
                suggestions.append("What are the key assets and liabilities?")
                suggestions.append("How has the debt-to-equity ratio changed?")
            if "cash_flow" in table_types:
                suggestions.append("What was the operating cash flow?")
                suggestions.append("How much was spent on capital expenditures?")
        
        # Add generic financial analysis suggestions
        suggestions.extend([
            "Provide a summary of key financial highlights",
            "What trends do you see in the financial data?",
            "Are there any concerning financial indicators?"
        ])
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all chat service components.
        
        Returns:
            Dict[str, Any]: Complete health status
        """
        rag_health = self.rag_service.health_check()
        llm_health = self.llm_service.health_check()
        
        return {
            "chat_service_healthy": rag_health["qdrant_healthy"] and llm_health["service_healthy"],
            "rag_service": rag_health,
            "llm_service": llm_health
        }