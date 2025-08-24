"""
Chat and conversational AI endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ...models import (
    ApiResponse, success_response, error_response,
    ChatRequest, ChatData, DocumentOverviewData, ComparePeriodsRequest, ComparePeriodsData,
    TableQueryRequest, SearchResultData
)
from ...services.chat_service import ChatService

router = APIRouter()

# Initialize chat service
chat_service = ChatService()


@router.post("/chat", response_model=ApiResponse)
async def chat_query(request: ChatRequest):
    """
    Process a chat query using RAG pipeline.
    
    Args:
        request: Chat request with query and filters
    
    Returns:
        ApiResponse: AI-generated response with context
    """
    try:
        result = chat_service.chat(
            user_query=request.query,
            num_results=request.num_results,
            content_type=request.content_type,
            table_type=request.table_type,
            source_file=request.source_file,
            min_confidence=request.min_confidence
        )
        
        if not result["success"]:
            return error_response(
                message="Chat query failed",
                errors=[result.get("error", "Unknown error occurred")]
            )
        
        # Convert context documents to API format
        context_docs = []
        for doc in result.get("context_documents", []):
            context_doc = SearchResultData(
                id=doc.get("id", ""),
                content=doc.get("content", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {})
            )
            context_docs.append(context_doc)
        
        chat_data = ChatData(
            query=result["query"],
            response=result["response"],
            sources_found=result["sources_found"],
            context_documents=context_docs,
            filters_applied=result.get("filters_applied", {})
        )
        
        return success_response(
            data=chat_data,
            message=f"Generated response for query: '{request.query}'"
        )
        
    except Exception as e:
        return error_response(
            message="Chat query failed",
            errors=[str(e)]
        )


@router.post("/chat/suggestions", response_model=ApiResponse)
async def chat_with_suggestions(request: ChatRequest):
    """
    Process a chat query and provide follow-up suggestions.
    
    Args:
        request: Chat request with query and filters
    
    Returns:
        ApiResponse: AI response with follow-up suggestions
    """
    try:
        result = chat_service.chat_with_suggestions(
            user_query=request.query,
            num_results=request.num_results,
            content_type=request.content_type,
            table_type=request.table_type,
            source_file=request.source_file,
            min_confidence=request.min_confidence
        )
        
        if not result["success"]:
            return error_response(
                message="Chat query with suggestions failed",
                errors=[result.get("error", "Unknown error occurred")]
            )
        
        # Convert context documents to API format
        context_docs = []
        for doc in result.get("context_documents", []):
            context_doc = SearchResultData(
                id=doc.get("id", ""),
                content=doc.get("content", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {})
            )
            context_docs.append(context_doc)
        
        chat_data = ChatData(
            query=result["query"],
            response=result["response"],
            sources_found=result["sources_found"],
            context_documents=context_docs,
            filters_applied=result.get("filters_applied", {}),
            follow_up_suggestions=result.get("follow_up_suggestions")
        )
        
        return success_response(
            data=chat_data,
            message=f"Generated response with suggestions for query: '{request.query}'"
        )
        
    except Exception as e:
        return error_response(
            message="Chat query with suggestions failed",
            errors=[str(e)]
        )


@router.get("/chat/document/{document_name}/overview", response_model=ApiResponse)
async def get_document_overview(document_name: str):
    """
    Get AI-generated overview of a financial document.
    
    Args:
        document_name: Name of the document to summarize
    
    Returns:
        ApiResponse: AI-generated document overview
    """
    try:
        result = chat_service.get_document_overview(document_name)
        
        if not result.get("found", False):
            return error_response(
                message="Document not found",
                errors=[f"Document '{document_name}' not found in the database"]
            )
        
        if not result.get("success", False):
            return error_response(
                message="Failed to generate document overview",
                errors=[result.get("error", "Unknown error occurred")]
            )
        
        overview_data = DocumentOverviewData(
            document_name=result["document_name"],
            found=result["found"],
            ai_summary=result.get("ai_summary"),
            document_stats=result.get("document_stats")
        )
        
        return success_response(
            data=overview_data,
            message=f"Generated overview for document: '{document_name}'"
        )
        
    except Exception as e:
        return error_response(
            message="Document overview generation failed",
            errors=[str(e)]
        )


@router.post("/chat/tables", response_model=ApiResponse)
async def query_tables(request: TableQueryRequest):
    """
    Ask questions specifically about tables.
    
    Args:
        request: Table-focused query request
    
    Returns:
        ApiResponse: AI response focused on table data
    """
    try:
        result = chat_service.ask_about_tables(
            user_query=request.query,
            table_type=request.table_type
        )
        
        if not result["success"]:
            return error_response(
                message="Table query failed",
                errors=[result.get("error", "Unknown error occurred")]
            )
        
        # Convert context documents to API format
        context_docs = []
        for doc in result.get("context_documents", []):
            context_doc = SearchResultData(
                id=doc.get("id", ""),
                content=doc.get("content", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {})
            )
            context_docs.append(context_doc)
        
        chat_data = ChatData(
            query=result["query"],
            response=result["response"],
            sources_found=result["sources_found"],
            context_documents=context_docs,
            filters_applied=result.get("filters_applied", {})
        )
        
        return success_response(
            data=chat_data,
            message=f"Generated table-focused response for query: '{request.query}'"
        )
        
    except Exception as e:
        return error_response(
            message="Table query failed",
            errors=[str(e)]
        )


@router.post("/chat/compare", response_model=ApiResponse)
async def compare_periods(request: ComparePeriodsRequest):
    """
    Compare data across multiple time periods/documents.
    
    Args:
        request: Period comparison request
    
    Returns:
        ApiResponse: Comparative analysis response
    """
    try:
        result = chat_service.compare_periods(
            user_query=request.query,
            source_files=request.source_files
        )
        
        if not result["success"]:
            return error_response(
                message="Period comparison failed",
                errors=[result.get("error", "Unknown error occurred")]
            )
        
        # Convert context documents to API format
        context_docs = []
        for doc in result.get("context_documents", []):
            context_doc = SearchResultData(
                id=doc.get("id", ""),
                content=doc.get("content", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {})
            )
            context_docs.append(context_doc)
        
        compare_data = ComparePeriodsData(
            query=result["query"],
            response=result["response"],
            sources_compared=result["sources_compared"],
            total_sources_found=result["total_sources_found"],
            context_documents=context_docs
        )
        
        return success_response(
            data=compare_data,
            message=f"Generated comparison for: '{request.query}'"
        )
        
    except Exception as e:
        return error_response(
            message="Period comparison failed",
            errors=[str(e)]
        )


@router.get("/chat/suggestions/financial", response_model=List[str])
async def get_financial_suggestions():
    """
    Get common financial analysis question suggestions.
    
    Returns:
        List[str]: List of suggested financial questions
    """
    suggestions = [
        "What was the revenue growth in the last quarter?",
        "Show me the key expense categories and their trends",
        "What are the main assets and liabilities?",
        "How has the debt-to-equity ratio changed?",
        "What was the operating cash flow?",
        "Analyze the profitability margins",
        "What are the main risk factors mentioned?",
        "Compare current performance to previous periods",
        "What capital investments were made?",
        "Summarize the key financial highlights"
    ]
    
    return suggestions