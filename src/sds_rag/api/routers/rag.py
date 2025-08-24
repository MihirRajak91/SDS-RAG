"""
RAG (Retrieval Augmented Generation) endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ...models import (
    ApiResponse, success_response, error_response,
    SearchRequest, SearchData, SearchResultData
)
from ...services.rag_service import RAGService

router = APIRouter()

# Initialize RAG service
rag_service = RAGService()


@router.post("/search", response_model=ApiResponse)
async def search_documents(request: SearchRequest):
    """
    Search financial documents using semantic similarity.
    
    Args:
        request: Search request with query and filters
    
    Returns:
        ApiResponse: Search results with relevance scores
    """
    try:
        results = rag_service.search_financial_data(
            query=request.query,
            limit=request.limit,
            content_type=request.content_type,
            table_type=request.table_type,
            source_file=request.source_file,
            min_confidence=request.min_confidence
        )
        
        # Convert results to API format
        search_results = []
        for result in results:
            search_result = SearchResultData(
                id=result.get("id", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {})
            )
            search_results.append(search_result)
        
        search_data = SearchData(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            filters_applied={
                "content_type": request.content_type,
                "table_type": request.table_type,
                "source_file": request.source_file,
                "min_confidence": request.min_confidence,
                "limit": request.limit
            }
        )
        
        message = f"Found {len(search_results)} results for query: '{request.query}'"
        
        return success_response(
            data=search_data,
            message=message
        )
        
    except Exception as e:
        return error_response(
            message="Search operation failed",
            errors=[str(e)]
        )


@router.get("/search/tables", response_model=ApiResponse)
async def search_tables(
    query: str,
    limit: int = 10,
    table_type: str = None,
    source_file: str = None,
    min_confidence: float = 0.7
):
    """
    Search specifically for table content.
    
    Args:
        query: Search query
        limit: Maximum number of results
        table_type: Filter by table type
        source_file: Filter by source file
        min_confidence: Minimum confidence threshold
    
    Returns:
        SearchResponse: Table search results
    """
    try:
        results = rag_service.search_financial_data(
            query=query,
            limit=limit,
            content_type="table_summary",  # Focus on table summaries
            table_type=table_type,
            source_file=source_file,
            min_confidence=min_confidence
        )
        
        # Convert results to API format
        search_results = []
        for result in results:
            search_result = SearchResultData(
                id=result.get("id", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {})
            )
            search_results.append(search_result)
        
        search_data = SearchData(
            query=query,
            results=search_results,
            total_results=len(search_results),
            filters_applied={
                "content_type": "table_summary",
                "table_type": table_type,
                "source_file": source_file,
                "min_confidence": min_confidence,
                "limit": limit
            }
        )
        
        message = f"Found {len(search_results)} table results for query: '{query}'"
        
        return success_response(
            data=search_data,
            message=message
        )
        
    except Exception as e:
        return error_response(
            message="Table search failed",
            errors=[str(e)]
        )


@router.get("/search/text", response_model=ApiResponse)
async def search_text(
    query: str,
    limit: int = 10,
    source_file: str = None,
    min_confidence: float = 0.6
):
    """
    Search specifically for narrative text content.
    
    Args:
        query: Search query
        limit: Maximum number of results
        source_file: Filter by source file
        min_confidence: Minimum confidence threshold
    
    Returns:
        SearchResponse: Text search results
    """
    try:
        results = rag_service.search_financial_data(
            query=query,
            limit=limit,
            content_type="narrative_text",  # Focus on text content
            source_file=source_file,
            min_confidence=min_confidence
        )
        
        # Convert results to API format
        search_results = []
        for result in results:
            search_result = SearchResultData(
                id=result.get("id", ""),
                content=result.get("content", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {})
            )
            search_results.append(search_result)
        
        search_data = SearchData(
            query=query,
            results=search_results,
            total_results=len(search_results),
            filters_applied={
                "content_type": "narrative_text",
                "source_file": source_file,
                "min_confidence": min_confidence,
                "limit": limit
            }
        )
        
        message = f"Found {len(search_results)} text results for query: '{query}'"
        
        return success_response(
            data=search_data,
            message=message
        )
        
    except Exception as e:
        return error_response(
            message="Text search failed",
            errors=[str(e)]
        )


@router.get("/embeddings/status", response_model=ApiResponse)
async def get_embeddings_status():
    """
    Get status of embeddings and vector database.
    
    Returns:
        dict: Embeddings and vector database status
    """
    try:
        health_info = rag_service.health_check()
        
        status_data = {
            "qdrant_healthy": health_info.get("qdrant_healthy", False),
            "collection_info": health_info.get("collection_info", {}),
            "embedding_model": health_info.get("embedding_model", ""),
            "status": "healthy" if health_info.get("qdrant_healthy", False) else "unhealthy"
        }
        
        is_healthy = health_info.get("qdrant_healthy", False)
        message = "Embeddings service is healthy" if is_healthy else "Embeddings service is unhealthy"
        
        return success_response(
            data=status_data,
            message=message
        )
        
    except Exception as e:
        return error_response(
            message="Failed to get embeddings status",
            errors=[str(e)]
        )