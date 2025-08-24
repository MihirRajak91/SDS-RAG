"""
Vector storage and database management endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from ...models import ApiResponse, success_response, error_response
from ...services.vector_storage_service import VectorStorageService

router = APIRouter()

# Initialize vector storage service
vector_storage = VectorStorageService()


@router.get("/vectors/health", response_model=ApiResponse)
async def vector_health_check():
    """
    Check health of vector database.
    
    Returns:
        dict: Vector database health status
    """
    try:
        is_healthy = vector_storage.health_check()
        
        health_data = {
            "healthy": is_healthy,
            "service": "qdrant"
        }
        
        message = "Vector database is healthy" if is_healthy else "Vector database is unhealthy"
        
        return success_response(
            data=health_data,
            message=message
        )
        
    except Exception as e:
        return error_response(
            message="Vector database health check failed",
            errors=[str(e)]
        )


@router.get("/vectors/collection/info", response_model=dict)
async def get_collection_info():
    """
    Get information about the vector collection.
    
    Returns:
        dict: Collection statistics and configuration
    """
    try:
        info = vector_storage.get_collection_info()
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


@router.get("/vectors/documents/by-source/{source_file}", response_model=ApiResponse)
async def get_documents_by_source(source_file: str):
    """
    Get all vector documents from a specific source file.
    
    Args:
        source_file: Name of the source file
    
    Returns:
        List[Dict[str, Any]]: Documents from the source file
    """
    try:
        documents = vector_storage.get_documents_by_source(source_file)
        return success_response(
            data=documents,
            message=f"Retrieved {len(documents)} documents from source: {source_file}"
        )
        
    except Exception as e:
        return error_response(
            message="Failed to get documents by source",
            errors=[str(e)]
        )


@router.delete("/vectors/documents/by-source/{source_file}", response_model=dict)
async def delete_documents_by_source(source_file: str):
    """
    Delete all vector documents from a specific source file.
    
    Args:
        source_file: Name of the source file
    
    Returns:
        dict: Deletion results
    """
    try:
        deleted_count = vector_storage.delete_by_source(source_file)
        
        return {
            "source_file": source_file,
            "documents_deleted": deleted_count,
            "success": deleted_count > 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")


@router.get("/vectors/search/raw", response_model=List[Dict[str, Any]])
async def raw_similarity_search(
    query: str,
    limit: int = 10,
    content_type: str = None,
    table_type: str = None,
    source_file: str = None,
    min_confidence: float = None
):
    """
    Perform raw similarity search in vector database.
    
    Args:
        query: Search query
        limit: Maximum number of results
        content_type: Filter by content type
        table_type: Filter by table type
        source_file: Filter by source file
        min_confidence: Minimum confidence threshold
    
    Returns:
        List[Dict[str, Any]]: Raw search results with vectors
    """
    try:
        results = vector_storage.similarity_search(
            query=query,
            limit=limit,
            content_type=content_type,
            table_type=table_type,
            source=source_file,
            min_confidence=min_confidence
        )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Raw similarity search failed: {str(e)}")


@router.get("/vectors/stats", response_model=dict)
async def get_vector_database_stats():
    """
    Get comprehensive statistics about the vector database.
    
    Returns:
        dict: Vector database statistics
    """
    try:
        # Get collection info
        collection_info = vector_storage.get_collection_info()
        
        # Get health status
        is_healthy = vector_storage.health_check()
        
        return {
            "collection_info": collection_info,
            "health_status": {
                "healthy": is_healthy,
                "status": "healthy" if is_healthy else "unhealthy"
            },
            "embedding_model": vector_storage.embedding_service.model_name,
            "vector_dimensions": collection_info.get("vector_size", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get vector database stats: {str(e)}")


@router.post("/vectors/recreate-collection", response_model=dict)
async def recreate_collection():
    """
    Recreate the vector collection (WARNING: This will delete all data).
    
    Returns:
        dict: Recreation results
    """
    try:
        # Delete existing collection
        try:
            vector_storage.client.delete_collection(vector_storage.collection_name)
        except:
            pass  # Collection might not exist
        
        # Recreate collection
        vector_storage._ensure_collection_exists()
        
        return {
            "success": True,
            "message": f"Collection '{vector_storage.collection_name}' recreated successfully",
            "warning": "All previous data has been deleted"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recreate collection: {str(e)}")


@router.get("/vectors/collections", response_model=List[str])
async def list_collections():
    """
    List all available vector collections.
    
    Returns:
        List[str]: Names of all collections
    """
    try:
        collections = vector_storage.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        return collection_names
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")