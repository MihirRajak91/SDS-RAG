"""
Health check endpoints.
"""
from fastapi import APIRouter, HTTPException
from ...models import ApiResponse, success_response, error_response, HealthData
from ...services.rag_service import RAGService
from ...services.chat_service import ChatService

router = APIRouter()

# Initialize services for health checks
rag_service = RAGService()
chat_service = ChatService()


@router.get("/health", response_model=ApiResponse)
async def health_check():
    """
    Comprehensive health check for all services.
    
    Returns:
        ApiResponse: Overall health status and individual service health
    """
    try:
        # Check RAG service health
        rag_health = rag_service.health_check()
        
        # Check chat service health
        chat_health = chat_service.health_check()
        
        # Determine overall status
        overall_healthy = (
            rag_health.get("qdrant_healthy", False) and
            chat_health.get("chat_service_healthy", False)
        )
        
        health_data = HealthData(
            services={
                "rag_service": rag_health,
                "chat_service": chat_health
            },
            overall_healthy=overall_healthy
        )
        
        message = "All services are healthy" if overall_healthy else "Some services are unhealthy"
        
        return success_response(
            data=health_data,
            message=message
        )
        
    except Exception as e:
        return error_response(
            message="Health check failed",
            errors=[str(e)]
        )


@router.get("/health/rag", response_model=ApiResponse)
async def rag_health_check():
    """
    RAG service specific health check.
    
    Returns:
        ApiResponse: RAG service health details
    """
    try:
        health_info = rag_service.health_check()
        is_healthy = health_info.get("qdrant_healthy", False)
        message = "RAG service is healthy" if is_healthy else "RAG service is unhealthy"
        
        return success_response(
            data=health_info,
            message=message
        )
    except Exception as e:
        return error_response(
            message="RAG service health check failed",
            errors=[str(e)]
        )


@router.get("/health/chat", response_model=ApiResponse)
async def chat_health_check():
    """
    Chat service specific health check.
    
    Returns:
        ApiResponse: Chat service health details
    """
    try:
        health_info = chat_service.health_check()
        is_healthy = health_info.get("chat_service_healthy", False)
        message = "Chat service is healthy" if is_healthy else "Chat service is unhealthy"
        
        return success_response(
            data=health_info,
            message=message
        )
    except Exception as e:
        return error_response(
            message="Chat service health check failed",
            errors=[str(e)]
        )