"""
Custom exceptions and error handlers for the API.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

from .response import error_response, ResponseStatus

logger = logging.getLogger(__name__)


class DocumentNotFoundError(Exception):
    """Raised when a document is not found."""
    pass


class ProcessingError(Exception):
    """Raised when document processing fails."""
    pass


class VectorSearchError(Exception):
    """Raised when vector search fails."""
    pass


class ServiceUnavailableError(Exception):
    """Raised when a required service is unavailable."""
    pass


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error for {request.url}: {exc.errors()}")
    
    # Extract error messages from validation errors
    error_messages = []
    for error in exc.errors():
        field = " -> ".join([str(loc) for loc in error["loc"]])
        message = f"{field}: {error['msg']}"
        error_messages.append(message)
    
    response = error_response(
        message="Validation failed",
        errors=error_messages,
        metadata={
            "path": str(request.url),
            "validation_details": exc.errors()
        }
    )
    
    return JSONResponse(
        status_code=422,
        content=response.dict()
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception for {request.url}: {str(exc)}", exc_info=True)
    
    response = error_response(
        message="An unexpected error occurred",
        errors=[str(exc)],
        metadata={
            "path": str(request.url),
            "exception_type": type(exc).__name__
        }
    )
    
    return JSONResponse(
        status_code=500,
        content=response.dict()
    )


async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    """Handle document not found errors."""
    logger.warning(f"Document not found for {request.url}: {str(exc)}")
    
    response = error_response(
        message="Document not found",
        errors=[str(exc)],
        metadata={
            "path": str(request.url)
        }
    )
    
    return JSONResponse(
        status_code=404,
        content=response.dict()
    )


async def processing_error_handler(request: Request, exc: ProcessingError):
    """Handle document processing errors."""
    logger.error(f"Processing error for {request.url}: {str(exc)}")
    
    response = error_response(
        message="Document processing failed",
        errors=[str(exc)],
        metadata={
            "path": str(request.url)
        }
    )
    
    return JSONResponse(
        status_code=422,
        content=response.dict()
    )


async def vector_search_error_handler(request: Request, exc: VectorSearchError):
    """Handle vector search errors."""
    logger.error(f"Vector search error for {request.url}: {str(exc)}")
    
    response = error_response(
        message="Search operation failed",
        errors=[str(exc)],
        metadata={
            "path": str(request.url)
        }
    )
    
    return JSONResponse(
        status_code=500,
        content=response.dict()
    )


async def service_unavailable_handler(request: Request, exc: ServiceUnavailableError):
    """Handle service unavailable errors."""
    logger.error(f"Service unavailable for {request.url}: {str(exc)}")
    
    response = error_response(
        message="Service temporarily unavailable",
        errors=[str(exc)],
        metadata={
            "path": str(request.url)
        }
    )
    
    return JSONResponse(
        status_code=503,
        content=response.dict()
    )