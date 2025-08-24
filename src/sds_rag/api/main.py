"""
FastAPI application for SDS-RAG API endpoints.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from .routers import documents, rag, chat, health, vector_storage
from ..models import (
    validation_exception_handler, general_exception_handler,
    document_not_found_handler, processing_error_handler,
    vector_search_error_handler, service_unavailable_handler,
    DocumentNotFoundError, ProcessingError, VectorSearchError, ServiceUnavailableError
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="SDS-RAG API",
    description="Simple Document Search with Retrieval Augmented Generation API",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(vector_storage.router, prefix="/api/v1", tags=["vector-storage"])

# Add exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(DocumentNotFoundError, document_not_found_handler)
app.add_exception_handler(ProcessingError, processing_error_handler)
app.add_exception_handler(VectorSearchError, vector_search_error_handler)
app.add_exception_handler(ServiceUnavailableError, service_unavailable_handler)
app.add_exception_handler(Exception, general_exception_handler)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SDS-RAG API",
        "version": "0.1.0",
        "docs": "/docs"
    }