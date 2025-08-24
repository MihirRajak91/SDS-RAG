"""
Models module exports for SDS-RAG system.
"""

# Core domain models
from .schemas import (
    TableType, ExtractionMethod, ConfidenceLevel,
    RawTableData, TableClassificationResult, DataValidationResult,
    ConfidenceMetrics, ProcessedTable, TextChunk, DocumentMetadata,
    ExtractionStatistics, ProcessedDocument
)

# API request and response models
from .api_models import (
    # Request models
    ProcessDocumentRequest, BatchProcessRequest, SearchRequest,
    ChatRequest, DocumentSummaryRequest, ComparePeriodsRequest,
    RemoveDocumentRequest, TableQueryRequest,
    
    # Data models for responses
    ProcessDocumentData, BatchProcessData, SearchResultData,
    SearchData, ChatData, DocumentSummaryData, DocumentOverviewData,
    ComparePeriodsData, RemoveDocumentData, HealthData,
    VectorHealthData, CollectionInfoData, VectorStatsData
)

# Response wrappers
from .response import (
    ResponseStatus, ApiResponse, SuccessResponse, ErrorResponse, WarningResponse,
    success_response, error_response, warning_response
)

# Exceptions and error handlers
from .exceptions import (
    DocumentNotFoundError, ProcessingError, VectorSearchError, ServiceUnavailableError,
    validation_exception_handler, general_exception_handler,
    document_not_found_handler, processing_error_handler,
    vector_search_error_handler, service_unavailable_handler
)

__all__ = [
    # Core domain models
    "TableType", "ExtractionMethod", "ConfidenceLevel",
    "RawTableData", "TableClassificationResult", "DataValidationResult",
    "ConfidenceMetrics", "ProcessedTable", "TextChunk", "DocumentMetadata",
    "ExtractionStatistics", "ProcessedDocument",
    
    # API request models
    "ProcessDocumentRequest", "BatchProcessRequest", "SearchRequest",
    "ChatRequest", "DocumentSummaryRequest", "ComparePeriodsRequest",
    "RemoveDocumentRequest", "TableQueryRequest",
    
    # API data models
    "ProcessDocumentData", "BatchProcessData", "SearchResultData",
    "SearchData", "ChatData", "DocumentSummaryData", "DocumentOverviewData",
    "ComparePeriodsData", "RemoveDocumentData", "HealthData",
    "VectorHealthData", "CollectionInfoData", "VectorStatsData",
    
    # Response wrappers
    "ResponseStatus", "ApiResponse", "SuccessResponse", "ErrorResponse", "WarningResponse",
    "success_response", "error_response", "warning_response",
    
    # Exceptions and error handlers
    "DocumentNotFoundError", "ProcessingError", "VectorSearchError", "ServiceUnavailableError",
    "validation_exception_handler", "general_exception_handler",
    "document_not_found_handler", "processing_error_handler",
    "vector_search_error_handler", "service_unavailable_handler"
]