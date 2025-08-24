"""
API request and response models for SDS-RAG endpoints.
All responses now use the standardized ApiResponse format.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

from .schemas import TableType, ConfidenceLevel


# Request Models 

class ProcessDocumentRequest(BaseModel):
    """Request model for document processing."""
    file_path: str = Field(..., description="Path to the PDF file to process")


class BatchProcessRequest(BaseModel):
    """Request model for batch document processing."""
    directory_path: str = Field(..., description="Path to directory containing PDF files")


class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., min_length=1, description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    content_type: Optional[str] = Field(None, description="Filter by content type")
    table_type: Optional[str] = Field(None, description="Filter by table type")
    source_file: Optional[str] = Field(None, description="Filter by source file")
    min_confidence: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")


class ChatRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., min_length=1, description="User's question")
    num_results: int = Field(5, ge=1, le=20, description="Number of context documents to retrieve")
    content_type: Optional[str] = Field(None, description="Filter by content type")
    table_type: Optional[str] = Field(None, description="Filter by table type")
    source_file: Optional[str] = Field(None, description="Filter by source file")
    min_confidence: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")


class DocumentSummaryRequest(BaseModel):
    """Request model for document summary."""
    file_name: str = Field(..., description="Name of the document to summarize")


class ComparePeriodsRequest(BaseModel):
    """Request model for period comparison."""
    query: str = Field(..., min_length=1, description="Comparison question")
    source_files: List[str] = Field(..., min_items=2, description="Documents to compare")


class RemoveDocumentRequest(BaseModel):
    """Request model for document removal."""
    file_name: str = Field(..., description="Name of the document to remove")


class TableQueryRequest(BaseModel):
    """Request model for table-focused queries."""
    query: str = Field(..., min_length=1, description="Question about tables")
    table_type: Optional[str] = Field(None, description="Specific table type to focus on")


# Data Models (for use within the standardized response data field)

class ProcessDocumentData(BaseModel):
    """Data model for document processing results."""
    file_name: str = Field(..., description="Name of the processed file")
    file_path: str = Field(..., description="Path to the processed file")
    tables_processed: Optional[int] = Field(None, description="Number of tables processed")
    text_chunks_processed: Optional[int] = Field(None, description="Number of text chunks processed")
    embedded_documents_created: Optional[int] = Field(None, description="Number of embedded documents created")
    vector_points_stored: Optional[int] = Field(None, description="Number of vector points stored")
    high_confidence_tables: Optional[int] = Field(None, description="Number of high confidence tables")
    average_confidence: Optional[float] = Field(None, description="Average confidence score")
    success_rate: Optional[float] = Field(None, description="Processing success rate")
    total_pages: Optional[int] = Field(None, description="Total pages in document")
    processing_timestamp: Optional[str] = Field(None, description="Processing timestamp")


class BatchProcessData(BaseModel):
    """Data model for batch processing results."""
    results: List[ProcessDocumentData] = Field(..., description="Processing results for each file")
    total_files: int = Field(..., description="Total number of files processed")
    successful_files: int = Field(..., description="Number of successfully processed files")
    failed_files: int = Field(..., description="Number of failed files")


class SearchResultData(BaseModel):
    """Individual search result data model."""
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class SearchData(BaseModel):
    """Data model for search results."""
    query: str = Field(..., description="Original search query")
    results: List[SearchResultData] = Field(..., description="Search results")
    total_results: int = Field(..., description="Number of results found")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")


class ChatData(BaseModel):
    """Data model for chat responses."""
    query: str = Field(..., description="User's question")
    response: str = Field(..., description="Generated response")
    sources_found: int = Field(..., description="Number of source documents found")
    context_documents: List[SearchResultData] = Field(default_factory=list, description="Context documents used")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    follow_up_suggestions: Optional[List[str]] = Field(None, description="Follow-up question suggestions")


class DocumentSummaryData(BaseModel):
    """Data model for document summary."""
    file_name: str = Field(..., description="Document file name")
    found: bool = Field(..., description="Whether the document was found")
    total_documents: Optional[int] = Field(None, description="Total number of document chunks")
    content_types: Optional[Dict[str, int]] = Field(None, description="Content type breakdown")
    table_types: Optional[Dict[str, int]] = Field(None, description="Table type breakdown")
    pages_covered: Optional[int] = Field(None, description="Number of pages covered")
    page_numbers: Optional[List[int]] = Field(None, description="Page numbers covered")


class DocumentOverviewData(BaseModel):
    """Data model for AI-generated document overview."""
    document_name: str = Field(..., description="Document name")
    found: bool = Field(..., description="Whether the document was found")
    ai_summary: Optional[str] = Field(None, description="AI-generated summary")
    document_stats: Optional[DocumentSummaryData] = Field(None, description="Document statistics")


class ComparePeriodsData(BaseModel):
    """Data model for period comparison."""
    query: str = Field(..., description="Comparison question")
    response: str = Field(..., description="Comparative analysis response")
    sources_compared: List[str] = Field(..., description="Documents that were compared")
    total_sources_found: int = Field(..., description="Total number of source documents found")
    context_documents: List[SearchResultData] = Field(default_factory=list, description="Context documents used")


class RemoveDocumentData(BaseModel):
    """Data model for document removal."""
    file_name: str = Field(..., description="Document file name")
    documents_removed: int = Field(..., description="Number of document chunks removed")


class HealthData(BaseModel):
    """Data model for health check results."""
    services: Dict[str, Any] = Field(default_factory=dict, description="Individual service health")
    overall_healthy: bool = Field(..., description="Overall system health")


class VectorHealthData(BaseModel):
    """Data model for vector database health."""
    healthy: bool = Field(..., description="Vector database health status")
    service: str = Field(..., description="Service name")


class CollectionInfoData(BaseModel):
    """Data model for collection information."""
    name: str = Field(..., description="Collection name")
    points_count: int = Field(..., description="Number of points in collection")
    vector_size: int = Field(..., description="Vector dimensions")
    distance: str = Field(..., description="Distance metric")


class VectorStatsData(BaseModel):
    """Data model for vector database statistics."""
    collection_info: CollectionInfoData = Field(..., description="Collection information")
    health_status: VectorHealthData = Field(..., description="Health status")
    embedding_model: str = Field(..., description="Embedding model name")
    vector_dimensions: int = Field(..., description="Vector dimensions")