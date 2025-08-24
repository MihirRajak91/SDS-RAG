"""
Data models and schemas for the document processing system.
Defines all data structures used across services.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

class TableType(Enum):
    """Enumeration of supported financial table types"""
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    PRODUCT_REVENUE = "product_revenue"
    GEOGRAPHIC_SEGMENT = "geographic_segment"
    COMPREHENSIVE_INCOME = "comprehensive_income"
    SHAREHOLDERS_EQUITY = "shareholders_equity"
    SEGMENT_PERFORMANCE = "segment_performance"
    QUARTERLY_COMPARISON = "quarterly_comparison"
    OTHER_FINANCIAL = "other_financial"

class ExtractionMethod(Enum):
    """Enumeration of table extraction methods"""
    TEXT_BASED = "text_based_extraction"
    LINES_BASED = "lines_based_extraction"
    DEFAULT = "default_extraction"

class ConfidenceLevel(Enum):
    """Confidence level categories"""
    HIGH = "high"      # > 0.7
    MEDIUM = "medium"  # 0.4 - 0.7
    LOW = "low"        # < 0.4

class RawTableData(BaseModel):
    """Raw extracted table data from PDF"""
    data: List[List[str]] = Field(..., description="Raw table data as nested lists")
    page: int = Field(..., ge=1, description="Page number (1-indexed)")
    table_index: int = Field(..., ge=0, description="Table index on the page")
    extraction_method: ExtractionMethod = Field(..., description="Extraction method used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('data')
    @classmethod
    def validate_data(cls, v):
        if not v or not any(row for row in v if row):
            raise ValueError("Table data cannot be empty")
        return v

class TableClassificationResult(BaseModel):
    """Result of table classification process"""
    table_type: TableType = Field(..., description="Classified table type")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords that matched for classification")
    classification_metadata: Dict[str, Any] = Field(default_factory=dict, description="Classification process metadata")

class DataValidationResult(BaseModel):
    """Result of data validation checks"""
    is_valid: bool = Field(..., description="Whether the data passed validation")
    issues: List[str] = Field(default_factory=list, description="List of validation issues found")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Validation metrics")

class ConfidenceMetrics(BaseModel):
    """Comprehensive confidence metrics for table extraction"""
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    data_preservation: float = Field(..., ge=0.0, le=1.0, description="Data preservation score")
    classification_confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    structure_consistency: float = Field(..., ge=0.0, le=1.0, description="Structure consistency score")
    content_density: float = Field(..., ge=0.0, le=1.0, description="Content density score")

class ProcessedTable(BaseModel):
    """Fully processed table with all metadata"""
    page: int = Field(..., ge=1, description="Page number (1-indexed)")
    table_index: int = Field(..., ge=0, description="Table index on the page")
    table_type: TableType = Field(..., description="Classified table type")
    headers: List[str] = Field(..., description="Table column headers")
    data: pd.DataFrame = Field(..., description="Processed table data")
    raw_table: List[List[str]] = Field(..., description="Original raw table data")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    confidence_level: ConfidenceLevel = Field(default=None, description="Confidence level category")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True
    
    @model_validator(mode='before')
    @classmethod
    def set_confidence_level(cls, values):
        """Set confidence level based on score"""
        if isinstance(values, dict):
            confidence_score = values.get('confidence_score', 0)
            if confidence_score > 0.7:
                values['confidence_level'] = ConfidenceLevel.HIGH
            elif confidence_score >= 0.4:
                values['confidence_level'] = ConfidenceLevel.MEDIUM
            else:
                values['confidence_level'] = ConfidenceLevel.LOW
        return values
    
    @field_validator('headers')
    @classmethod
    def validate_headers(cls, v):
        if not v:
            raise ValueError("Headers cannot be empty")
        return v

class TextChunk(BaseModel):
    """Extracted text content chunk"""
    type: str = Field(..., description="Type of text chunk")
    content: str = Field(..., min_length=1, description="Text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

class DocumentMetadata(BaseModel):
    """Document-level metadata"""
    file_path: str = Field(..., description="Full path to the processed file")
    file_name: str = Field(..., description="File name")
    extraction_method: str = Field(..., description="Method used for extraction")
    processing_timestamp: str = Field(..., description="ISO timestamp of processing")
    processor_version: str = Field(..., description="Version of processor used")
    total_pages: int = Field(..., ge=0, description="Total number of pages in document")
    pdf_metadata: Dict[str, Any] = Field(default_factory=dict, description="PDF metadata from document")

class ExtractionStatistics(BaseModel):
    """Statistics about the extraction process"""
    total_pages: int = Field(..., ge=0, description="Total pages processed")
    tables_extracted: int = Field(..., ge=0, description="Number of tables extracted")
    text_chunks: int = Field(..., ge=0, description="Number of text chunks extracted")
    table_types: List[str] = Field(default_factory=list, description="Types of tables found")
    confidence_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution of confidence levels")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    success_rate: float = Field(..., ge=0.0, le=100.0, description="Success rate percentage")
    pages_with_tables: int = Field(..., ge=0, description="Number of pages containing tables")
    pages_with_text: int = Field(..., ge=0, description="Number of pages containing text")

class ProcessedDocument(BaseModel):
    """Complete processed document with all components"""
    structured_tables: List[ProcessedTable] = Field(default_factory=list, description="Processed tables")
    text_chunks: List[TextChunk] = Field(default_factory=list, description="Extracted text chunks")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    extraction_stats: ExtractionStatistics = Field(..., description="Extraction statistics")
    
    def get_high_confidence_tables(self) -> List[ProcessedTable]:
        """Get tables with high confidence scores"""
        return [t for t in self.structured_tables if t.confidence_level == ConfidenceLevel.HIGH]
    
    def get_tables_by_type(self, table_type: TableType) -> List[ProcessedTable]:
        """Get tables of a specific type"""
        return [t for t in self.structured_tables if t.table_type == table_type]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get document processing summary"""
        return {
            'total_tables': len(self.structured_tables),
            'high_confidence_tables': len(self.get_high_confidence_tables()),
            'table_types_found': list(set(t.table_type.value for t in self.structured_tables)),
            'success_rate': self.extraction_stats.success_rate,
            'pages_processed': self.extraction_stats.total_pages
        }

# Type aliases for cleaner code
TableData = List[List[str]]
HeaderRow = List[str]
TableMetrics = Dict[str, float]