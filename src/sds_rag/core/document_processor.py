"""
Document Processing Orchestrator - Main microservice coordinator.

This module serves as the central orchestrator for the document processing pipeline,
coordinating all microservices to provide a clean, high-level API for extracting
structured data from financial PDF documents.

The orchestrator manages the complete pipeline:
1. Table Extraction: Multi-strategy PDF table extraction
2. Table Processing: Cleaning, header detection, and DataFrame creation
3. Classification: Intelligent table type identification
4. Validation: Quality assessment and business rule validation
5. Text Processing: Comprehensive text extraction and chunking
6. Document Assembly: Final document creation with metadata and statistics

Key Features:
- Complete document processing pipeline orchestration
- Microservice coordination and error handling
- Comprehensive metadata and statistics generation
- High-level API abstraction over complex processing steps
- Quality assessment and confidence scoring

Classes:
    DocumentProcessingOrchestrator: Main orchestrator service

Functions:
    process_financial_pdf: Convenience function for quick document processing
    extract_tables_only: Extract only tables without full document processing
"""

import PyPDF2
import pandas as pd
import logging
import warnings
from typing import List, Dict, Any
from pathlib import Path

from ..utils import StructuredLogger, Timer, log_performance, validate_pdf_file

# Import all services
from src.sds_rag.services.extraction_service import TableExtractionService, TableCleaningService
from src.sds_rag.services.parsing_service import (
    FinancialNumberParsingService, HeaderProcessingService, DataFrameService
)
from src.sds_rag.services.classification_service import (
    TableClassificationService, ConfidenceCalculationService
)
from src.sds_rag.services.validation_service import (
    DataValidationService, TableStructureValidationService, BusinessRuleValidationService
)

# Import models
from src.sds_rag.models.schemas import (
    ProcessedDocument, ProcessedTable, TextChunk, DocumentMetadata, 
    ExtractionStatistics, ConfidenceLevel
)

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pdfplumber")

class DocumentProcessingOrchestrator:
    """Orchestrates all microservices for complete document processing."""
    
    def __init__(self):
        """Initialize orchestrator with all required microservices."""
        # Initialize all microservices
        self.table_extractor = TableExtractionService()
        self.table_cleaner = TableCleaningService()
        self.parser = FinancialNumberParsingService()
        self.header_processor = HeaderProcessingService()
        self.dataframe_service = DataFrameService()
        self.classifier = TableClassificationService()
        self.confidence_calculator = ConfidenceCalculationService()
        self.data_validator = DataValidationService()
        self.structure_validator = TableStructureValidationService()
        self.business_validator = BusinessRuleValidationService()
        
        logger.info("Document processing orchestrator initialized with all microservices")
    
    @log_performance
    def process_document(self, pdf_path: str) -> ProcessedDocument:
        """
        Process PDF document through complete pipeline.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            ProcessedDocument: Processed document with tables, text, and metadata
        """
        # Validate PDF file first
        is_valid, validation_errors = validate_pdf_file(pdf_path)
        if not is_valid:
            raise ValueError(f"Invalid PDF file: {', '.join(validation_errors)}")
        
        structured_logger.log_document_processing(
            file_path=pdf_path,
            status="started",
            details={"pipeline": "document_processing"}
        )
        
        try:
            with Timer(f"Processing document {Path(pdf_path).name}") as timer:
                # Step 1: Extract and process tables
                tables = self._process_tables_pipeline(pdf_path)
                
                # Step 2: Extract text content
                text_chunks = self._process_text_pipeline(pdf_path)
                
                # Step 3: Generate document metadata
                metadata = self._generate_document_metadata(pdf_path)
                
                # Step 4: Calculate extraction statistics
                stats = self._calculate_extraction_statistics(pdf_path, tables, text_chunks)
                
                # Step 5: Create final document
                document = ProcessedDocument(
                    structured_tables=tables,
                    text_chunks=text_chunks,
                    metadata=metadata,
                    extraction_stats=stats
                )
                
                # Generate summary
                summary = {
                    "file_path": pdf_path,
                    "file_name": metadata.file_name,
                    "processing_successful": True,
                    "tables_processed": len(tables),
                    "text_chunks_processed": len(text_chunks),
                    "high_confidence_tables": len([t for t in tables if t.confidence_level.value == "high"]),
                    "average_confidence": stats.average_confidence,
                    "success_rate": stats.success_rate,
                    "total_pages": stats.total_pages,
                    "processing_time": timer.elapsed_human
                }
                
                structured_logger.log_document_processing(
                    file_path=pdf_path,
                    status="success",
                    details=summary
                )
                
                logger.info(f"Processing complete: {len(tables)} tables, {len(text_chunks)} text chunks")
                return document
            
        except Exception as e:
            structured_logger.log_document_processing(
                file_path=pdf_path,
                status="failed",
                details={"error": str(e)}
            )
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            raise
    
    def _process_tables_pipeline(self, pdf_path: str) -> List[ProcessedTable]:
        """
        Extract and process all tables from PDF.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            List[ProcessedTable]: List of processed tables
        """
        processed_tables = []
        
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Successfully opened PDF with {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        logger.debug(f"Processing page {page_num + 1}")
                        
                        # Extract raw tables using extraction service
                        raw_tables = self.table_extractor.extract_tables_from_page(page, page_num + 1)
                        logger.debug(f"Found {len(raw_tables)} raw tables on page {page_num + 1}")
                        
                        # Process each raw table through complete pipeline
                        for raw_table in raw_tables:
                            if self.structure_validator.is_valid_financial_table(raw_table.data):
                                try:
                                    processed_table = self._process_single_table(raw_table)
                                    if processed_table:
                                        processed_tables.append(processed_table)
                                except Exception as e:
                                    logger.warning(f"Failed to process table {raw_table.table_index} on page {page_num + 1}: {e}")
                                    continue
                                    
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            raise
        
        logger.info(f"Successfully processed {len(processed_tables)} tables")
        return processed_tables
    
    def _process_single_table(self, raw_table) -> ProcessedTable:
        """
        Process single table through all microservices.
        
        Args:
            raw_table (RawTableData): Raw extracted table
            
        Returns:
            ProcessedTable: Fully processed table or None if processing fails
        """
        
        # Step 1: Clean raw table data
        cleaned_data = self.table_cleaner.clean_raw_table(raw_table)
        if not cleaned_data:
            return None
        
        # Step 2: Process headers and create DataFrame
        header_row_idx, headers = self.header_processor.identify_header_row(cleaned_data)
        data_rows = cleaned_data[header_row_idx + 1:] if header_row_idx + 1 < len(cleaned_data) else []
        
        if not data_rows:
            # If no data rows, treat entire table as data with generic headers
            headers = [f"Column_{i+1}" for i in range(len(cleaned_data[0]))] if cleaned_data else []
            data_rows = cleaned_data
        
        # Step 3: Create and validate DataFrame
        df = self.dataframe_service.create_dataframe(headers, data_rows)
        if df.empty:
            return None
        
        # Step 4: Validate data quality
        validation_result = self.data_validator.validate_table_data(df)
        if not validation_result.is_valid:
            logger.debug(f"Table validation failed: {validation_result.issues}")
        
        # Step 5: Classify table type
        classification = self.classifier.classify_table(df, headers)
        
        # Step 6: Calculate confidence metrics
        confidence_metrics = self.confidence_calculator.calculate_extraction_confidence(
            raw_table, df, classification
        )
        
        # Step 7: Business rule validation
        business_issues = self.business_validator.validate_financial_consistency(
            df, classification.table_type.value
        )
        
        # Step 8: Generate comprehensive metadata
        table_metadata = self._generate_table_metadata(
            raw_table, df, classification, confidence_metrics, validation_result, business_issues
        )
        
        return ProcessedTable(
            page=raw_table.page,
            table_index=raw_table.table_index,
            table_type=classification.table_type,
            headers=headers,
            data=df,
            raw_table=raw_table.data,
            confidence_score=confidence_metrics.overall_confidence,
            confidence_level=ConfidenceLevel.HIGH,  # Will be set by Pydantic root_validator
            metadata=table_metadata
        )
    
    def _process_text_pipeline(self, pdf_path: str) -> List[TextChunk]:
        """
        Extract text content from PDF.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            List[TextChunk]: List of text chunks with metadata
        """
        text_chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"Extracting text from {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            # Split into paragraphs
                            paragraphs = [p.strip() for p in page_text.split('\\n\\n') if p.strip()]
                            
                            for para_idx, paragraph in enumerate(paragraphs):
                                if len(paragraph) > 100:  # Only substantial paragraphs
                                    chunk = TextChunk(
                                        type='narrative_text',
                                        content=paragraph,
                                        metadata={
                                            'page': page_num + 1,
                                            'paragraph_index': para_idx,
                                            'word_count': len(paragraph.split()),
                                            'source': Path(pdf_path).name
                                        }
                                    )
                                    text_chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        
        logger.info(f"Extracted {len(text_chunks)} text chunks")
        return text_chunks
    
    def _generate_document_metadata(self, pdf_path: str) -> DocumentMetadata:
        """
        Generate document metadata.
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            DocumentMetadata: Document metadata including file info and processing details
        """
        metadata_dict = {
            'file_path': pdf_path,
            'file_name': Path(pdf_path).name,
            'extraction_method': 'Microservice Architecture v2.0',
            'processing_timestamp': structured_logger.utc_now_iso(),
            'processor_version': '2.0.0-microservice',
            'total_pages': 0,
            'pdf_metadata': {}
        }
        
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                metadata_dict.update({
                    'total_pages': len(pdf.pages),
                    'pdf_metadata': pdf.metadata or {}
                })
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
        
        return DocumentMetadata(**metadata_dict)
    
    def _calculate_extraction_statistics(
        self, 
        pdf_path: str, 
        tables: List[ProcessedTable],
        text_chunks: List[TextChunk]
    ) -> ExtractionStatistics:
        """
        Calculate extraction statistics and success metrics.
        
        Args:
            pdf_path (str): Path to PDF file
            tables (List[ProcessedTable]): Processed tables
            text_chunks (List[TextChunk]): Text chunks
            
        Returns:
            ExtractionStatistics: Comprehensive extraction statistics
        """
        
        # Get page count
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
        except:
            total_pages = 0
        
        # Table statistics
        table_types = list(set(t.table_type.value for t in tables))
        confidence_scores = [t.confidence_score for t in tables]
        
        # Confidence distribution
        high_confidence = sum(1 for t in tables if t.confidence_level == ConfidenceLevel.HIGH)
        medium_confidence = sum(1 for t in tables if t.confidence_level == ConfidenceLevel.MEDIUM)
        low_confidence = sum(1 for t in tables if t.confidence_level == ConfidenceLevel.LOW)
        
        return ExtractionStatistics(
            total_pages=total_pages,
            tables_extracted=len(tables),
            text_chunks=len(text_chunks),
            table_types=table_types,
            confidence_distribution={
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence
            },
            average_confidence=sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            success_rate=(high_confidence / len(confidence_scores) * 100) if confidence_scores else 0,
            pages_with_tables=len(set(t.page for t in tables)),
            pages_with_text=len(set(chunk.metadata['page'] for chunk in text_chunks))
        )
    
    def _generate_table_metadata(
        self, 
        raw_table,
        df: pd.DataFrame,
        classification,
        confidence_metrics,
        validation_result,
        business_issues: List[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for the extracted table"""
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        return {
            'page_number': raw_table.page,
            'table_type': classification.table_type.value,
            'extraction_method': raw_table.extraction_method.value,
            'dimensions': f"{len(df)} rows × {len(df.columns)} columns",
            'has_numeric_data': len(numeric_cols) > 0,
            'numeric_columns': numeric_cols,
            'column_count': len(df.columns),
            'row_count': len(df),
            'matched_keywords': classification.matched_keywords,
            'confidence_metrics': {
                'overall': confidence_metrics.overall_confidence,
                'data_preservation': confidence_metrics.data_preservation,
                'classification': confidence_metrics.classification_confidence,
                'structure_consistency': confidence_metrics.structure_consistency,
                'content_density': confidence_metrics.content_density
            },
            'classification_metadata': classification.classification_metadata,
            'validation_result': {
                'is_valid': validation_result.is_valid,
                'issues': validation_result.issues,
                'metrics': validation_result.metrics
            },
            'business_validation': {
                'issues': business_issues
            },
            'extraction_timestamp': structured_logger.utc_now_iso(),
            'data_quality': {
                'missing_data_percentage': (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0,
                'content_density': (df.count().sum() / df.size * 100) if df.size > 0 else 0
            }
        }

# Convenience functions for backward compatibility and RAG integration
def process_financial_pdf(pdf_path: str) -> ProcessedDocument:
    """
    Process financial PDF with default settings.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        ProcessedDocument: Complete processed document
    """
    orchestrator = DocumentProcessingOrchestrator()
    return orchestrator.process_document(pdf_path)

def extract_tables_only(pdf_path: str) -> List[ProcessedTable]:
    """
    Extract only tables without text processing.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        List[ProcessedTable]: List of processed tables
    """
    orchestrator = DocumentProcessingOrchestrator()
    result = orchestrator.process_document(pdf_path)
    return result.structured_tables

def process_and_store_rag(pdf_path: str, qdrant_host: str = "localhost", qdrant_port: int = 6333) -> dict:
    """
    Process PDF and store in RAG vector database.
    
    Args:
        pdf_path (str): Path to PDF file
        qdrant_host (str): Qdrant server host
        qdrant_port (int): Qdrant server port
        
    Returns:
        dict: Processing summary with statistics
    """
    from src.services.rag_service import RAGService
    rag_service = RAGService(qdrant_host=qdrant_host, qdrant_port=qdrant_port)
    return rag_service.process_and_store_document(pdf_path)