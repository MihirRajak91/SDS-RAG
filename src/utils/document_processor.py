"""
Main document processor orchestrator - coordinates all components for financial PDF processing.
Clean, modular architecture with single responsibility principle.
"""

import PyPDF2
import pandas as pd
import logging
import warnings
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass

from .table_extractor import TableExtractor, TableCleaner, RawTable
from .financial_parser import FinancialNumberParser, HeaderProcessor
from .table_classifier import FinancialTableClassifier, TableValidator, ConfidenceCalculator

# Set up logging and suppress warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pdfplumber")

@dataclass
class ExtractedTable:
    """Container for fully processed table data"""
    page: int
    table_index: int
    table_type: str
    headers: List[str]
    data: pd.DataFrame
    raw_table: List[List[str]]
    confidence_score: float
    metadata: Dict[str, Any]

@dataclass
class ProcessedDocument:
    """Container for complete document processing results"""
    structured_tables: List[ExtractedTable]
    text_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    extraction_stats: Dict[str, Any]

class DocumentProcessor:
    """Main orchestrator for financial document processing"""
    
    def __init__(self):
        # Initialize all components
        self.table_extractor = TableExtractor()
        self.table_cleaner = TableCleaner()
        self.financial_parser = FinancialNumberParser()
        self.header_processor = HeaderProcessor()
        self.table_classifier = FinancialTableClassifier()
        self.table_validator = TableValidator()
        
        logger.info("Document processor initialized with all components")
    
    def process_document(self, pdf_path: str) -> ProcessedDocument:
        """Main processing pipeline for financial PDF documents"""
        logger.info(f"Processing financial PDF: {pdf_path}")
        
        try:
            # Extract structured tables
            tables = self._extract_and_process_tables(pdf_path)
            
            # Extract text content
            text_chunks = self._extract_text_content(pdf_path)
            
            # Generate document metadata
            metadata = self._extract_document_metadata(pdf_path)
            
            # Calculate extraction statistics
            stats = self._calculate_extraction_stats(pdf_path, tables, text_chunks)
            
            result = ProcessedDocument(
                structured_tables=tables,
                text_chunks=text_chunks,
                metadata=metadata,
                extraction_stats=stats
            )
            
            logger.info(f"Processing complete: {len(tables)} tables, {len(text_chunks)} text chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _extract_and_process_tables(self, pdf_path: str) -> List[ExtractedTable]:
        """Extract and fully process all tables from the PDF"""
        processed_tables = []
        
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Successfully opened PDF with {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        logger.debug(f"Processing page {page_num + 1}")
                        
                        # Extract raw tables
                        raw_tables = self.table_extractor.extract_tables_from_page(page, page_num + 1)
                        logger.debug(f"Found {len(raw_tables)} raw tables on page {page_num + 1}")
                        
                        # Process each raw table
                        for raw_table in raw_tables:
                            if self.table_validator.is_valid_financial_table(raw_table.data):
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
            logger.error(f"Failed to open or process PDF {pdf_path}: {e}")
            raise
        
        logger.info(f"Successfully processed {len(processed_tables)} tables")
        return processed_tables
    
    def _process_single_table(self, raw_table: RawTable) -> ExtractedTable:
        """Process a single raw table through the complete pipeline"""
        
        # 1. Clean the raw table data
        cleaned_data = self.table_cleaner.clean_raw_table(raw_table)
        
        if not cleaned_data:
            return None
        
        # 2. Identify headers and data rows
        header_row_idx, headers = self.header_processor.identify_header_row(cleaned_data)
        data_rows = cleaned_data[header_row_idx + 1:] if header_row_idx + 1 < len(cleaned_data) else []
        
        if not data_rows:
            # If no data rows, treat entire table as data with generic headers
            headers = [f"Column_{i+1}" for i in range(len(cleaned_data[0]))] if cleaned_data else []
            data_rows = cleaned_data
        
        # 3. Create DataFrame
        df = self._create_dataframe(headers, data_rows)
        
        if df.empty:
            return None
        
        # 4. Convert financial columns to numeric
        df = self.financial_parser.convert_dataframe_columns(df)
        
        # 5. Classify table type and calculate confidence
        classification = self.table_classifier.classify_table(df, headers)
        
        # 6. Calculate comprehensive confidence metrics
        confidence_metrics = ConfidenceCalculator.calculate_extraction_confidence(
            raw_table.data, df, classification
        )
        
        # 7. Generate table metadata
        table_metadata = self._generate_table_metadata(
            raw_table, df, classification, confidence_metrics
        )
        
        return ExtractedTable(
            page=raw_table.page,
            table_index=raw_table.table_index,
            table_type=classification.table_type,
            headers=headers,
            data=df,
            raw_table=raw_table.data,
            confidence_score=confidence_metrics['overall_confidence'],
            metadata=table_metadata
        )
    
    def _create_dataframe(self, headers: List[str], data_rows: List[List[str]]) -> pd.DataFrame:
        """Create pandas DataFrame from headers and data rows"""
        if not data_rows:
            return pd.DataFrame()
        
        # Ensure consistent column count
        max_cols = max(len(headers), max(len(row) for row in data_rows) if data_rows else 0)
        
        # Pad headers if needed
        while len(headers) < max_cols:
            headers.append(f"Column_{len(headers) + 1}")
        
        # Pad data rows if needed
        padded_rows = []
        for row in data_rows:
            padded_row = row[:]
            while len(padded_row) < max_cols:
                padded_row.append('')
            padded_rows.append(padded_row[:max_cols])  # Trim if too long
        
        df = pd.DataFrame(padded_rows, columns=headers[:max_cols])
        
        # Clean DataFrame
        df = df.replace('', pd.NA)
        df = df.dropna(how='all')  # Remove completely empty rows
        
        return df
    
    def _generate_table_metadata(
        self, 
        raw_table: RawTable,
        df: pd.DataFrame,
        classification,
        confidence_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for the extracted table"""
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        return {
            'page_number': raw_table.page,
            'table_type': classification.table_type,
            'extraction_method': raw_table.extraction_method,
            'dimensions': f"{len(df)} rows × {len(df.columns)} columns",
            'has_numeric_data': len(numeric_cols) > 0,
            'numeric_columns': numeric_cols,
            'column_count': len(df.columns),
            'row_count': len(df),
            'matched_keywords': classification.matched_keywords,
            'confidence_metrics': confidence_metrics,
            'classification_metadata': classification.classification_metadata,
            'extraction_timestamp': pd.Timestamp.now().isoformat(),
            'data_quality': {
                'missing_data_percentage': (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0,
                'content_density': (df.count().sum() / df.size * 100) if df.size > 0 else 0
            }
        }
    
    def _extract_text_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text content for narrative sections"""
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
                                    text_chunks.append({
                                        'type': 'narrative_text',
                                        'content': paragraph,
                                        'metadata': {
                                            'page': page_num + 1,
                                            'paragraph_index': para_idx,
                                            'word_count': len(paragraph.split()),
                                            'source': Path(pdf_path).name
                                        }
                                    })
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
        
        logger.info(f"Extracted {len(text_chunks)} text chunks")
        return text_chunks
    
    def _extract_document_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document-level metadata"""
        metadata = {
            'file_path': pdf_path,
            'file_name': Path(pdf_path).name,
            'extraction_method': 'Multi-strategy pdfplumber + PyPDF2',
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'processor_version': '2.0.0'
        }
        
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                metadata.update({
                    'total_pages': len(pdf.pages),
                    'pdf_metadata': pdf.metadata or {}
                })
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
        
        return metadata
    
    def _calculate_extraction_stats(
        self, 
        pdf_path: str, 
        tables: List[ExtractedTable],
        text_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive extraction statistics"""
        
        # Get page count
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
        except:
            total_pages = 0
        
        # Table statistics
        table_types = list(set(table.table_type for table in tables))
        confidence_scores = [table.confidence_score for table in tables]
        
        # Confidence distribution
        high_confidence = sum(1 for score in confidence_scores if score > 0.7)
        medium_confidence = sum(1 for score in confidence_scores if 0.4 <= score <= 0.7)
        low_confidence = sum(1 for score in confidence_scores if score < 0.4)
        
        return {
            'total_pages': total_pages,
            'tables_extracted': len(tables),
            'text_chunks': len(text_chunks),
            'table_types': table_types,
            'confidence_distribution': {
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence
            },
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            'success_rate': (high_confidence / len(confidence_scores) * 100) if confidence_scores else 0,
            'pages_with_tables': len(set(table.page for table in tables)),
            'pages_with_text': len(set(chunk['metadata']['page'] for chunk in text_chunks))
        }

# Convenience functions for backward compatibility
def process_financial_pdf(pdf_path: str) -> ProcessedDocument:
    """Quick function to process a financial PDF"""
    processor = DocumentProcessor()
    return processor.process_document(pdf_path)

def extract_tables_only(pdf_path: str) -> List[ExtractedTable]:
    """Extract only tables from financial PDF"""
    processor = DocumentProcessor()
    result = processor.process_document(pdf_path)
    return result.structured_tables