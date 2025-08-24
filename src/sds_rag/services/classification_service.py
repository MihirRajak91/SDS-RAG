"""
Table Classification Service - Microservice for classifying financial tables.

This module provides intelligent classification of financial tables extracted from PDFs.
It identifies table types (income statement, balance sheet, cash flow, etc.) using
keyword matching and contextual analysis, supporting accurate categorization for
RAG-based financial document processing.

Key Features:
- Multi-type financial statement classification
- Keyword-based content analysis with confidence scoring
- Comprehensive metadata generation for classification decisions
- Support for company-specific and geographic segment tables
- Robust confidence metrics for extraction quality assessment

Classes:
    TableClassificationService: Main classification service
    ConfidenceCalculationService: Confidence metrics calculator
"""

import pandas as pd
import logging
from typing import Dict, List

from src.sds_rag.models.schemas import (
    TableType, TableClassificationResult, ConfidenceMetrics, 
    RawTableData, HeaderRow
)

logger = logging.getLogger(__name__)

class TableClassificationService:
    """Service for classifying financial tables into specific categories"""
    
    def __init__(self):
        self.financial_keywords = {
            TableType.INCOME_STATEMENT: [
                'net sales', 'revenue', 'gross margin', 'operating income', 
                'net income', 'earnings per share', 'cost of sales', 'cost of goods',
                'operating expenses', 'research and development', 'selling general'
            ],
            TableType.BALANCE_SHEET: [
                'assets', 'liabilities', 'shareholders equity', 'current assets',
                'cash and cash equivalents', 'marketable securities', 'accounts receivable',
                'inventory', 'property plant equipment', 'goodwill', 'current liabilities',
                'accounts payable', 'accrued expenses'
            ],
            TableType.CASH_FLOW: [
                'cash flow', 'operating activities', 'investing activities',
                'financing activities', 'cash generated', 'cash used',
                'net cash provided', 'net cash used', 'cash payments',
                'depreciation', 'amortization', 'capital expenditures'
            ],
            TableType.PRODUCT_REVENUE: [
                'iphone', 'mac', 'ipad', 'services', 'wearables', 'home and accessories',
                'product sales', 'service revenue', 'hardware', 'software',
                'accessories', 'apple watch', 'airpods'
            ],
            TableType.GEOGRAPHIC_SEGMENT: [
                'americas', 'europe', 'greater china', 'japan', 'rest of asia pacific',
                'united states', 'international', 'domestic', 'regional',
                'geographic', 'segment revenue', 'by region'
            ],
            TableType.COMPREHENSIVE_INCOME: [
                'comprehensive income', 'other comprehensive income', 
                'foreign currency translation', 'unrealized gains', 'unrealized losses',
                'hedging instruments', 'available for sale', 'pension adjustments'
            ],
            TableType.SHAREHOLDERS_EQUITY: [
                'common stock', 'retained earnings', 'accumulated other comprehensive',
                'share repurchase', 'dividends', 'stockholders equity',
                'additional paid in capital', 'treasury stock'
            ],
            TableType.SEGMENT_PERFORMANCE: [
                'segment', 'division', 'business unit', 'operating segment',
                'segment revenue', 'segment income', 'segment assets'
            ],
            TableType.QUARTERLY_COMPARISON: [
                'quarter', 'quarterly', 'three months', 'nine months',
                'year over year', 'sequential', 'prior quarter', 'q1', 'q2', 'q3', 'q4'
            ]
        }
        logger.info("Table classification service initialized")
    
    def classify_table(self, df: pd.DataFrame, headers: HeaderRow) -> TableClassificationResult:
        """
        Classify table type and calculate confidence score.
        
        Args:
            df (pd.DataFrame): DataFrame containing table data
            headers (HeaderRow): Table headers
            
        Returns:
            TableClassificationResult: Classification result with type, confidence, and metadata
        """
        
        # Extract text content for analysis
        text_content = self._extract_classification_text(df, headers)
        
        # Score each table type
        type_scores = {}
        matched_keywords_by_type = {}
        
        for table_type, keywords in self.financial_keywords.items():
            matches = []
            score = 0
            
            for keyword in keywords:
                if keyword.lower() in text_content.lower():
                    matches.append(keyword)
                    # Weight longer, more specific keywords higher
                    weight = len(keyword.split()) + 1
                    score += weight
            
            if matches:
                type_scores[table_type] = score
                matched_keywords_by_type[table_type] = matches
        
        # Determine best classification
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            best_score = type_scores[best_type]
            matched_keywords = matched_keywords_by_type[best_type]
        else:
            best_type = TableType.OTHER_FINANCIAL
            best_score = 0
            matched_keywords = []
        
        # Calculate confidence score
        confidence = self._calculate_classification_confidence(
            df, best_type, best_score, text_content
        )
        
        # Generate classification metadata
        metadata = self._generate_classification_metadata(
            df, headers, type_scores, text_content
        )
        
        logger.debug(f"Classified as {best_type.value} with confidence {confidence:.2f}")
        
        return TableClassificationResult(
            table_type=best_type,
            confidence_score=confidence,
            matched_keywords=matched_keywords,
            classification_metadata=metadata
        )
    
    def _extract_classification_text(self, df: pd.DataFrame, headers: HeaderRow) -> str:
        """
        Extract text content for classification analysis.
        
        Args:
            df (pd.DataFrame): DataFrame containing table data
            headers (HeaderRow): Table headers
            
        Returns:
            str: Extracted text content for classification
        """
        text_parts = []
        
        # Add headers
        if headers:
            text_parts.append(' '.join(str(h) for h in headers if h))
        
        # Add sample of data content
        if not df.empty:
            # Get first column content (often contains row labels)
            first_col = df.iloc[:, 0].dropna().head(10)
            text_parts.extend(str(val) for val in first_col if str(val).strip())
            
            # Get some cell content from first few rows
            sample_data = df.head(5).values.flatten()
            text_parts.extend(
                str(val) for val in sample_data 
                if pd.notna(val) and str(val).strip() and not str(val).isdigit()
            )
        
        return ' '.join(text_parts)
    
    def _calculate_classification_confidence(
        self, 
        df: pd.DataFrame, 
        table_type: TableType, 
        keyword_score: int,
        text_content: str
    ) -> float:
        """
        Calculate confidence score for table classification.
        
        Args:
            df (pd.DataFrame): DataFrame containing table data
            table_type (TableType): Classified table type
            keyword_score (int): Keyword matching score
            text_content (str): Extracted text content for classification
            
        Returns:
            float: Confidence score for classification
        """
        if df.empty:
            return 0.0
        
        confidence = 0.0
        
        # Base score from data structure
        if len(df.columns) >= 2:
            confidence += 0.2
        if len(df) >= 2:
            confidence += 0.2
        
        # Keyword matching score (normalized)
        if keyword_score > 0:
            content_length = len(text_content.split())
            keyword_confidence = min(keyword_score / max(content_length * 0.1, 1), 0.4)
            confidence += keyword_confidence
        
        # Content density bonus
        total_cells = df.size
        non_null_cells = df.count().sum()
        content_density = non_null_cells / max(total_cells, 1)
        if content_density > 0.5:
            confidence += 0.1
        
        # Numeric data presence bonus
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            confidence += 0.1
        
        # Penalize if classified as 'other_financial' with no specific matches
        if table_type == TableType.OTHER_FINANCIAL and keyword_score == 0:
            confidence *= 0.5
        
        return min(confidence, 1.0)
    
    def _generate_classification_metadata(
        self,
        df: pd.DataFrame,
        headers: HeaderRow,
        type_scores: Dict[TableType, int],
        text_content: str
    ) -> Dict[str, any]:
        """
        Generate metadata about the classification process.
        
        Args:
            df (pd.DataFrame): DataFrame containing table data
            headers (HeaderRow): Table headers
            type_scores (Dict[TableType, int]): Classification scores for each table type
            text_content (str): Extracted text content for classification
            
        Returns:
            Dict[str, any]: Classification metadata including scores, alternatives, and content sample
        """
        
        # Convert enum keys to strings for JSON serialization
        score_dict = {table_type.value: score for table_type, score in type_scores.items()}
        alt_types = sorted(score_dict.keys(), key=score_dict.get, reverse=True)[:3]
        
        return {
            'classification_scores': score_dict,
            'alternative_types': alt_types,
            'text_content_length': len(text_content),
            'has_numeric_columns': len(df.select_dtypes(include=['number']).columns) > 0,
            'content_sample': text_content[:200] + "..." if len(text_content) > 200 else text_content
        }

class ConfidenceCalculationService:
    """
    Service for calculating comprehensive confidence metrics.
    
    This service provides detailed confidence metrics for table classification,
    including data preservation, classification accuracy, and content quality.
    """
    
    def calculate_extraction_confidence(
        self,
        raw_table: RawTableData,
        processed_df: pd.DataFrame,
        classification: TableClassificationResult
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics.
        
        Args:
            raw_table (RawTableData): Raw extracted table data
            processed_df (pd.DataFrame): Processed DataFrame from table
            classification (TableClassificationResult): Classification result
            
        Returns:
            ConfidenceMetrics: Comprehensive confidence metrics including data preservation, classification accuracy, and content quality
        """
        
        # Data preservation score
        if raw_table.data and not processed_df.empty:
            raw_cells = sum(1 for row in raw_table.data for cell in row if cell and cell.strip())
            processed_cells = processed_df.count().sum()
            data_preservation = min(processed_cells / max(raw_cells, 1), 1.0)
        else:
            data_preservation = 0.0
        
        # Classification confidence
        classification_confidence = classification.confidence_score
        
        # Structure consistency
        if not processed_df.empty and raw_table.data:
            expected_cols = len(processed_df.columns)
            actual_cols = [len(row) for row in raw_table.data if row]
            if actual_cols:
                col_consistency = sum(1 for c in actual_cols if c == expected_cols) / len(actual_cols)
            else:
                col_consistency = 0.0
        else:
            col_consistency = 0.0
        
        # Content density
        if not processed_df.empty:
            total_cells = processed_df.size
            non_null_cells = processed_df.count().sum()
            content_density = non_null_cells / max(total_cells, 1)
        else:
            content_density = 0.0
        
        # Overall confidence (weighted average)
        weights = {
            'data_preservation': 0.3,
            'classification_confidence': 0.4,
            'structure_consistency': 0.2,
            'content_density': 0.1
        }
        
        overall_confidence = (
            data_preservation * weights['data_preservation'] +
            classification_confidence * weights['classification_confidence'] +
            col_consistency * weights['structure_consistency'] +
            content_density * weights['content_density']
        )
        
        logger.debug(f"Confidence metrics: overall={overall_confidence:.2f}")
        
        return ConfidenceMetrics(
            overall_confidence=overall_confidence,
            data_preservation=data_preservation,
            classification_confidence=classification_confidence,
            structure_consistency=col_consistency,
            content_density=content_density
        )