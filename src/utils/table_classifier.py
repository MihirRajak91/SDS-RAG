"""
Financial table classification and validation utilities.
Identifies table types and calculates confidence scores for extraction quality.
"""

import pandas as pd
import re
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from .financial_parser import DataValidator

logger = logging.getLogger(__name__)

@dataclass
class TableClassification:
    """Result of table classification"""
    table_type: str
    confidence_score: float
    matched_keywords: List[str]
    classification_metadata: Dict[str, Any]

class FinancialTableClassifier:
    """Classifies financial tables into specific categories"""
    
    def __init__(self):
        self.financial_keywords = {
            'income_statement': [
                'net sales', 'revenue', 'gross margin', 'operating income', 
                'net income', 'earnings per share', 'cost of sales', 'cost of goods',
                'operating expenses', 'research and development', 'selling general'
            ],
            'balance_sheet': [
                'assets', 'liabilities', 'shareholders equity', 'current assets',
                'cash and cash equivalents', 'marketable securities', 'accounts receivable',
                'inventory', 'property plant equipment', 'goodwill', 'current liabilities',
                'accounts payable', 'accrued expenses'
            ],
            'cash_flow': [
                'cash flow', 'operating activities', 'investing activities',
                'financing activities', 'cash generated', 'cash used',
                'net cash provided', 'net cash used', 'cash payments',
                'depreciation', 'amortization', 'capital expenditures'
            ],
            'product_revenue': [
                'iphone', 'mac', 'ipad', 'services', 'wearables', 'home and accessories',
                'product sales', 'service revenue', 'hardware', 'software',
                'accessories', 'apple watch', 'airpods'
            ],
            'geographic_segment': [
                'americas', 'europe', 'greater china', 'japan', 'rest of asia pacific',
                'united states', 'international', 'domestic', 'regional',
                'geographic', 'segment revenue', 'by region'
            ],
            'comprehensive_income': [
                'comprehensive income', 'other comprehensive income', 
                'foreign currency translation', 'unrealized gains', 'unrealized losses',
                'hedging instruments', 'available for sale', 'pension adjustments'
            ],
            'shareholders_equity': [
                'common stock', 'retained earnings', 'accumulated other comprehensive',
                'share repurchase', 'dividends', 'stockholders equity',
                'additional paid in capital', 'treasury stock'
            ],
            'segment_performance': [
                'segment', 'division', 'business unit', 'operating segment',
                'segment revenue', 'segment income', 'segment assets'
            ],
            'quarterly_comparison': [
                'quarter', 'quarterly', 'three months', 'nine months',
                'year over year', 'sequential', 'prior quarter', 'q1', 'q2', 'q3', 'q4'
            ]
        }
        
        self.validator = DataValidator()
    
    def classify_table(self, df: pd.DataFrame, headers: List[str]) -> TableClassification:
        """Classify table type and calculate confidence score"""
        
        # Combine headers and sample data for analysis
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
            best_type = 'other_financial'
            best_score = 0
            matched_keywords = []
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(df, best_type, best_score, text_content)
        
        # Generate classification metadata
        metadata = self._generate_classification_metadata(
            df, headers, type_scores, text_content
        )
        
        return TableClassification(
            table_type=best_type,
            confidence_score=confidence,
            matched_keywords=matched_keywords,
            classification_metadata=metadata
        )
    
    def _extract_classification_text(self, df: pd.DataFrame, headers: List[str]) -> str:
        """Extract text content for classification analysis"""
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
    
    def _calculate_confidence_score(
        self, 
        df: pd.DataFrame, 
        table_type: str, 
        keyword_score: int,
        text_content: str
    ) -> float:
        """Calculate confidence score for table classification"""
        if df.empty:
            return 0.0
        
        confidence = 0.0
        
        # Base score from data validation
        validation_result = self.validator.validate_table_data(df)
        if validation_result['is_valid']:
            confidence += 0.4
        
        # Structure quality score
        if len(df.columns) >= 2:
            confidence += 0.2
        if len(df) >= 2:
            confidence += 0.2
        
        # Keyword matching score (normalized)
        if keyword_score > 0:
            # Normalize based on content length and keyword strength
            content_length = len(text_content.split())
            keyword_confidence = min(keyword_score / max(content_length * 0.1, 1), 0.3)
            confidence += keyword_confidence
        
        # Content density bonus
        metrics = validation_result.get('metrics', {})
        content_density = metrics.get('content_density', 0)
        if content_density > 50:
            confidence += 0.1
        
        # Financial data presence bonus
        numeric_percentage = metrics.get('numeric_percentage', 0)
        if numeric_percentage > 30:  # Has substantial numeric data
            confidence += 0.1
        
        # Penalize if classified as 'other_financial' with no specific matches
        if table_type == 'other_financial' and keyword_score == 0:
            confidence *= 0.5
        
        return min(confidence, 1.0)
    
    def _generate_classification_metadata(
        self,
        df: pd.DataFrame,
        headers: List[str],
        type_scores: Dict[str, int],
        text_content: str
    ) -> Dict[str, Any]:
        """Generate metadata about the classification process"""
        validation_result = self.validator.validate_table_data(df)
        
        return {
            'classification_scores': type_scores,
            'alternative_types': sorted(type_scores.keys(), key=type_scores.get, reverse=True)[:3],
            'text_content_length': len(text_content),
            'validation_metrics': validation_result.get('metrics', {}),
            'data_quality_issues': validation_result.get('issues', []),
            'has_descriptive_headers': validation_result['metrics'].get('has_descriptive_headers', False),
            'content_sample': text_content[:200] + "..." if len(text_content) > 200 else text_content
        }

class TableValidator:
    """Validates if extracted tables contain meaningful financial data"""
    
    def __init__(self):
        self.min_content_threshold = 0.1  # At least 10% non-empty cells
        self.min_dimensions = (1, 2)      # At least 1 row, 2 columns
    
    def is_valid_financial_table(self, table: List[List[str]]) -> bool:
        """Check if extracted table contains valid financial data"""
        if not table or len(table) < 1:
            return False
        
        # Basic dimension check
        non_empty_rows = [row for row in table if row and any(cell and cell.strip() for cell in row)]
        if len(non_empty_rows) < self.min_dimensions[0]:
            return False
        
        # Check for minimum column count
        has_multiple_cols = any(
            len([cell for cell in row if cell and cell.strip()]) >= self.min_dimensions[1] 
            for row in non_empty_rows
        )
        if not has_multiple_cols:
            return False
        
        # Check content quality
        table_text = self._extract_table_text(non_empty_rows)
        if not table_text.strip():
            return False
        
        # Look for financial indicators
        return self._has_financial_indicators(table_text)
    
    def _extract_table_text(self, table: List[List[str]]) -> str:
        """Extract text content from table for analysis"""
        text_parts = []
        for row in table:
            for cell in row:
                if cell and cell.strip():
                    text_parts.append(cell.strip())
        return ' '.join(text_parts)
    
    def _has_financial_indicators(self, text: str) -> bool:
        """Check if text contains financial indicators"""
        text_lower = text.lower()
        
        # Financial patterns
        has_currency = bool(re.search(r'[\$€£¥]', text))
        has_numbers = bool(re.search(r'\d', text))
        has_financial_numbers = bool(re.search(
            r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\(\d+\)|\d+\.\d+', text
        ))
        
        # Financial terms (basic set)
        financial_terms = [
            'revenue', 'income', 'cash', 'assets', 'sales', 'cost', 'expenses',
            'profit', 'loss', 'margin', 'quarter', 'year', 'million', 'billion'
        ]
        has_financial_terms = any(term in text_lower for term in financial_terms)
        
        # Date/period indicators
        has_date_terms = bool(re.search(
            r'\b(quarter|q[1-4]|202[0-9]|ended|months|year)\b', text_lower
        ))
        
        # Need at least numbers plus one financial indicator
        return has_numbers and (
            has_currency or has_financial_terms or 
            has_financial_numbers or has_date_terms
        )

class ConfidenceCalculator:
    """Calculates various confidence metrics for extracted tables"""
    
    @staticmethod
    def calculate_extraction_confidence(
        raw_table: List[List[str]],
        processed_df: pd.DataFrame,
        classification: TableClassification
    ) -> Dict[str, float]:
        """Calculate comprehensive confidence metrics"""
        
        metrics = {}
        
        # Data preservation score
        if raw_table and not processed_df.empty:
            raw_cells = sum(1 for row in raw_table for cell in row if cell and cell.strip())
            processed_cells = processed_df.count().sum()
            metrics['data_preservation'] = min(processed_cells / max(raw_cells, 1), 1.0)
        else:
            metrics['data_preservation'] = 0.0
        
        # Classification confidence
        metrics['classification_confidence'] = classification.confidence_score
        
        # Structure confidence
        if not processed_df.empty:
            # Consistent column count
            expected_cols = len(processed_df.columns)
            actual_cols = [len(row) for row in raw_table if row] if raw_table else [0]
            if actual_cols:
                col_consistency = sum(1 for c in actual_cols if c == expected_cols) / len(actual_cols)
                metrics['structure_consistency'] = col_consistency
            else:
                metrics['structure_consistency'] = 0.0
            
            # Content density
            total_cells = processed_df.size
            non_null_cells = processed_df.count().sum()
            metrics['content_density'] = non_null_cells / max(total_cells, 1)
        else:
            metrics['structure_consistency'] = 0.0
            metrics['content_density'] = 0.0
        
        # Overall confidence (weighted average)
        weights = {
            'data_preservation': 0.3,
            'classification_confidence': 0.4,
            'structure_consistency': 0.2,
            'content_density': 0.1
        }
        
        metrics['overall_confidence'] = sum(
            metrics.get(key, 0) * weight 
            for key, weight in weights.items()
        )
        
        return metrics