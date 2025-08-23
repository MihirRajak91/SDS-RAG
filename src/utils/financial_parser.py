"""
Financial number parsing and data type conversion utilities.
Handles various financial number formats, currencies, and data validation.
"""

import re
import pandas as pd
import logging
from typing import Union, Optional, List, Dict
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

class FinancialNumberParser:
    """Parser for financial numbers in various formats"""
    
    def __init__(self):
        # Currency symbols pattern
        self.currency_pattern = r'^[\$€£¥₹]'
        
        # Common financial dash representations for zero/null
        self.null_values = {
            '—', '–', '-', 'n/a', 'N/A', 'n.a.', 'N.A.', 'na', 'NA', 
            'nil', 'Nil', 'NULL', 'null', '', ' '
        }
        
        # Magnitude multipliers
        self.magnitude_multipliers = {
            'k': 1_000, 'K': 1_000,
            'm': 1_000_000, 'M': 1_000_000,
            'b': 1_000_000_000, 'B': 1_000_000_000,
            't': 1_000_000_000_000, 'T': 1_000_000_000_000
        }
    
    def is_financial_number(self, value: str) -> bool:
        """Check if a string represents a financial number"""
        if not value or not isinstance(value, str):
            return False
        
        value = value.strip()
        
        if not value:
            return False
        
        # Handle null values
        if value in self.null_values:
            return True
        
        # Remove currency symbols and clean
        clean_value = re.sub(self.currency_pattern, '', value).strip()
        
        # Handle parentheses for negatives
        if clean_value.startswith('(') and clean_value.endswith(')'):
            clean_value = clean_value[1:-1]
        
        # Remove commas, spaces, and percentage signs
        clean_value = re.sub(r'[,\s%]', '', clean_value)
        
        # Check for number patterns with optional magnitude suffix
        patterns = [
            r'^-?\d+(\.\d+)?[MBKTmkbt]?$',  # Standard numbers with magnitude
            r'^\d{1,3}(,\d{3})*(\.\d+)?$',  # Comma-separated numbers
            r'^\d+$',                       # Simple integers
            r'^\d*\.\d+$'                   # Decimals
        ]
        
        return any(re.match(pattern, clean_value, re.IGNORECASE) for pattern in patterns)
    
    def parse_financial_number(self, value: Union[str, int, float]) -> Optional[float]:
        """Parse financial number string to float"""
        if pd.isna(value) or value is None:
            return pd.NA
        
        # Handle already numeric values
        if isinstance(value, (int, float)):
            return float(value)
        
        value_str = str(value).strip()
        
        # Handle null values
        if value_str.lower() in {v.lower() for v in self.null_values}:
            return 0.0
        
        try:
            # Track if number is negative (parentheses)
            is_negative = False
            if value_str.startswith('(') and value_str.endswith(')'):
                value_str = value_str[1:-1]
                is_negative = True
            
            # Remove currency symbols
            clean_value = re.sub(self.currency_pattern, '', value_str).strip()
            
            # Handle percentage
            is_percentage = clean_value.endswith('%')
            if is_percentage:
                clean_value = clean_value[:-1]
            
            # Remove commas and spaces
            clean_value = re.sub(r'[,\s]', '', clean_value)
            
            # Handle magnitude suffixes
            multiplier = 1
            if clean_value and clean_value[-1].upper() in self.magnitude_multipliers:
                suffix = clean_value[-1]
                multiplier = self.magnitude_multipliers[suffix]
                clean_value = clean_value[:-1]
            
            # Parse the base number
            if not clean_value:
                return 0.0
            
            try:
                number = float(clean_value) * multiplier
                
                if is_percentage:
                    number = number / 100
                
                return -number if is_negative else number
                
            except ValueError:
                # Try with Decimal for high precision
                try:
                    number = float(Decimal(clean_value)) * multiplier
                    if is_percentage:
                        number = number / 100
                    return -number if is_negative else number
                except (InvalidOperation, ValueError):
                    logger.debug(f"Could not parse financial number: {value}")
                    return pd.NA
                    
        except Exception as e:
            logger.debug(f"Error parsing financial number '{value}': {e}")
            return pd.NA
    
    def detect_numeric_columns(self, df: pd.DataFrame, threshold: float = 0.6) -> List[str]:
        """Detect columns that contain mostly financial numbers"""
        numeric_columns = []
        
        for col in df.columns:
            if df[col].empty:
                continue
            
            # Sample non-null values
            sample_values = df[col].dropna().astype(str).head(10)
            if len(sample_values) == 0:
                continue
            
            # Count financial numbers
            financial_count = sum(1 for val in sample_values if self.is_financial_number(val))
            
            # If majority are financial numbers, mark as numeric column
            if financial_count / len(sample_values) >= threshold:
                numeric_columns.append(col)
        
        return numeric_columns
    
    def convert_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns with financial data to numeric types"""
        df_copy = df.copy()
        
        numeric_columns = self.detect_numeric_columns(df_copy)
        
        for col in numeric_columns:
            logger.debug(f"Converting column '{col}' to numeric")
            df_copy[col] = df_copy[col].apply(self.parse_financial_number)
        
        logger.info(f"Converted {len(numeric_columns)} columns to numeric: {numeric_columns}")
        return df_copy

class DataValidator:
    """Validates financial data quality and consistency"""
    
    def __init__(self):
        self.parser = FinancialNumberParser()
    
    def validate_table_data(self, df: pd.DataFrame) -> dict:
        """Comprehensive validation of table data quality"""
        if df.empty:
            return {
                'is_valid': False,
                'issues': ['Empty DataFrame'],
                'metrics': {}
            }
        
        issues = []
        metrics = {}
        
        # Basic structure validation
        metrics['total_cells'] = df.size
        metrics['total_rows'] = len(df)
        metrics['total_columns'] = len(df.columns)
        
        # Missing data analysis
        null_count = df.isnull().sum().sum()
        metrics['missing_data_percentage'] = (null_count / df.size * 100) if df.size > 0 else 0
        
        if metrics['missing_data_percentage'] > 80:
            issues.append(f"High missing data: {metrics['missing_data_percentage']:.1f}%")
        
        # Content validation
        non_empty_cells = df.count().sum()
        metrics['content_density'] = (non_empty_cells / df.size * 100) if df.size > 0 else 0
        
        # Numeric data validation
        numeric_columns = self.parser.detect_numeric_columns(df)
        metrics['numeric_columns'] = len(numeric_columns)
        metrics['numeric_percentage'] = (len(numeric_columns) / len(df.columns) * 100) if df.columns.size > 0 else 0
        
        # Header validation
        metrics['has_descriptive_headers'] = self._has_descriptive_headers(df.columns.tolist())
        
        # Determine overall validity
        is_valid = (
            metrics['content_density'] >= 20 and  # At least 20% content
            metrics['missing_data_percentage'] <= 90 and  # Not too sparse
            len(df) >= 1 and  # Has at least one row
            len(df.columns) >= 2  # Has at least two columns
        )
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'metrics': metrics
        }
    
    def _has_descriptive_headers(self, headers: List[str]) -> bool:
        """Check if headers contain descriptive financial terms"""
        financial_terms = [
            'revenue', 'income', 'cash', 'assets', 'liabilities', 'equity',
            'sales', 'expenses', 'cost', 'margin', 'profit', 'loss',
            'quarter', 'year', 'period', 'ended', 'total', 'net'
        ]
        
        descriptive_count = 0
        for header in headers:
            if header and isinstance(header, str):
                header_lower = header.lower()
                if any(term in header_lower for term in financial_terms):
                    descriptive_count += 1
        
        return descriptive_count > 0 and descriptive_count >= len(headers) * 0.3

class HeaderProcessor:
    """Processes and improves table headers"""
    
    def __init__(self):
        self.financial_keywords = [
            'revenue', 'income', 'cash', 'assets', 'liabilities', 'equity',
            'sales', 'expenses', 'cost', 'margin', 'profit', 'loss',
            'quarter', 'year', 'period', 'ended', 'total', 'net'
        ]
        self.parser = FinancialNumberParser()
    
    def identify_header_row(self, table: List[List[str]]) -> tuple[int, List[str]]:
        """Intelligently identify the header row in a table"""
        if not table or len(table) < 1:
            return 0, []
        
        header_candidates = []
        
        for i, row in enumerate(table[:4]):  # Check first 4 rows
            if not row or all(not cell for cell in row):
                continue
            
            score = self._score_header_row(row)
            header_candidates.append((i, score, row))
        
        if not header_candidates:
            # Fallback to first non-empty row
            for i, row in enumerate(table):
                if row and any(cell and cell.strip() for cell in row):
                    return i, self._clean_headers(row)
            return 0, []
        
        # Select best candidate
        header_candidates.sort(key=lambda x: x[1], reverse=True)
        header_idx, score, header_row = header_candidates[0]
        
        # If best candidate has very low score, create generic headers
        if score < 0:
            generic_headers = [f"Column_{j+1}" for j in range(len(header_row))]
            return header_idx, generic_headers
        
        return header_idx, self._clean_headers(header_row)
    
    def _score_header_row(self, row: List[str]) -> float:
        """Score a row based on its likelihood of being headers"""
        if not row:
            return -10
        
        score = 0
        text_cells = 0
        numeric_cells = 0
        currency_cells = 0
        descriptive_cells = 0
        
        for cell in row:
            if not cell or not cell.strip():
                continue
            
            cell_clean = cell.strip().lower()
            
            # Check for descriptive header text
            if any(keyword in cell_clean for keyword in self.financial_keywords):
                descriptive_cells += 1
                text_cells += 1
            # Check for pure currency symbols (likely not headers)
            elif cell_clean in ['$', '€', '£', '¥'] or re.match(r'^[\$€£¥]?[\d,]+$', cell_clean):
                currency_cells += 1
            # Check for financial numbers
            elif self.parser.is_financial_number(cell):
                numeric_cells += 1
            else:
                text_cells += 1
        
        # Score calculation
        score += descriptive_cells * 3  # Descriptive terms are good
        score += text_cells * 2         # Text content is good for headers
        score -= numeric_cells * 1      # Pure numbers less likely to be headers
        score -= currency_cells * 2     # Currency symbols unlikely to be headers
        
        # Penalize rows that are mostly numbers/currency
        total_content = text_cells + numeric_cells + currency_cells + descriptive_cells
        if total_content > 0:
            numeric_ratio = (numeric_cells + currency_cells) / total_content
            if numeric_ratio > 0.7:
                score -= 5
        
        return score
    
    def _clean_headers(self, headers: List[str]) -> List[str]:
        """Clean and standardize header names"""
        cleaned_headers = []
        
        for i, header in enumerate(headers):
            if header and header.strip():
                clean_header = header.strip()
                # Remove currency symbols from headers
                clean_header = re.sub(r'^[\$€£¥]+', '', clean_header).strip()
                # If header is just a number, make it descriptive
                if self.parser.is_financial_number(clean_header):
                    clean_header = f"Value_{i+1}"
                cleaned_headers.append(clean_header if clean_header else f"Column_{i+1}")
            else:
                cleaned_headers.append(f"Column_{i+1}")
        
        return cleaned_headers