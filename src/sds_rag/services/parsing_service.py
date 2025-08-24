"""
Financial Number Parsing Service - Microservice for parsing financial data formats.

This module provides comprehensive parsing and processing of financial data formats
commonly found in PDF documents. It handles number conversion, intelligent header
detection, and DataFrame construction with financial data type awareness.

Key Features:
- Multi-format financial number parsing (currency, percentages, magnitudes)
- Intelligent header row detection using financial keyword analysis
- Automatic numeric column detection and conversion
- DataFrame creation with proper data types
- Support for various financial notation formats (parentheses for negatives, etc.)

Classes:
    FinancialNumberParsingService: Core number parsing and conversion
    HeaderProcessingService: Intelligent table header detection
    DataFrameService: DataFrame creation and type management

"""

import re
import pandas as pd
import logging
from typing import Union, List, Tuple
from decimal import Decimal, InvalidOperation

from src.sds_rag.models.schemas import HeaderRow, TableData

logger = logging.getLogger(__name__)

class FinancialNumberParsingService:
    """Service for parsing financial numbers in various formats"""
    
    def __init__(self):
        # Currency symbols pattern
        self.currency_pattern = r'^[\$€£¥₹]'
        
        # Common financial null values
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
        
        logger.info("Financial number parsing service initialized")
    
    def is_financial_number(self, value: str) -> bool:
        """
        Check if a string represents a financial number.
        
        Args:
            value (str): String to check
            
        Returns:
            bool: True if string represents a financial number, False otherwise
        """
        if not value or not isinstance(value, str):
            return False
        
        value = value.strip()
        if not value or value in self.null_values:
            return True
        
        # Remove currency and clean
        clean_value = re.sub(self.currency_pattern, '', value).strip()
        
        # Handle parentheses
        if clean_value.startswith('(') and clean_value.endswith(')'):
            clean_value = clean_value[1:-1]
        
        # Remove formatting
        clean_value = re.sub(r'[,\s%]', '', clean_value)
        
        # Check patterns
        patterns = [
            r'^-?\d+(\.\d+)?[MBKTmkbt]?$',  # With magnitude
            r'^\d{1,3}(,\d{3})*(\.\d+)?$',  # Comma-separated
            r'^\d+$',                       # Simple integers
            r'^\d*\.\d+$'                   # Decimals
        ]
        
        return any(re.match(pattern, clean_value, re.IGNORECASE) for pattern in patterns)
    
    def parse_financial_number(self, value: Union[str, int, float]) -> float:
        """
        Parse financial number string to float.
        
        Args:
            value (Union[str, int, float]): Value to parse
            
        Returns:
            float: Parsed financial number
        """
        if pd.isna(value) or value is None:
            return pd.NA
        
        if isinstance(value, (int, float)):
            return float(value)
        
        value_str = str(value).strip()
        
        if value_str.lower() in {v.lower() for v in self.null_values}:
            return 0.0
        
        try:
            # Track negative (parentheses)
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
            
            if not clean_value:
                return 0.0
            
            try:
                number = float(clean_value) * multiplier
                if is_percentage:
                    number = number / 100
                return -number if is_negative else number
                
            except ValueError:
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
        """
        Detect columns that contain mostly financial numbers.
        
        Args:
            df (pd.DataFrame): DataFrame to detect numeric columns
            threshold (float): Threshold for financial number detection (default: 0.6)
            
        Returns:
            List[str]: List of numeric columns
        """
        numeric_columns = []
        
        for col in df.columns:
            if df[col].empty:
                continue
            
            sample_values = df[col].dropna().astype(str).head(10)
            if len(sample_values) == 0:
                continue
            
            financial_count = sum(1 for val in sample_values if self.is_financial_number(val))
            
            if financial_count / len(sample_values) >= threshold:
                numeric_columns.append(col)
        
        return numeric_columns
    
    def convert_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert DataFrame columns with financial data to numeric types.
        
        Args:
            df (pd.DataFrame): DataFrame to convert columns
            
        Returns:
            pd.DataFrame: DataFrame with numeric columns converted
        """
        df_copy = df.copy()
        numeric_columns = self.detect_numeric_columns(df_copy)
        
        for col in numeric_columns:
            logger.debug(f"Converting column '{col}' to numeric")
            df_copy[col] = df_copy[col].apply(self.parse_financial_number)
        
        logger.info(f"Converted {len(numeric_columns)} columns to numeric: {numeric_columns}")
        return df_copy

class HeaderProcessingService:
    """Service for processing and identifying table headers"""
    
    def __init__(self):
        self.financial_keywords = [
            'revenue', 'income', 'cash', 'assets', 'liabilities', 'equity',
            'sales', 'expenses', 'cost', 'margin', 'profit', 'loss',
            'quarter', 'year', 'period', 'ended', 'total', 'net'
        ]
        self.parser = FinancialNumberParsingService()
        logger.info("Header processing service initialized")
    
    def identify_header_row(self, table: TableData) -> Tuple[int, HeaderRow]:
        """
        Intelligently identify the header row in a table.
        
        Args:
            table (TableData): Table data to identify header row
            
        Returns:
            Tuple[int, HeaderRow]: Tuple of header row index and header row data
        """
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
        """
        Score a row based on its likelihood of being headers.
        
        Args:
            row (List[str]): Row to score
            
        Returns:
            float: Score of row
        """
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
            # Check for pure currency symbols
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
    
    def _clean_headers(self, headers: List[str]) -> HeaderRow:
        """
        Clean and standardize header names.
        
        Args:
            headers (List[str]): List of header names
            
        Returns:
            HeaderRow: Cleaned header names
        """
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

class DataFrameService:
    """Service for creating and managing pandas DataFrames"""
    
    def __init__(self):
        self.parser = FinancialNumberParsingService()
        logger.info("DataFrame service initialized")
    
    def create_dataframe(self, headers: HeaderRow, data_rows: TableData) -> pd.DataFrame:
        """
        Create pandas DataFrame from headers and data rows.
        
        Args:
            headers (HeaderRow): List of header names
            data_rows (TableData): List of data rows
            
        Returns:
            pd.DataFrame: Created DataFrame
        """
        if not data_rows:
            return pd.DataFrame()
        
        # Ensure consistent column count
        max_cols = max(len(headers), max(len(row) for row in data_rows) if data_rows else 0)
        
        # Pad headers if needed
        headers_padded = headers[:]
        while len(headers_padded) < max_cols:
            headers_padded.append(f"Column_{len(headers_padded) + 1}")
        
        # Pad data rows if needed
        padded_rows = []
        for row in data_rows:
            padded_row = row[:]
            while len(padded_row) < max_cols:
                padded_row.append('')
            padded_rows.append(padded_row[:max_cols])  # Trim if too long
        
        df = pd.DataFrame(padded_rows, columns=headers_padded[:max_cols])
        
        # Clean DataFrame
        df = df.replace('', pd.NA)
        df = df.dropna(how='all')  # Remove completely empty rows
        
        # Convert financial columns
        df = self.parser.convert_dataframe_columns(df)
        
        logger.debug(f"Created DataFrame: {len(df)} rows × {len(df.columns)} columns")
        return df