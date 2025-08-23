"""
Data Validation Service - Microservice for validating financial data quality.

This module provides comprehensive validation of extracted financial table data,
ensuring quality and consistency before downstream processing. It implements
multiple validation layers including structure validation, content quality checks,
and business rule validation specific to financial statements.

Key Features:
- Comprehensive data quality assessment
- Table structure validation for financial tables
- Business rule validation for financial consistency
- Content density and completeness analysis
- Financial data pattern recognition and validation

Classes:
    DataValidationService: Core data quality validation
    TableStructureValidationService: Structure and format validation  
    BusinessRuleValidationService: Financial business rule validation

"""

import re
import pandas as pd
import logging
from typing import List

from src.models.schemas import DataValidationResult, TableData, HeaderRow

logger = logging.getLogger(__name__)

class DataValidationService:
    """Service for comprehensive data quality validation"""
    
    def __init__(self):
        self.min_content_threshold = 0.1  # At least 10% non-empty cells
        self.min_dimensions = (1, 2)      # At least 1 row, 2 columns
        logger.info("Data validation service initialized")
    
    def validate_table_data(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Comprehensive validation of table data quality.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            DataValidationResult: Validation result
        """
        if df.empty:
            return DataValidationResult(
                is_valid=False,
                issues=['Empty DataFrame'],
                metrics={}
            )
        
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
        numeric_columns = df.select_dtypes(include=['number']).columns
        metrics['numeric_columns'] = len(numeric_columns)
        metrics['numeric_percentage'] = (len(numeric_columns) / len(df.columns) * 100) if df.columns.size > 0 else 0
        
        # Header validation
        metrics['has_descriptive_headers'] = self._has_descriptive_headers(df.columns.tolist())
        
        # Data consistency checks
        consistency_issues = self._check_data_consistency(df)
        issues.extend(consistency_issues)
        
        # Determine overall validity
        is_valid = (
            metrics['content_density'] >= 20 and  # At least 20% content
            metrics['missing_data_percentage'] <= 90 and  # Not too sparse
            len(df) >= 1 and  # Has at least one row
            len(df.columns) >= 2 and  # Has at least two columns
            len(consistency_issues) == 0  # No critical consistency issues
        )
        
        logger.debug(f"Table validation: valid={is_valid}, issues={len(issues)}")
        
        return DataValidationResult(
            is_valid=is_valid,
            issues=issues,
            metrics=metrics
        )
    
    def _has_descriptive_headers(self, headers: HeaderRow) -> bool:
        """
        Check if headers contain descriptive financial terms.
        
        Args:
            headers (HeaderRow): List of header names
            
        Returns:
            bool: True if headers contain descriptive financial terms, False otherwise
        """
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
    
    def _check_data_consistency(self, df: pd.DataFrame) -> List[str]:
        """
        Check for data consistency issues.
        
        Args:
            df (pd.DataFrame): DataFrame to check for consistency issues
            
        Returns:
            List[str]: List of consistency issues
        """
        issues = []
        
        # Check for columns with all identical values (except NaN)
        for col in df.columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 1:
                unique_values = non_null_values.nunique()
                if unique_values == 1:
                    issues.append(f"Column '{col}' has all identical values")
        
        # Check for suspiciously uniform data patterns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 3:
                # Check if all values are the same
                if col_data.std() == 0:
                    issues.append(f"Numeric column '{col}' has zero variance")
        
        return issues

class TableStructureValidationService:
    """Service for validating table structure and format"""
    
    def __init__(self):
        self.min_content_threshold = 0.1
        self.min_dimensions = (1, 2)
        logger.info("Table structure validation service initialized")
    
    def is_valid_financial_table(self, table: TableData) -> bool:
        """
        Check if extracted table contains valid financial data.
        
        Args:
            table (TableData): Table data to validate
            
        Returns:
            bool: True if table contains valid financial data, False otherwise
        """
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
        is_financial = self._has_financial_indicators(table_text)
        
        logger.debug(f"Table structure validation: financial_indicators={is_financial}")
        return is_financial
    
    def _extract_table_text(self, table: TableData) -> str:
        """
        Extract text content from table for analysis.
        
        Args:
            table (TableData): Table data to extract text from
            
        Returns:
            str: Extracted text content
        """
        text_parts = []
        for row in table:
            for cell in row:
                if cell and cell.strip():
                    text_parts.append(cell.strip())
        return ' '.join(text_parts)
    
    def _has_financial_indicators(self, text: str) -> bool:
        """
        Check if text contains financial indicators.
        
        Args:
            text (str): Text to check for financial indicators
            
        Returns:
            bool: True if text contains financial indicators, False otherwise
        """
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

class BusinessRuleValidationService:
    """Service for validating business rules and financial data integrity"""
    
    def __init__(self):
        logger.info("Business rule validation service initialized")
    
    def validate_financial_consistency(self, df: pd.DataFrame, table_type: str) -> List[str]:
        """
        Validate financial data consistency based on table type.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            table_type (str): Type of table (balance_sheet, income_statement, cash_flow)
            
        Returns:
            List[str]: List of validation issues
        """
        issues = []
        
        # Balance sheet specific validations
        if table_type == 'balance_sheet':
            issues.extend(self._validate_balance_sheet(df))
        
        # Income statement specific validations
        elif table_type == 'income_statement':
            issues.extend(self._validate_income_statement(df))
        
        # Cash flow specific validations
        elif table_type == 'cash_flow':
            issues.extend(self._validate_cash_flow(df))
        
        # General financial validations
        issues.extend(self._validate_general_financial_rules(df))
        
        logger.debug(f"Business rule validation: {len(issues)} issues found")
        return issues
    
    def _validate_balance_sheet(self, df: pd.DataFrame) -> List[str]:
        """
        Validate balance sheet specific rules.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            List[str]: List of validation issues
        """
        issues = []
        
        # Look for assets and liabilities columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # Check for negative assets (usually not expected)
                negative_values = (col_data < 0).sum()
                if negative_values > len(col_data) * 0.5:  # More than 50% negative
                    col_name = str(col).lower()
                    if any(term in col_name for term in ['asset', 'cash', 'inventory']):
                        issues.append(f"Column '{col}' has unexpectedly high negative values for assets")
        
        return issues
    
    def _validate_income_statement(self, df: pd.DataFrame) -> List[str]:
        """
        Validate income statement specific rules.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            List[str]: List of validation issues
        """
        issues = []
        
        # Check for reasonable revenue patterns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            col_name = str(col).lower()
            if 'revenue' in col_name or 'sales' in col_name:
                col_data = df[col].dropna()
                if len(col_data) > 0 and (col_data < 0).any():
                    issues.append(f"Revenue column '{col}' contains negative values")
        
        return issues
    
    def _validate_cash_flow(self, df: pd.DataFrame) -> List[str]:
        """
        Validate cash flow specific rules.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            List[str]: List of validation issues
        """
        issues = []
        
        # Cash flow statements can have both positive and negative values
        # This is normal, so fewer strict validations
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Check for extremely large outliers
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 2:
                q75, q25 = col_data.quantile([0.75, 0.25])
                iqr = q75 - q25
                if iqr > 0:
                    outliers = col_data[(col_data < (q25 - 1.5 * iqr)) | 
                                       (col_data > (q75 + 1.5 * iqr))]
                    if len(outliers) > len(col_data) * 0.3:  # More than 30% outliers
                        issues.append(f"Column '{col}' has unusually high number of outliers")
        
        return issues
    
    def _validate_general_financial_rules(self, df: pd.DataFrame) -> List[str]:
        """
        Validate general financial data rules.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            List[str]: List of validation issues
        """
        issues = []
        
        # Check for unreasonably large numbers (could indicate parsing errors)
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                max_val = col_data.max()
                if max_val > 1e15:  # More than quadrillion
                    issues.append(f"Column '{col}' has unreasonably large values (possible parsing error)")
        
        return issues