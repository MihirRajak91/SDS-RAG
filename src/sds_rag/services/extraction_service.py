"""
Table Extraction Service - Microservice for extracting tables from PDF pages.

This module provides robust table extraction from PDF documents using multiple
extraction strategies. It employs a multi-strategy approach to maximize extraction
success rate and includes comprehensive table validation and deduplication.

Key Features:
- Multi-strategy extraction (text-based, lines-based, default)
- Table structure validation and quality assessment
- Duplicate table detection and removal
- Merged cell handling and table cleaning
- Comprehensive extraction metadata

Classes:
    TableExtractionService: Main extraction service with multiple strategies
    TableCleaningService: Post-extraction cleaning and standardization
"""

import re
import logging
from typing import List, Dict, Any

from src.sds_rag.models.schemas import RawTableData, ExtractionMethod

logger = logging.getLogger(__name__)

class TableExtractionService:
    """Service for extracting tables from PDF pages using multiple strategies"""
    
    def __init__(self):
        self.extraction_strategies = [
            (ExtractionMethod.TEXT_BASED, self._text_based_extraction),
            (ExtractionMethod.LINES_BASED, self._lines_based_extraction),
            (ExtractionMethod.DEFAULT, self._default_extraction)
        ]
        logger.info("Table extraction service initialized")
    
    def extract_tables_from_page(self, page, page_num: int) -> List[RawTableData]:
        """
        Extract tables using multiple strategies.
        
        Args:
            page (pdfplumber.Page): PDF page to extract tables from
            page_num (int): Page number for logging and metadata
            
        Returns:
            List[RawTableData]: List of extracted tables with metadata
        """
        all_tables = []
        
        for method, strategy in self.extraction_strategies:
            try:
                strategy_name = method.value.replace('_', ' ').title()
                logger.debug(f"Applying {strategy_name} to page {page_num}")
                
                tables = strategy(page)
                
                for _, table in enumerate(tables):
                    if table and self._is_valid_table_structure(table):
                        raw_table = RawTableData(
                            data=table,
                            page=page_num,
                            table_index=len(all_tables),
                            extraction_method=method,
                            metadata={
                                'raw_dimensions': f"{len(table)}x{len(table[0]) if table else 0}"
                            }
                        )
                        all_tables.append(raw_table)
                        
            except Exception as e:
                logger.warning(f"Strategy {method.value} failed on page {page_num}: {e}")
                continue
        
        # Remove duplicate tables
        unique_tables = self._remove_duplicate_tables(all_tables)
        logger.debug(f"Page {page_num}: {len(all_tables)} raw tables → {len(unique_tables)} unique tables")
        
        return unique_tables
    
    def _text_based_extraction(self, page) -> List[List[List[str]]]:
        """
        Text-based extraction for borderless tables.
        
        Args:
            page (pdfplumber.Page): PDF page to extract tables from
            
        Returns:
            List[List[List[str]]]: List of extracted tables
        """
        settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 3
        }
        return page.extract_tables(settings) or []
    
    def _lines_based_extraction(self, page) -> List[List[List[str]]]:
        """
        Lines-based extraction for bordered tables.
        
        Args:
            page (pdfplumber.Page): PDF page to extract tables from
            
        Returns:
            List[List[List[str]]]: List of extracted tables
        """
        settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 2,
            "join_tolerance": 2,
            "edge_min_length": 5,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
            "intersection_tolerance": 2,
        }
        return page.extract_tables(settings) or []
    
    def _default_extraction(self, page) -> List[List[List[str]]]:
        """
        Default pdfplumber extraction as fallback.
        
        Args:
            page (pdfplumber.Page): PDF page to extract tables from
            
        Returns:
            List[List[List[str]]]: List of extracted tables
        """
        return page.extract_tables() or []
    
    def _is_valid_table_structure(self, table: List[List[str]]) -> bool:
        """
        Basic validation for table structure.
        
        Args:
            table (List[List[str]]): Table data to validate
            
        Returns:
            bool: True if table has valid structure, False otherwise
        """
        if not table or len(table) < 1:
            return False
        
        non_empty_rows = [row for row in table if row and any(cell and cell.strip() for cell in row)]
        if len(non_empty_rows) < 1:
            return False
        
        has_multiple_cols = any(len([cell for cell in row if cell and cell.strip()]) >= 2 for row in non_empty_rows)
        return has_multiple_cols
    
    def _remove_duplicate_tables(self, tables: List[RawTableData]) -> List[RawTableData]:
        """
        Remove duplicate tables based on content similarity.
        
        Args:
            tables (List[RawTableData]): List of raw tables to deduplicate
            
        Returns:
            List[RawTableData]: List of unique tables
        """
        if len(tables) <= 1:
            return tables
        
        unique_tables = []
        
        for table in tables:
            is_duplicate = False
            table_content = self._normalize_table_content(table.data)
            
            for existing in unique_tables:
                existing_content = self._normalize_table_content(existing.data)
                
                if self._tables_are_similar(table_content, existing_content):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _normalize_table_content(self, table: List[List[str]]) -> str:
        """
        Normalize table content for comparison.
        
        Args:
            table (List[List[str]]): Table data to normalize
            
        Returns:
            str: Normalized table content
        """
        content = []
        for row in table[:5]:  # Compare first 5 rows
            row_content = []
            for cell in row[:5]:  # First 5 columns
                if cell and cell.strip():
                    normalized = re.sub(r'\s+', ' ', cell.strip().lower())
                    row_content.append(normalized)
            if row_content:
                content.append('|'.join(row_content))
        return '\n'.join(content)
    
    def _tables_are_similar(self, content1: str, content2: str, threshold: float = 0.8) -> bool:
        """
        Check if two table contents are similar.
        
        Args:
            content1 (str): First table content for comparison
            content2 (str): Second table content for comparison
            threshold (float): Similarity threshold (default: 0.8)
            
        Returns:
            bool: True if tables are similar, False otherwise
        """
        if not content1 or not content2:
            return False
        
        lines1 = set(content1.split('\n'))
        lines2 = set(content2.split('\n'))
        
        if not lines1 or not lines2:
            return False
        
        intersection = len(lines1.intersection(lines2))
        union = len(lines1.union(lines2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

class TableCleaningService:
    """Service for cleaning and processing raw table data"""
    
    def clean_raw_table(self, raw_table: RawTableData) -> List[List[str]]:
        """
        Clean and standardize raw table data with merged cell handling.
        
        Args:
            raw_table (RawTableData): Raw table data to clean
            
        Returns:
            List[List[str]]: Cleaned table data
        """
        if not raw_table.data:
            return []
        
        # Basic cleaning
        cleaned = self._basic_cleaning(raw_table.data)
        
        # Handle merged cells
        if len(cleaned) > 1:
            cleaned = self._fix_merged_cells(cleaned)
        
        logger.debug(f"Cleaned table: {len(raw_table.data)} → {len(cleaned)} rows")
        return cleaned
    
    def _basic_cleaning(self, raw_table: List[List[str]]) -> List[List[str]]:
        """
        Basic cell cleaning and row filtering.
        
        Args:
            raw_table (List[List[str]]): Raw table data to clean
            
        Returns:
            List[List[str]]: Cleaned table data
        """
        cleaned = []
        
        for row in raw_table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append('')
                else:
                    cleaned_cell = str(cell).strip()
                    cleaned_cell = re.sub(r'\s+', ' ', cleaned_cell)
                    cleaned_cell = cleaned_cell.replace('\n', ' ').replace('\r', ' ')
                    cleaned_row.append(cleaned_cell)
            
            if any(cell and len(cell.strip()) > 0 for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        return cleaned
    
    def _fix_merged_cells(self, table: List[List[str]]) -> List[List[str]]:
        """
        Attempt to fix common merged cell issues.
        
        Args:
            table (List[List[str]]): Table data to fix merged cells
            
        Returns:
            List[List[str]]: Table data with merged cells fixed
        """
        if not table or len(table) < 2:
            return table
        
        max_cols = max(len(row) for row in table)
        
        # Pad all rows to same length
        padded_table = []
        for row in table:
            padded_row = row[:]
            while len(padded_row) < max_cols:
                padded_row.append('')
            padded_table.append(padded_row)
        
        # Look for split rows and merge them
        fixed_table = []
        i = 0
        while i < len(padded_table):
            current_row = padded_table[i]
            non_empty_count = sum(1 for cell in current_row if cell and cell.strip())
            
            # Try to merge with next row if current is sparse
            if (i + 1 < len(padded_table) and 
                non_empty_count > 0 and non_empty_count < max_cols // 2 and
                current_row[0] and current_row[0].strip()):
                
                next_row = padded_table[i + 1]
                next_non_empty = sum(1 for cell in next_row if cell and cell.strip())
                
                if next_non_empty > 0 and next_non_empty <= max_cols // 2:
                    merged_row = self._merge_rows(current_row, next_row, max_cols)
                    merged_non_empty = sum(1 for cell in merged_row if cell and cell.strip())
                    
                    if merged_non_empty > max(non_empty_count, next_non_empty):
                        fixed_table.append(merged_row)
                        i += 2  # Skip next row
                        continue
            
            fixed_table.append(current_row)
            i += 1
        
        return fixed_table
    
    def _merge_rows(self, row1: List[str], row2: List[str], max_cols: int) -> List[str]:
        """
        Merge two rows by combining non-empty cells.
        
        Args:
            row1 (List[str]): First row to merge
            row2 (List[str]): Second row to merge
            max_cols (int): Maximum number of columns
            
        Returns:
            List[str]: Merged row
        """
        merged_row = []
        for j in range(max_cols):
            cell1 = row1[j] if j < len(row1) else ''
            cell2 = row2[j] if j < len(row2) else ''
            
            if cell1 and cell1.strip():
                merged_row.append(cell1)
            elif cell2 and cell2.strip():
                merged_row.append(cell2)
            else:
                merged_row.append('')
        
        return merged_row