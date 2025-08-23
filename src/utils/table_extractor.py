"""
Table extraction utilities with multiple strategies for PDF processing.
Handles complex table structures, merged cells, and financial document layouts.
"""

import pdfplumber
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RawTable:
    """Container for raw extracted table data"""
    data: List[List[str]]
    page: int
    table_index: int
    extraction_method: str
    metadata: Dict[str, Any]

class TableExtractor:
    """Multi-strategy table extractor for financial PDFs"""
    
    def __init__(self):
        self.extraction_strategies = [
            self._text_based_extraction,
            self._lines_based_extraction,
            self._default_extraction
        ]
    
    def extract_tables_from_page(self, page, page_num: int) -> List[RawTable]:
        """Extract tables using multiple strategies for better coverage"""
        all_tables = []
        
        for strategy_idx, strategy in enumerate(self.extraction_strategies):
            try:
                strategy_name = strategy.__name__.replace('_', ' ').title()
                logger.debug(f"Applying {strategy_name} to page {page_num}")
                
                tables = strategy(page)
                
                for table_idx, table in enumerate(tables):
                    if table and self._is_valid_table_structure(table):
                        raw_table = RawTable(
                            data=table,
                            page=page_num,
                            table_index=len(all_tables),
                            extraction_method=strategy_name,
                            metadata={
                                'strategy_index': strategy_idx,
                                'raw_dimensions': f"{len(table)}x{len(table[0]) if table else 0}"
                            }
                        )
                        all_tables.append(raw_table)
                        
            except Exception as e:
                logger.warning(f"Strategy {strategy.__name__} failed on page {page_num}: {e}")
                continue
        
        # Remove duplicate tables
        unique_tables = self._remove_duplicate_tables(all_tables)
        logger.debug(f"Page {page_num}: {len(all_tables)} raw tables → {len(unique_tables)} unique tables")
        
        return unique_tables
    
    def _text_based_extraction(self, page) -> List[List[List[str]]]:
        """Text-based extraction for borderless tables"""
        settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 3
        }
        return page.extract_tables(settings) or []
    
    def _lines_based_extraction(self, page) -> List[List[List[str]]]:
        """Lines-based extraction for bordered tables"""
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
        """Default pdfplumber extraction as fallback"""
        return page.extract_tables() or []
    
    def _is_valid_table_structure(self, table: List[List[str]]) -> bool:
        """Basic validation for table structure"""
        if not table or len(table) < 1:
            return False
        
        # Check if we have meaningful content
        non_empty_rows = [row for row in table if row and any(cell and cell.strip() for cell in row)]
        if len(non_empty_rows) < 1:
            return False
        
        # Check if we have at least 2 columns in any row
        has_multiple_cols = any(len([cell for cell in row if cell and cell.strip()]) >= 2 for row in non_empty_rows)
        return has_multiple_cols
    
    def _remove_duplicate_tables(self, tables: List[RawTable]) -> List[RawTable]:
        """Remove duplicate tables based on content similarity"""
        if len(tables) <= 1:
            return tables
        
        unique_tables = []
        
        for table in tables:
            is_duplicate = False
            table_content = self._normalize_table_content(table.data)
            
            for existing in unique_tables:
                existing_content = self._normalize_table_content(existing.data)
                
                # Check for content similarity
                if self._tables_are_similar(table_content, existing_content):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _normalize_table_content(self, table: List[List[str]]) -> str:
        """Normalize table content for comparison"""
        content = []
        for row in table[:5]:  # Compare first 5 rows
            row_content = []
            for cell in row[:5]:  # First 5 columns
                if cell and cell.strip():
                    # Normalize whitespace and case
                    normalized = re.sub(r'\s+', ' ', cell.strip().lower())
                    row_content.append(normalized)
            if row_content:
                content.append('|'.join(row_content))
        return '\n'.join(content)
    
    def _tables_are_similar(self, content1: str, content2: str, threshold: float = 0.8) -> bool:
        """Check if two table contents are similar"""
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

class TableCleaner:
    """Handles table cleaning and merged cell recovery"""
    
    def clean_raw_table(self, raw_table: RawTable) -> List[List[str]]:
        """Clean and standardize raw table data with merged cell handling"""
        if not raw_table.data:
            return []
        
        # First pass: basic cleaning
        cleaned = self._basic_cleaning(raw_table.data)
        
        # Second pass: handle merged cells
        if len(cleaned) > 1:
            cleaned = self._fix_merged_cells(cleaned)
        
        return cleaned
    
    def _basic_cleaning(self, raw_table: List[List[str]]) -> List[List[str]]:
        """Basic cell cleaning and row filtering"""
        cleaned = []
        
        for row in raw_table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append('')
                else:
                    # Clean whitespace and standardize
                    cleaned_cell = str(cell).strip()
                    cleaned_cell = re.sub(r'\s+', ' ', cleaned_cell)
                    # Handle line breaks in cells
                    cleaned_cell = cleaned_cell.replace('\n', ' ').replace('\r', ' ')
                    cleaned_row.append(cleaned_cell)
            
            # Only add rows with meaningful content
            if any(cell and len(cell.strip()) > 0 for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        return cleaned
    
    def _fix_merged_cells(self, table: List[List[str]]) -> List[List[str]]:
        """Attempt to fix common merged cell issues"""
        if not table or len(table) < 2:
            return table
        
        # Find the maximum number of columns
        max_cols = max(len(row) for row in table)
        
        # Pad all rows to same length
        padded_table = []
        for row in table:
            padded_row = row[:]
            while len(padded_row) < max_cols:
                padded_row.append('')
            padded_table.append(padded_row)
        
        # Look for patterns where content might be split across rows
        fixed_table = []
        i = 0
        while i < len(padded_table):
            current_row = padded_table[i]
            
            # Check if current row looks incomplete (too many empty cells)
            non_empty_count = sum(1 for cell in current_row if cell and cell.strip())
            
            # If row has content in first column but sparse elsewhere, might be merged
            if (i + 1 < len(padded_table) and 
                non_empty_count > 0 and non_empty_count < max_cols // 2 and
                current_row[0] and current_row[0].strip()):
                
                next_row = padded_table[i + 1]
                next_non_empty = sum(1 for cell in next_row if cell and cell.strip())
                
                # If next row complements this one, try to merge
                if next_non_empty > 0 and next_non_empty <= max_cols // 2:
                    merged_row = self._merge_rows(current_row, next_row, max_cols)
                    
                    # Only use merged row if it's better than either individual row
                    merged_non_empty = sum(1 for cell in merged_row if cell and cell.strip())
                    if merged_non_empty > max(non_empty_count, next_non_empty):
                        fixed_table.append(merged_row)
                        i += 2  # Skip next row since we merged it
                        continue
            
            fixed_table.append(current_row)
            i += 1
        
        return fixed_table
    
    def _merge_rows(self, row1: List[str], row2: List[str], max_cols: int) -> List[str]:
        """Merge two rows by combining non-empty cells"""
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