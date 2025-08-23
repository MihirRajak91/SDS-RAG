"""
Document processing utilities with specialized financial PDF extraction using pdfplumber.
Optimized for structured financial documents like 10-Q, 10-K reports.
"""

import pdfplumber
import pandas as pd
import PyPDF2
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path
import logging
from dataclasses import dataclass
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress PDF parsing warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pdfplumber")

@dataclass
class ExtractedTable:
    """Container for extracted table data"""
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
    """Container for processed document data"""
    structured_tables: List[ExtractedTable]
    text_chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    extraction_stats: Dict[str, Any]

class FinancialPDFProcessor:
    """Specialized processor for financial PDFs with complex table structures"""
    
    def __init__(self):
        self.financial_keywords = {
            'income_statement': [
                'net sales', 'revenue', 'gross margin', 'operating income', 
                'net income', 'earnings per share', 'cost of sales'
            ],
            'balance_sheet': [
                'assets', 'liabilities', 'shareholders equity', 'current assets',
                'cash and cash equivalents', 'marketable securities'
            ],
            'cash_flow': [
                'cash flow', 'operating activities', 'investing activities',
                'financing activities', 'cash generated', 'cash used'
            ],
            'product_revenue': [
                'iphone', 'mac', 'ipad', 'services', 'wearables', 'home and accessories'
            ],
            'geographic_segment': [
                'americas', 'europe', 'greater china', 'japan', 'rest of asia pacific'
            ],
            'comprehensive_income': [
                'comprehensive income', 'other comprehensive income', 
                'foreign currency translation', 'unrealized gains'
            ],
            'shareholders_equity': [
                'common stock', 'retained earnings', 'accumulated other comprehensive',
                'share repurchase', 'dividends'
            ]
        }
    
    def process_document(self, pdf_path: str) -> ProcessedDocument:
        """Main method to process financial PDF document"""
        logger.info(f"Processing financial PDF: {pdf_path}")
        
        try:
            # Extract structured tables
            tables = self.extract_financial_tables(pdf_path)
            
            # Extract text content
            text_chunks = self.extract_text_content(pdf_path)
            
            # Document metadata
            metadata = self.extract_document_metadata(pdf_path)
            
            # Extraction statistics
            stats = {
                'total_pages': len(self._get_pages(pdf_path)),
                'tables_extracted': len(tables),
                'text_chunks': len(text_chunks),
                'table_types': list(set(table.table_type for table in tables))
            }
            
            return ProcessedDocument(
                structured_tables=tables,
                text_chunks=text_chunks,
                metadata=metadata,
                extraction_stats=stats
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def extract_financial_tables(self, pdf_path: str) -> List[ExtractedTable]:
        """Extract and classify financial tables using pdfplumber"""
        extracted_tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Successfully opened PDF with {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        logger.debug(f"Processing page {page_num + 1}")
                        
                        # Extract tables with multiple strategies
                        tables = self._extract_tables_from_page(page)
                        logger.debug(f"Found {len(tables)} potential tables on page {page_num + 1}")
                        
                        for table_idx, raw_table in enumerate(tables):
                            if self._is_valid_financial_table(raw_table):
                                try:
                                    extracted_table = self._process_raw_table(
                                        raw_table, page_num + 1, table_idx
                                    )
                                    extracted_tables.append(extracted_table)
                                except Exception as e:
                                    logger.warning(f"Failed to process table {table_idx} on page {page_num + 1}: {e}")
                                    continue
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to open or process PDF {pdf_path}: {e}")
            raise
        
        logger.info(f"Successfully extracted {len(extracted_tables)} tables")
        return extracted_tables
    
    def _extract_tables_from_page(self, page) -> List[List[List[str]]]:
        """Extract tables using multiple strategies for better coverage"""
        tables = []
        
        # Strategy 1: Text-based extraction (for borderless tables)
        text_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 3
        }
        text_tables = page.extract_tables(text_settings)
        if text_tables:
            tables.extend(text_tables)
        
        # Strategy 2: Lines-based extraction (for bordered tables)
        lines_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 2,
            "join_tolerance": 2,
            "edge_min_length": 5,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
            "intersection_tolerance": 2,
        }
        lines_tables = page.extract_tables(lines_settings)
        if lines_tables:
            for table in lines_tables:
                if table and table not in tables:
                    tables.append(table)
        
        # Strategy 3: Default extraction as fallback
        try:
            default_tables = page.extract_tables()
            if default_tables:
                for table in default_tables:
                    if table and table not in tables:
                        tables.append(table)
        except:
            pass  # Default extraction fallback
        
        return tables
    
    def _is_valid_financial_table(self, table: List[List[str]]) -> bool:
        """Check if extracted table contains valid financial data"""
        if not table or len(table) < 1:
            return False
        
        # Be more lenient with dimensions - single row tables can be valid
        non_empty_rows = [row for row in table if row and any(cell and cell.strip() for cell in row)]
        if len(non_empty_rows) < 1:
            return False
        
        # Check if we have at least 2 columns in any row
        has_multiple_cols = any(len([cell for cell in row if cell and cell.strip()]) >= 2 for row in non_empty_rows)
        if not has_multiple_cols:
            return False
        
        # Check for financial patterns in the content
        table_text = ' '.join([
            ' '.join([cell.strip() if cell else '' for cell in row]) 
            for row in non_empty_rows
        ]).lower()
        
        if not table_text.strip():
            return False
        
        # Look for financial indicators
        has_currency = bool(re.search(r'[\$€£¥]', table_text))
        has_numbers = bool(re.search(r'\d', table_text))  # Any digit
        has_financial_numbers = bool(re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\(\d+\)|\d+\.\d+', table_text))
        has_financial_terms = any(
            keyword in table_text 
            for keywords in self.financial_keywords.values()
            for keyword in keywords
        )
        has_date_terms = bool(re.search(r'\b(quarter|q[1-4]|2022|2023|2024|ended|months|year)\b', table_text))
        
        # More lenient criteria - any financial indicator is enough
        return has_numbers and (has_currency or has_financial_terms or has_financial_numbers or has_date_terms)
    
    def _process_raw_table(self, raw_table: List[List[str]], page: int, table_idx: int) -> ExtractedTable:
        """Process raw table data into structured format"""
        
        # Clean and prepare table data
        cleaned_table = self._clean_table_data(raw_table)
        
        # Extract headers and data
        headers, data_rows = self._separate_headers_and_data(cleaned_table)
        
        # Create DataFrame
        df = self._create_dataframe(headers, data_rows)
        
        # Identify table type
        table_type = self._classify_table_type(headers, df)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(df, table_type)
        
        # Generate metadata
        metadata = self._generate_table_metadata(df, page, table_type)
        
        return ExtractedTable(
            page=page,
            table_index=table_idx,
            table_type=table_type,
            headers=headers,
            data=df,
            raw_table=raw_table,
            confidence_score=confidence,
            metadata=metadata
        )
    
    def _clean_table_data(self, raw_table: List[List[str]]) -> List[List[str]]:
        """Clean and standardize table data with better merged cell handling"""
        if not raw_table:
            return []
        
        cleaned = []
        
        # First pass: clean cells and identify structure
        for row_idx, row in enumerate(raw_table):
            cleaned_row = []
            for cell_idx, cell in enumerate(row):
                if cell is None:
                    cleaned_row.append('')
                else:
                    # Clean whitespace and standardize
                    cleaned_cell = str(cell).strip()
                    cleaned_cell = re.sub(r'\s+', ' ', cleaned_cell)
                    # Handle line breaks in cells
                    cleaned_cell = cleaned_cell.replace('\n', ' ').replace('\r', ' ')
                    cleaned_row.append(cleaned_cell)
            
            # Only add non-empty rows or rows with meaningful content
            if any(cell and len(cell.strip()) > 0 for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        # Second pass: handle merged cell patterns
        if len(cleaned) > 1:
            cleaned = self._fix_merged_cells(cleaned)
        
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
                    merged_row = []
                    for j in range(max_cols):
                        current_cell = current_row[j] if j < len(current_row) else ''
                        next_cell = next_row[j] if j < len(next_row) else ''
                        
                        if current_cell and current_cell.strip():
                            merged_row.append(current_cell)
                        elif next_cell and next_cell.strip():
                            merged_row.append(next_cell)
                        else:
                            merged_row.append('')
                    
                    # Only use merged row if it's better than either individual row
                    merged_non_empty = sum(1 for cell in merged_row if cell and cell.strip())
                    if merged_non_empty > max(non_empty_count, next_non_empty):
                        fixed_table.append(merged_row)
                        i += 2  # Skip next row since we merged it
                        continue
            
            fixed_table.append(current_row)
            i += 1
        
        return fixed_table
    
    def _separate_headers_and_data(self, table: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """Separate headers from data rows intelligently"""
        if not table or len(table) < 2:
            return [], table if table else []
        
        # Find header row using multiple criteria
        header_candidates = []
        
        for i, row in enumerate(table[:4]):  # Check first 4 rows
            if not row or all(not cell for cell in row):
                continue
                
            # Count different types of content
            text_cells = 0
            numeric_cells = 0
            currency_cells = 0
            descriptive_cells = 0
            
            for cell in row:
                if not cell or not cell.strip():
                    continue
                    
                cell_clean = cell.strip().lower()
                
                # Check for descriptive header text
                if any(word in cell_clean for word in ['revenue', 'income', 'cash', 'assets', 'liabilities', 
                                                      'equity', 'sales', 'expenses', 'cost', 'margin',
                                                      'quarter', 'year', 'period', 'ended']):
                    descriptive_cells += 1
                    text_cells += 1
                # Check for pure currency symbols (likely not headers)
                elif cell_clean in ['$', '€', '£', '¥'] or re.match(r'^[\$€£¥]?[\d,]+$', cell_clean):
                    currency_cells += 1
                # Check for financial numbers
                elif self._is_financial_number(cell):
                    numeric_cells += 1
                # Check for text content
                elif not self._is_financial_number(cell):
                    text_cells += 1
            
            # Score based on header likelihood
            # Headers should have descriptive text, minimal pure numbers
            score = descriptive_cells * 3 + text_cells * 2 - numeric_cells * 1 - currency_cells * 2
            
            # Penalize rows that are mostly currency symbols or numbers
            total_content = text_cells + numeric_cells + currency_cells + descriptive_cells
            if total_content > 0:
                if (numeric_cells + currency_cells) / total_content > 0.7:
                    score -= 5
            
            header_candidates.append((i, score, row))
        
        # Select best header row, or use first row if no clear winner
        if header_candidates:
            header_candidates.sort(key=lambda x: x[1], reverse=True)
            header_idx = header_candidates[0][0]
            
            # If top candidate has negative score, likely no proper headers
            if header_candidates[0][1] < 0:
                # Use first non-empty row and treat as data
                for i, row in enumerate(table):
                    if row and any(cell and cell.strip() for cell in row):
                        headers = [f"Column_{j+1}" for j in range(len(row))]
                        return headers, table[i:]
        else:
            header_idx = 0
        
        headers = table[header_idx]
        data_rows = table[header_idx + 1:] if header_idx + 1 < len(table) else []
        
        # Clean and improve headers
        cleaned_headers = []
        for i, header in enumerate(headers):
            if header and header.strip():
                clean_header = header.strip()
                # Remove currency symbols from headers
                clean_header = re.sub(r'^[\$€£¥]+', '', clean_header).strip()
                # If header is just a number, make it descriptive
                if self._is_financial_number(clean_header):
                    clean_header = f"Value_{i+1}"
                cleaned_headers.append(clean_header if clean_header else f"Column_{i + 1}")
            else:
                cleaned_headers.append(f"Column_{i + 1}")
        
        return cleaned_headers, data_rows
    
    def _is_numeric(self, value: str) -> bool:
        """Check if value is numeric (including financial formats)"""
        return self._is_financial_number(value)
    
    def _is_financial_number(self, value: str) -> bool:
        """Enhanced financial number detection"""
        if not value or not value.strip():
            return False
        
        value = value.strip()
        
        # Handle common financial representations
        # Remove currency symbols and spaces
        clean_value = re.sub(r'^[\$€£¥]\s*', '', value)
        
        # Handle parentheses for negatives: (123) -> -123
        if clean_value.startswith('(') and clean_value.endswith(')'):
            clean_value = clean_value[1:-1]
        
        # Handle dashes for zeros
        if clean_value in ['—', '–', '-', 'n/a', 'N/A', 'n.a.', 'N.A.']:
            return True
        
        # Remove commas, spaces, and percentage signs
        clean_value = re.sub(r'[,\s%]', '', clean_value)
        
        # Check for number patterns
        # Integer or decimal, possibly with trailing 'M', 'B', 'K' for millions/billions/thousands
        if re.match(r'^-?\d+(\.\d+)?[MBK]?$', clean_value, re.IGNORECASE):
            return True
        
        # Standard float check
        try:
            float(clean_value)
            return True
        except ValueError:
            return False
    
    def _create_dataframe(self, headers: List[str], data_rows: List[List[str]]) -> pd.DataFrame:
        """Create cleaned pandas DataFrame from table data with financial number conversion"""
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
        df = df.replace('', pd.NA)  # Replace empty strings with NA
        df = df.dropna(how='all')  # Remove completely empty rows
        
        # Convert financial columns to numeric
        df = self._convert_financial_columns(df)
        
        return df
    
    def _convert_financial_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns with financial data to proper numeric types"""
        for col in df.columns:
            # Check if column contains mostly financial numbers
            sample_size = min(10, len(df))
            if sample_size == 0:
                continue
                
            sample_values = df[col].dropna().head(sample_size)
            if len(sample_values) == 0:
                continue
                
            # Count how many values look like financial numbers
            financial_count = sum(1 for val in sample_values if self._is_financial_number(str(val)))
            
            # If majority are financial numbers, convert the column
            if financial_count / len(sample_values) >= 0.6:
                df[col] = df[col].apply(self._parse_financial_number)
        
        return df
    
    def _parse_financial_number(self, value) -> float:
        """Parse financial number string to float"""
        if pd.isna(value) or not value:
            return pd.NA
        
        value_str = str(value).strip()
        
        # Handle dashes and N/A values
        if value_str.lower() in ['—', '–', '-', 'n/a', 'n.a.', 'na', '']:
            return 0.0
        
        # Track if number is negative (parentheses)
        is_negative = False
        if value_str.startswith('(') and value_str.endswith(')'):
            value_str = value_str[1:-1]
            is_negative = True
        
        # Remove currency symbols and clean
        clean_value = re.sub(r'^[\$€£¥]\s*', '', value_str)
        clean_value = re.sub(r'[,\s%]', '', clean_value)
        
        # Handle magnitude suffixes (M, B, K)
        multiplier = 1
        if clean_value.endswith(('M', 'm')):
            multiplier = 1_000_000
            clean_value = clean_value[:-1]
        elif clean_value.endswith(('B', 'b')):
            multiplier = 1_000_000_000
            clean_value = clean_value[:-1]
        elif clean_value.endswith(('K', 'k')):
            multiplier = 1_000
            clean_value = clean_value[:-1]
        
        try:
            number = float(clean_value) * multiplier
            return -number if is_negative else number
        except ValueError:
            return pd.NA
    
    def _classify_table_type(self, headers: List[str], df: pd.DataFrame) -> str:
        """Classify the type of financial table"""
        
        # Combine headers and first few rows for analysis
        text_content = ' '.join(headers).lower()
        if not df.empty:
            first_rows = df.head(3).to_string().lower()
            text_content += ' ' + first_rows
        
        # Score each table type
        scores = {}
        for table_type, keywords in self.financial_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_content)
            if score > 0:
                scores[table_type] = score
        
        # Return best match or 'other_financial'
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'other_financial'
    
    def _calculate_confidence_score(self, df: pd.DataFrame, table_type: str) -> float:
        """Calculate confidence score for table extraction accuracy"""
        if df.empty:
            return 0.0
        
        score = 0.0
        
        # Basic structure score
        if len(df.columns) >= 2:
            score += 0.3
        if len(df) >= 2:
            score += 0.3
        
        # Data quality score
        non_empty_cells = df.count().sum()
        total_cells = df.size
        if total_cells > 0:
            fill_rate = non_empty_cells / total_cells
            score += 0.4 * fill_rate
        
        return min(score, 1.0)
    
    def _generate_table_metadata(self, df: pd.DataFrame, page: int, table_type: str) -> Dict[str, Any]:
        """Generate metadata for the extracted table"""
        return {
            'page_number': page,
            'table_type': table_type,
            'dimensions': f"{len(df)} rows × {len(df.columns)} columns",
            'has_numeric_data': any(df.select_dtypes(include=['number']).columns),
            'column_count': len(df.columns),
            'row_count': len(df),
            'extraction_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def extract_text_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text content for narrative sections"""
        text_chunks = []
        
        try:
            # Use PyPDF2 for text extraction
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"Extracting text from {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        if page_text and page_text.strip():
                            # Split into paragraphs
                            paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                            
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
            # Don't raise - text extraction failure shouldn't stop table extraction
            
        logger.info(f"Extracted {len(text_chunks)} text chunks")
        return text_chunks
    
    def extract_document_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract document-level metadata"""
        metadata = {
            'file_path': pdf_path,
            'file_name': Path(pdf_path).name,
            'extraction_method': 'pdfplumber + PyPDF2',
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata.update({
                    'total_pages': len(pdf.pages),
                    'pdf_metadata': pdf.metadata or {}
                })
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {e}")
        
        return metadata
    
    def _get_pages(self, pdf_path: str) -> List:
        """Get page count for statistics"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return pdf.pages
        except Exception:
            return []

# Convenience functions for quick usage
def process_financial_pdf(pdf_path: str) -> ProcessedDocument:
    """Quick function to process a financial PDF"""
    processor = FinancialPDFProcessor()
    return processor.process_document(pdf_path)

def extract_tables_only(pdf_path: str) -> List[ExtractedTable]:
    """Extract only tables from financial PDF"""
    processor = FinancialPDFProcessor()
    return processor.extract_financial_tables(pdf_path)
