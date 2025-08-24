"""
Validation utilities for SDS-RAG system.
"""

import re
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


def validate_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if file exists
    """
    return Path(file_path).exists() and Path(file_path).is_file()


def validate_directory_exists(directory_path: Union[str, Path]) -> bool:
    """
    Validate that a directory exists.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        bool: True if directory exists
    """
    return Path(directory_path).exists() and Path(directory_path).is_dir()


def validate_pdf_file(file_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate a PDF file comprehensively.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        errors.append(f"File does not exist: {file_path}")
        return False, errors
    
    # Check if it's a file (not directory)
    if not file_path.is_file():
        errors.append(f"Path is not a file: {file_path}")
        return False, errors
    
    # Check file extension
    if file_path.suffix.lower() != '.pdf':
        errors.append(f"File is not a PDF (extension: {file_path.suffix})")
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type != 'application/pdf':
        errors.append(f"File MIME type is not PDF: {mime_type}")
    
    # Check file size (not empty, not too large)
    try:
        file_size = file_path.stat().st_size
        if file_size == 0:
            errors.append("PDF file is empty")
        elif file_size > 100 * 1024 * 1024:  # 100MB limit
            errors.append(f"PDF file is too large: {file_size / (1024*1024):.1f}MB")
    except OSError as e:
        errors.append(f"Cannot read file stats: {e}")
    
    # Try to read first few bytes to check PDF header
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                errors.append("File does not have valid PDF header")
    except IOError as e:
        errors.append(f"Cannot read file header: {e}")
    
    return len(errors) == 0, errors


def validate_confidence_score(confidence: float) -> bool:
    """
    Validate a confidence score.
    
    Args:
        confidence: Confidence score to validate
        
    Returns:
        bool: True if confidence score is valid (0.0 - 1.0)
    """
    return 0.0 <= confidence <= 1.0


def validate_limit_parameter(limit: int, min_limit: int = 1, max_limit: int = 100) -> bool:
    """
    Validate a limit parameter for queries.
    
    Args:
        limit: Limit value to validate
        min_limit: Minimum allowed limit
        max_limit: Maximum allowed limit
        
    Returns:
        bool: True if limit is valid
    """
    return min_limit <= limit <= max_limit


def validate_search_query(query: str, min_length: int = 1, max_length: int = 1000) -> Tuple[bool, List[str]]:
    """
    Validate a search query.
    
    Args:
        query: Search query to validate
        min_length: Minimum query length
        max_length: Maximum query length
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    if not query or not query.strip():
        errors.append("Query cannot be empty")
        return False, errors
    
    query_length = len(query.strip())
    
    if query_length < min_length:
        errors.append(f"Query too short (minimum {min_length} characters)")
    
    if query_length > max_length:
        errors.append(f"Query too long (maximum {max_length} characters)")
    
    # Check for potentially problematic characters
    if re.search(r'[<>"\']', query):
        logger.warning(f"Query contains potentially unsafe characters: {query}")
    
    return len(errors) == 0, errors


def validate_table_data(table_data: List[List[str]]) -> Tuple[bool, List[str]]:
    """
    Validate table data structure.
    
    Args:
        table_data: Table data to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    if not table_data:
        errors.append("Table data is empty")
        return False, errors
    
    if not isinstance(table_data, list):
        errors.append("Table data must be a list")
        return False, errors
    
    # Check if all rows are lists
    for i, row in enumerate(table_data):
        if not isinstance(row, list):
            errors.append(f"Row {i} is not a list")
    
    # Check for consistent row lengths (allow some variance)
    if len(table_data) > 1:
        row_lengths = [len(row) for row in table_data]
        min_length, max_length = min(row_lengths), max(row_lengths)
        
        if max_length - min_length > 2:  # Allow some variance for merged cells
            errors.append(f"Inconsistent row lengths: min={min_length}, max={max_length}")
    
    # Check for completely empty table
    has_content = False
    for row in table_data:
        for cell in row:
            if cell and str(cell).strip():
                has_content = True
                break
        if has_content:
            break
    
    if not has_content:
        errors.append("Table contains no content")
    
    return len(errors) == 0, errors


def validate_metadata(metadata: Dict[str, Any], required_fields: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validate metadata dictionary.
    
    Args:
        metadata: Metadata dictionary to validate
        required_fields: List of required field names
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    if not isinstance(metadata, dict):
        errors.append("Metadata must be a dictionary")
        return False, errors
    
    # Check required fields
    if required_fields:
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
            elif metadata[field] is None:
                errors.append(f"Required field is None: {field}")
    
    # Validate specific known fields
    if 'confidence_score' in metadata:
        if not validate_confidence_score(metadata['confidence_score']):
            errors.append("Invalid confidence_score in metadata")
    
    if 'page' in metadata:
        page = metadata['page']
        if not isinstance(page, int) or page < 1:
            errors.append("Invalid page number in metadata")
    
    return len(errors) == 0, errors


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if email format is valid
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if URL format is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_api_key(api_key: str, min_length: int = 20) -> Tuple[bool, List[str]]:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        min_length: Minimum length for API key
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    
    if not api_key or not api_key.strip():
        errors.append("API key cannot be empty")
        return False, errors
    
    if len(api_key) < min_length:
        errors.append(f"API key too short (minimum {min_length} characters)")
    
    # Check for suspicious patterns
    if api_key.lower() in ['test', 'demo', 'example', 'placeholder']:
        errors.append("API key appears to be a placeholder")
    
    return len(errors) == 0, errors


def validate_processing_result(result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate document processing result structure.
    
    Args:
        result: Processing result to validate
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    required_fields = ['processing_successful', 'file_name']
    
    # Check required fields
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing required field: {field}")
    
    # Validate success field
    if 'processing_successful' in result:
        if not isinstance(result['processing_successful'], bool):
            errors.append("processing_successful must be boolean")
    
    # If processing failed, error message should be present
    if result.get('processing_successful') is False:
        if 'error' not in result or not result['error']:
            errors.append("Error message required when processing fails")
    
    # Validate numeric fields if present
    numeric_fields = ['tables_processed', 'text_chunks_processed', 'total_pages']
    for field in numeric_fields:
        if field in result:
            value = result[field]
            if value is not None and (not isinstance(value, int) or value < 0):
                errors.append(f"Invalid {field}: must be non-negative integer")
    
    # Validate confidence scores
    confidence_fields = ['average_confidence', 'success_rate']
    for field in confidence_fields:
        if field in result:
            value = result[field]
            if value is not None and not validate_confidence_score(value):
                errors.append(f"Invalid {field}: must be between 0.0 and 1.0")
    
    return len(errors) == 0, errors


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors


def validate_or_raise(validator_func, *args, **kwargs):
    """
    Run a validator function and raise ValidationError if validation fails.
    
    Args:
        validator_func: Validation function to run
        *args: Arguments for validator function
        **kwargs: Keyword arguments for validator function
        
    Raises:
        ValidationError: If validation fails
    """
    is_valid, errors = validator_func(*args, **kwargs)
    if not is_valid:
        raise ValidationError(f"Validation failed: {validator_func.__name__}", errors)