"""
File handling utilities for SDS-RAG system.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Union, Generator
import logging

logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path: The directory path as a Path object
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File extension (without the dot)
    """
    return Path(file_path).suffix.lower().lstrip('.')


def is_pdf_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is a PDF based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if file is a PDF
    """
    return get_file_extension(file_path) == 'pdf'


def find_pdf_files(directory: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    Find all PDF files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List[Path]: List of PDF file paths
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory_path}")
        return []
    
    if not directory_path.is_dir():
        logger.warning(f"Path is not a directory: {directory_path}")
        return []
    
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(directory_path.glob(pattern))
    
    logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
    return pdf_files


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        int: File size in bytes
    """
    return Path(file_path).stat().st_size


def get_file_size_human(file_path: Union[str, Path]) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Human-readable file size (e.g., "1.5 MB")
    """
    size_bytes = get_file_size(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def create_temp_file(suffix: str = "", prefix: str = "sds_rag_", content: Optional[bytes] = None) -> str:
    """
    Create a temporary file.
    
    Args:
        suffix: File suffix/extension
        prefix: File prefix
        content: Optional content to write to file
        
    Returns:
        str: Path to the temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix) as temp_file:
        if content:
            temp_file.write(content)
        temp_file_path = temp_file.name
    
    logger.debug(f"Created temporary file: {temp_file_path}")
    return temp_file_path


def cleanup_temp_file(file_path: str) -> bool:
    """
    Clean up a temporary file.
    
    Args:
        file_path: Path to the temporary file
        
    Returns:
        bool: True if file was successfully deleted
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
            return True
        return False
    except OSError as e:
        logger.error(f"Failed to cleanup temporary file {file_path}: {e}")
        return False


def safe_filename(filename: str, max_length: int = 255) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        str: Safe filename
    """
    # Replace problematic characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    safe_name = "".join(c if c in safe_chars else "_" for c in filename)
    
    # Remove consecutive underscores
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    
    # Trim to max length
    if len(safe_name) > max_length:
        name_part = safe_name[:max_length-4]
        safe_name = name_part + "_..."
    
    return safe_name.strip("_")


def copy_file(source: Union[str, Path], destination: Union[str, Path]) -> bool:
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        bool: True if copy was successful
    """
    try:
        source_path = Path(source)
        dest_path = Path(destination)
        
        # Ensure destination directory exists
        ensure_directory_exists(dest_path.parent)
        
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied file from {source_path} to {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to copy file from {source} to {destination}: {e}")
        return False


def move_file(source: Union[str, Path], destination: Union[str, Path]) -> bool:
    """
    Move a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        bool: True if move was successful
    """
    try:
        source_path = Path(source)
        dest_path = Path(destination)
        
        # Ensure destination directory exists
        ensure_directory_exists(dest_path.parent)
        
        shutil.move(str(source_path), str(dest_path))
        logger.info(f"Moved file from {source_path} to {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to move file from {source} to {destination}: {e}")
        return False


class TempFileManager:
    """Context manager for handling temporary files with automatic cleanup."""
    
    def __init__(self, suffix: str = "", prefix: str = "sds_rag_", content: Optional[bytes] = None):
        self.suffix = suffix
        self.prefix = prefix
        self.content = content
        self.temp_file_path: Optional[str] = None
    
    def __enter__(self) -> str:
        self.temp_file_path = create_temp_file(self.suffix, self.prefix, self.content)
        return self.temp_file_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_file_path:
            cleanup_temp_file(self.temp_file_path)


def batch_process_files(
    file_paths: List[Union[str, Path]], 
    process_func, 
    *args, 
    **kwargs
) -> Generator[tuple, None, None]:
    """
    Process multiple files in batch, yielding results as they complete.
    
    Args:
        file_paths: List of file paths to process
        process_func: Function to process each file
        *args: Additional arguments for process_func
        **kwargs: Additional keyword arguments for process_func
        
    Yields:
        tuple: (file_path, result, success, error)
    """
    for file_path in file_paths:
        try:
            logger.info(f"Processing file: {file_path}")
            result = process_func(file_path, *args, **kwargs)
            yield (file_path, result, True, None)
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            yield (file_path, None, False, str(e))