"""
Utility functions and classes for SDS-RAG system.
"""

# File handling utilities
from .file_utils import (
    ensure_directory_exists, get_file_extension, is_pdf_file, find_pdf_files,
    get_file_size, get_file_size_human, create_temp_file, cleanup_temp_file,
    safe_filename, copy_file, move_file, TempFileManager, batch_process_files
)

# Logging utilities
from .logging_utils import (
    ColoredFormatter, setup_logging, get_logger, log_function_call,
    log_performance, StructuredLogger, configure_service_logging,
    silence_noisy_loggers, debug, info, warning, error, critical
)

# Validation utilities
from .validation_utils import (
    validate_file_exists, validate_directory_exists, validate_pdf_file,
    validate_confidence_score, validate_limit_parameter, validate_search_query,
    validate_table_data, validate_metadata, validate_email, validate_url,
    validate_api_key, validate_processing_result, ValidationError, validate_or_raise
)

# Date/time utilities
from .datetime_utils import (
    utc_now, utc_now_iso, format_timestamp, parse_iso_timestamp,
    seconds_to_human, time_ago, start_of_day, end_of_day, days_between,
    is_same_day, add_business_days, get_week_boundaries, get_month_boundaries,
    Timer, RateLimiter
)

__all__ = [
    # File utilities
    "ensure_directory_exists", "get_file_extension", "is_pdf_file", "find_pdf_files",
    "get_file_size", "get_file_size_human", "create_temp_file", "cleanup_temp_file",
    "safe_filename", "copy_file", "move_file", "TempFileManager", "batch_process_files",
    
    # Logging utilities
    "ColoredFormatter", "setup_logging", "get_logger", "log_function_call",
    "log_performance", "StructuredLogger", "configure_service_logging",
    "silence_noisy_loggers", "debug", "info", "warning", "error", "critical",
    
    # Validation utilities
    "validate_file_exists", "validate_directory_exists", "validate_pdf_file",
    "validate_confidence_score", "validate_limit_parameter", "validate_search_query",
    "validate_table_data", "validate_metadata", "validate_email", "validate_url",
    "validate_api_key", "validate_processing_result", "ValidationError", "validate_or_raise",
    
    # Date/time utilities
    "utc_now", "utc_now_iso", "format_timestamp", "parse_iso_timestamp",
    "seconds_to_human", "time_ago", "start_of_day", "end_of_day", "days_between",
    "is_same_day", "add_business_days", "get_week_boundaries", "get_month_boundaries",
    "Timer", "RateLimiter"
]