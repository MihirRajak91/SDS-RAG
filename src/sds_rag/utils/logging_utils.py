"""
Logging utilities for SDS-RAG system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    colored_console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        log_format: Custom log format string
        colored_console: Whether to use colored console output
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        logging.Logger: Configured root logger
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if colored_console:
        console_formatter = ColoredFormatter(log_format)
    else:
        console_formatter = logging.Formatter(log_format)
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        # File formatter (no colors)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        function: Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Log function call
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
        
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func_name} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func_name} failed after {execution_time:.3f}s: {str(e)}")
            raise
    
    return wrapper


def log_performance(operation_name: str):
    """
    Context manager for logging operation performance.
    
    Args:
        operation_name: Name of the operation being timed
    """
    class PerformanceLogger:
        def __init__(self, name: str):
            self.name = name
            self.logger = logging.getLogger(__name__)
            self.start_time = None
        
        def __enter__(self):
            self.start_time = datetime.now()
            self.logger.info(f"Starting {self.name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            execution_time = (datetime.now() - self.start_time).total_seconds()
            if exc_type is None:
                self.logger.info(f"{self.name} completed in {execution_time:.3f}s")
            else:
                self.logger.error(f"{self.name} failed after {execution_time:.3f}s: {exc_val}")
    
    return PerformanceLogger(operation_name)


class StructuredLogger:
    """Logger for structured logging with consistent format."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_app_startup(self, host: str, port: int, debug: bool, command: str):
        """Log application startup event."""
        message = f"App Startup | Host: {host} | Port: {port} | Debug: {debug} | Command: '{command}'"
        self.logger.info(message)
    
    def log_document_processing(self, 
                              file_path: str, 
                              status: str, 
                              details: Optional[Dict[str, Any]] = None):
        """Log document processing events."""
        message = f"Document processing | File: {file_path} | Status: {status}"
        if details:
            detail_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
            message += f" | {detail_str}"
        
        if status.lower() in ['success', 'completed']:
            self.logger.info(message)
        elif status.lower() in ['failed', 'error']:
            self.logger.error(message)
        else:
            self.logger.warning(message)
    
    def log_api_request(self, 
                       endpoint: str, 
                       method: str, 
                       status_code: int, 
                       execution_time: float,
                       details: Optional[Dict[str, Any]] = None):
        """Log API request events."""
        message = f"API Request | {method} {endpoint} | Status: {status_code} | Time: {execution_time:.3f}s"
        if details:
            detail_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
            message += f" | {detail_str}"
        
        if 200 <= status_code < 400:
            self.logger.info(message)
        elif 400 <= status_code < 500:
            self.logger.warning(message)
        else:
            self.logger.error(message)
    
    def log_search_operation(self, 
                           query: str, 
                           results_count: int, 
                           execution_time: float,
                           filters: Optional[Dict[str, Any]] = None):
        """Log search operation events."""
        message = f"Search | Query: '{query}' | Results: {results_count} | Time: {execution_time:.3f}s"
        if filters:
            filter_str = " | ".join([f"{k}: {v}" for k, v in filters.items() if v is not None])
            if filter_str:
                message += f" | Filters: {filter_str}"
        
        self.logger.info(message)
    
    def log_chat_interaction(self, 
                           query: str, 
                           response_length: int, 
                           sources_used: int,
                           execution_time: float):
        """Log chat interaction events."""
        message = f"Chat | Query: '{query[:50]}...' | Response: {response_length} chars | Sources: {sources_used} | Time: {execution_time:.3f}s"
        self.logger.info(message)
    
    def log_component_init(self, component: str, config: Dict[str, Any]):
        """Log component initialization."""
        config_str = " | ".join([f"{k}: {v}" for k, v in config.items()])
        message = f"Component Init | {component} | {config_str}"
        self.logger.info(message)
    
    def log_chat_query(self, query: str, **kwargs):
        """Log chat query."""
        details = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
        message = f"Chat Query | '{query}'"
        if details:
            message += f" | {details}"
        self.logger.info(message)
    
    def log_chat_response(self, response: str, sources_found: int = 0, execution_time: float = 0.0, **kwargs):
        """Log chat response."""
        message = f"Chat Response | Length: {len(response)} chars | Sources: {sources_found}"
        if execution_time > 0:
            message += f" | Time: {execution_time:.3f}s"
        for k, v in kwargs.items():
            message += f" | {k}: {v}"
        self.logger.info(message)
    
    def log_embedding_operation(self, **kwargs):
        """Log embedding operation."""
        details = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
        message = f"Embedding Operation | {details}"
        self.logger.info(message)


def configure_service_logging(service_name: str, log_level: str = "INFO") -> StructuredLogger:
    """
    Configure logging for a specific service.
    
    Args:
        service_name: Name of the service
        log_level: Logging level for the service
        
    Returns:
        StructuredLogger: Configured structured logger
    """
    logger = logging.getLogger(f"sds_rag.services.{service_name}")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    return StructuredLogger(f"sds_rag.services.{service_name}")


def silence_noisy_loggers():
    """Silence overly verbose third-party loggers."""
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool',
        'qdrant_client',
        'sentence_transformers',
        'transformers.tokenization_utils',
        'transformers.tokenization_utils_base',
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Module-level convenience functions
def debug(message: str, logger_name: str = __name__):
    """Log debug message."""
    logging.getLogger(logger_name).debug(message)


def info(message: str, logger_name: str = __name__):
    """Log info message."""
    logging.getLogger(logger_name).info(message)


def warning(message: str, logger_name: str = __name__):
    """Log warning message."""
    logging.getLogger(logger_name).warning(message)


def error(message: str, logger_name: str = __name__, exc_info: bool = False):
    """Log error message."""
    logging.getLogger(logger_name).error(message, exc_info=exc_info)


def critical(message: str, logger_name: str = __name__):
    """Log critical message."""
    logging.getLogger(logger_name).critical(message)