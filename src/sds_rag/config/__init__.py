"""
SDS-RAG Configuration Package

Provides centralized configuration management with constants and settings.
"""

# Import main settings and constants
from .settings import (
    # Main settings classes
    AppSettings,
    QdrantSettings, 
    EmbeddingSettings,
    LLMSettings,
    APISettings,
    StreamlitSettings,
    ProcessingSettings,
    LoggingSettings,
    SecuritySettings,
    PerformanceSettings,
    
    # Global settings instance
    settings,
    
    # Convenience functions
    get_qdrant_config,
    get_embedding_config,
    get_llm_config,
    get_api_config,
    get_streamlit_config,
    get_processing_config,
    get_logging_config,
    get_performance_config,
    
    # Utilities
    create_settings,
    validate_environment,
    print_settings_summary
)

from .constants import (
    # Default values
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_VECTOR_SIZE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_STREAMLIT_HOST,
    DEFAULT_STREAMLIT_PORT,
    
    # Content and classification types
    CONTENT_TYPES,
    FINANCIAL_STATEMENT_TYPES,
    TABLE_CLASSIFICATION_KEYWORDS,
    
    # Error and success messages
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    
    # UI constants
    STREAMLIT_CONFIG,
    NAVIGATION_PAGES,
    QUICK_QUESTIONS,
    
    # Performance thresholds
    PERFORMANCE_THRESHOLDS,
    TIMEOUTS,
    
    # Validation constants
    SUPPORTED_EMBEDDING_MODELS,
    ALLOWED_FILE_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    
    # Utility functions
    get_supported_models,
    get_model_dimension,
    is_valid_file_type,
    get_content_type_display_name,
    get_statement_type_display_name,
    
    # System metadata
    VERSION,
    SYSTEM_NAME,
    SYSTEM_DESCRIPTION
)

# Legacy compatibility - import the old config wrapper
from .config import config

# Backward compatibility
__all__ = [
    # Settings classes
    'AppSettings', 'QdrantSettings', 'EmbeddingSettings', 'LLMSettings',
    'APISettings', 'StreamlitSettings', 'ProcessingSettings', 'LoggingSettings',
    'SecuritySettings', 'PerformanceSettings',
    
    # Global instances
    'settings', 'config',
    
    # Configuration functions
    'get_qdrant_config', 'get_embedding_config', 'get_llm_config',
    'get_api_config', 'get_streamlit_config', 'get_processing_config', 
    'get_logging_config', 'get_performance_config',
    
    # Utilities
    'create_settings', 'validate_environment', 'print_settings_summary',
    
    # Constants
    'CONTENT_TYPES', 'FINANCIAL_STATEMENT_TYPES', 'ERROR_MESSAGES', 
    'SUCCESS_MESSAGES', 'STREAMLIT_CONFIG', 'PERFORMANCE_THRESHOLDS',
    'SUPPORTED_EMBEDDING_MODELS', 'DEFAULT_EMBEDDING_MODEL',
    'VERSION', 'SYSTEM_NAME'
]