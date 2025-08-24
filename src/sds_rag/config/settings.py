"""
SDS-RAG Application Settings

Comprehensive settings management for the SDS-RAG system using Pydantic for validation.
Integrates with constants.py for default values and provides environment-based configuration.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
try:
    from pydantic_settings import BaseSettings
    from pydantic import validator, Field
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, validator, Field
from dotenv import load_dotenv

from .constants import (
    # Default values
    DEFAULT_EMBEDDING_MODEL, EMBEDDING_VECTOR_SIZE, DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_QDRANT_HOST,
    DEFAULT_QDRANT_PORT, DEFAULT_COLLECTION_NAME, DEFAULT_API_HOST,
    DEFAULT_API_PORT, DEFAULT_STREAMLIT_HOST, DEFAULT_STREAMLIT_PORT,
    DEFAULT_TEMP_DIR, DEFAULT_BATCH_SIZE, MAX_CONCURRENT_PROCESSES,
    DEFAULT_MIN_CONFIDENCE, DEFAULT_LOG_LEVEL, DEFAULT_LOG_DIR,
    MAX_LOG_FILES, LOG_FORMAT, MAX_FILE_SIZE_MB,
    
    # Validation constants
    SUPPORTED_EMBEDDING_MODELS, LLM_TEMPERATURE_RANGE, 
    REQUIRED_ENV_VARS, OPTIONAL_ENV_VARS, ALLOWED_FILE_EXTENSIONS,
    
    # System constants
    STREAMLIT_CONFIG, PERFORMANCE_THRESHOLDS, TIMEOUTS
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""
    
    host: str = Field(default=DEFAULT_QDRANT_HOST, description="Qdrant server hostname")
    port: int = Field(default=DEFAULT_QDRANT_PORT, ge=1024, le=65535, description="Qdrant server port")
    collection_name: str = Field(default=DEFAULT_COLLECTION_NAME, description="Vector collection name")
    vector_size: int = Field(default=EMBEDDING_VECTOR_SIZE, ge=1, le=2048, description="Vector dimension size")
    distance_metric: str = Field(default="COSINE", description="Distance metric for similarity search")
    timeout: float = Field(default=TIMEOUTS["QDRANT_CONNECTION"], ge=1.0, description="Connection timeout in seconds")
    
    @validator('host')
    def validate_host(cls, v):
        if not v or not v.strip():
            raise ValueError("Qdrant host cannot be empty")
        return v.strip()
    
    @validator('collection_name') 
    def validate_collection_name(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Collection name must be at least 3 characters long")
        return v.strip().lower().replace(' ', '_')
    
    @validator('distance_metric')
    def validate_distance_metric(cls, v):
        allowed_metrics = ['COSINE', 'DOT', 'EUCLID']
        if v.upper() not in allowed_metrics:
            raise ValueError(f"Distance metric must be one of: {allowed_metrics}")
        return v.upper()
    
    class Config:
        env_prefix = "QDRANT_"
        extra = "ignore"  # Allow extra environment variables


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    
    embedding_model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL, description="HuggingFace embedding model name")
    device: str = Field(default="cpu", description="Device for embedding model (cpu/cuda)")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize embeddings")
    batch_size: int = Field(default=32, ge=1, le=128, description="Batch size for embedding generation")
    max_length: int = Field(default=512, ge=128, le=2048, description="Maximum sequence length")
    
    @validator('embedding_model_name')
    def validate_embedding_model_name(cls, v):
        if v not in SUPPORTED_EMBEDDING_MODELS:
            logger.warning(f"Embedding model '{v}' not in supported list. Proceeding anyway.")
        return v
    
    @validator('device')
    def validate_device(cls, v):
        allowed_devices = ['cpu', 'cuda', 'mps']
        if v not in allowed_devices:
            raise ValueError(f"Device must be one of: {allowed_devices}")
        return v
    
    class Config:
        env_prefix = "EMBEDDING_"
        extra = "ignore"
        protected_namespaces = ()


class LLMSettings(BaseSettings):
    """Large Language Model configuration."""
    
    google_api_key: Optional[str] = Field(default=None, description="Google AI API key")
    llm_model_name: str = Field(default=DEFAULT_LLM_MODEL, description="LLM model name")
    temperature: float = Field(default=DEFAULT_LLM_TEMPERATURE, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=8192, description="Maximum output tokens")
    timeout: float = Field(default=TIMEOUTS["LLM_REQUEST"], ge=5.0, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Number of retry attempts")
    
    @validator('google_api_key', pre=True)
    def validate_api_key(cls, v):
        # Get from environment if not provided
        if not v:
            v = os.getenv("GOOGLE_API_KEY")
        if not v:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        min_temp, max_temp = LLM_TEMPERATURE_RANGE
        if not (min_temp <= v <= max_temp):
            raise ValueError(f"Temperature must be between {min_temp} and {max_temp}")
        return v
    
    class Config:
        env_prefix = "LLM_"
        extra = "ignore"
        protected_namespaces = ()


class APISettings(BaseSettings):
    """API server configuration."""
    
    host: str = Field(default=DEFAULT_API_HOST, description="API server host")
    port: int = Field(default=DEFAULT_API_PORT, ge=1024, le=65535, description="API server port")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per minute limit")
    docs_enabled: bool = Field(default=True, description="Enable API documentation")
    
    @validator('host')
    def validate_host(cls, v):
        if not v or not v.strip():
            raise ValueError("API host cannot be empty")
        return v.strip()
    
    class Config:
        env_prefix = "API_"
        extra = "ignore"


class StreamlitSettings(BaseSettings):
    """Streamlit application configuration."""
    
    host: str = Field(default=DEFAULT_STREAMLIT_HOST, description="Streamlit server host")
    port: int = Field(default=DEFAULT_STREAMLIT_PORT, ge=1024, le=65535, description="Streamlit server port")
    title: str = Field(default=STREAMLIT_CONFIG["PAGE_TITLE"], description="Application title")
    page_icon: str = Field(default=STREAMLIT_CONFIG["PAGE_ICON"], description="Page icon")
    layout: str = Field(default=STREAMLIT_CONFIG["LAYOUT"], description="Page layout")
    sidebar_state: str = Field(default=STREAMLIT_CONFIG["SIDEBAR_STATE"], description="Initial sidebar state")
    theme_base: str = Field(default=STREAMLIT_CONFIG["THEME_BASE"], description="Theme base")
    primary_color: str = Field(default=STREAMLIT_CONFIG["PRIMARY_COLOR"], description="Primary theme color")
    max_upload_size_mb: int = Field(default=MAX_FILE_SIZE_MB, ge=1, le=500, description="Maximum file upload size in MB")
    
    @validator('layout')
    def validate_layout(cls, v):
        allowed_layouts = ['centered', 'wide']
        if v not in allowed_layouts:
            raise ValueError(f"Layout must be one of: {allowed_layouts}")
        return v
    
    @validator('sidebar_state')
    def validate_sidebar_state(cls, v):
        allowed_states = ['auto', 'expanded', 'collapsed']
        if v not in allowed_states:
            raise ValueError(f"Sidebar state must be one of: {allowed_states}")
        return v
    
    @validator('primary_color')
    def validate_primary_color(cls, v):
        if not v.startswith('#') or len(v) != 7:
            raise ValueError("Primary color must be a valid hex color code (e.g., #1f77b4)")
        return v
    
    class Config:
        env_prefix = "STREAMLIT_"
        extra = "ignore"


class ProcessingSettings(BaseSettings):
    """Document processing configuration."""
    
    temp_dir: str = Field(default=DEFAULT_TEMP_DIR, description="Temporary files directory")
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, ge=1, le=100, description="Batch processing size")
    max_concurrent: int = Field(default=MAX_CONCURRENT_PROCESSES, ge=1, le=16, description="Maximum concurrent processes")
    pdf_validation: bool = Field(default=True, description="Enable PDF validation")
    min_confidence: float = Field(default=DEFAULT_MIN_CONFIDENCE, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_file_size_mb: int = Field(default=MAX_FILE_SIZE_MB, ge=1, le=500, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(default=ALLOWED_FILE_EXTENSIONS, description="Allowed file extensions")
    processing_timeout: float = Field(default=TIMEOUTS["FILE_PROCESSING"], ge=60.0, description="Processing timeout in seconds")
    
    @validator('temp_dir')
    def validate_temp_dir(cls, v):
        temp_path = Path(v)
        temp_path.mkdir(parents=True, exist_ok=True)
        if not temp_path.is_dir():
            raise ValueError(f"Cannot create or access temp directory: {v}")
        return str(temp_path)
    
    @validator('allowed_extensions')
    def validate_extensions(cls, v):
        if not v:
            raise ValueError("At least one file extension must be allowed")
        return [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in v]
    
    class Config:
        env_prefix = "PROCESSING_"
        extra = "ignore"


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field(default=DEFAULT_LOG_LEVEL, description="Logging level")
    log_dir: str = Field(default=DEFAULT_LOG_DIR, description="Log files directory")
    max_files: int = Field(default=MAX_LOG_FILES, ge=1, le=100, description="Maximum number of log files")
    format: str = Field(default=LOG_FORMAT, description="Log message format")
    structured_logs: bool = Field(default=True, description="Enable structured JSON logging")
    file_logging: bool = Field(default=True, description="Enable logging to files")
    console_logging: bool = Field(default=True, description="Enable console logging")
    
    @validator('level')
    def validate_level(cls, v):
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    @validator('log_dir')
    def validate_log_dir(cls, v):
        log_path = Path(v)
        log_path.mkdir(parents=True, exist_ok=True)
        if not log_path.is_dir():
            raise ValueError(f"Cannot create or access log directory: {v}")
        return str(log_path)
    
    class Config:
        env_prefix = "LOG_"
        extra = "ignore"


class SecuritySettings(BaseSettings):
    """Security and authentication configuration."""
    
    api_key_required: bool = Field(default=False, description="Require API key for requests")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    allowed_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    max_request_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum request size in MB")
    
    class Config:
        env_prefix = "SECURITY_"
        extra = "ignore"


class PerformanceSettings(BaseSettings):
    """Performance monitoring and optimization settings."""
    
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    slow_query_threshold: float = Field(default=2.0, ge=0.1, description="Slow query threshold in seconds")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl_seconds: int = Field(default=3600, ge=60, description="Cache TTL in seconds")
    max_cache_size_mb: int = Field(default=100, ge=10, le=1000, description="Maximum cache size in MB")
    
    # Performance thresholds from constants
    document_processing_threshold: float = Field(default=PERFORMANCE_THRESHOLDS["DOCUMENT_PROCESSING_SECONDS"])
    embedding_generation_threshold: float = Field(default=PERFORMANCE_THRESHOLDS["EMBEDDING_GENERATION_SECONDS"])
    vector_search_threshold: float = Field(default=PERFORMANCE_THRESHOLDS["VECTOR_SEARCH_SECONDS"])
    llm_response_threshold: float = Field(default=PERFORMANCE_THRESHOLDS["LLM_RESPONSE_SECONDS"])
    
    class Config:
        env_prefix = "PERFORMANCE_"
        extra = "ignore"


class AppSettings(BaseSettings):
    """Main application settings container."""
    
    # Component settings
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    # Global settings
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Enable debug mode")
    version: str = Field(default="1.0.0", description="Application version")
    
    def __init__(self, **data):
        """Initialize settings with environment-based configuration."""
        super().__init__(**data)
        self._validate_configuration()
        self._setup_directories()
    
    def _validate_configuration(self):
        """Validate overall configuration consistency."""
        errors = []
        
        # Check for required environment variables
        for env_var in REQUIRED_ENV_VARS:
            if not os.getenv(env_var):
                errors.append(f"Required environment variable {env_var} is not set")
        
        # Validate port conflicts
        ports = {
            'API': self.api.port,
            'Streamlit': self.streamlit.port,
            'Qdrant': self.qdrant.port
        }
        
        port_values = list(ports.values())
        if len(port_values) != len(set(port_values)):
            duplicates = [name for name, port in ports.items() 
                         if port_values.count(port) > 1]
            errors.append(f"Port conflicts detected for: {', '.join(duplicates)}")
        
        # Validate embedding model compatibility
        if self.embedding.embedding_model_name in SUPPORTED_EMBEDDING_MODELS:
            expected_size = SUPPORTED_EMBEDDING_MODELS[self.embedding.embedding_model_name]
            if self.qdrant.vector_size != expected_size:
                errors.append(f"Vector size mismatch: {self.embedding.embedding_model_name} "
                            f"produces {expected_size}D vectors, but Qdrant is configured for {self.qdrant.vector_size}D")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    def _setup_directories(self):
        """Create required directories."""
        directories = [
            self.processing.temp_dir,
            self.logging.log_dir,
            "data",
            "data/uploads",
            "data/processed"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.dict()
    
    def get_database_url(self) -> str:
        """Get Qdrant database URL."""
        return f"http://{self.qdrant.host}:{self.qdrant.port}"
    
    def get_api_url(self) -> str:
        """Get API server URL."""
        return f"http://{self.api.host}:{self.api.port}"
    
    def get_streamlit_url(self) -> str:
        """Get Streamlit application URL."""
        return f"http://{self.streamlit.host}:{self.streamlit.port}"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ('dev', 'development', 'local')
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() in ('prod', 'production')
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
        protected_namespaces = ()


# =============================================================================
# SETTINGS FACTORY AND UTILITIES
# =============================================================================

def create_settings(env_file: Optional[str] = None, **overrides) -> AppSettings:
    """
    Create application settings with optional overrides.
    
    Args:
        env_file (str, optional): Path to environment file
        **overrides: Settings overrides
        
    Returns:
        AppSettings: Configured application settings
    """
    if env_file:
        load_dotenv(env_file)
    
    return AppSettings(**overrides)


def validate_environment() -> Dict[str, str]:
    """
    Validate environment variables and return status.
    
    Returns:
        Dict[str, str]: Validation results
    """
    results = {}
    
    # Check required variables
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value:
            results[var] = "Set"
        else:
            results[var] = "Missing"
    
    # Check optional variables
    for var, default in OPTIONAL_ENV_VARS.items():
        value = os.getenv(var, default)
        results[var] = f"{value}"
    
    return results


def print_settings_summary(settings: AppSettings):
    """Print a comprehensive settings summary."""
    print("=" * 70)
    print("SDS-RAG Application Settings Summary")
    print("=" * 70)
    
    print(f"Environment: {settings.environment}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Version: {settings.version}")
    print()
    
    print(f"Qdrant: {settings.get_database_url()}")
    print(f"Collection: {settings.qdrant.collection_name}")
    print(f"Vector Size: {settings.qdrant.vector_size}")
    print()
    
    print(f"Embedding Model: {settings.embedding.embedding_model_name}")
    print(f"Device: {settings.embedding.device}")
    print(f"Batch Size: {settings.embedding.batch_size}")
    print()
    
    print(f"LLM Model: {settings.llm.llm_model_name}")
    print(f"Temperature: {settings.llm.temperature}")
    print(f"Max Tokens: {settings.llm.max_tokens}")
    print(f"API Key: {settings.llm.google_api_key}")
    print()
    
    print(f"API Server: {settings.get_api_url()}")
    print(f"Streamlit: {settings.get_streamlit_url()}")
    print()
    
    print(f"Temp Directory: {settings.processing.temp_dir}")
    print(f"Log Level: {settings.logging.level}")
    print(f"Log Directory: {settings.logging.log_dir}")
    print()
    
    print("=" * 70)


# =============================================================================
# GLOBAL SETTINGS INSTANCE
# =============================================================================

# Create global settings instance
try:
    settings = create_settings()
    logger.info("Application settings loaded successfully")
except Exception as e:
    logger.error(f"Failed to load application settings: {e}")
    # Create minimal settings for testing/development
    settings = AppSettings(
        qdrant=QdrantSettings(),
        embedding=EmbeddingSettings(),
        llm=LLMSettings(google_api_key="dummy_key_for_testing"),
        api=APISettings(),
        streamlit=StreamlitSettings(),
        processing=ProcessingSettings(),
        logging=LoggingSettings(),
        security=SecuritySettings(),
        performance=PerformanceSettings()
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_qdrant_config() -> QdrantSettings:
    """Get Qdrant configuration."""
    return settings.qdrant


def get_embedding_config() -> EmbeddingSettings:
    """Get embedding configuration.""" 
    return settings.embedding


def get_llm_config() -> LLMSettings:
    """Get LLM configuration."""
    return settings.llm


def get_api_config() -> APISettings:
    """Get API configuration."""
    return settings.api


def get_streamlit_config() -> StreamlitSettings:
    """Get Streamlit configuration."""
    return settings.streamlit


def get_processing_config() -> ProcessingSettings:
    """Get processing configuration."""
    return settings.processing


def get_logging_config() -> LoggingSettings:
    """Get logging configuration."""
    return settings.logging


def get_performance_config() -> PerformanceSettings:
    """Get performance configuration."""
    return settings.performance


if __name__ == "__main__":
    # Print settings summary when run directly
    print_settings_summary(settings)
    print("\nEnvironment Variables:")
    env_status = validate_environment()
    for var, status in env_status.items():
        print(f"   {var}: {status}")