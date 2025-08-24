"""
SDS-RAG Configuration (Legacy Compatibility)

Legacy configuration module for backward compatibility.
For new code, use sds_rag.config.settings instead.
"""

import warnings
from typing import Dict, Any

# Import from new settings module
from .settings import settings, AppSettings
from .constants import *

# Legacy compatibility - issue deprecation warning
warnings.warn(
    "Using config.py is deprecated. Please import from 'sds_rag.config.settings' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Legacy compatibility classes and functions
class LegacyConfig:
    """Legacy configuration wrapper for backward compatibility."""
    
    def __init__(self, settings_instance: AppSettings):
        self._settings = settings_instance
    
    @property
    def qdrant(self):
        return self._settings.qdrant
    
    @property
    def embedding(self):
        return self._settings.embedding
    
    @property
    def llm(self):
        return self._settings.llm
    
    @property
    def streamlit(self):
        return self._settings.streamlit
    
    @property
    def processing(self):
        return self._settings.processing
    
    @property
    def logging(self):
        return self._settings.logging
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._settings.to_dict()
    
    def get_qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        return self._settings.get_database_url()
    
    def get_streamlit_url(self) -> str:
        """Get Streamlit application URL."""
        return self._settings.get_streamlit_url()

# Create legacy config instance
config = LegacyConfig(settings)

# Legacy compatibility functions
def get_qdrant_config():
    """Get Qdrant configuration."""
    return config.qdrant

def get_embedding_config():
    """Get embedding configuration."""
    return config.embedding

def get_llm_config():
    """Get LLM configuration."""
    return config.llm

def get_streamlit_config():
    """Get Streamlit configuration."""
    return config.streamlit

def get_processing_config():
    """Get processing configuration."""
    return config.processing

def get_logging_config():
    """Get logging configuration."""
    return config.logging

def print_config_summary():
    """Print a summary of the current configuration."""
    from sds_rag.config.settings import print_settings_summary
    print_settings_summary(settings)

if __name__ == "__main__":
    print_config_summary()