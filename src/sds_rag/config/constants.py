"""
SDS-RAG System Constants

Centralized constants for the SDS-RAG application.
All constants used throughout the system should be defined here.
"""

from typing import Dict, List, Any

# =============================================================================
# DEFAULT VALUES AND SYSTEM LIMITS
# =============================================================================

# Embedding Models
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_VECTOR_SIZE = 384  # Default dimension for all-MiniLM-L6-v2
SUPPORTED_EMBEDDING_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/all-distilroberta-v1": 768,
}

# LLM Configuration
DEFAULT_LLM_MODEL = "gemini-1.5-flash"
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048
LLM_TEMPERATURE_RANGE = (0.0, 2.0)
MAX_CONTEXT_LENGTH = 30000

# Vector Database (Qdrant)
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_COLLECTION_NAME = "financial_documents"
VECTOR_DISTANCE_METRIC = "COSINE"
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100

# API Configuration
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_STREAMLIT_HOST = "localhost"
DEFAULT_STREAMLIT_PORT = 8501

# File Processing
MAX_FILE_SIZE_MB = 100
SUPPORTED_FILE_TYPES = [".pdf"]
DEFAULT_TEMP_DIR = "temp"
DEFAULT_BATCH_SIZE = 10
MAX_CONCURRENT_PROCESSES = 4

# Confidence and Quality Thresholds
DEFAULT_MIN_CONFIDENCE = 0.6
HIGH_CONFIDENCE_THRESHOLD = 0.8
TABLE_EXTRACTION_MIN_CONFIDENCE = 0.5
TEXT_CHUNK_MIN_LENGTH = 50
MAX_TEXT_CHUNK_LENGTH = 4000

# =============================================================================
# CONTENT TYPES AND CLASSIFICATIONS
# =============================================================================

# Document Content Types
CONTENT_TYPES = {
    "TABLE_SUMMARY": "table_summary",
    "TABLE_ROW": "table_row", 
    "NARRATIVE_TEXT": "narrative_text",
    "METADATA": "metadata",
    "HEADER": "header",
    "FOOTER": "footer"
}

# Financial Statement Types
FINANCIAL_STATEMENT_TYPES = {
    "INCOME_STATEMENT": "income_statement",
    "BALANCE_SHEET": "balance_sheet", 
    "CASH_FLOW": "cash_flow",
    "EQUITY_STATEMENT": "equity_statement",
    "NOTES": "notes",
    "OTHER": "other",
    "UNKNOWN": "unknown"
}

# Table Classification Categories
TABLE_CLASSIFICATION_KEYWORDS = {
    "income_statement": [
        "revenue", "income", "sales", "profit", "loss", "earnings", "ebitda",
        "operating income", "net income", "gross profit", "operating expenses",
        "cost of goods sold", "depreciation", "amortization"
    ],
    "balance_sheet": [
        "assets", "liabilities", "equity", "shareholders equity", "stockholders equity",
        "current assets", "non-current assets", "current liabilities", "long-term debt",
        "retained earnings", "cash and equivalents", "accounts receivable", "inventory"
    ],
    "cash_flow": [
        "cash flow", "operating activities", "investing activities", "financing activities",
        "net cash provided", "net cash used", "cash generated", "cash payments",
        "capital expenditures", "dividends paid"
    ]
}

# =============================================================================
# PROMPT TEMPLATES AND MESSAGES
# =============================================================================

# LLM Prompt Templates
FINANCIAL_ANALYSIS_SYSTEM_PROMPT = """You are a financial analysis expert. Provide accurate, detailed responses based on the financial document context provided. Focus on:

1. Accurate financial data extraction and analysis
2. Clear explanations of financial concepts
3. Identification of trends and patterns
4. Professional financial terminology
5. Citation of source information when available

Always base your responses on the provided context and clearly indicate when information is not available."""

CHAT_RESPONSE_TEMPLATE = """Based on the financial document context below, please answer the user's question:

Context Documents:
{context}

User Question: {question}

Please provide a comprehensive answer based on the context. If the information needed to answer the question is not in the context, please say so clearly."""

FOLLOW_UP_SUGGESTIONS_TEMPLATE = """Based on this financial analysis conversation, suggest 3 relevant follow-up questions that would help the user understand the financial data better:

Previous Question: {question}
Response: {response}

Provide 3 concise, relevant follow-up questions."""

# =============================================================================
# ERROR MESSAGES AND STATUS CODES
# =============================================================================

# Error Messages
ERROR_MESSAGES = {
    "INVALID_FILE_TYPE": "Invalid file type. Only PDF files are supported.",
    "FILE_TOO_LARGE": f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB}MB.",
    "PROCESSING_FAILED": "Document processing failed. Please try again.",
    "EMBEDDING_FAILED": "Failed to generate embeddings for the document.",
    "VECTOR_STORAGE_FAILED": "Failed to store document in vector database.",
    "QDRANT_CONNECTION_FAILED": "Cannot connect to Qdrant vector database.",
    "LLM_SERVICE_UNAVAILABLE": "Language model service is currently unavailable.",
    "API_KEY_MISSING": "API key is missing or invalid.",
    "INVALID_SEARCH_QUERY": "Search query is invalid or empty.",
    "NO_RESULTS_FOUND": "No relevant documents found for the query.",
    "CONFIDENCE_TOO_LOW": "Search results have confidence scores below the threshold.",
}

# Success Messages
SUCCESS_MESSAGES = {
    "DOCUMENT_PROCESSED": "Document processed successfully.",
    "SEARCH_COMPLETED": "Search completed successfully.",
    "CHAT_RESPONSE_GENERATED": "Response generated successfully.",
    "DOCUMENT_REMOVED": "Document removed successfully from database.",
    "HEALTH_CHECK_PASSED": "All system components are healthy.",
}

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

# Log Levels
LOG_LEVELS = {
    "DEBUG": "DEBUG",
    "INFO": "INFO", 
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL"
}

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "logs"
MAX_LOG_FILES = 10
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance Monitoring
PERFORMANCE_THRESHOLDS = {
    "DOCUMENT_PROCESSING_SECONDS": 60.0,
    "EMBEDDING_GENERATION_SECONDS": 10.0,
    "VECTOR_SEARCH_SECONDS": 2.0,
    "LLM_RESPONSE_SECONDS": 15.0,
    "HEALTH_CHECK_SECONDS": 5.0
}

# =============================================================================
# UI AND DISPLAY CONSTANTS
# =============================================================================

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "PAGE_TITLE": "SDS-RAG: Financial Document Analysis",
    "PAGE_ICON": "📊",
    "LAYOUT": "wide",
    "SIDEBAR_STATE": "expanded",
    "THEME_BASE": "light",
    "PRIMARY_COLOR": "#1f77b4"
}

# Navigation Pages
NAVIGATION_PAGES = [
    "🏠 Home",
    "📄 Document Upload", 
    "🔍 Search Documents",
    "💬 Chat Assistant",
    "📋 Document Management",
    "🔧 System Status"
]

# Quick Question Suggestions
QUICK_QUESTIONS = [
    "What was the total revenue?",
    "Show me the operating expenses",
    "What are the key financial highlights?", 
    "How much cash was generated?",
    "What are the main assets?",
    "What is the net income for this period?",
    "Show me the debt-to-equity ratio",
    "What are the major expenses?"
]

# Display Limits
MAX_SEARCH_RESULTS_DISPLAY = 20
MAX_CHAT_HISTORY_DISPLAY = 50
CONTENT_PREVIEW_LENGTH = 200
MAX_TABLE_ROWS_DISPLAY = 100

# =============================================================================
# VALIDATION PATTERNS AND RULES
# =============================================================================

# File Validation
PDF_MIME_TYPES = ["application/pdf"]
ALLOWED_FILE_EXTENSIONS = [".pdf"]
MIN_FILE_SIZE_BYTES = 1024  # 1 KB minimum
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Text Validation  
MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 500
SPECIAL_CHARACTERS_ALLOWED = ".,!?;:()-_[]{}@#$%^&*+=<>/\\"

# Financial Data Patterns (for extraction validation)
CURRENCY_PATTERNS = [r'\$[\d,]+\.?\d*', r'USD\s*[\d,]+\.?\d*', r'[\d,]+\.?\d*\s*dollars?']
PERCENTAGE_PATTERNS = [r'\d+\.?\d*\s*%', r'\d+\.?\d*\s*percent']
DATE_PATTERNS = [
    r'\d{1,2}/\d{1,2}/\d{4}',
    r'\d{4}-\d{1,2}-\d{1,2}',
    r'[A-Za-z]+ \d{1,2},? \d{4}'
]

# =============================================================================
# SYSTEM HEALTH AND MONITORING
# =============================================================================

# Health Check Endpoints
HEALTH_CHECK_ENDPOINTS = {
    "QDRANT": "/health",
    "API": "/health",
    "EMBEDDING_SERVICE": "/health",
    "LLM_SERVICE": "/health"
}

# Service Status Codes
SERVICE_STATUS = {
    "HEALTHY": "healthy",
    "UNHEALTHY": "unhealthy", 
    "DEGRADED": "degraded",
    "UNKNOWN": "unknown"
}

# Timeout Values (seconds)
TIMEOUTS = {
    "QDRANT_CONNECTION": 5.0,
    "LLM_REQUEST": 30.0,
    "EMBEDDING_GENERATION": 60.0,
    "FILE_PROCESSING": 300.0,
    "API_REQUEST": 30.0,
    "HEALTH_CHECK": 10.0
}

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

# Required Environment Variables
REQUIRED_ENV_VARS = [
    "GOOGLE_API_KEY"
]

# Optional Environment Variables with Defaults
OPTIONAL_ENV_VARS = {
    "QDRANT_HOST": DEFAULT_QDRANT_HOST,
    "QDRANT_PORT": str(DEFAULT_QDRANT_PORT),
    "QDRANT_COLLECTION": DEFAULT_COLLECTION_NAME,
    "EMBEDDING_MODEL": DEFAULT_EMBEDDING_MODEL,
    "LLM_MODEL": DEFAULT_LLM_MODEL,
    "LLM_TEMPERATURE": str(DEFAULT_LLM_TEMPERATURE),
    "LLM_MAX_TOKENS": str(DEFAULT_MAX_TOKENS),
    "STREAMLIT_HOST": DEFAULT_STREAMLIT_HOST,
    "STREAMLIT_PORT": str(DEFAULT_STREAMLIT_PORT),
    "API_HOST": DEFAULT_API_HOST,
    "API_PORT": str(DEFAULT_API_PORT),
    "LOG_LEVEL": DEFAULT_LOG_LEVEL,
    "LOG_DIR": DEFAULT_LOG_DIR,
    "TEMP_DIR": DEFAULT_TEMP_DIR,
    "MIN_CONFIDENCE": str(DEFAULT_MIN_CONFIDENCE),
    "BATCH_SIZE": str(DEFAULT_BATCH_SIZE),
    "MAX_UPLOAD_SIZE_MB": str(MAX_FILE_SIZE_MB)
}

# =============================================================================
# API RESPONSE FORMATS
# =============================================================================

# Standard API Response Structure
API_RESPONSE_FIELDS = {
    "SUCCESS": ["status", "message", "data", "timestamp"],
    "ERROR": ["status", "error", "message", "timestamp", "details"]
}

# HTTP Status Codes
HTTP_STATUS_CODES = {
    "OK": 200,
    "CREATED": 201,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "NOT_FOUND": 404,
    "INTERNAL_ERROR": 500,
    "SERVICE_UNAVAILABLE": 503
}

# =============================================================================
# UTILITY FUNCTIONS FOR CONSTANTS
# =============================================================================

def get_supported_models() -> List[str]:
    """Get list of supported embedding models."""
    return list(SUPPORTED_EMBEDDING_MODELS.keys())

def get_model_dimension(model_name: str) -> int:
    """Get vector dimension for a specific embedding model."""
    return SUPPORTED_EMBEDDING_MODELS.get(model_name, EMBEDDING_VECTOR_SIZE)

def is_valid_file_type(filename: str) -> bool:
    """Check if file type is supported."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_FILE_EXTENSIONS)

def get_content_type_display_name(content_type: str) -> str:
    """Get display name for content type."""
    display_names = {
        "table_summary": "Table Summary",
        "table_row": "Table Row",
        "narrative_text": "Narrative Text",
        "metadata": "Metadata",
        "header": "Header",
        "footer": "Footer"
    }
    return display_names.get(content_type, content_type.title())

def get_statement_type_display_name(statement_type: str) -> str:
    """Get display name for financial statement type.""" 
    display_names = {
        "income_statement": "Income Statement",
        "balance_sheet": "Balance Sheet", 
        "cash_flow": "Cash Flow Statement",
        "equity_statement": "Statement of Equity",
        "notes": "Notes to Financial Statements",
        "other": "Other",
        "unknown": "Unknown"
    }
    return display_names.get(statement_type, statement_type.title())

# =============================================================================
# VERSION AND METADATA
# =============================================================================

VERSION = "1.0.0"
SYSTEM_NAME = "SDS-RAG"
SYSTEM_DESCRIPTION = "Simple Document Search with Retrieval Augmented Generation"
AUTHOR = "SDS-RAG Development Team"