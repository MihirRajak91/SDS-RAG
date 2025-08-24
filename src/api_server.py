"""
API server entry point for SDS-RAG.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import the FastAPI app
from sds_rag.api.main import app

if __name__ == "__main__":
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print(f"Starting SDS-RAG API server on {host}:{port}")
    print(f"API Documentation available at: http://{host}:{port}/docs")
    print(f"Interactive API docs at: http://{host}:{port}/redoc")
    
    # Run the server
    uvicorn.run(
        "sds_rag.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )