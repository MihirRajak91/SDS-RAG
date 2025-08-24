"""
SDS-RAG Application Launcher

This script provides a comprehensive launcher for the SDS-RAG Streamlit application
with proper service initialization, health checks, and error handling.
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from typing import Optional
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sds_rag.utils import StructuredLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    
    # Mapping from package name to import name
    required_packages = {
        'streamlit': 'streamlit',
        'langchain': 'langchain',
        'qdrant-client': 'qdrant_client',
        'sentence-transformers': 'sentence_transformers',
        'google-generativeai': 'google.generativeai',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'pdfplumber': 'pdfplumber',
        'pypdf2': 'PyPDF2'  # Note the case difference
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: poetry install")
        return False
    
    logger.info("✅ All dependencies are installed")
    return True

def check_qdrant_connection(host: str = "localhost", port: int = 6333) -> bool:
    """Check if Qdrant server is accessible."""
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port)
        collections = client.get_collections()
        logger.info(f"✅ Qdrant connection successful at {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        logger.info("Make sure Qdrant server is running. Start with: docker run -p 6333:6333 qdrant/qdrant")
        return False

def setup_environment():
    """Set up environment variables and configuration."""
    
    # Set Streamlit configuration
    os.environ.setdefault("STREAMLIT_THEME_BASE", "light")
    os.environ.setdefault("STREAMLIT_THEME_PRIMARY_COLOR", "#1f77b4")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    
    # Create required directories
    directories = ["logs", "temp", "data"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("✅ Environment setup complete")

def create_streamlit_config():
    """Create Streamlit configuration file."""
    
    config_dir = Path.home() / ".streamlit"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "config.toml"
    
    config_content = """
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 100

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[logger]
level = "info"
"""
    
    with open(config_file, "w") as f:
        f.write(config_content.strip())
    
    logger.info(f"✅ Streamlit config created at {config_file}")

def run_health_checks() -> bool:
    """Run comprehensive health checks before starting the app."""
    
    logger.info("🔍 Running system health checks...")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check Qdrant connection
    if not check_qdrant_connection():
        logger.warning("⚠️ Qdrant not available. Some features may not work.")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            return False
    
    # Test basic imports
    try:
        from src.sds_rag.services.rag_service import RAGService
        from src.sds_rag.services.chat_service import ChatService
        logger.info("✅ Core services import successfully")
    except Exception as e:
        logger.error(f"❌ Core services import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("✅ All health checks passed")
    return True

def start_streamlit_app(port: int = 8501, host: str = "localhost", debug: bool = False):
    """Start the Streamlit application."""
    
    # Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "src/app.py",
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true" if not debug else "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    if debug:
        cmd.extend(["--logger.level", "debug"])
    
    logger.info(f"Starting Streamlit app on http://{host}:{port}")
    
    try:
        # Log startup
        structured_logger.log_app_startup(
            host=host,
            port=port,
            debug=debug,
            command=" ".join(cmd)
        )
        
        # Start the app
        process = subprocess.run(cmd, check=True)
        return process.returncode == 0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit app: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
        return True

def main():
    """Main launcher function."""
    
    parser = argparse.ArgumentParser(description="SDS-RAG Streamlit Application Launcher")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the app on")
    parser.add_argument("--host", default="localhost", help="Host to bind the app to")
    parser.add_argument("--skip-health-checks", action="store_true", help="Skip health checks")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--setup-only", action="store_true", help="Only setup environment, don't start app")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SDS-RAG Application Launcher")
    print("Financial Document Analysis with RAG")
    print("=" * 60)
    
    # Setup environment
    logger.info("Setting up environment...")
    setup_environment()
    create_streamlit_config()
    
    if args.setup_only:
        logger.info("Environment setup complete. Exiting.")
        return
    
    # Health checks
    if not args.skip_health_checks:
        if not run_health_checks():
            logger.error("Health checks failed. Use --skip-health-checks to override.")
            sys.exit(1)
    
    # Start the application
    success = start_streamlit_app(
        port=args.port,
        host=args.host,
        debug=args.debug
    )
    
    if success:
        logger.info("Application started successfully")
    else:
        logger.error("Application failed to start")
        sys.exit(1)

if __name__ == "__main__":
    main()