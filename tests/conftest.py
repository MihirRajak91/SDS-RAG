"""
Pytest configuration and shared fixtures for SDS-RAG tests.
"""

import pytest
import requests
from typing import Generator


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Base URL for API endpoints."""
    return "http://localhost:8000/api/v1"


@pytest.fixture(scope="session")
def root_url() -> str:
    """Root URL for API."""
    return "http://localhost:8000"


@pytest.fixture(scope="session", autouse=True)
def check_api_server(root_url: str) -> Generator[None, None, None]:
    """Ensure API server is running before running any tests."""
    try:
        response = requests.get(root_url, timeout=10)
        if response.status_code != 200:
            pytest.exit("API server is not responding correctly", 1)
    except requests.exceptions.RequestException:
        pytest.exit(
            "API server is not running. Please start it with: python src/api_server.py", 
            1
        )
    
    yield
    # Teardown code could go here if needed


@pytest.fixture
def sample_document_request():
    """Sample document processing request."""
    return {
        "file_path": "/path/to/sample.pdf"
    }


@pytest.fixture
def sample_search_request():
    """Sample search request."""
    return {
        "query": "revenue growth",
        "limit": 5,
        "min_confidence": 0.7
    }


@pytest.fixture
def sample_chat_request():
    """Sample chat request."""
    return {
        "query": "What was the revenue last quarter?",
        "num_results": 3,
        "min_confidence": 0.6
    }