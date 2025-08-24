"""
Integration tests for SDS-RAG API endpoints.
"""

import requests
import pytest
import json
import sys
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000/api/v1"
ROOT_URL = "http://localhost:8000"


class TestAPIEndpoints:
    """Test class for API endpoint integration tests."""

    @pytest.fixture(scope="class", autouse=True)
    def check_server(self):
        """Check if API server is running before tests."""
        try:
            response = requests.get(ROOT_URL, timeout=5)
            assert response.status_code == 200, "API server is not responding"
        except requests.exceptions.RequestException:
            pytest.fail("API server is not running. Start with: python src/api_server.py")

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = requests.get(ROOT_URL)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "SDS-RAG API"
        
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check standardized response format
        assert "status" in data
        assert "message" in data
        assert "data" in data
        assert "timestamp" in data
        
        # Check specific health data
        if data["status"] == "success":
            assert "overall_healthy" in data["data"]
            assert "services" in data["data"]
            
    def test_rag_health_endpoint(self):
        """Test RAG service health endpoint."""
        response = requests.get(f"{BASE_URL}/health/rag")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check standardized response format
        assert "status" in data
        assert "message" in data
        assert "data" in data
        assert "timestamp" in data
        
    def test_chat_health_endpoint(self):
        """Test chat service health endpoint."""
        response = requests.get(f"{BASE_URL}/health/chat")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check standardized response format
        assert "status" in data
        assert "message" in data
        assert "data" in data
        assert "timestamp" in data
        
    def test_vector_health_endpoint(self):
        """Test vector database health endpoint."""
        response = requests.get(f"{BASE_URL}/vectors/health")
        
        # Should work even if Qdrant is not running (will return error format)
        assert response.status_code in [200, 503]
        data = response.json()
        
        # Check standardized response format
        assert "status" in data
        assert "message" in data
        assert "timestamp" in data
        
        if response.status_code == 200:
            assert "data" in data
            assert "healthy" in data["data"]
            assert "service" in data["data"]
        else:
            assert "errors" in data
            
    def test_collection_info_endpoint(self):
        """Test collection info endpoint."""
        response = requests.get(f"{BASE_URL}/vectors/collection/info")
        
        # Should work even if collection doesn't exist (returns collection info or error)
        assert response.status_code in [200, 500]
        
        # Note: This endpoint returns collection info directly, not in standardized format
        # This might need to be updated to use standardized format
        
    def test_chat_suggestions_endpoint(self):
        """Test chat suggestions endpoint."""
        response = requests.get(f"{BASE_URL}/chat/suggestions/financial")
        
        assert response.status_code == 200
        data = response.json()
        
        # This endpoint returns a list directly
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Each suggestion should be a string
        for suggestion in data:
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0
            
    def test_search_endpoint_validation(self):
        """Test search endpoint with invalid data."""
        # Test with missing required field
        response = requests.post(f"{BASE_URL}/search", json={
            "limit": 5  # Missing required 'query' field
        })
        
        assert response.status_code == 422
        data = response.json()
        
        # Check standardized error response format
        assert "status" in data
        assert "message" in data
        assert "errors" in data
        assert "timestamp" in data
        assert data["status"] == "error"
        assert isinstance(data["errors"], list)
        assert len(data["errors"]) > 0
        
    def test_chat_endpoint_validation(self):
        """Test chat endpoint with invalid data."""
        # Test with invalid limit
        response = requests.post(f"{BASE_URL}/chat", json={
            "query": "test query",
            "num_results": -1  # Invalid negative number
        })
        
        assert response.status_code == 422
        data = response.json()
        
        # Check standardized error response format
        assert "status" in data
        assert "message" in data
        assert "errors" in data
        assert "timestamp" in data
        assert data["status"] == "error"
        
    def test_document_upload_validation(self):
        """Test document upload with invalid file."""
        # Test without file
        response = requests.post(f"{BASE_URL}/documents/upload")
        
        assert response.status_code == 422
        data = response.json()
        
        # Check standardized error response format
        assert "status" in data
        assert "message" in data
        assert "errors" in data
        assert "timestamp" in data
        assert data["status"] == "error"
        

# Standalone test runner for manual execution
def run_manual_tests():
    """Run tests manually without pytest."""
    print("🧪 Testing SDS-RAG API Endpoints")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(ROOT_URL, timeout=5)
    except requests.exceptions.RequestException:
        print("❌ API server is not running!")
        print("Please start the server with: python src/api_server.py")
        return False
    
    test_instance = TestAPIEndpoints()
    
    tests = [
        ("Root Endpoint", test_instance.test_root_endpoint),
        ("Health Check", test_instance.test_health_endpoint),
        ("RAG Health", test_instance.test_rag_health_endpoint),
        ("Chat Health", test_instance.test_chat_health_endpoint),
        ("Vector Health", test_instance.test_vector_health_endpoint),
        ("Collection Info", test_instance.test_collection_info_endpoint),
        ("Chat Suggestions", test_instance.test_chat_suggestions_endpoint),
        ("Search Validation", test_instance.test_search_endpoint_validation),
        ("Chat Validation", test_instance.test_chat_endpoint_validation),
        ("Upload Validation", test_instance.test_document_upload_validation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the API server logs for details.")
        return False


if __name__ == "__main__":
    success = run_manual_tests()
    sys.exit(0 if success else 1)