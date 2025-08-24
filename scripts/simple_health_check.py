#!/usr/bin/env python3
"""
Simple Health Check - Check basic service availability without full dependencies.
"""

import os
import sys
import urllib.request
import socket
from urllib.error import URLError, HTTPError


def check_port_open(host, port, timeout=5):
    """Check if a port is open and accessible."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def check_qdrant_simple():
    """Simple Qdrant health check using HTTP request."""
    print("QDRANT SIMPLE HEALTH CHECK")
    print("-" * 30)
    
    # Check if port is open
    port_open = check_port_open("localhost", 6333)
    
    if not port_open:
        print("[FAIL] Qdrant port 6333: NOT ACCESSIBLE")
        print("   Possible solutions:")
        print("   - Start Qdrant with Docker:")
        print("     docker run -p 6333:6333 qdrant/qdrant")
        print("   - Or use docker-compose:")
        print("     docker compose up -d")
        return False
    
    print("[OK] Qdrant port 6333: ACCESSIBLE")
    
    # Try to access health endpoint
    try:
        with urllib.request.urlopen("http://localhost:6333/health", timeout=10) as response:
            if response.status == 200:
                print("[OK] Qdrant health endpoint: HEALTHY")
                print("   Qdrant is running and responsive")
                return True
            else:
                print(f"[WARN] Qdrant health endpoint: HTTP {response.status}")
                return False
                
    except HTTPError as e:
        print(f"[WARN] Qdrant health endpoint: HTTP {e.code}")
        return False
    except URLError as e:
        print(f"[FAIL] Qdrant health endpoint: Connection failed - {e.reason}")
        return False
    except Exception as e:
        print(f"[FAIL] Qdrant health endpoint: Error - {e}")
        return False


def check_mongodb_simple():
    """Simple MongoDB health check."""
    print("\nMONGODB SIMPLE HEALTH CHECK")
    print("-" * 30)
    
    # Check if port is open
    port_open = check_port_open("localhost", 27017)
    
    if not port_open:
        print("[FAIL] MongoDB port 27017: NOT ACCESSIBLE")
        print("   Possible solutions:")
        print("   - Start MongoDB with Docker:")
        print("     docker run -d -p 27017:27017 mongo:7.0")
        print("   - Or use docker-compose:")
        print("     docker compose up -d")
        return False
    
    print("[OK] MongoDB port 27017: ACCESSIBLE")
    print("   MongoDB appears to be running")
    return True


def check_google_api_key():
    """Check if Google AI API key is configured."""
    print("\nGOOGLE AI API KEY CHECK")
    print("-" * 30)
    
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    if not api_key:
        print("[FAIL] Google AI API Key: NOT SET")
        print("   Set environment variable:")
        print("   - Windows: set GOOGLE_AI_API_KEY=your_key_here")
        print("   - Linux/Mac: export GOOGLE_AI_API_KEY=your_key_here")
        return False
    
    if len(api_key) < 20:
        print("[WARN] Google AI API Key: TOO SHORT (possibly invalid)")
        print(f"   Current key length: {len(api_key)} characters")
        return False
    
    print("[OK] Google AI API Key: CONFIGURED")
    print(f"   Key length: {len(api_key)} characters")
    print(f"   Key prefix: {api_key[:8]}...")
    return True


def check_docker_services():
    """Check if Docker services are running."""
    print("\nDOCKER SERVICES CHECK")
    print("-" * 25)
    
    import subprocess
    
    try:
        # Check if docker command is available
        result = subprocess.run(["docker", "--version"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("[FAIL] Docker: NOT AVAILABLE")
            print("   Install Docker Desktop from: https://www.docker.com/products/docker-desktop")
            return False
        
        print("[OK] Docker: AVAILABLE")
        print(f"   Version: {result.stdout.strip()}")
        
        # Check if our containers are running
        try:
            result = subprocess.run(["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}"],
                                  capture_output=True, text=True, timeout=10)
            
            if "sds-rag-qdrant" in result.stdout:
                print("[OK] Qdrant container: RUNNING")
            else:
                print("[WARN] Qdrant container: NOT RUNNING")
                
            if "sds-rag-mongodb" in result.stdout:
                print("[OK] MongoDB container: RUNNING")
            else:
                print("[WARN] MongoDB container: NOT RUNNING")
                
        except subprocess.TimeoutExpired:
            print("[WARN] Docker ps command timed out")
        
        return True
        
    except FileNotFoundError:
        print("[FAIL] Docker: NOT INSTALLED")
        print("   Install Docker Desktop from: https://www.docker.com/products/docker-desktop")
        return False
    except subprocess.TimeoutExpired:
        print("[WARN] Docker command timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Docker check failed: {e}")
        return False


def main():
    """Run simple health checks without heavy dependencies."""
    print("SDS-RAG SIMPLE HEALTH CHECK")
    print("=" * 50)
    print("Checking basic service availability...\n")
    
    # Track service status
    status = {
        "docker": False,
        "qdrant": False,
        "mongodb": False,
        "google_api": False
    }
    
    # Run checks
    status["docker"] = check_docker_services()
    status["qdrant"] = check_qdrant_simple()
    status["mongodb"] = check_mongodb_simple()
    status["google_api"] = check_google_api_key()
    
    # Summary
    print("\n" + "=" * 50)
    print("SIMPLE HEALTH CHECK SUMMARY")
    print("=" * 50)
    
    healthy_count = sum(status.values())
    total_components = len(status)
    
    for component, is_healthy in status.items():
        status_icon = "[OK]" if is_healthy else "[FAIL]"
        component_name = component.upper().replace('_', ' ')
        print(f"{status_icon} {component_name}: {'HEALTHY' if is_healthy else 'NEEDS ATTENTION'}")
    
    print(f"\nOverall Status: {healthy_count}/{total_components} components ready")
    
    # Recommendations
    print("\nNEXT STEPS:")
    
    if not status["docker"]:
        print("1. Install Docker Desktop")
        print("2. Run setup_services.bat (Windows) or docker compose up -d")
    elif not status["qdrant"] or not status["mongodb"]:
        print("1. Start services: docker compose up -d")
        print("2. Or run: setup_services.bat")
    
    if not status["google_api"]:
        print("3. Set Google AI API key environment variable")
    
    if all(status.values()):
        print("All basic components are ready!")
        print("4. Install dependencies: poetry install")
        print("5. Run full health check: poetry run python health_check.py")
    
    print("\n" + "=" * 50)
    return healthy_count == total_components


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nHealth check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Health check failed: {e}")
        sys.exit(1)