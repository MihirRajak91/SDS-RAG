@echo off
echo Starting SDS-RAG Services Setup...
echo =====================================

REM Check if Docker is available
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop and try again
    echo Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo Docker found! Starting services...

REM Start services with Docker Compose
docker compose up -d

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to start services
    echo Check docker-compose.yml and try again
    pause
    exit /b 1
)

echo.
echo Services started successfully!
echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Check service health
echo Checking Qdrant health...
curl -s http://localhost:6333/health >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✓ Qdrant is healthy
) else (
    echo ✗ Qdrant health check failed
)

echo.
echo Checking MongoDB connection...
docker exec sds-rag-mongodb mongosh --eval "db.adminCommand('ping')" --quiet >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✓ MongoDB is healthy
) else (
    echo ✗ MongoDB health check failed
)

echo.
echo =====================================
echo Service Setup Complete!
echo.
echo Next steps:
echo 1. Set your Google AI API key:
echo    set GOOGLE_AI_API_KEY=your_api_key_here
echo.
echo 2. Install Python dependencies:
echo    poetry install
echo.
echo 3. Run health check:
echo    poetry run python health_check.py
echo.
echo 4. Access services:
echo    Qdrant Web UI: http://localhost:6333/dashboard
echo    MongoDB: mongodb://admin:password123@localhost:27017
echo.
pause