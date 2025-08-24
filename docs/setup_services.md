# SDS-RAG Services Setup

## Prerequisites
- Docker Desktop installed on your system
- Poetry for dependency management

## Quick Setup

### 1. Install Dependencies
```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

### 2. Start Services with Docker
```bash
# Start Qdrant and MongoDB
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs
```

### 3. Verify Services are Running
```bash
# Check Qdrant health
curl http://localhost:6333/health

# Check MongoDB connection
docker exec -it sds-rag-mongodb mongosh --eval "db.adminCommand('ping')"
```

### 4. Run Health Check
```bash
# Set Google AI API key (required for LLM service)
export GOOGLE_AI_API_KEY=your_api_key_here

# Run comprehensive health check
poetry run python health_check.py
```

## Service Details

### Qdrant Vector Database
- **Port**: 6333 (HTTP), 6334 (gRPC)
- **Storage**: Persistent volume `qdrant_storage`
- **Health Check**: http://localhost:6333/health

### MongoDB Database
- **Port**: 27017
- **Username**: admin
- **Password**: password123
- **Storage**: Persistent volume `mongodb_data`

## Manual Setup (if Docker is unavailable)

### Qdrant
```bash
# Using Docker directly
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### MongoDB
```bash
# Using Docker directly
docker run -d -p 27017:27017 \
    -v mongodb_data:/data/db \
    -e MONGO_INITDB_ROOT_USERNAME=admin \
    -e MONGO_INITDB_ROOT_PASSWORD=password123 \
    mongo:7.0
```

## Troubleshooting

### Common Issues
1. **Port conflicts**: Stop other services using ports 6333, 6334, or 27017
2. **Docker not running**: Start Docker Desktop
3. **Permission issues**: Run commands with appropriate permissions
4. **API key missing**: Set GOOGLE_AI_API_KEY environment variable

### Health Check Failures
- **Qdrant not healthy**: Check if container is running and port 6333 is accessible
- **LLM not healthy**: Verify Google AI API key is set and valid
- **Embedding not healthy**: Ensure sentence-transformers model can be downloaded

### Stopping Services
```bash
# Stop all services
docker compose down

# Stop and remove volumes (data will be lost)
docker compose down -v
```