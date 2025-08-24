# Standardized Response Format

All API endpoints now return responses in a consistent format using the `ApiResponse` wrapper.

## Response Structure

```json
{
  "status": "success|error|warning",
  "message": "Human-readable description",
  "data": {}, // Response payload (null for errors)
  "errors": [], // Array of error messages (null for success)
  "metadata": {}, // Additional context/debugging info (optional)
  "timestamp": "2024-01-01T12:00:00.000000Z", // ISO 8601 timestamp
  "request_id": "optional-request-id" // For request tracing (optional)
}
```

## Success Response Examples

### Document Processing Success
```json
{
  "status": "success",
  "message": "Document 'financial_report.pdf' processed successfully",
  "data": {
    "file_name": "financial_report.pdf",
    "file_path": "/path/to/financial_report.pdf",
    "tables_processed": 15,
    "text_chunks_processed": 42,
    "embedded_documents_created": 57,
    "vector_points_stored": 57,
    "high_confidence_tables": 12,
    "average_confidence": 0.85,
    "success_rate": 95.2,
    "total_pages": 25,
    "processing_timestamp": "2024-01-01T12:00:00Z"
  },
  "errors": null,
  "metadata": null,
  "timestamp": "2024-01-01T12:00:00.000000Z"
}
```

### Search Results Success
```json
{
  "status": "success",
  "message": "Found 3 results for query: 'revenue growth'",
  "data": {
    "query": "revenue growth",
    "results": [
      {
        "id": "doc_123",
        "content": "Revenue increased by 15% year-over-year...",
        "score": 0.95,
        "metadata": {
          "source": "q3_report.pdf",
          "page": 5,
          "content_type": "table_summary"
        }
      }
    ],
    "total_results": 3,
    "filters_applied": {
      "content_type": "table_summary",
      "min_confidence": 0.7,
      "limit": 10
    }
  },
  "errors": null,
  "metadata": null,
  "timestamp": "2024-01-01T12:00:00.000000Z"
}
```

### Health Check Success
```json
{
  "status": "success",
  "message": "All services are healthy",
  "data": {
    "services": {
      "rag_service": {
        "qdrant_healthy": true,
        "collection_info": {
          "name": "financial_documents",
          "points_count": 1250
        }
      },
      "chat_service": {
        "chat_service_healthy": true
      }
    },
    "overall_healthy": true
  },
  "errors": null,
  "metadata": null,
  "timestamp": "2024-01-01T12:00:00.000000Z"
}
```

## Error Response Examples

### Validation Error
```json
{
  "status": "error",
  "message": "Validation failed",
  "data": null,
  "errors": [
    "query: Field required",
    "limit: Input should be greater than 0"
  ],
  "metadata": {
    "path": "http://localhost:8000/api/v1/search",
    "validation_details": [
      {
        "type": "missing",
        "loc": ["body", "query"],
        "msg": "Field required"
      }
    ]
  },
  "timestamp": "2024-01-01T12:00:00.000000Z"
}
```

### Processing Error
```json
{
  "status": "error",
  "message": "Document processing failed",
  "data": null,
  "errors": [
    "PDF file is corrupted or unreadable"
  ],
  "metadata": {
    "path": "http://localhost:8000/api/v1/documents/process"
  },
  "timestamp": "2024-01-01T12:00:00.000000Z"
}
```

### File Not Found Error
```json
{
  "status": "error",
  "message": "File not found",
  "data": null,
  "errors": [
    "File not found: /path/to/nonexistent.pdf"
  ],
  "metadata": null,
  "timestamp": "2024-01-01T12:00:00.000000Z"
}
```

### Service Unavailable Error
```json
{
  "status": "error",
  "message": "Service temporarily unavailable",
  "data": null,
  "errors": [
    "Vector database connection failed"
  ],
  "metadata": {
    "path": "http://localhost:8000/api/v1/search"
  },
  "timestamp": "2024-01-01T12:00:00.000000Z"
}
```

## Warning Response Example

```json
{
  "status": "warning",
  "message": "Processing completed with warnings",
  "data": {
    "file_name": "partial_report.pdf",
    "tables_processed": 8,
    "text_chunks_processed": 15,
    "success_rate": 75.0
  },
  "errors": [
    "Some tables had low confidence scores",
    "Page 10 could not be processed due to image quality"
  ],
  "metadata": {
    "low_confidence_pages": [10, 15],
    "processing_warnings": 2
  },
  "timestamp": "2024-01-01T12:00:00.000000Z"
}
```

## Benefits

### ✅ **Consistency**
- All endpoints return the same structure
- Predictable error handling
- Consistent status codes and messages

### ✅ **Client-Friendly**
- Easy to parse and handle in client applications
- Clear separation between success data and error information
- Human-readable messages for user interfaces

### ✅ **Debugging**
- Timestamps for all responses
- Metadata field for additional context
- Request tracing support with optional request_id

### ✅ **Error Handling**
- Structured error messages
- Multiple error support (validation errors)
- Contextual metadata for debugging

## Status Field Values

- **`success`**: Request completed successfully
- **`error`**: Request failed due to client or server error
- **`warning`**: Request completed but with issues/warnings

## HTTP Status Code Mapping

- **200**: Success responses (`status: "success"`)
- **422**: Validation errors (`status: "error"`)
- **404**: Not found errors (`status: "error"`)
- **500**: Server errors (`status: "error"`)
- **503**: Service unavailable (`status: "error"`)

## Usage in Client Code

### Python Example
```python
import requests

response = requests.post("http://localhost:8000/api/v1/chat", json={
    "query": "What was the revenue last quarter?"
})

data = response.json()

if data["status"] == "success":
    print(f"Answer: {data['data']['response']}")
    print(f"Sources: {data['data']['sources_found']}")
elif data["status"] == "error":
    print(f"Error: {data['message']}")
    for error in data.get("errors", []):
        print(f"  - {error}")
```

### JavaScript Example
```javascript
const response = await fetch('/api/v1/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: 'cash flow analysis' })
});

const data = await response.json();

if (data.status === 'success') {
    console.log(`Found ${data.data.total_results} results`);
    data.data.results.forEach(result => {
        console.log(`Score: ${result.score}, Content: ${result.content}`);
    });
} else {
    console.error(`Error: ${data.message}`);
    data.errors?.forEach(error => console.error(`- ${error}`));
}
```