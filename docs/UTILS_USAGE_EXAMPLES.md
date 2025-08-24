# Utils Integration Examples

## ✅ **How We're Now Using Utilities Throughout SDS-RAG**

### **🗂️ File Handling Improvements**

#### **Before (documents.py):**
```python
# Manual tempfile handling
with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
    content = await file.read()
    tmp_file.write(content)
    tmp_file_path = tmp_file.name

try:
    # Process file
    result = process_document(tmp_file_path)
finally:
    # Manual cleanup
    if os.path.exists(tmp_file_path):
        os.unlink(tmp_file_path)
```

#### **After (documents.py):**
```python
# Automatic file management with validation
from ...utils import TempFileManager, validate_pdf_file

content = await file.read()
with TempFileManager(suffix='.pdf', content=content) as tmp_file_path:
    # Process file - automatic cleanup guaranteed
    result = process_document(tmp_file_path)
```

### **⏱️ Performance Monitoring**

#### **Before:**
```python
# Manual timing
import time
start_time = time.time()
result = process_document(file_path)
duration = time.time() - start_time
logger.info(f"Processing took {duration:.3f} seconds")
```

#### **After:**
```python
# Automatic timing with human-readable output
from ...utils import Timer

with Timer(f"Processing {file_path}") as timer:
    result = process_document(file_path)
    # Automatic: "Processing document.pdf completed in 2m 30s"
```

### **📋 File Validation**

#### **Before:**
```python
# Basic validation
if not file_path.endswith('.pdf'):
    return error("Not a PDF")
if not os.path.exists(file_path):
    return error("File not found")
```

#### **After:**
```python
# Comprehensive validation
from ...utils import validate_pdf_file

is_valid, errors = validate_pdf_file(file_path)
if not is_valid:
    return error_response("Invalid PDF", errors)
    # Errors: ["File too large: 150.2MB", "Invalid PDF header", etc.]
```

### **📊 Structured Logging**

#### **Before:**
```python
logger.info(f"Processed {file_name}: Success")
```

#### **After:**
```python
from ...utils import StructuredLogger

api_logger = StructuredLogger(__name__)
api_logger.log_document_processing(
    file_path=file_path,
    status="success",
    details={
        "processing_time": "2m 30s",
        "tables_processed": 15,
        "file_size": "45.2MB",
        "pages": 25
    }
)
# Output: "Document processing | File: report.pdf | Status: success | processing_time: 2m 30s | tables_processed: 15 | file_size: 45.2MB | pages: 25"
```

### **🕒 Timestamp Consistency**

#### **Before (response.py):**
```python
timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
```

#### **After (response.py):**
```python
from ..utils.datetime_utils import utc_now_iso

timestamp: str = Field(default_factory=utc_now_iso)
```

### **📁 Directory Processing**

#### **Before (rag_service.py):**
```python
# Manual PDF discovery
pdf_dir = Path(directory)
if not pdf_dir.exists():
    raise ValueError(f"Directory does not exist: {directory}")
pdf_files = list(pdf_dir.glob("*.pdf"))
```

#### **After (rag_service.py):**
```python
# Utility-based PDF discovery
from ...utils import find_pdf_files

pdf_files = find_pdf_files(directory)
if not pdf_files:
    raise ValueError(f"No PDF files found in: {directory}")
```

## 🎯 **Benefits Achieved**

### **🔧 Code Consistency**
- All file operations use the same validation logic
- Consistent timestamp formatting across API responses
- Uniform error handling and logging patterns

### **⚡ Performance Insights**
- Automatic timing for all major operations
- Human-readable duration formatting ("2m 30s" vs "150.3")
- Structured performance logging for analysis

### **🛡️ Robustness**
- Comprehensive PDF validation (size, header, MIME type)
- Automatic cleanup of temporary files
- Better error messages with detailed validation results

### **📈 Maintainability**
- Common operations centralized in utils
- Easy to add new validation rules
- Consistent logging format for debugging

## 🚀 **Usage in New Code**

```python
from sds_rag.utils import (
    # File operations
    validate_pdf_file, TempFileManager, find_pdf_files,
    
    # Performance monitoring
    Timer, log_performance,
    
    # Logging
    StructuredLogger,
    
    # Time utilities
    utc_now_iso, seconds_to_human, time_ago,
    
    # Validation
    validate_search_query, validate_confidence_score
)

class NewService:
    def __init__(self):
        self.logger = StructuredLogger(__name__)
    
    def process_files(self, directory: str):
        # Find files with utility
        pdf_files = find_pdf_files(directory, recursive=True)
        
        # Process with timing and logging
        with Timer(f"Processing {len(pdf_files)} files") as timer:
            for pdf_file in pdf_files:
                # Validate before processing
                is_valid, errors = validate_pdf_file(pdf_file)
                if not is_valid:
                    self.logger.log_document_processing(
                        str(pdf_file), "validation_failed", {"errors": errors}
                    )
                    continue
                
                # Process with performance logging
                with log_performance(f"Processing {pdf_file.name}"):
                    result = self.process_single_file(pdf_file)
                
                # Log results
                self.logger.log_document_processing(
                    str(pdf_file),
                    "success" if result else "failed"
                )
        
        return f"Completed batch in {timer.elapsed_human}"
```

## 📦 **What's Available**

- **File Utils**: PDF validation, temp files, directory scanning, size formatting
- **Logging Utils**: Structured logging, colored console, performance timing  
- **Validation Utils**: Comprehensive validators for files, queries, confidence scores
- **DateTime Utils**: UTC timestamps, human durations, time ago formatting

All utilities are now integrated and being used throughout the SDS-RAG codebase for better consistency, performance monitoring, and maintainability! 🎉