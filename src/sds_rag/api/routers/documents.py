"""
Document processing endpoints.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
import os

from ...models import (
    ApiResponse, success_response, error_response,
    ProcessDocumentRequest, ProcessDocumentData,
    BatchProcessRequest, BatchProcessData,
    RemoveDocumentRequest, RemoveDocumentData,
    DocumentSummaryRequest, DocumentSummaryData
)
from ...services.rag_service import RAGService
from ...utils import (
    validate_pdf_file, TempFileManager, get_file_size_human,
    StructuredLogger, Timer
)

router = APIRouter()

# Initialize RAG service and logger
rag_service = RAGService()
api_logger = StructuredLogger(__name__)


@router.post("/documents/upload", response_model=ApiResponse)
async def upload_and_process_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF document.
    
    Args:
        file: PDF file to upload and process
    
    Returns:
        ApiResponse: Processing results
    """
    if not file.filename.endswith('.pdf'):
        return error_response(
            message="Invalid file type",
            errors=["Only PDF files are supported"]
        )
    
    file_size = len(await file.read())
    await file.seek(0)  # Reset file pointer
    
    if file_size == 0:
        return error_response(
            message="Invalid file",
            errors=["File is empty"]
        )
    
    if file_size > 100 * 1024 * 1024:  # 100MB limit
        return error_response(
            message="File too large",
            errors=[f"File size ({get_file_size_human(file_size)}) exceeds 100MB limit"]
        )
    
    try:
        # Create temporary file and process document
        content = await file.read()
        with TempFileManager(suffix='.pdf', content=content) as tmp_file_path:
            with Timer(f"Processing document {file.filename}") as timer:
                result = rag_service.process_and_store_document(tmp_file_path)
            
            # Log the operation
            api_logger.log_document_processing(
                file_path=file.filename,
                status="success" if result.get("processing_successful") else "failed",
                details={
                    "file_size": get_file_size_human(file_size),
                    "processing_time": timer.elapsed_human,
                    "tables_processed": result.get("tables_processed", 0),
                    "pages": result.get("total_pages", 0)
                }
            )
            
            # Check if processing was successful
            if not result.get("processing_successful", False):
                return error_response(
                    message="Document processing failed",
                    errors=[result.get("error", "Unknown processing error")]
                )
            
            # Convert result to data model
            doc_data = ProcessDocumentData(
                file_name=result.get("file_name", file.filename),
                file_path=result.get("file_path", tmp_file_path),
                tables_processed=result.get("tables_processed"),
                text_chunks_processed=result.get("text_chunks_processed"),
                embedded_documents_created=result.get("embedded_documents_created"),
                vector_points_stored=result.get("vector_points_stored"),
                high_confidence_tables=result.get("high_confidence_tables"),
                average_confidence=result.get("average_confidence"),
                success_rate=result.get("success_rate"),
                total_pages=result.get("total_pages"),
                processing_timestamp=result.get("processing_timestamp")
            )
            
            return success_response(
                data=doc_data,
                message=f"Document '{file.filename}' processed successfully in {timer.elapsed_human}"
            )
                
    except Exception as e:
        return error_response(
            message="Document upload failed",
            errors=[str(e)]
        )


@router.post("/documents/process", response_model=ApiResponse)
async def process_document(request: ProcessDocumentRequest):
    """
    Process a PDF document from file path.
    
    Args:
        request: Document processing request
    
    Returns:
        ApiResponse: Processing results
    """
    # Validate the PDF file comprehensively
    is_valid, validation_errors = validate_pdf_file(request.file_path)
    if not is_valid:
        return error_response(
            message="Invalid PDF file",
            errors=validation_errors
        )
    
    try:
        with Timer(f"Processing document {Path(request.file_path).name}") as timer:
            result = rag_service.process_and_store_document(request.file_path)
        
        # Log the operation
        api_logger.log_document_processing(
            file_path=request.file_path,
            status="success" if result.get("processing_successful") else "failed",
            details={
                "processing_time": timer.elapsed_human,
                "tables_processed": result.get("tables_processed", 0),
                "pages": result.get("total_pages", 0)
            }
        )
        
        # Check if processing was successful
        if not result.get("processing_successful", False):
            return error_response(
                message="Document processing failed",
                errors=[result.get("error", "Unknown processing error")]
            )
        
        # Convert result to data model
        doc_data = ProcessDocumentData(
            file_name=result.get("file_name", Path(request.file_path).name),
            file_path=result.get("file_path", request.file_path),
            tables_processed=result.get("tables_processed"),
            text_chunks_processed=result.get("text_chunks_processed"),
            embedded_documents_created=result.get("embedded_documents_created"),
            vector_points_stored=result.get("vector_points_stored"),
            high_confidence_tables=result.get("high_confidence_tables"),
            average_confidence=result.get("average_confidence"),
            success_rate=result.get("success_rate"),
            total_pages=result.get("total_pages"),
            processing_timestamp=result.get("processing_timestamp")
        )
        
        return success_response(
            data=doc_data,
            message=f"Document '{Path(request.file_path).name}' processed successfully in {timer.elapsed_human}"
        )
        
    except Exception as e:
        return error_response(
            message="Document processing failed",
            errors=[str(e)]
        )


@router.post("/documents/batch", response_model=ApiResponse)
async def batch_process_documents(request: BatchProcessRequest):
    """
    Process multiple PDF documents in a directory.
    
    Args:
        request: Batch processing request
    
    Returns:
        ApiResponse: Batch processing results
    """
    if not os.path.exists(request.directory_path):
        return error_response(
            message="Directory not found",
            errors=[f"Directory not found: {request.directory_path}"]
        )
    
    try:
        results = rag_service.batch_process_documents(request.directory_path)
        
        # Convert results to data models
        response_results = []
        successful = 0
        
        for result in results:
            doc_data = ProcessDocumentData(
                file_name=result.get("file_name", ""),
                file_path=result.get("file_path", ""),
                tables_processed=result.get("tables_processed"),
                text_chunks_processed=result.get("text_chunks_processed"),
                embedded_documents_created=result.get("embedded_documents_created"),
                vector_points_stored=result.get("vector_points_stored"),
                high_confidence_tables=result.get("high_confidence_tables"),
                average_confidence=result.get("average_confidence"),
                success_rate=result.get("success_rate"),
                total_pages=result.get("total_pages"),
                processing_timestamp=result.get("processing_timestamp")
            )
            response_results.append(doc_data)
            
            if result.get("processing_successful", False):
                successful += 1
        
        batch_data = BatchProcessData(
            results=response_results,
            total_files=len(results),
            successful_files=successful,
            failed_files=len(results) - successful
        )
        
        message = f"Processed {successful}/{len(results)} documents successfully"
        
        return success_response(
            data=batch_data,
            message=message
        )
        
    except Exception as e:
        return error_response(
            message="Batch processing failed",
            errors=[str(e)]
        )


@router.get("/documents/{file_name}/summary", response_model=ApiResponse)
async def get_document_summary(file_name: str):
    """
    Get summary of a processed document.
    
    Args:
        file_name: Name of the document
    
    Returns:
        ApiResponse: Document summary
    """
    try:
        summary = rag_service.get_document_summary(file_name)
        
        if not summary.get("found", False):
            return error_response(
                message="Document not found",
                errors=[f"Document '{file_name}' not found in the database"]
            )
        
        summary_data = DocumentSummaryData(
            file_name=file_name,
            found=summary.get("found", False),
            total_documents=summary.get("total_documents"),
            content_types=summary.get("content_types"),
            table_types=summary.get("table_types"),
            pages_covered=summary.get("pages_covered"),
            page_numbers=summary.get("page_numbers")
        )
        
        return success_response(
            data=summary_data,
            message=f"Retrieved summary for document '{file_name}'"
        )
        
    except Exception as e:
        return error_response(
            message="Failed to get document summary",
            errors=[str(e)]
        )


@router.delete("/documents/{file_name}", response_model=ApiResponse)
async def remove_document(file_name: str):
    """
    Remove a document from the vector database.
    
    Args:
        file_name: Name of the document to remove
    
    Returns:
        ApiResponse: Removal results
    """
    try:
        removed_count = rag_service.remove_document(file_name)
        
        if removed_count == 0:
            return error_response(
                message="Document not found",
                errors=[f"No documents found for '{file_name}' to remove"]
            )
        
        removal_data = RemoveDocumentData(
            file_name=file_name,
            documents_removed=removed_count
        )
        
        return success_response(
            data=removal_data,
            message=f"Successfully removed {removed_count} document chunks for '{file_name}'"
        )
        
    except Exception as e:
        return error_response(
            message="Failed to remove document",
            errors=[str(e)]
        )


@router.get("/documents", response_model=ApiResponse)
async def list_documents():
    """
    List all processed documents in the vector database.
    
    Returns:
        ApiResponse: List of processed documents with metadata
    """
    try:
        # This would require implementing a method in the vector storage service
        # For now, return a placeholder
        placeholder_data = {
            "note": "This endpoint would list all processed documents with their metadata",
            "status": "not_implemented"
        }
        
        return success_response(
            data=placeholder_data,
            message="Document listing not yet implemented"
        )
        
    except Exception as e:
        return error_response(
            message="Failed to list documents",
            errors=[str(e)]
        )