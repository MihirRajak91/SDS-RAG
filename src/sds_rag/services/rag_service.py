"""
RAG Service - Complete RAG pipeline for financial documents.

Orchestrates document processing, embedding generation, and vector storage
to provide a complete RAG pipeline for financial document analysis.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.sds_rag.core.document_processor import DocumentProcessingOrchestrator
from src.sds_rag.services.embedding_service import EmbeddingService
from src.sds_rag.services.vector_storage_service import VectorStorageService
from src.sds_rag.models.schemas import ProcessedDocument
from src.sds_rag.utils import (
    find_pdf_files, validate_pdf_file, Timer,
    StructuredLogger, log_performance
)

logger = logging.getLogger(__name__)
structured_logger = StructuredLogger(__name__)


class RAGService:
    """Complete RAG pipeline for financial documents."""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG service with all components.
        
        Args:
            qdrant_host (str): Qdrant server host
            qdrant_port (int): Qdrant server port
            embedding_model (str): HuggingFace embedding model name
        """
        self.document_processor = DocumentProcessingOrchestrator()
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.vector_storage = VectorStorageService(
            host=qdrant_host,
            port=qdrant_port
        )
        logger.info(f"RAG service initialized: embedding_model={embedding_model}, qdrant={qdrant_host}:{qdrant_port}")
    
    def process_and_store_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete pipeline: process PDF and store in vector database.
        
        Args:
            pdf_path (str): Path to PDF file to process
            
        Returns:
            Dict[str, Any]: Processing summary with statistics
        """
        logger.info(f"Starting complete RAG pipeline for: {pdf_path}")
        
        try:
            # Step 1: Process document
            processed_doc = self.document_processor.process_document(pdf_path)
            
            # Step 2: Generate embeddings
            embedded_docs = self.embedding_service.embed_document(processed_doc)
            
            # Step 3: Store in vector database
            point_ids = self.vector_storage.store_documents(embedded_docs)
            
            # Generate summary
            summary = {
                "file_path": pdf_path,
                "file_name": processed_doc.metadata.file_name,
                "processing_successful": True,
                "tables_processed": len(processed_doc.structured_tables),
                "text_chunks_processed": len(processed_doc.text_chunks),
                "embedded_documents_created": len(embedded_docs),
                "vector_points_stored": len(point_ids),
                "high_confidence_tables": len(processed_doc.get_high_confidence_tables()),
                "average_confidence": processed_doc.extraction_stats.average_confidence,
                "success_rate": processed_doc.extraction_stats.success_rate,
                "total_pages": processed_doc.extraction_stats.total_pages,
                "processing_timestamp": processed_doc.metadata.processing_timestamp
            }
            
            logger.info(f"RAG pipeline completed successfully: {processed_doc.metadata.file_name}")
            return summary
            
        except Exception as e:
            logger.error(f"RAG pipeline failed for {pdf_path}: {e}")
            return {
                "file_path": pdf_path,
                "file_name": Path(pdf_path).name,
                "processing_successful": False,
                "error": str(e)
            }
    
    def search_financial_data(
        self,
        query: str,
        limit: int = 10,
        content_type: Optional[str] = None,
        table_type: Optional[str] = None,
        source_file: Optional[str] = None,
        min_confidence: Optional[float] = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search financial documents using semantic similarity.
        
        Args:
            query (str): Natural language search query
            limit (int): Maximum number of results
            content_type (str, optional): Filter by content type (table_summary, table_row, narrative_text)
            table_type (str, optional): Filter by table type (income_statement, balance_sheet, etc.)
            source_file (str, optional): Filter by source file
            min_confidence (float, optional): Minimum confidence threshold
            
        Returns:
            List[Dict[str, Any]]: Search results with relevance scores
        """
        return self.vector_storage.similarity_search(
            query=query,
            limit=limit,
            content_type=content_type,
            table_type=table_type,
            source=source_file,
            min_confidence=min_confidence
        )
    
    def get_document_summary(self, file_name: str) -> Dict[str, Any]:
        """
        Get summary of processed document in vector database.
        
        Args:
            file_name (str): Name of source file
            
        Returns:
            Dict[str, Any]: Document summary with content breakdown
        """
        try:
            documents = self.vector_storage.get_documents_by_source(file_name)
            
            if not documents:
                return {"file_name": file_name, "found": False}
            
            # Analyze content types
            content_types = {}
            table_types = {}
            pages = set()
            
            for doc in documents:
                metadata = doc["metadata"]
                
                # Count content types
                content_type = metadata.get("content_type", "unknown")
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                # Count table types
                if "table_type" in metadata:
                    table_type = metadata["table_type"]
                    table_types[table_type] = table_types.get(table_type, 0) + 1
                
                # Track pages
                if "page" in metadata:
                    pages.add(metadata["page"])
            
            return {
                "file_name": file_name,
                "found": True,
                "total_documents": len(documents),
                "content_types": content_types,
                "table_types": table_types,
                "pages_covered": len(pages),
                "page_numbers": sorted(list(pages))
            }
            
        except Exception as e:
            logger.error(f"Error getting document summary: {e}")
            return {"file_name": file_name, "found": False, "error": str(e)}
    
    def remove_document(self, file_name: str) -> int:
        """
        Remove all content from a document from vector database.
        
        Args:
            file_name (str): Name of source file to remove
            
        Returns:
            int: Number of documents removed
        """
        return self.vector_storage.delete_by_source(file_name)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all RAG components.
        
        Returns:
            Dict[str, Any]: Health status of each component
        """
        return {
            "qdrant_healthy": self.vector_storage.health_check(),
            "collection_info": self.vector_storage.get_collection_info(),
            "embedding_model": self.embedding_service.model_name
        }
    
    def batch_process_documents(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """
        Process multiple PDF documents in a directory.
        
        Args:
            pdf_directory (str): Directory containing PDF files
            
        Returns:
            List[Dict[str, Any]]: Processing results for each file
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise ValueError(f"Directory does not exist: {pdf_directory}")
        
        pdf_files = find_pdf_files(pdf_directory)
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_directory}")
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.process_and_store_document(str(pdf_file))
                results.append(result)
                logger.info(f"Processed {pdf_file.name}: {'Success' if result['processing_successful'] else 'Failed'}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                results.append({
                    "file_path": str(pdf_file),
                    "file_name": pdf_file.name,
                    "processing_successful": False,
                    "error": str(e)
                })
        
        successful = sum(1 for r in results if r["processing_successful"])
        logger.info(f"Batch processing complete: {successful}/{len(results)} files processed successfully")
        
        return results