"""
Embedding Service - Generate embeddings for financial document content.

Creates vector embeddings for tables and text chunks using LangChain embeddings.
Supports different embedding models and handles content preprocessing.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from src.models.schemas import ProcessedDocument, ProcessedTable, TextChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings for financial document content."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding service with specified model.
        
        Args:
            model_name (str): HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Embedding service initialized with model: {model_name}")
    
    def embed_document(self, document: ProcessedDocument) -> List[Document]:
        """
        Create embeddings for entire processed document.
        
        Args:
            document (ProcessedDocument): Processed document with tables and text
            
        Returns:
            List[Document]: List of LangChain documents with embeddings
        """
        langchain_docs = []
        
        # Process tables
        for table in document.structured_tables:
            table_docs = self._embed_table(table, document.metadata.file_name)
            langchain_docs.extend(table_docs)
        
        # Process text chunks
        for chunk in document.text_chunks:
            text_doc = self._embed_text_chunk(chunk, document.metadata.file_name)
            langchain_docs.append(text_doc)
        
        logger.info(f"Created {len(langchain_docs)} embedded documents from {document.metadata.file_name}")
        return langchain_docs
    
    def _embed_table(self, table: ProcessedTable, source_file: str) -> List[Document]:
        """
        Create embeddings for a processed table.
        
        Args:
            table (ProcessedTable): Processed table to embed
            source_file (str): Source file name
            
        Returns:
            List[Document]: List of documents for table content
        """
        docs = []
        
        # Create document for table summary
        table_summary = self._create_table_summary(table)
        summary_doc = Document(
            page_content=table_summary,
            metadata={
                "source": source_file,
                "content_type": "table_summary",
                "page": table.page,
                "table_index": table.table_index,
                "table_type": table.table_type.value,
                "confidence_score": table.confidence_score,
                "confidence_level": table.confidence_level.value,
                "dimensions": f"{len(table.data)} rows × {len(table.data.columns)} columns"
            }
        )
        docs.append(summary_doc)
        
        # Create documents for significant rows (if table is large)
        if len(table.data) > 5:
            row_docs = self._embed_table_rows(table, source_file)
            docs.extend(row_docs)
        
        return docs
    
    def _embed_table_rows(self, table: ProcessedTable, source_file: str) -> List[Document]:
        """
        Create embeddings for individual table rows.
        
        Args:
            table (ProcessedTable): Processed table
            source_file (str): Source file name
            
        Returns:
            List[Document]: Documents for table rows
        """
        docs = []
        
        # Sample every few rows for large tables
        step = max(1, len(table.data) // 10)  # Max 10 row samples
        
        for i in range(0, len(table.data), step):
            row = table.data.iloc[i]
            row_content = self._format_table_row(table.headers, row)
            
            row_doc = Document(
                page_content=row_content,
                metadata={
                    "source": source_file,
                    "content_type": "table_row",
                    "page": table.page,
                    "table_index": table.table_index,
                    "table_type": table.table_type.value,
                    "row_index": i,
                    "confidence_score": table.confidence_score
                }
            )
            docs.append(row_doc)
        
        return docs
    
    def _embed_text_chunk(self, chunk: TextChunk, source_file: str) -> Document:
        """
        Create embedding for text chunk.
        
        Args:
            chunk (TextChunk): Text chunk to embed
            source_file (str): Source file name
            
        Returns:
            Document: LangChain document with metadata
        """
        return Document(
            page_content=chunk.content,
            metadata={
                "source": source_file,
                "content_type": chunk.type,
                "page": chunk.metadata.get("page", 0),
                "word_count": chunk.metadata.get("word_count", 0),
                "paragraph_index": chunk.metadata.get("paragraph_index", 0)
            }
        )
    
    def _create_table_summary(self, table: ProcessedTable) -> str:
        """
        Create text summary of table for embedding.
        
        Args:
            table (ProcessedTable): Table to summarize
            
        Returns:
            str: Text summary of table content
        """
        summary_parts = [
            f"Table Type: {table.table_type.value.replace('_', ' ').title()}",
            f"Headers: {', '.join(table.headers)}",
            f"Dimensions: {len(table.data)} rows, {len(table.data.columns)} columns"
        ]
        
        # Add sample of numeric data if available
        numeric_cols = table.data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            sample_data = []
            for col in numeric_cols[:3]:  # First 3 numeric columns
                values = table.data[col].dropna().head(3)
                if len(values) > 0:
                    sample_data.append(f"{col}: {', '.join(map(str, values))}")
            
            if sample_data:
                summary_parts.append(f"Sample Data: {'; '.join(sample_data)}")
        
        return "\n".join(summary_parts)
    
    def _format_table_row(self, headers: List[str], row) -> str:
        """
        Format table row for embedding.
        
        Args:
            headers (List[str]): Table headers
            row: Pandas Series representing a table row
            
        Returns:
            str: Formatted row content
        """
        row_parts = []
        for header, value in zip(headers, row):
            if value is not None and str(value).strip():
                row_parts.append(f"{header}: {value}")
        
        return ", ".join(row_parts)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for search query.
        
        Args:
            query (str): Search query text
            
        Returns:
            List[float]: Query embedding vector
        """
        return self.embeddings.embed_query(query)