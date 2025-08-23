"""
Vector Storage Service - Manage vector storage with Qdrant.

Handles vector database operations including storing embeddings,
similarity search, and metadata filtering using Qdrant.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchRequest
)
from langchain.schema import Document
import uuid

from src.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStorageService:
    """Manage vector storage operations with Qdrant."""
    
    def __init__(
        self, 
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "financial_documents",
        vector_size: int = 384  # all-MiniLM-L6-v2 dimension
    ):
        """
        Initialize vector storage service with Qdrant.
        
        Args:
            host (str): Qdrant server host
            port (int): Qdrant server port
            collection_name (str): Name of vector collection
            vector_size (int): Dimension of embedding vectors
        """
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_service = EmbeddingService()
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
        logger.info(f"Vector storage service initialized: {host}:{port}/{collection_name}")
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def store_documents(self, documents: List[Document]) -> List[str]:
        """
        Store documents with embeddings in vector database.
        
        Args:
            documents (List[Document]): Documents to store
            
        Returns:
            List[str]: List of point IDs for stored documents
        """
        if not documents:
            return []
        
        # Generate embeddings for all documents
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_service.embed_texts(texts)
        
        # Create points for Qdrant
        points = []
        point_ids = []
        
        for doc, embedding in zip(documents, embeddings):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": doc.page_content,
                    **doc.metadata
                }
            )
            points.append(point)
        
        # Store in Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} documents in vector database")
            return point_ids
            
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        limit: int = 10,
        content_type: Optional[str] = None,
        table_type: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search with optional filters.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            content_type (str, optional): Filter by content type
            table_type (str, optional): Filter by table type  
            source (str, optional): Filter by source file
            min_confidence (float, optional): Minimum confidence score
            
        Returns:
            List[Dict[str, Any]]: Search results with content and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)
        
        # Build filter conditions
        filter_conditions = []
        
        if content_type:
            filter_conditions.append(
                FieldCondition(key="content_type", match=MatchValue(value=content_type))
            )
        
        if table_type:
            filter_conditions.append(
                FieldCondition(key="table_type", match=MatchValue(value=table_type))
            )
        
        if source:
            filter_conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source))
            )
        
        if min_confidence is not None:
            filter_conditions.append(
                FieldCondition(key="confidence_score", range={"gte": min_confidence})
            )
        
        # Create filter if conditions exist
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        try:
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "content": result.payload.get("content", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "content"}
                })
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        Retrieve all documents from specific source file.
        
        Args:
            source (str): Source file name
            
        Returns:
            List[Dict[str, Any]]: All documents from source
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
                with_payload=True,
                with_vectors=False
            )
            
            documents = []
            for point in results[0]:  # results is tuple (points, next_page_offset)
                documents.append({
                    "id": point.id,
                    "content": point.payload.get("content", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "content"}
                })
            
            logger.info(f"Retrieved {len(documents)} documents from source: {source}")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents by source: {e}")
            raise
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents from specific source file.
        
        Args:
            source (str): Source file name
            
        Returns:
            int: Number of deleted documents
        """
        try:
            # Get documents to delete
            documents = self.get_documents_by_source(source)
            
            if not documents:
                return 0
            
            # Delete by IDs
            point_ids = [doc["id"] for doc in documents]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            
            logger.info(f"Deleted {len(point_ids)} documents from source: {source}")
            return len(point_ids)
            
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector collection.
        
        Returns:
            Dict[str, Any]: Collection statistics and info
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.name,
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if Qdrant service is healthy.
        
        Returns:
            bool: True if service is healthy
        """
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False