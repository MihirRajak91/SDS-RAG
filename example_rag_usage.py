#!/usr/bin/env python3
"""
Example RAG Usage - Demonstrates complete RAG pipeline.

Shows how to:
1. Process financial PDFs
2. Store embeddings in Qdrant  
3. Perform semantic search
4. Query financial data
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.services.rag_service import RAGService
from src.core.document_processor import process_and_store_rag


def example_rag_pipeline():
    """Demonstrate complete RAG pipeline usage."""
    print("🚀 Financial Document RAG Pipeline Example")
    print("=" * 50)
    
    # Initialize RAG service
    print("1. Initializing RAG service...")
    rag_service = RAGService()
    
    # Check health
    health = rag_service.health_check()
    print(f"   Qdrant healthy: {health['qdrant_healthy']}")
    print(f"   Collection: {health['collection_info']['name']}")
    print(f"   Embedding model: {health['embedding_model']}")
    
    # Process and store document (if available)
    pdf_path = "data/uploads/2022 Q3 AAPL.pdf"
    if Path(pdf_path).exists():
        print(f"\n2. Processing and storing document: {Path(pdf_path).name}")
        
        # Use convenience function
        result = process_and_store_rag(pdf_path)
        
        if result["processing_successful"]:
            print(f"   ✅ Success! Processed {result['tables_processed']} tables")
            print(f"   📊 High confidence tables: {result['high_confidence_tables']}")
            print(f"   🎯 Success rate: {result['success_rate']:.1f}%")
            print(f"   📝 Created {result['embedded_documents_created']} embedded documents")
            print(f"   💾 Stored {result['vector_points_stored']} vectors in database")
        else:
            print(f"   ❌ Processing failed: {result.get('error', 'Unknown error')}")
            return
        
        # Demonstrate search capabilities
        print(f"\n3. Demonstrating semantic search...")
        
        # Search examples
        search_queries = [
            "revenue and sales data",
            "balance sheet assets",
            "cash flow from operations",
            "quarterly results Q3"
        ]
        
        for query in search_queries:
            print(f"\n   Query: '{query}'")
            results = rag_service.search_financial_data(query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result['score']:.3f}")
                    print(f"      Type: {result['metadata'].get('content_type', 'unknown')}")
                    print(f"      Page: {result['metadata'].get('page', 'unknown')}")
                    print(f"      Content: {result['content'][:100]}...")
            else:
                print("   No results found")
        
        # Get document summary
        print(f"\n4. Document summary:")
        summary = rag_service.get_document_summary(Path(pdf_path).name)
        if summary["found"]:
            print(f"   📄 Total documents: {summary['total_documents']}")
            print(f"   📊 Content types: {summary['content_types']}")
            print(f"   🏷️  Table types: {summary['table_types']}")
            print(f"   📖 Pages covered: {summary['pages_covered']}")
        
        # Filter searches
        print(f"\n5. Filtered searches:")
        
        # Search only table summaries
        table_results = rag_service.search_financial_data(
            "financial performance", 
            limit=3,
            content_type="table_summary",
            min_confidence=0.7
        )
        print(f"   Table summaries found: {len(table_results)}")
        
        # Search specific table type
        income_results = rag_service.search_financial_data(
            "net income earnings",
            limit=3,
            table_type="income_statement"
        )
        print(f"   Income statement results: {len(income_results)}")
        
    else:
        print(f"\n⚠️  Sample PDF not found at: {pdf_path}")
        print("   To test the full pipeline:")
        print("   1. Place a financial PDF at data/uploads/")
        print("   2. Make sure Qdrant is running (docker run -p 6333:6333 qdrant/qdrant)")
        print("   3. Run this script again")
        
        # Demonstrate search on existing data
        print(f"\n   Checking for existing data...")
        try:
            info = rag_service.vector_storage.get_collection_info()
            print(f"   Collection has {info['points_count']} documents")
            
            if info['points_count'] > 0:
                print(f"\n   Testing search on existing data:")
                results = rag_service.search_financial_data("revenue", limit=5)
                print(f"   Found {len(results)} results for 'revenue'")
                
                for i, result in enumerate(results[:2], 1):
                    print(f"   {i}. {result['content'][:80]}...")
        except Exception as e:
            print(f"   Error checking existing data: {e}")
    
    print(f"\n✅ RAG Pipeline Example Complete!")
    print(f"\nNext steps:")
    print(f"- Integrate with your frontend application")
    print(f"- Add more sophisticated query processing")
    print(f"- Implement query expansion and filtering")
    print(f"- Add response generation with LLMs")


if __name__ == "__main__":
    try:
        example_rag_pipeline()
    except Exception as e:
        print(f"❌ Error running RAG example: {e}")
        import traceback
        traceback.print_exc()