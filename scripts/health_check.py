#!/usr/bin/env python3
"""
Health Check Script - Check status of all RAG system components.

Performs comprehensive health checks on:
1. Qdrant vector database
2. Google Gemini LLM service
3. Embedding service
4. Complete RAG pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from sds_rag.services.vector_storage_service import VectorStorageService
from sds_rag.services.llm_service import LLMService
from sds_rag.services.embedding_service import EmbeddingService
from sds_rag.services.chat_service import ChatService
from sds_rag.services.rag_service import RAGService
import traceback


def check_qdrant_health():
    """Check Qdrant vector database health."""
    print("🔍 QDRANT HEALTH CHECK")
    print("-" * 30)
    
    try:
        # Initialize vector storage service
        vector_service = VectorStorageService()
        
        # Basic health check
        is_healthy = vector_service.health_check()
        
        if is_healthy:
            print("✅ Qdrant connection: HEALTHY")
            
            # Get collection info
            collection_info = vector_service.get_collection_info()
            print(f"   📊 Collection: {collection_info['name']}")
            print(f"   🔢 Documents: {collection_info['points_count']}")
            print(f"   📐 Vector size: {collection_info['vector_size']}")
            print(f"   📏 Distance: {collection_info['distance']}")
            
            # Test basic operations
            print("\n🧪 Testing basic operations...")
            
            # Test search (if we have data)
            if collection_info['points_count'] > 0:
                try:
                    results = vector_service.similarity_search("test query", limit=1)
                    print(f"   ✅ Search test: Found {len(results)} results")
                except Exception as e:
                    print(f"   ⚠️  Search test failed: {e}")
            else:
                print("   ℹ️  Search test skipped (no documents in database)")
                
        else:
            print("❌ Qdrant connection: FAILED")
            print("   Check if Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
            
        return is_healthy
        
    except Exception as e:
        print(f"❌ Qdrant health check failed: {e}")
        print("   Possible issues:")
        print("   - Qdrant not running")
        print("   - Wrong host/port configuration")
        print("   - Network connectivity issues")
        return False


def check_llm_health():
    """Check Google Gemini LLM service health."""
    print("\n🤖 GOOGLE GEMINI LLM HEALTH CHECK")
    print("-" * 40)
    
    try:
        # Check API key
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            print("❌ Google AI API Key: NOT SET")
            print("   Set environment variable: export GOOGLE_AI_API_KEY='your_key'")
            return False
        
        print("✅ Google AI API Key: CONFIGURED")
        print(f"   Key length: {len(api_key)} characters")
        print(f"   Key prefix: {api_key[:8]}...")
        
        # Initialize LLM service
        llm_service = LLMService(api_key=api_key)
        
        # Perform health check
        health_status = llm_service.health_check()
        
        if health_status["service_healthy"]:
            print("✅ Gemini service: HEALTHY")
            print(f"   🧠 Model: {health_status['model_name']}")
            print(f"   🧪 Test response: {health_status['test_response']}")
            
            # Test financial response generation
            print("\n🧪 Testing financial response generation...")
            test_prompt = "Analyze the following financial data: Revenue $100M, Expenses $70M, Net Income $30M"
            
            try:
                response = llm_service.generate_response(test_prompt, max_tokens=100)
                print(f"   ✅ Financial analysis test: {response[:100]}...")
                return True
                
            except Exception as e:
                print(f"   ⚠️  Financial analysis test failed: {e}")
                return False
                
        else:
            print("❌ Gemini service: FAILED")
            print(f"   Error: {health_status.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ LLM health check failed: {e}")
        print("   Possible issues:")
        print("   - Invalid API key")
        print("   - API quota exceeded")
        print("   - Network connectivity issues")
        print("   - Google AI service outage")
        return False


def check_embedding_health():
    """Check embedding service health."""
    print("\n🎯 EMBEDDING SERVICE HEALTH CHECK")
    print("-" * 35)
    
    try:
        # Initialize embedding service
        embedding_service = EmbeddingService()
        
        print("✅ Embedding service: INITIALIZED")
        print(f"   🧠 Model: {embedding_service.model_name}")
        
        # Test embedding generation
        print("\n🧪 Testing embedding generation...")
        
        test_texts = [
            "Apple reported revenue of $83 billion in Q3 2022",
            "The balance sheet shows assets of $352 billion"
        ]
        
        embeddings = embedding_service.embed_texts(test_texts)
        
        if embeddings and len(embeddings) == 2:
            print(f"   ✅ Text embedding test: Generated {len(embeddings)} embeddings")
            print(f"   📐 Vector dimensions: {len(embeddings[0])}")
            
            # Test query embedding
            query_embedding = embedding_service.embed_query("financial performance")
            print(f"   ✅ Query embedding test: Generated {len(query_embedding)}-dim vector")
            
            return True
        else:
            print("   ❌ Embedding generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Embedding service health check failed: {e}")
        print("   Possible issues:")
        print("   - HuggingFace model download failed")
        print("   - Insufficient memory/storage")
        print("   - Network connectivity for model download")
        return False


def check_rag_pipeline_health():
    """Check complete RAG pipeline health."""
    print("\n🚀 COMPLETE RAG PIPELINE HEALTH CHECK")
    print("-" * 40)
    
    try:
        # Initialize RAG service
        rag_service = RAGService()
        
        # Get health status
        health_status = rag_service.health_check()
        
        print(f"📊 Qdrant: {'✅ Healthy' if health_status['qdrant_healthy'] else '❌ Unhealthy'}")
        print(f"🧠 Embedding model: {health_status['embedding_model']}")
        print(f"💾 Collection: {health_status['collection_info']['name']}")
        print(f"🔢 Documents: {health_status['collection_info']['points_count']}")
        
        # Test search if we have data
        if health_status['collection_info']['points_count'] > 0:
            print("\n🧪 Testing RAG search...")
            
            results = rag_service.search_financial_data("revenue", limit=3)
            print(f"   ✅ Search test: Found {len(results)} results")
            
            if results:
                print(f"   📋 Sample result: {results[0]['content'][:80]}...")
                print(f"   🎯 Score: {results[0]['score']:.3f}")
                
        return health_status['qdrant_healthy']
        
    except Exception as e:
        print(f"❌ RAG pipeline health check failed: {e}")
        return False


def check_chatbot_health():
    """Check complete chatbot health."""
    print("\n💬 CHATBOT HEALTH CHECK")
    print("-" * 25)
    
    try:
        # Check API key first
        api_key = os.getenv("GOOGLE_AI_API_KEY")
        if not api_key:
            print("⚠️  Chatbot partially available (no LLM)")
            print("   Set GOOGLE_AI_API_KEY to enable full chatbot functionality")
            return False
        
        # Initialize chat service
        chat_service = ChatService(google_api_key=api_key)
        
        # Get comprehensive health status
        health_status = chat_service.health_check()
        
        print(f"🤖 Chatbot: {'✅ Healthy' if health_status['chat_service_healthy'] else '❌ Unhealthy'}")
        print(f"💾 RAG Service: {'✅ OK' if health_status['rag_service']['qdrant_healthy'] else '❌ Failed'}")
        print(f"🧠 LLM Service: {'✅ OK' if health_status['llm_service']['service_healthy'] else '❌ Failed'}")
        
        # Test chatbot if everything is healthy
        if health_status['chat_service_healthy']:
            # Check if we have data to test with
            doc_count = health_status['rag_service']['collection_info']['points_count']
            
            if doc_count > 0:
                print(f"\n🧪 Testing chatbot with {doc_count} documents...")
                
                test_query = "What is the financial performance?"
                response = chat_service.chat(test_query, num_results=2)
                
                if response["success"]:
                    print(f"   ✅ Chat test: Generated response ({len(response['response'])} chars)")
                    print(f"   📚 Sources found: {response['sources_found']}")
                    print(f"   💬 Sample response: {response['response'][:100]}...")
                    return True
                else:
                    print(f"   ❌ Chat test failed: {response.get('error', 'Unknown error')}")
                    return False
            else:
                print("   ℹ️  Chat test skipped (no documents in database)")
                print("   Chatbot is ready but needs documents to be processed first")
                return True
        else:
            print("   ❌ Chatbot components not healthy")
            return False
            
    except Exception as e:
        print(f"❌ Chatbot health check failed: {e}")
        return False


def main():
    """Run comprehensive health checks."""
    print("🏥 SDS-RAG SYSTEM HEALTH CHECK")
    print("=" * 50)
    print("Checking all system components...\n")
    
    # Track component health
    health_status = {
        "qdrant": False,
        "llm": False,
        "embedding": False,
        "rag_pipeline": False,
        "chatbot": False
    }
    
    # Run individual health checks
    health_status["qdrant"] = check_qdrant_health()
    health_status["llm"] = check_llm_health()
    health_status["embedding"] = check_embedding_health()
    health_status["rag_pipeline"] = check_rag_pipeline_health()
    health_status["chatbot"] = check_chatbot_health()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 HEALTH CHECK SUMMARY")
    print("=" * 50)
    
    healthy_count = sum(health_status.values())
    total_components = len(health_status)
    
    for component, is_healthy in health_status.items():
        status_icon = "✅" if is_healthy else "❌"
        print(f"{status_icon} {component.upper().replace('_', ' ')}: {'HEALTHY' if is_healthy else 'UNHEALTHY'}")
    
    print(f"\n🎯 Overall System Health: {healthy_count}/{total_components} components healthy")
    
    if healthy_count == total_components:
        print("🎉 All systems operational! Your RAG chatbot is fully functional.")
        print("\nNext steps:")
        print("• Process financial documents with: python test_pdf_extraction.py")
        print("• Try the interactive chatbot: python example_chatbot_usage.py")
        
    elif healthy_count >= 3:
        print("⚠️  System partially operational. Some features may be limited.")
        print("\nRecommendations:")
        
        if not health_status["qdrant"]:
            print("• Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        if not health_status["llm"]:
            print("• Set Google AI API key: export GOOGLE_AI_API_KEY='your_key'")
            
    else:
        print("🚨 System requires attention. Multiple components are unhealthy.")
        print("\nTroubleshooting:")
        print("• Check Docker and Qdrant installation")
        print("• Verify Google AI API key")
        print("• Check network connectivity")
        print("• Review error messages above")
    
    print(f"\n{'✅' if healthy_count == total_components else '⚠️'} Health check complete!")
    
    return healthy_count == total_components


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Health check interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Health check failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)