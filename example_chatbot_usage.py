#!/usr/bin/env python3
"""
Example Chatbot Usage - Demonstrates complete RAG chatbot with Gemini.

Shows how to:
1. Initialize the chat service with RAG + LLM
2. Process financial documents 
3. Ask questions and get AI responses
4. Use specialized query methods
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.services.chat_service import ChatService
from src.services.rag_service import RAGService


def example_chatbot():
    """Demonstrate complete RAG chatbot usage."""
    print("🤖 Financial RAG Chatbot with Google Gemini")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("⚠️  GOOGLE_AI_API_KEY environment variable not set!")
        print("   Please set your Google AI API key:")
        print("   export GOOGLE_AI_API_KEY='your_api_key_here'")
        return
    
    try:
        # Initialize chat service
        print("1. Initializing RAG chatbot...")
        chat_service = ChatService(google_api_key=api_key)
        
        # Health check
        health = chat_service.health_check()
        print(f"   💾 Qdrant: {'✅ Healthy' if health['rag_service']['qdrant_healthy'] else '❌ Unhealthy'}")
        print(f"   🧠 Gemini: {'✅ Healthy' if health['llm_service']['service_healthy'] else '❌ Unhealthy'}")
        print(f"   📊 Documents in DB: {health['rag_service']['collection_info']['points_count']}")
        
        if not health["chat_service_healthy"]:
            print("⚠️  Chat service is not fully healthy. Some components may not work.")
            return
        
        # Process document if available
        pdf_path = "data/uploads/2022 Q3 AAPL.pdf"
        document_name = None
        
        if Path(pdf_path).exists():
            print(f"\n2. Processing document for chatbot context...")
            rag_service = RAGService()
            result = rag_service.process_and_store_document(pdf_path)
            
            if result["processing_successful"]:
                document_name = result["file_name"]
                print(f"   ✅ Processed: {document_name}")
                print(f"   📊 Created {result['embedded_documents_created']} searchable chunks")
            else:
                print(f"   ❌ Processing failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"\n2. Sample document not found at: {pdf_path}")
            print("   Checking for existing data in database...")
        
        # Get document overview if we have data
        if health['rag_service']['collection_info']['points_count'] > 0:
            print(f"\n3. Available documents in database:")
            
            # If we have a specific document, get its overview
            if document_name:
                print(f"\n   Getting AI overview of {document_name}...")
                overview = chat_service.get_document_overview(document_name)
                
                if overview["found"]:
                    print(f"   📋 Document Stats:")
                    stats = overview["document_stats"]
                    print(f"      - Total chunks: {stats['total_documents']}")
                    print(f"      - Content types: {list(stats['content_types'].keys())}")
                    print(f"      - Table types: {list(stats['table_types'].keys())}")
                    
                    print(f"\n   🤖 AI Summary:")
                    print(f"   {overview['ai_summary'][:300]}...")
        
        # Interactive chat examples
        print(f"\n4. Chatbot Examples:")
        
        # Example queries
        example_queries = [
            "What was Apple's revenue in Q3 2022?",
            "Show me the key financial highlights from the quarterly results",
            "What were the major expenses in Q3?",
            "How did iPhone sales perform this quarter?",
            "What is the cash flow from operations?"
        ]
        
        for i, query in enumerate(example_queries, 1):
            print(f"\n   Example {i}: '{query}'")
            
            # Get response from chatbot
            response = chat_service.chat_with_suggestions(
                user_query=query,
                num_results=5,
                min_confidence=0.6
            )
            
            if response["success"]:
                print(f"   🤖 Response:")
                print(f"   {response['response'][:400]}...")
                print(f"   📚 Sources found: {response['sources_found']}")
                
                # Show follow-up suggestions if available
                if "follow_up_suggestions" in response:
                    print(f"   💡 Follow-up suggestions:")
                    for suggestion in response["follow_up_suggestions"][:3]:
                        print(f"      • {suggestion}")
            else:
                print(f"   ❌ Error: {response.get('error', 'Unknown error')}")
            
            print()
        
        # Specialized queries
        print(f"\n5. Specialized Query Examples:")
        
        # Table-focused query
        print(f"\n   📊 Table Analysis:")
        table_response = chat_service.ask_about_tables(
            "Analyze the income statement data and key metrics"
        )
        
        if table_response["success"] and table_response["sources_found"] > 0:
            print(f"   🤖 Table Analysis:")
            print(f"   {table_response['response'][:300]}...")
            print(f"   📋 Table sources: {table_response['sources_found']}")
        else:
            print(f"   ℹ️  No table data available for analysis")
        
        # Interactive mode
        print(f"\n6. Interactive Chat Mode:")
        print(f"   Type 'quit' to exit, 'help' for commands")
        
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    print_help_commands()
                    continue
                elif not user_input:
                    continue
                
                # Process user query
                print("   🤖 Thinking...")
                response = chat_service.chat_with_suggestions(
                    user_query=user_input,
                    num_results=5,
                    min_confidence=0.5
                )
                
                if response["success"]:
                    print(f"\n   🤖 {response['response']}")
                    
                    if response["sources_found"] > 0:
                        print(f"\n   📚 Found {response['sources_found']} relevant sources")
                        
                        # Show source details if requested
                        show_sources = input("   Show source details? (y/n): ").lower() == 'y'
                        if show_sources:
                            for i, doc in enumerate(response["context_documents"][:3], 1):
                                metadata = doc["metadata"]
                                print(f"   Source {i}: {metadata.get('content_type', 'unknown')} "
                                      f"(Page {metadata.get('page', '?')}, "
                                      f"Score: {doc['score']:.2f})")
                    
                    # Show follow-up suggestions
                    if "follow_up_suggestions" in response:
                        print(f"\n   💡 You might also ask:")
                        for suggestion in response["follow_up_suggestions"][:3]:
                            print(f"      • {suggestion}")
                else:
                    print(f"   ❌ {response['response']}")
                    
            except KeyboardInterrupt:
                print(f"\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        import traceback
        traceback.print_exc()


def print_help_commands():
    """Print available commands."""
    print("""
   Available commands:
   • Ask any financial question about your documents
   • 'quit' or 'exit' - Exit the chat
   • 'help' - Show this help message
   
   Example questions:
   • "What was the revenue growth this quarter?"
   • "Show me the balance sheet highlights"
   • "What are the key risks mentioned?"
   • "Compare expenses year over year"
   """)


if __name__ == "__main__":
    print("🚀 Starting Financial RAG Chatbot Example")
    print("Make sure you have:")
    print("  1. GOOGLE_AI_API_KEY environment variable set")
    print("  2. Qdrant running (docker run -p 6333:6333 qdrant/qdrant)")
    print("  3. Financial documents processed in the system")
    print()
    
    example_chatbot()