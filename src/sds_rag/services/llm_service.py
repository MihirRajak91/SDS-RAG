"""
LLM Service - Google Gemini integration for response generation.

Handles LLM interactions using Google Gemini Pro for generating responses
based on retrieved financial document context from vector database.
"""

import logging
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)


class LLMService:
    """Google Gemini LLM service for response generation."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """
        Initialize LLM service with Google Gemini.
        
        Args:
            api_key (str, optional): Google AI API key. Uses GOOGLE_AI_API_KEY env var if not provided
            model_name (str): Gemini model name to use
        """
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        if not self.api_key:
            raise ValueError("Google AI API key is required. Set GOOGLE_AI_API_KEY environment variable or pass api_key parameter")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        logger.info(f"LLM service initialized with model: {model_name}")
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate response using Gemini model.
        
        Args:
            prompt (str): Input prompt for the model
            max_tokens (int): Maximum tokens in response
            
        Returns:
            str: Generated response text
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1  # Low temperature for factual responses
                )
            )
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini model")
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def generate_financial_response(
        self, 
        user_query: str, 
        context_documents: List[Dict[str, Any]], 
        document_name: Optional[str] = None
    ) -> str:
        """
        Generate financial analysis response using retrieved context.
        
        Args:
            user_query (str): User's financial question
            context_documents (List[Dict]): Retrieved documents from vector search
            document_name (str, optional): Name of the source document
            
        Returns:
            str: Generated financial response with context
        """
        # Build context from retrieved documents
        context_text = self._build_context_text(context_documents)
        
        # Create comprehensive prompt
        prompt = self._create_financial_prompt(user_query, context_text, document_name)
        
        # Generate response
        return self.generate_response(prompt, max_tokens=1500)
    
    def _build_context_text(self, documents: List[Dict[str, Any]]) -> str:
        """
        Build context text from retrieved documents.
        
        Args:
            documents (List[Dict]): Retrieved documents with content and metadata
            
        Returns:
            str: Formatted context text
        """
        if not documents:
            return "No relevant financial data found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Format document context
            doc_context = f"Document {i}:\n"
            doc_context += f"Content: {content}\n"
            
            # Add relevant metadata
            if metadata.get("content_type"):
                doc_context += f"Type: {metadata['content_type']}\n"
            if metadata.get("table_type"):
                doc_context += f"Table Type: {metadata['table_type']}\n"
            if metadata.get("page"):
                doc_context += f"Page: {metadata['page']}\n"
            if metadata.get("confidence_score"):
                doc_context += f"Confidence: {metadata['confidence_score']:.2f}\n"
            
            context_parts.append(doc_context)
        
        return "\n" + "="*50 + "\n".join(context_parts)
    
    def _create_financial_prompt(
        self, 
        user_query: str, 
        context_text: str, 
        document_name: Optional[str] = None
    ) -> str:
        """
        Create comprehensive prompt for financial analysis.
        
        Args:
            user_query (str): User's question
            context_text (str): Retrieved context
            document_name (str, optional): Source document name
            
        Returns:
            str: Complete prompt for the LLM
        """
        doc_info = f" from the document '{document_name}'" if document_name else ""
        
        prompt = f"""You are a financial analyst AI assistant specializing in analyzing financial documents and data. Your role is to provide accurate, insightful responses based on the retrieved financial information.

USER QUESTION: {user_query}

RETRIEVED FINANCIAL DATA{doc_info}:
{context_text}

INSTRUCTIONS:
1. Base your response ONLY on the provided financial data context
2. If the context doesn't contain sufficient information to answer the question, clearly state this
3. Provide specific numbers, percentages, and financial metrics when available
4. Explain financial trends, patterns, or relationships you observe in the data
5. Use professional financial terminology appropriately
6. Structure your response clearly with bullet points or sections when helpful
7. If comparing periods or categories, highlight the differences clearly
8. Always cite which part of the data supports your statements

IMPORTANT GUIDELINES:
- Do not make assumptions or provide information not present in the context
- If asked about data not in the context, say "This information is not available in the provided data"
- Focus on accuracy and precision in financial analysis
- Provide actionable insights when possible
- Use appropriate financial analysis frameworks (ratios, trends, comparisons)

Please provide a comprehensive response based on the financial data provided:"""

        return prompt
    
    def generate_summary_response(self, documents: List[Dict[str, Any]], document_name: str) -> str:
        """
        Generate document summary response.
        
        Args:
            documents (List[Dict]): All documents from a financial report
            document_name (str): Name of the document
            
        Returns:
            str: Executive summary of the financial document
        """
        context_text = self._build_context_text(documents[:20])  # Limit to first 20 docs
        
        prompt = f"""You are a financial analyst providing an executive summary of the financial document '{document_name}'.

FINANCIAL DATA:
{context_text}

Please provide a comprehensive executive summary that includes:

1. **Document Overview**: Type of financial report and reporting period
2. **Key Financial Highlights**: Most important financial metrics and figures
3. **Performance Analysis**: Revenue, profitability, and growth trends
4. **Balance Sheet Summary**: Key assets, liabilities, and equity positions (if available)
5. **Cash Flow Insights**: Operating, investing, and financing cash flows (if available)
6. **Notable Items**: Any unusual or significant items that stand out

Structure your response professionally and focus on the most material financial information. Use specific numbers and percentages from the data to support your analysis.

EXECUTIVE SUMMARY:"""
        
        return self.generate_response(prompt, max_tokens=2000)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check LLM service health.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Test with simple prompt
            test_response = self.generate_response("Hello, please respond with 'OK' if you're working.", max_tokens=10)
            
            return {
                "service_healthy": True,
                "model_name": self.model_name,
                "api_key_configured": bool(self.api_key),
                "test_response": test_response[:50] if test_response else "No response"
            }
        except Exception as e:
            return {
                "service_healthy": False,
                "model_name": self.model_name,
                "api_key_configured": bool(self.api_key),
                "error": str(e)
            }