import os
import google.generativeai as genai
from typing import List, Dict
from dotenv import load_dotenv

from .utils import (
    LoggerFactory, 
    ConfigManager, 
    DataValidator
)

load_dotenv()

class RAGPipeline:
    def __init__(self, api_key: str = None):
        """
        Initialize RAG pipeline with utility functions
        
        :param api_key: Google Generative AI API key
        """
        # Utility logger
        self.logger = LoggerFactory.get_logger('rag_pipeline')
        
        # Load configuration
        config = ConfigManager.load_config()
        rag_config = config.get('rag', {})
        
        # API Key validation
        if not api_key:
            api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            self.logger.error("No API key provided")
            raise ValueError("Google API key is required")
        
        try:
            genai.configure(api_key=api_key)
            
            # Model configuration
            self.temperature = rag_config.get('temperature', 0.7)
            self.max_tokens = rag_config.get('max_tokens', 1024)
            
            # Initialize Gemini Pro model
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            self.logger.info("RAG Pipeline initialized successfully")
        
        except Exception as e:
            self.logger.error(f"RAG Pipeline initialization failed: {e}")
            raise
    
    def generate_response(
        self, 
        query: str, 
        context_articles: List[Dict]
    ) -> str:
        """
        Generate contextual response using RAG
        
        :param query: User's input query
        :param context_articles: List of relevant articles
        :return: Generated response
        """
        # Clean query
        clean_query = DataValidator.clean_text(query)
        
        # Construct context
        context = "\n\n".join([
            f"Title: {article.get('title', 'Untitled')}\n"
            f"Content: {DataValidator.clean_text(article.get('text', '')[:500])}..."
            for article in context_articles
        ])
        
        # Prompt engineering
        prompt = f"""
        Context Information:
        {context}
        
        User Query: {clean_query}
        
        Generate a comprehensive and precise answer based on the provided context.
        If the context does not contain sufficient information, clearly state that.
        
        Response Guidelines:
        1. Be factual and objective
        2. Use clear and concise language
        3. Cite sources if possible
        """
        
        try:
            # Generate response with configured parameters
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_tokens
                }
            )
            
            # Clean and return response
            return DataValidator.clean_text(response.text)
        
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return f"Error generating response: {e}"

# Example usage
if __name__ == "__main__":
    from embedder import TextEmbedder
    from scraper import WebScraper
    
    # Scrape and embed articles
    scraper = WebScraper("https://techcrunch.com")
    articles = scraper.extract_articles()
    
    embedder = TextEmbedder()
    embeddings = embedder.create_embeddings(articles)
    
    # RAG Pipeline
    rag_pipeline = RAGPipeline()
    query = "What are the latest trends in AI technology?"
    
    response = rag_pipeline.generate_response(query, articles)
    print(response)