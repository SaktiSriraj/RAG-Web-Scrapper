import os
import sys
import streamlit as st

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import WebScraper
from src.embedder import TextEmbedder
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

from src.utils import (
    LoggerFactory, 
    ConfigManager, 
    DataValidator
)

def main():
    # Utility logger
    logger = LoggerFactory.get_logger('streamlit_app')
    
    # Load configuration
    config = ConfigManager.load_config()
    app_config = config.get('app', {})
    
    # Streamlit configuration
    st.set_page_config(
        page_title=app_config.get('title', 'RAG Information System'),
        page_icon=app_config.get('icon', ':robot:')
    )
    
    # Title and description
    st.title(app_config.get('title', 'RAG Information System'))
    st.write("An AI-powered information retrieval and generation system")
    
    # Website URL input
    website_url = st.text_input("Enter Website URL to Scrape", "https://techcrunch.com")
    num_articles = st.slider("Number of Articles", 1, 20, 5)
    
    # Query input
    query = st.text_input("Enter your query:")
    
    if st.button("Retrieve and Generate"):
        try:
            # Validate URL
            if not DataValidator.validate_url(website_url):
                st.error("Invalid URL. Please enter a valid website URL.")
                return
            
            # Display progress
            progress_bar = st.progress(0)
            
            # Web Scraping
            with st.spinner("Scraping website..."):
                scraper = WebScraper(website_url)
                articles = scraper.extract_articles()[:num_articles]
                progress_bar.progress(25)
            
            # Embedding
            with st.spinner("Creating embeddings..."):
                embedder = TextEmbedder()
                embeddings = embedder.create_embeddings(articles)
                embedder.save_embeddings(embeddings, articles)
                progress_bar.progress(50)
            
            # Vector Store
            with st.spinner("Indexing documents..."):
                vector_store = VectorStore()
                vector_store.add_documents(
                    [article['text'] for article in articles],
                    metadata=articles
                )
                progress_bar.progress(75)
            
            # Semantic Search
            with st.spinner("Performing semantic search..."):
                context_articles = vector_store.search(query, top_k=3)
                progress_bar.progress(85)
            
            # RAG Pipeline
            with st.spinner("Generating response..."):
                rag_pipeline = RAGPipeline()
                
                # Prepare context for RAG
                context_docs = [
                    article['metadata'] for article in context_articles
                ]
                
                # Generate response
                response = rag_pipeline.generate_response(query, context_docs)
                progress_bar.progress(100)
            
            # Display results
            st.subheader("Retrieved Context")
            for article in context_docs:
                st.write(f"**{article.get('title', 'Untitled')}**")
                st.write(article.get('text', '')[:300] + "...")
            
            st.subheader("AI-Generated Response")
            st.write(response)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()