import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from typing import List, Dict

from .utils import (
    LoggerFactory, 
    ConfigManager, 
    FileManager,
    DataValidator
)

class TextEmbedder:
    def __init__(
        self, 
        model_name: str = None, 
        dimension: int = 384
    ):
        """
        Initialize text embedder with utility functions
        
        :param model_name: Sentence Transformer model name
        :param dimension: Embedding vector dimension
        """
        # Utility logger
        self.logger = LoggerFactory.get_logger('text_embedder')
        
        # Load configuration
        config = ConfigManager.load_config()
        embedder_config = config.get('embedder', {})
        
        # Model selection
        self.model_name = model_name or embedder_config.get('model_name', 'all-MiniLM-L6-v2')
        self.dimension = dimension
        
        # Initialize model
        try:
            # Use CPU explicitly to avoid CUDA conflicts
            self.model = SentenceTransformer(self.model_name, device='cpu')
            self.logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
        
        # Ensure directories
        FileManager.ensure_directory('embeddings')
        FileManager.ensure_directory('vector_index')
    
    def create_embeddings(self, articles: List[Dict]) -> np.ndarray:
        """
        Create embeddings for articles
        
        :param articles: List of article dictionaries
        :return: Numpy array of embeddings
        """
        # Clean and extract text
        texts = [
            DataValidator.clean_text(article.get('text', '')) 
            for article in articles
        ]
        
        # Generate embeddings
        try:
            embeddings = self.model.encode(texts)
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return np.array([])
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        articles: List[Dict]
    ):
        """
        Save embeddings and associated metadata
        
        :param embeddings: Numpy array of embeddings
        :param articles: List of article dictionaries
        """
        try:
            # Generate unique filename
            filename = os.path.join(
                'embeddings', 
                f'embeddings_{DataValidator.generate_unique_id({"count": len(articles)})}.npy'
            )
            
            # Save embeddings
            np.save(filename, embeddings)
            
            # Save metadata
            metadata_filename = filename.replace('.npy', '_metadata.json')
            FileManager.save_json(articles, metadata_filename)
            
            self.logger.info(f"Saved {len(embeddings)} embeddings to {filename}")
        except Exception as e:
            self.logger.error(f"Embedding saving failed: {e}")
    
    def load_embeddings(self, filename: str = None):
        """
        Load previously saved embeddings
        
        :param filename: Specific embedding file to load
        :return: Tuple of embeddings and metadata
        """
        try:
            # If no filename, get latest
            if not filename:
                embedding_files = sorted(
                    [f for f in os.listdir('embeddings') if f.endswith('.npy')],
                    reverse=True
                )
                filename = os.path.join('embeddings', embedding_files[0])
            
            # Load embeddings
            embeddings = np.load(filename)
            
            # Load metadata
            metadata_filename = filename.replace('.npy', '_metadata.json')
            metadata = FileManager.load_json(metadata_filename)
            
            self.logger.info(f"Loaded {len(embeddings)} embeddings from {filename}")
            return embeddings, metadata
        
        except Exception as e:
            self.logger.error(f"Embedding loading failed: {e}")
            return None, None