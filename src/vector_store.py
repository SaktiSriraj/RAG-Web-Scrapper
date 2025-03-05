import numpy as np
import faiss
from typing import List, Dict

from .utils import (
    LoggerFactory, 
    FileManager,
    DataValidator
)

class VectorStore:
    def __init__(self, dimension: int = 384):
        """
        Initialize vector store with FAISS
        
        :param dimension: Embedding vector dimension
        """
        self.logger = LoggerFactory.get_logger('vector_store')
        self.dimension = dimension
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Metadata storage
        self.metadata = []
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """
        Add documents to the vector store
        
        :param texts: List of text documents
        :param metadata: Optional metadata for each document
        """
        # Ensure metadata is provided
        if metadata is None:
            metadata = [{}] * len(texts)
        
        # Create embeddings (you might want to use a separate embedder)
        from .embedder import TextEmbedder
        
        embedder = TextEmbedder()
        
        # Prepare documents with metadata
        prepared_docs = [
            {
                'text': text,
                'metadata': meta
            } for text, meta in zip(texts, metadata)
        ]
        
        # Generate embeddings
        embeddings = embedder.create_embeddings(prepared_docs)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata.extend(metadata)
        
        self.logger.info(f"Added {len(texts)} documents to vector store")
    
    def search(self, query: str, top_k: int = 3):
        """
        Search vector store
        
        :param query: Search query
        :param top_k: Number of top results to return
        :return: List of most relevant documents
        """
        # Create embedder
        from .embedder import TextEmbedder
        
        embedder = TextEmbedder()
        
        # Generate query embedding
        query_embedding = embedder.create_embeddings([{'text': query}])
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve metadata for top results
        results = [
            {
                'metadata': self.metadata[idx],
                'distance': distances[0][i]
            } 
            for i, idx in enumerate(indices[0])
        ]
        
        return results
    
    def save(self, path: str = 'vector_store'):
        """
        Save vector store to disk
        
        :param path: Path to save vector store
        """
        FileManager.ensure_directory(path)
        
        # Save FAISS index
        faiss.write_index(self.index, f'{path}/faiss_index.bin')
        
        # Save metadata
        FileManager.save_json(self.metadata, f'{path}/metadata.json')
        
        self.logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str = 'vector_store'):
        """
        Load vector store from disk
        
        :param path: Path to load vector store from
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f'{path}/faiss_index.bin')
            
            # Load metadata
            self.metadata = FileManager.load_json(f'{path}/metadata.json')
            
            self.logger.info(f"Vector store loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            raise