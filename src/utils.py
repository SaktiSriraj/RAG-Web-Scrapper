import os
import logging
from rich.logging import RichHandler
import hashlib
import re
from typing import Dict, Any, List
import json
from datetime import datetime

class ConfigManager:
    """
    Centralized configuration management with environment variable support
    """
    @staticmethod
    def load_config(config_path: str = None) -> Dict[str, Any]:
        """
        Load configuration from JSON or environment variables
        
        :param config_path: Path to configuration file
        :return: Configuration dictionary
        """
        # Default configuration
        default_config = {
            'scraper': {
                'user_agent': 'RAG-System-Scraper/1.0',
                'max_articles': 20,
                'timeout': 10
            },
            'embedder': {
                'model_name': 'all-MiniLM-L6-v2',
                'max_sequence_length': 256
            },
            'rag': {
                'temperature': 0.7,
                'max_tokens': 1024
            }
        }
        
        # Check for JSON config
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON in {config_path}. Using default configuration.")
        
        # Override with environment variables
        for section in default_config:
            for key in default_config[section]:
                env_key = f"{section.upper()}_{key.upper()}"
                env_value = os.getenv(env_key)
                if env_value:
                    default_config[section][key] = env_value
        
        return default_config

class LoggerFactory:
    """
    Centralized logging configuration
    """
    @staticmethod
    def get_logger(name: str = 'rag_system', log_level: str = 'INFO') -> logging.Logger:
        """
        Create a configured logger
        
        :param name: Logger name
        :param log_level: Logging level
        :return: Configured logger
        """
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers to prevent duplicate logging
        logger.handlers.clear()
        
        # Rich handler for colorful logging
        rich_handler = RichHandler(
            rich_tracebacks=True,
            tracebacks_show_locals=True
        )
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        rich_handler.setFormatter(formatter)
        
        # Add handler
        logger.addHandler(rich_handler)
        
        return logger

class DataValidator:
    """
    Data validation and cleaning utilities
    """
    @staticmethod
    def generate_unique_id(data: Dict[str, Any]) -> str:
        """
        Generate a unique hash for a dictionary
        
        :param data: Input dictionary
        :return: Unique hash
        """
        # Convert dictionary to a sorted, hashable representation
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        
        :param text: Input text
        :return: Cleaned text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        
        return text
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format
        
        :param url: URL to validate
        :return: Boolean indicating URL validity
        """
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return url_pattern.match(url) is not None

class FileManager:
    """
    File and directory management utilities
    """
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Create directory if it doesn't exist
        
        :param directory: Directory path
        """
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str) -> None:
        """
        Save dictionary to JSON file
        
        :param data: Data to save
        :param filepath: Output file path
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        Load JSON file
        
        :param filepath: Input file path
        :return: Loaded dictionary
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

# Demonstration of usage
if __name__ == "__main__":
    # Logger demonstration
    logger = LoggerFactory.get_logger()
    logger.info("Utility module loaded successfully")
    
    # Configuration demonstration
    config = ConfigManager.load_config()
    logger.info(f"Loaded configuration: {config}")
    
    # Validator demonstration
    url = "https://example.com"
    logger.info(f"URL Validation: {DataValidator.validate_url(url)}")