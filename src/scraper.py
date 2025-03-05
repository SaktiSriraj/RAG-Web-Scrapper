import requests
from bs4 import BeautifulSoup
from newspaper import Article
from typing import List, Dict
import os

from .utils import (
    LoggerFactory, 
    ConfigManager, 
    DataValidator, 
    FileManager
)

class WebScraper:
    def __init__(self, base_url: str):
        """
        Initialize WebScraper with utility functions
        
        :param base_url: Base URL to scrape
        """
        # Utility logger
        self.logger = LoggerFactory.get_logger('web_scraper')
        
        # Validate URL
        if not DataValidator.validate_url(base_url):
            self.logger.error(f"Invalid URL: {base_url}")
            raise ValueError("Invalid URL format")
        
        self.base_url = base_url
        
        # Load configuration
        config = ConfigManager.load_config()
        scraper_config = config.get('scraper', {})
        
        # Configuration parameters
        self.max_articles = scraper_config.get('max_articles', 10)
        self.timeout = scraper_config.get('timeout', 10)
        self.user_agent = scraper_config.get('user_agent', 'RAG-System-Scraper/1.0')
    
    def extract_articles(self) -> List[Dict]:
        """
        Extract articles from the specified website
        
        :return: List of scraped articles
        """
        try:
            # Custom headers
            headers = {
                'User-Agent': self.user_agent
            }
            
            # Fetch webpage
            response = requests.get(
                self.base_url, 
                headers=headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            
            scraped_articles = []
            
            for link in article_links[:self.max_articles]:
                full_url = requests.compat.urljoin(self.base_url, link['href'])
                
                try:
                    article = Article(full_url)
                    article.download()
                    article.parse()
                    
                    # Clean text
                    clean_text = DataValidator.clean_text(article.text)
                    
                    # Only add if article has meaningful content
                    if clean_text and len(clean_text) > 100:
                        scraped_article = {
                            'url': full_url,
                            'title': article.title,
                            'text': clean_text,
                            'authors': article.authors,
                            'publish_date': str(article.publish_date) if article.publish_date else None
                        }
                        
                        scraped_articles.append(scraped_article)
                
                except Exception as article_error:
                    self.logger.warning(f"Could not scrape article {full_url}: {article_error}")
            
            return scraped_articles
        
        except Exception as e:
            self.logger.error(f"Scraping error: {e}")
            return []
    
    def save_articles(self, articles: List[Dict], output_dir: str = 'scraped_data'):
        """
        Save scraped articles using FileManager
        
        :param articles: List of article dictionaries
        :param output_dir: Directory to save articles
        """
        # Ensure directory exists
        FileManager.ensure_directory(output_dir)
        
        for i, article in enumerate(articles):
            # Generate unique filename
            filename = os.path.join(
                output_dir, 
                f'article_{DataValidator.generate_unique_id(article)}.json'
            )
            
            # Save article
            FileManager.save_json(article, filename)
        
        self.logger.info(f"Saved {len(articles)} articles to {output_dir}")

# Example usage
if __name__ == "__main__":
    scraper = WebScraper("https://techcrunch.com")
    articles = scraper.extract_articles()
    scraper.save_articles(articles)