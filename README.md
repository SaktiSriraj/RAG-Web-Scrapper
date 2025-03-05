# Web Scraping RAG System

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that:
1. Scrapes text data from websites
2. Indexes scraped content in a vector database
3. Provides semantic search and context-aware responses

## Technology Stack
- Web Scraping: `requests`, `beautifulsoup4`
- Vector Database: `faiss-cpu`
- Embedding Generation: `sentence-transformers`
- LLM Integration: `langchain`
- UI: `streamlit`

## Project Structure
```
rag-web-scraper/
│
├── src/
│   ├── __init__.py
│   ├── web_scraper.py        # Web scraping logic
│   ├── embedder.py           # Text embedding generation
│   ├── vector_store.py       # Vector database management
│   ├── rag_pipeline.py       # RAG implementation
│   └── app.py                # Streamlit UI
│
├── data/
│   └── scraped_content/      # Stored scraped documents
│
├── requirements.txt
└── README.md
```

## Setup Instructions
1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Streamlit app: `streamlit run src/app.py`# Web Scraping RAG System

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline that:
1. Scrapes text data from websites
2. Indexes scraped content in a vector database
3. Provides semantic search and context-aware responses

## Technology Stack
- Web Scraping: `requests`, `beautifulsoup4`
- Vector Database: `faiss-cpu`
- Embedding Generation: `sentence-transformers`
- LLM Integration: `langchain`
- UI: `streamlit`

## Project Structure
```
rag-web-scraper/
│
├── src/
│   ├── __init__.py
│   ├── web_scraper.py        # Web scraping logic
│   ├── embedder.py           # Text embedding generation
│   ├── vector_store.py       # Vector database management
│   ├── rag_pipeline.py       # RAG implementation
│   └── app.py                # Streamlit UI
│
├── data/
│   └── scraped_content/      # Stored scraped documents
│
├── requirements.txt
└── README.md
```

## Setup Instructions
1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run the Streamlit app: `streamlit run src/app.py`