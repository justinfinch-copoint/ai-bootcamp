"""
Configuration settings for the RAG Agent.
"""

import os

# ChromaDB Configuration
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "ai_course_docs"

# Azure OpenAI Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-35-turbo"  # Azure deployment name

# Retrieval Configuration
TOP_K_RESULTS = 4

# Azure AI Configuration
AZURE_AI_PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-10-21"
