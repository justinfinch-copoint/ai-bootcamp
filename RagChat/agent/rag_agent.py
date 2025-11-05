"""
RAG Agent implementation for the AI Course Q&A system.

This module implements a Retrieval Augmented Generation (RAG) agent
that answers questions using Azure AI Foundry and ChromaDB.
"""

import os
import chromadb
from chromadb.config import Settings
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from .config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TOP_K_RESULTS,
    AZURE_AI_PROJECT_ENDPOINT,
    AZURE_OPENAI_API_VERSION
)


class RAGAgent:
    """
    RAG Agent for answering questions about the AI training course.

    Uses Azure AI Foundry for embeddings and chat completion,
    and ChromaDB for vector storage and retrieval.
    """

    def __init__(self):
        """Initialize the RAG Agent with Azure AI and ChromaDB clients."""
        self.project_client = self._initialize_azure_client()
        self.collection = self._load_chroma_db()

    def _initialize_azure_client(self) -> AIProjectClient:
        """Initialize Azure AI Foundry client."""
        if not AZURE_AI_PROJECT_ENDPOINT:
            raise ValueError(
                "AZURE_AI_PROJECT_ENDPOINT environment variable not set. "
                "Please set your Azure AI Foundry project endpoint."
            )

        try:
            credential = DefaultAzureCredential()
            project_client = AIProjectClient(
                credential=credential,
                endpoint=AZURE_AI_PROJECT_ENDPOINT
            )
            return project_client
        except Exception as e:
            raise RuntimeError(
                f"Error connecting to Azure AI Foundry: {str(e)}. "
                "Make sure you've run 'az login' or configured your Azure credentials."
            )

    def _load_chroma_db(self):
        """Load the ChromaDB vector store."""
        if not os.path.exists(CHROMA_DB_PATH):
            raise FileNotFoundError(
                f"Vector database not found at {CHROMA_DB_PATH}. "
                "Please run `python RagChat/build_vectordb.py` first to build the vector database."
            )

        try:
            chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            collection = chroma_client.get_collection(name=COLLECTION_NAME)
            return collection
        except Exception as e:
            raise RuntimeError(f"Error loading vector database: {str(e)}")

    def _retrieve_context(self, query: str, k: int = TOP_K_RESULTS) -> list[dict]:
        """
        Retrieve relevant documents from ChromaDB using semantic search.

        Args:
            query: The user's question
            k: Number of top results to retrieve

        Returns:
            List of documents with content and metadata
        """
        # Create embedding for the query
        with self.project_client.get_openai_client(api_version=AZURE_OPENAI_API_VERSION) as openai_client:
            response = openai_client.embeddings.create(
                input=query,
                model=EMBEDDING_MODEL
            )
            query_embedding = response.data[0].embedding

        # Query ChromaDB for similar documents
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Format results
        context_docs = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                context_docs.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })

        return context_docs

    def _generate_response(self, query: str, context_docs: list[dict]) -> str:
        """
        Generate a response using Azure OpenAI with retrieved context.

        Args:
            query: The user's question
            context_docs: Retrieved context documents

        Returns:
            Generated answer string
        """
        # Build context string from retrieved documents
        context_str = "\n\n".join([doc['content'] for doc in context_docs])

        # Create system prompt with context
        system_prompt = (
            "You are a helpful AI assistant answering questions about an AI training course. "
            "Use the following context from the course transcript to answer the question. "
            "If you don't know the answer based on the context, say so - don't make up information. "
            "Keep your answer concise and relevant.\n\n"
            f"Context:\n{context_str}"
        )

        # Generate response using Azure OpenAI
        with self.project_client.get_openai_client(api_version=AZURE_OPENAI_API_VERSION) as openai_client:
            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )

        return response.choices[0].message.content

    def ask(self, query: str) -> dict:
        """
        Ask a question and get an answer with source context.

        Args:
            query: The user's question

        Returns:
            Dictionary with 'answer' and 'context_docs' keys
        """
        # Retrieve relevant context
        context_docs = self._retrieve_context(query)

        # Generate answer
        answer = self._generate_response(query, context_docs)

        return {
            'answer': answer,
            'context_docs': context_docs
        }
