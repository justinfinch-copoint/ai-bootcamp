"""
RAG Agent implementation using Microsoft Agent Framework.

This module demonstrates how to use the Agent Framework to build a RAG system
where the agent orchestrates retrieval and generation, rather than manual orchestration.

Key Difference from standard approach:
- Standard: Manual retrieve → construct prompt → call LLM
- Agent Framework: Agent decides when to retrieve and orchestrates the flow
"""

import os
import asyncio
from typing import Annotated
import chromadb
from chromadb.config import Settings
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from agent_framework import ChatAgent
from agent_framework.azure import AzureOpenAIChatClient
from pydantic import Field

from .config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TOP_K_RESULTS,
    AZURE_AI_PROJECT_ENDPOINT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION
)


class RAGAgent:
    """
    RAG Agent using Microsoft Agent Framework.

    Unlike the standard implementation, this agent:
    - Uses Agent Framework's ChatAgent
    - Implements retrieval as a tool/function
    - Lets the agent orchestrate when to retrieve context
    """

    def __init__(self):
        """Initialize the RAG Agent with Agent Framework."""
        self.project_client = self._initialize_azure_client()
        self.collection = self._load_chroma_db()
        self.agent = self._create_agent()

    def _initialize_azure_client(self) -> AIProjectClient:
        """Initialize Azure AI Foundry client (for embeddings)."""
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

    def _create_retrieval_tool(self):
        """
        Create the retrieval tool function.

        KEY DIFFERENCE: In Agent Framework, retrieval becomes a tool that
        the agent can choose to call, rather than us always calling it manually.
        """
        def retrieve_context(
            query: Annotated[str, Field(description="The user's question to search for relevant course content")]
        ) -> str:
            """
            Retrieve relevant context from the AI course transcript.

            Use this tool to search the AI training course materials for information
            relevant to answering the user's question.
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
                n_results=TOP_K_RESULTS
            )

            # Format results as a string for the agent
            if results['documents'] and len(results['documents']) > 0:
                context_parts = []
                for i, doc in enumerate(results['documents'][0], 1):
                    context_parts.append(f"[Chunk {i}]\n{doc}\n")
                return "\n".join(context_parts)
            else:
                return "No relevant context found in the course materials."

        return retrieve_context

    def _create_agent(self) -> ChatAgent:
        """
        Create the Agent Framework agent with retrieval tool.

        KEY DIFFERENCE: We create an agent that orchestrates the workflow,
        rather than manually orchestrating ourselves.
        """
        if not AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT environment variable not set. "
                "This should be your Azure OpenAI endpoint (e.g., https://<resource>.openai.azure.com)"
            )

        # Create the chat client
        chat_client = AzureOpenAIChatClient(
            credential=DefaultAzureCredential(),
            endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=CHAT_MODEL,
            api_version=AZURE_OPENAI_API_VERSION
        )

        # Create the retrieval tool
        retrieval_tool = self._create_retrieval_tool()

        # Create the agent with instructions and tools
        agent = ChatAgent(
            chat_client=chat_client,
            instructions=(
                "You are a helpful AI assistant that answers questions about an AI training course. "
                "When answering questions, use the retrieve_context tool to search the course materials. "
                "Base your answers on the retrieved context. "
                "If the context doesn't contain relevant information, say so - don't make up information. "
                "Keep your answers concise and relevant."
            ),
            tools=[retrieval_tool]
        )

        return agent

    def ask(self, query: str) -> dict:
        """
        Ask a question and get an answer.

        KEY DIFFERENCE: We just pass the query to the agent, and it orchestrates
        everything (deciding when to retrieve, how to use context, etc.)

        Args:
            query: The user's question

        Returns:
            Dictionary with 'answer' key
        """
        # Run the agent - it will orchestrate the retrieval and generation
        result = asyncio.run(self._ask_async(query))
        return result

    async def _ask_async(self, query: str) -> dict:
        """Async version of ask."""
        # The agent orchestrates: retrieval → reasoning → response
        response = await self.agent.run(query)

        return {
            'answer': str(response),
            'context_docs': []  # Agent Framework handles context internally
        }
