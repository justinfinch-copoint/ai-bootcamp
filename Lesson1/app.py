"""
AI Q&A Chatbot - Streamlit UI (Azure AI Foundry + Agent Framework Version)

A chatbot that answers questions about the AI training course content
using RAG (Retrieval Augmented Generation) with Azure AI Foundry and ChromaDB.

Run with:
    streamlit run Lesson1/app.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import chromadb
from chromadb.config import Settings
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Configuration
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "ai_course_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-35-turbo"  # Azure deployment name
TOP_K_RESULTS = 4

# Page configuration
st.set_page_config(
    page_title="AI Course Q&A Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Course Q&A Chatbot")
st.caption(
    "Ask me anything about the AI training course! (Powered by Azure AI Foundry)")


@st.cache_resource
def initialize_azure_client():
    """Initialize Azure AI Foundry client."""
    project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if not project_endpoint:
        st.error("❌ AZURE_AI_PROJECT_ENDPOINT environment variable not set")
        st.info("Please set your Azure AI Foundry project endpoint.")
        st.stop()

    try:
        credential = DefaultAzureCredential()
        project_client = AIProjectClient(
            credential=credential,
            endpoint=project_endpoint
        )
        return project_client
    except Exception as e:
        st.error(f"❌ Error connecting to Azure AI Foundry: {str(e)}")
        st.info("Make sure you've run 'az login' or configured your Azure credentials.")
        st.stop()


@st.cache_resource
def load_chroma_db():
    """Load the ChromaDB vector store."""
    if not os.path.exists(CHROMA_DB_PATH):
        st.error(f"❌ Vector database not found at {CHROMA_DB_PATH}")
        st.info(
            "Please run `python Lesson1/build_vectordb.py` first to build the vector database.")
        st.stop()

    try:
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        return collection
    except Exception as e:
        st.error(f"❌ Error loading vector database: {str(e)}")
        st.stop()


def retrieve_context(collection, query: str, project_client: AIProjectClient, k: int = TOP_K_RESULTS) -> list[dict]:
    """Retrieve relevant documents from ChromaDB using semantic search."""
    # Create embedding for the query
    with project_client.get_openai_client(api_version="2024-10-21") as openai_client:
        response = openai_client.embeddings.create(
            input=query,
            model=EMBEDDING_MODEL
        )
        query_embedding = response.data[0].embedding

    # Query ChromaDB for similar documents
    results = collection.query(
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


def generate_response(project_client: AIProjectClient, query: str, context_docs: list[dict]) -> str:
    """Generate a response using Azure OpenAI with retrieved context."""
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
    with project_client.get_openai_client(api_version="2024-10-21") as openai_client:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0
        )

    return response.choices[0].message.content


# Initialize Azure client and ChromaDB
project_client = initialize_azure_client()
collection = load_chroma_db()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the AI course..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant context
                context_docs = retrieve_context(
                    collection, prompt, project_client, TOP_K_RESULTS)

                # Generate answer
                answer = generate_response(
                    project_client, prompt, context_docs)

                # Display answer
                st.markdown(answer)

                # Show source context in expander
                if context_docs:
                    with st.expander("📚 View Source Context"):
                        for i, doc in enumerate(context_docs, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(doc['content'])
                            if doc.get('metadata'):
                                st.caption(f"Source: {doc['metadata']}")
                            st.divider()

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg})

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **Azure AI Foundry** for embeddings & chat
    - **ChromaDB** for vector storage
    - **Streamlit** for the UI
    - **Microsoft Agent Framework** architecture

    It answers questions based on the AI training course transcript.
    """)

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built for AI Training Course - Azure Edition")
