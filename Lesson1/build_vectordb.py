"""
Vector Database Builder for AI Q&A Chatbot - Azure AI Foundry Version

This script loads the Intro_To_AI_Transcript.pdf, splits it into chunks,
creates embeddings using Azure AI Foundry, and stores them in a ChromaDB vector database.

Run this script once to build the vector database:
    python Lesson1/build_vectordb.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configuration
PDF_PATH = "Intro_To_AI_Transcript.pdf"
CHROMA_DB_PATH = "chroma_db"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
COLLECTION_NAME = "ai_course_docs"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_pdf_text(pdf_path: str) -> list[dict]:
    """Load text from PDF file and return as list of page documents."""
    print(f"🔄 Loading PDF document from {pdf_path}...")
    reader = PdfReader(pdf_path)
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        documents.append({
            "page_content": text,
            "metadata": {
                "source": pdf_path,
                "page": page_num
            }
        })

    print(f"✅ Loaded {len(documents)} pages from PDF")
    return documents


def split_text_into_chunks(documents: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Split documents into smaller chunks with overlap."""
    print(f"\n🔄 Splitting documents into chunks (size={chunk_size}, overlap={overlap})...")
    chunks = []

    for doc in documents:
        text = doc["page_content"]
        metadata = doc["metadata"]

        # Split text into chunks
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append({
                    "page_content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": len(chunks)
                    }
                })

    print(f"✅ Created {len(chunks)} document chunks")
    return chunks


def create_embeddings_azure(chunks: list[dict], project_client: AIProjectClient, model: str) -> list[list[float]]:
    """Create embeddings for chunks using Azure AI Foundry."""
    print("\n🔄 Creating embeddings using Azure AI Foundry...")
    print("   (This may take a minute depending on document size)")

    # Get Azure OpenAI client from project
    with project_client.get_openai_client(api_version="2024-10-21") as openai_client:
        # Extract just the text content for embedding
        texts = [chunk["page_content"] for chunk in chunks]

        # Create embeddings in batches (Azure OpenAI has a limit of ~2048 inputs per request)
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"   Processing batch {i // batch_size + 1} of {(len(texts) + batch_size - 1) // batch_size}...")

            response = openai_client.embeddings.create(
                input=batch,
                model=model
            )

            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

    print(f"✅ Created {len(all_embeddings)} embeddings")
    return all_embeddings


def build_vector_database():
    """Load PDF, split into chunks, create embeddings, and build Chroma vector database."""

    # Check if Azure AI Project endpoint is set
    project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if not project_endpoint:
        print("❌ Error: AZURE_AI_PROJECT_ENDPOINT environment variable not set")
        print("   Please set your Azure AI Foundry project endpoint before running this script")
        print("   You can find it in the Azure AI Foundry portal under your project's Overview section")
        exit(1)

    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF file not found at {PDF_PATH}")
        exit(1)

    # Initialize Azure AI Foundry client
    print("🔄 Connecting to Azure AI Foundry...")
    try:
        credential = DefaultAzureCredential()
        project_client = AIProjectClient(
            credential=credential,
            endpoint=project_endpoint
        )
        print("✅ Connected to Azure AI Foundry")
    except Exception as e:
        print(f"❌ Error connecting to Azure AI Foundry: {str(e)}")
        print("   Make sure you've run 'az login' or configured your Azure credentials")
        exit(1)

    # Load PDF
    documents = load_pdf_text(PDF_PATH)

    # Split into chunks
    chunks = split_text_into_chunks(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # Create embeddings
    embeddings = create_embeddings_azure(chunks, project_client, EMBEDDING_MODEL)

    # Initialize ChromaDB
    print("\n🔄 Building ChromaDB vector database...")
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete collection if it exists (to rebuild fresh)
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print("   Deleted existing collection")
    except Exception:
        pass

    # Create collection
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "AI Course document embeddings"}
    )

    # Add documents to collection
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[chunk["page_content"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks]
    )

    print(f"✅ Vector database created and persisted to: {CHROMA_DB_PATH}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Total vectors: {len(chunks)}")
    print("\n🎉 Vector database build complete!")
    print("   You can now run the chatbot with: streamlit run Lesson1/app.py")


if __name__ == "__main__":
    build_vector_database()
