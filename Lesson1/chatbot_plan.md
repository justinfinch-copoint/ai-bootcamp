# AI Q&A Chatbot Implementation Plan (Azure AI Foundry Edition)

## Overview
Build a question-and-answer chatbot using Azure AI Foundry, Microsoft Agent Framework architecture, and ChromaDB to answer questions about the AI training course content from `Intro_To_AI_Transcript.pdf`.

## Architecture

### 1. Vector Database Setup (`build_vectordb.py`)
Separate script to create and persist the vector database using Azure AI Foundry:
- **Load PDF**: Use `pypdf.PdfReader` to load `Intro_To_AI_Transcript.pdf`
- **Split Documents**: Custom text splitter implementation
  - Chunk size: 200 characters
  - Overlap: 40 characters
- **Embed & Store**: Use Azure AI Foundry SDK for embeddings with ChromaDB vector store
  - Storage location: `Lesson1/chroma_db/`
  - Embeddings model: `text-embedding-3-small` (via Azure OpenAI)
  - Azure authentication: DefaultAzureCredential

### 2. Streamlit Chatbot UI (`app.py`)
Interactive chat interface with RAG capabilities using Azure AI Foundry:
- **Azure AI Foundry Client**: Initialize AIProjectClient with Azure credentials
- **Load Vector Store**: Connect to existing ChromaDB database
- **RAG Pipeline**:
  1. User submits question
  2. Create query embedding using Azure AI Foundry
  3. Retrieve top 4 similar chunks from ChromaDB
  4. Generate response using Azure OpenAI with retrieved context
- **Streamlit Interface**:
  - Chat history with session state
  - User input field
  - Display AI responses with source context
  - Show retrieved document chunks

## Technology Stack (2025 - Azure Edition)

### Core Libraries
- **azure-ai-projects**: Azure AI Foundry SDK for project management
- **azure-identity**: Azure authentication (DefaultAzureCredential)
- **azure-ai-inference**: Azure AI inference client
- **agent-framework**: Microsoft Agent Framework (preview)
- **chromadb**: Vector database backend
- **pypdf**: PDF parsing
- **streamlit**: Web UI framework
- **opentelemetry-sdk**: Observability and tracing

### Key Design Decisions
1. **Azure-Native Authentication**: Using DefaultAzureCredential for secure, keyless authentication
2. **Azure AI Foundry SDK**: Unified SDK for accessing Azure OpenAI models and AI services
3. **Agent Framework Architecture**: Structured to support future migration to full Agent Framework features
4. **Direct ChromaDB Integration**: Using ChromaDB API directly instead of LangChain wrapper
5. **Separate Vector DB Script**: Build once, use many times - faster iteration
6. **Persistent Storage**: ChromaDB persists to disk, no need to rebuild on every run

## Implementation Steps

1. ✅ **Document Plan** - Create this markdown file
2. ✅ **Update DevContainer** - Add Azure CLI feature and update dependencies
3. ✅ **Update Dependencies** - Replace LangChain with Azure AI packages
4. ✅ **Migrate Vector DB Builder** - Rewrite to use Azure AI Foundry SDK
5. ✅ **Migrate Chatbot App** - Rewrite to use Azure AI Foundry and Agent Framework patterns
6. **Build Vector Database** - Run `build_vectordb.py` once to create embeddings
7. **Launch Chatbot** - Run `streamlit run app.py` to start the UI

## Usage

### First Time Setup
```bash
# Rebuild devcontainer to get Azure CLI and new dependencies
# The post-create script will automatically install packages

# Authenticate with Azure
az login

# Set your Azure AI Foundry project endpoint
# Option 1: In secrets.env file
echo "AZURE_AI_PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project" > .devcontainer/secrets.env

# Option 2: Export environment variable
export AZURE_AI_PROJECT_ENDPOINT="https://your-project.services.ai.azure.com/api/projects/your-project"

# Build vector database (run once)
python Lesson1/build_vectordb.py
```

### Run Chatbot
```bash
# Launch Streamlit app
streamlit run Lesson1/app.py
```

## File Structure
```
Lesson1/
├── Intro_To_AI_Transcript.pdf    # Source document
├── chatbot_plan.md                # This file
├── build_vectordb.py              # Vector DB creation script (Azure version)
├── app.py                         # Streamlit chatbot UI (Azure version)
└── chroma_db/                     # Persisted vector database (generated)
```

## Environment Variables Required
- `AZURE_AI_PROJECT_ENDPOINT`: Azure AI Foundry project endpoint (required)
- Azure credentials via one of:
  - Azure CLI (`az login`) - recommended for local development
  - Service Principal (AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
  - Managed Identity (when running in Azure)

## Migration from OpenAI/LangChain

### What Changed
- ❌ **Removed**: OpenAI API key authentication
- ❌ **Removed**: All LangChain packages (langchain-openai, langchain-chroma, langchain-core)
- ✅ **Added**: Azure AI Foundry SDK (azure-ai-projects)
- ✅ **Added**: Azure authentication (azure-identity)
- ✅ **Added**: Microsoft Agent Framework (agent-framework)
- ✅ **Added**: Azure CLI for authentication

### What Stayed the Same
- ✅ **ChromaDB**: Still using ChromaDB for vector storage
- ✅ **Streamlit**: UI framework unchanged
- ✅ **RAG Pattern**: Same retrieval-augmented generation approach
- ✅ **Functionality**: Same chatbot capabilities and user experience

## Notes for Learning
- This is a learning project with production-quality patterns
- Azure AI Foundry provides enterprise-grade security and governance
- Agent Framework is in preview - architecture supports future enhancements
- Small chunk size (200) for fine-grained retrieval
- Overlap (40) ensures context isn't lost at chunk boundaries
- DefaultAzureCredential supports multiple auth methods for flexibility

## Future Enhancements
- Migrate to full Agent Framework RAG when features mature
- Add conversation memory using Agent Framework state management
- Implement multi-turn conversations with context preservation
- Add observability with OpenTelemetry tracing
- Deploy to Azure for production use with managed identity
