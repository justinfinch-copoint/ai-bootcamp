# AI Q&A Chatbot - Setup & Usage Instructions (Azure AI Foundry Edition)

A question-and-answer chatbot that uses RAG (Retrieval Augmented Generation) with Azure AI Foundry and Microsoft Agent Framework to answer questions about the AI training course content.

## Prerequisites

- Python 3.11+
- Azure account with an Azure AI Foundry project
- Azure CLI (automatically installed in devcontainer)

## What's Different in This Version?

This version has been migrated from OpenAI/LangChain to Azure AI Foundry/Microsoft Agent Framework:

### Removed
- OpenAI API key authentication
- LangChain framework and all related packages
- Direct OpenAI SDK usage

### Added
- Azure AI Foundry SDK (`azure-ai-projects`)
- Azure authentication via `DefaultAzureCredential`
- Microsoft Agent Framework architecture (preview)
- Azure CLI for authentication

### Unchanged
- ChromaDB for vector storage
- Streamlit for the UI
- Same RAG pattern and functionality

## Setup Instructions

### 1. Azure AI Foundry Project Setup

You need an Azure AI Foundry project with access to Azure OpenAI models. If you don't have one:

1. Go to [Azure AI Foundry](https://ai.azure.com)
2. Create a new project or select an existing one
3. Deploy the following models (if not already deployed):
   - `text-embedding-3-small` for embeddings
   - `gpt-35-turbo` for chat completions
4. Copy your project endpoint from the Overview section
   - Format: `https://your-project-name.services.ai.azure.com/api/projects/your-project-id`

### 2. Authenticate with Azure

**Option A: Using Azure CLI (Recommended for local development)**

```bash
# Sign in to Azure
az login
```

This will open a browser window for authentication. The `DefaultAzureCredential` will automatically use these credentials.

**Option B: Using Service Principal (For CI/CD or automation)**

Set these environment variables in `.devcontainer/secrets.env`:

```bash
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
```

### 3. Set Your Azure AI Foundry Project Endpoint

**Option A: Using secrets.env file (Recommended)**

1. Copy the template:
   ```bash
   cp .devcontainer/secrets.env.template .devcontainer/secrets.env
   ```

2. Edit `.devcontainer/secrets.env` and set your endpoint:
   ```bash
   AZURE_AI_PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project
   ```

**Option B: Export environment variable**

```bash
export AZURE_AI_PROJECT_ENDPOINT="https://your-project.services.ai.azure.com/api/projects/your-project"
```

Or add it to your Lesson1/.env file:

```bash
echo "AZURE_AI_PROJECT_ENDPOINT=https://your-project.services.ai.azure.com/api/projects/your-project" > Lesson1/.env
```

### 4. Install Dependencies

If you're using the devcontainer, dependencies are automatically installed. Otherwise:

```bash
pip install -r requirements.txt
```

This will install:
- azure-ai-projects
- azure-identity
- azure-ai-inference
- agent-framework (preview)
- chromadb
- streamlit
- pypdf
- opentelemetry-sdk
- And other required packages

### 5. Build the Vector Database

**Run this once** to create the ChromaDB vector database from the PDF:

```bash
python Lesson1/build_vectordb.py
```

This will:
- Load `Intro_To_AI_Transcript.pdf`
- Split it into 200-character chunks with 40-character overlap
- Create embeddings using Azure OpenAI via Azure AI Foundry
- Save the vector database to `Lesson1/chroma_db/`

Expected output:

```
🔄 Connecting to Azure AI Foundry...
✅ Connected to Azure AI Foundry
🔄 Loading PDF document...
✅ Loaded X pages from PDF
🔄 Splitting documents into chunks...
✅ Created X document chunks
🔄 Creating embeddings using Azure AI Foundry...
✅ Created X embeddings
🔄 Building ChromaDB vector database...
✅ Vector database created and persisted
🎉 Vector database build complete!
```

### 6. Launch the Chatbot

```bash
streamlit run Lesson1/app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the Chatbot

1. Type your question in the chat input at the bottom
2. Press Enter to submit
3. The chatbot will:
   - Retrieve relevant context from the course transcript using semantic search
   - Generate an answer using Azure OpenAI with the retrieved context
   - Show the answer and source context
4. Click "📚 View Source Context" to see the retrieved document chunks
5. Use "🗑️ Clear Chat History" in the sidebar to start fresh

## Example Questions

- "What is machine learning?"
- "Explain the difference between supervised and unsupervised learning"
- "What are neural networks?"
- "How does deep learning work?"

## Troubleshooting

### Error: AZURE_AI_PROJECT_ENDPOINT not set

Make sure you've set your Azure AI Foundry project endpoint:

```bash
# Check if it's set
echo $AZURE_AI_PROJECT_ENDPOINT

# Set it if needed
export AZURE_AI_PROJECT_ENDPOINT="https://your-project.services.ai.azure.com/api/projects/your-project"
```

### Error: DefaultAzureCredentialError

Make sure you've authenticated with Azure:

```bash
# Sign in with Azure CLI
az login

# Verify authentication
az account show
```

### Error: Vector database not found

Run the database build script first:

```bash
python Lesson1/build_vectordb.py
```

### Error: PDF file not found

Make sure `Intro_To_AI_Transcript.pdf` exists in the `Lesson1/` folder.

### Error: Model deployment not found

Make sure you have the required models deployed in your Azure AI Foundry project:
- `text-embedding-3-small` for embeddings
- `gpt-35-turbo` for chat completions

You may need to update the model names in `Lesson1/app.py` to match your deployment names.

## Project Structure

```
Lesson1/
├── Intro_To_AI_Transcript.pdf    # Source document
├── README.md                      # This file
├── chatbot_plan.md                # Implementation plan
├── build_vectordb.py              # Vector DB creation script (Azure version)
├── app.py                         # Streamlit chatbot UI (Azure version)
└── chroma_db/                     # Persisted vector database (generated)
```

## Technical Details

- **Vector Store**: ChromaDB with persistent storage
- **Embeddings**: Azure OpenAI text-embedding-3-small (via Azure AI Foundry)
- **LLM**: Azure OpenAI gpt-35-turbo (via Azure AI Foundry)
- **Authentication**: Azure DefaultAzureCredential (supports Azure CLI, service principal, managed identity)
- **Chunking**: Custom text splitter (200 chars, 40 overlap)
- **Retrieval**: Top 4 most similar chunks using semantic search
- **Framework**: Azure AI Foundry SDK + Microsoft Agent Framework architecture

## Authentication Flow

The `DefaultAzureCredential` tries authentication methods in this order:

1. **Environment Variables**: AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
2. **Azure CLI**: Credentials from `az login`
3. **Managed Identity**: When running in Azure (App Service, Container Apps, etc.)
4. **Visual Studio Code**: Azure Account extension credentials
5. **Azure PowerShell**: If authenticated via PowerShell

For local development, using `az login` (option 2) is the simplest approach.

## Notes

- This is a learning project with production-quality patterns
- Azure AI Foundry provides enterprise-grade security and governance
- Microsoft Agent Framework is in preview - the architecture supports future enhancements
- The vector database only needs to be built once
- You can rebuild the database anytime by running `build_vectordb.py` again
- Chat history is stored in session state (resets on page refresh)

## Cost Considerations

- **Azure OpenAI Service**: You'll be charged for:
  - Embedding API calls when building the vector database
  - Embedding API calls for each user query
  - Chat completion API calls for each response
- **ChromaDB**: Free (runs locally)
- **Streamlit**: Free (runs locally)

Check Azure OpenAI pricing at: https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/

## Future Lessons

This migration pattern can be applied to future lessons that use LangChain:
- Replace LangChain chains with Azure AI Foundry SDK calls
- Use ChromaDB directly instead of LangChain wrappers
- Adopt Agent Framework patterns for more complex agent workflows
- Leverage Azure authentication for secure, keyless access

## Additional Resources

- [Azure AI Foundry Documentation](https://learn.microsoft.com/azure/ai-foundry/)
- [Microsoft Agent Framework Documentation](https://learn.microsoft.com/agent-framework/)
- [Azure AI Foundry SDK for Python](https://learn.microsoft.com/python/api/overview/azure/ai-projects-readme)
- [DefaultAzureCredential Documentation](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential)
