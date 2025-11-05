# RAG Chat with Microsoft Agent Framework

This is a simplified proof-of-concept demonstrating the difference between **manual RAG orchestration** (in `RagChat/`) and **Agent Framework orchestration** (in this folder).

## 🎯 Purpose

This POC helps you understand:
- How Microsoft Agent Framework differs from direct Azure AI Project Client usage
- How agents orchestrate workflows vs manual orchestration
- When to use tools/functions in Agent Framework

## 🏗️ Architecture Comparison

### Standard Approach (RagChat/)

```python
# Manual orchestration - YOU control the flow
def ask(query):
    # 1. YOU create embeddings
    embedding = create_embedding(query)

    # 2. YOU retrieve context
    context = search_vectordb(embedding)

    # 3. YOU construct prompt with context
    prompt = f"Context: {context}\n\nQuestion: {query}"

    # 4. YOU call LLM
    response = llm.chat(prompt)

    return response
```

**Flow:** You → Embedding → Vector DB → You → Prompt Construction → You → LLM → You

### Agent Framework Approach (This folder)

```python
# Agent orchestration - AGENT controls the flow
def ask(query):
    # 1. YOU just ask the agent
    response = agent.run(query)

    # 2. AGENT decides when to retrieve
    # 3. AGENT calls retrieval tool
    # 4. AGENT uses context to answer

    return response
```

**Flow:** You → Agent → [Agent decides] → Retrieval Tool → Agent → Response

## 🔑 Key Differences

| Aspect | Standard (RagChat/) | Agent Framework (This folder) |
|--------|---------------------|-------------------------------|
| **Orchestration** | Manual - you control flow | Automatic - agent controls flow |
| **Client** | `AIProjectClient` | `AzureOpenAIChatClient` |
| **Retrieval** | Always called manually | Called by agent as needed |
| **Prompt Construction** | Manual string formatting | Agent handles internally |
| **Tools** | N/A | Retrieval is a function tool |
| **Flexibility** | You decide everything | Agent decides when to use tools |

## 📁 What's in This Folder

```
RagChat_AgentFramework/
├── agent/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Same config as RagChat
│   └── rag_agent.py          # Agent Framework implementation
└── app.py                    # Streamlit UI (adapted)
```

## 🚀 How to Run

Set the required environment variable:
```bash
export AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com"
```

Then run:
```bash
streamlit run RagChat_AgentFramework/app.py
```

**Note:** This version requires both `AZURE_AI_PROJECT_ENDPOINT` (for embeddings) and `AZURE_OPENAI_ENDPOINT` (for Agent Framework chat client).

## 🔍 Code Walkthrough

### 1. Creating the Retrieval Tool

In Agent Framework, retrieval becomes a **tool/function** that the agent can call:

```python
def retrieve_context(
    query: Annotated[str, Field(description="The user's question to search for")]
) -> str:
    """
    Retrieve relevant context from the AI course transcript.

    The agent calls this tool when it needs course information.
    """
    # Same ChromaDB search as before
    embedding = create_embedding(query)
    results = chroma_collection.query(embedding)
    return format_results(results)
```

**Key points:**
- Type annotations help the agent understand parameters
- Docstring describes WHEN to use the tool
- Returns string that agent can use in its reasoning

### 2. Creating the Agent

```python
agent = ChatAgent(
    chat_client=AzureOpenAIChatClient(
        credential=DefaultAzureCredential(),
        endpoint=AZURE_OPENAI_ENDPOINT,
        deployment_name=CHAT_MODEL
    ),
    instructions=(
        "You are a helpful AI assistant. "
        "Use the retrieve_context tool to search course materials. "
        "Base answers on retrieved context."
    ),
    tools=[retrieve_context]
)
```

**Key points:**
- Instructions tell the agent HOW to use tools
- Agent gets the tool as a function
- Agent decides WHEN to call it

### 3. Using the Agent

```python
# That's it! Agent orchestrates everything
response = await agent.run("What is deep learning?")
```

**What happens behind the scenes:**
1. Agent receives question
2. Agent thinks: "I need course information"
3. Agent calls `retrieve_context("What is deep learning?")`
4. Agent receives context
5. Agent formulates answer using context
6. Agent returns response

## 🤔 When to Use Each Approach

### Use Standard Approach (AIProjectClient) When:
- ✅ You have a simple, fixed pipeline (always retrieve → always generate)
- ✅ You want full control over every step
- ✅ You want to minimize dependencies
- ✅ The workflow is predictable and doesn't need decision-making

### Use Agent Framework When:
- ✅ The agent needs to DECIDE when to retrieve (not always needed)
- ✅ You have multiple tools the agent can choose from
- ✅ You want the agent to orchestrate complex workflows
- ✅ You're building conversational agents with tool use

## 💡 Learning Exercise

Try asking these questions to both versions:

1. **"What is deep learning?"**
   - Both should retrieve and answer

2. **"Hello, how are you?"**
   - Standard: Still retrieves (unnecessary)
   - Agent Framework: Agent might skip retrieval (smarter)

3. **"What is 2+2?"**
   - Standard: Retrieves course content (not needed)
   - Agent Framework: Agent might answer directly

**Observation:** Agent Framework can be more efficient by deciding when retrieval is actually needed!

## 📊 Pros and Cons

### Agent Framework Pros:
- ✅ Less boilerplate code
- ✅ Agent makes intelligent decisions
- ✅ Easy to add more tools
- ✅ Natural conversation flow
- ✅ Built-in tool orchestration

### Agent Framework Cons:
- ❌ Less transparent (agent decides internally)
- ❌ Harder to debug (what did the agent decide?)
- ❌ Additional dependency
- ❌ Potential for unexpected behavior

### Standard Approach Pros:
- ✅ Full transparency (you see every step)
- ✅ Predictable behavior
- ✅ Easy to debug
- ✅ Simple to understand

### Standard Approach Cons:
- ❌ More code to write and maintain
- ❌ Manual orchestration required
- ❌ Less flexible for complex workflows
- ❌ No intelligent decision-making

## 🎓 Key Takeaways

1. **Agent Framework adds orchestration**: Instead of you controlling the flow, the agent controls it

2. **Tools are the interface**: You define WHAT the agent can do (via tools), the agent decides WHEN to do it

3. **Trade-offs**: You get convenience and intelligence, but lose some control and transparency

4. **Not always needed**: For simple, fixed pipelines, the standard approach might be better

5. **Best for complexity**: Agent Framework shines when you have multiple tools or need decision-making

## 🔗 Comparison with AIProjectClient

### AIProjectClient (Standard)
- **Purpose**: Direct access to Azure AI services
- **Usage**: `project_client.get_openai_client()` → direct API calls
- **You control**: Everything - embeddings, retrieval, prompt construction, LLM calls

### AzureOpenAIChatClient (Agent Framework)
- **Purpose**: Create agents that orchestrate AI workflows
- **Usage**: `chat_client.create_agent(tools=[...])` → agent orchestrates
- **Agent controls**: Tool selection, workflow orchestration, context usage

## 📚 Next Steps

To learn more:
1. Try modifying the agent instructions - see how behavior changes
2. Add another tool (e.g., web search) - see agent choose between tools
3. Compare the code complexity between `RagChat/` and this folder
4. Read Microsoft Agent Framework docs: https://learn.microsoft.com/agent-framework

## 🎯 Bottom Line

**This POC shows that Agent Framework is essentially a layer on top of Azure OpenAI that adds intelligent orchestration.**

### Key Insights:

1. **Different Clients, Different Patterns**
   - **RagChat**: Uses `AIProjectClient` (project endpoint)
   - **Agent Framework**: Uses `AzureOpenAIChatClient` (OpenAI endpoint)
   - Both can achieve the same RAG result

2. **Agent Framework Simplifies Tool Orchestration**
   - Wraps function calling in a cleaner API
   - `ChatAgent` manages tool definitions and execution
   - Instructions guide agent behavior

3. **Control Flow Inverted**
   - **RagChat/**: You decide when to retrieve (always)
   - **RagChat_AgentFramework/**: Agent decides when to retrieve (intelligent)

4. **Trade-offs**
   - **Agent Framework**: Less code, smarter orchestration, additional abstraction
   - **Standard**: More control, simpler debugging, more transparent

Both achieve the same result (RAG Q&A), but the orchestration approach differs!
