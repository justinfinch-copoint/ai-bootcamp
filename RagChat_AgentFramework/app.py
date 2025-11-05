"""
AI Q&A Chatbot - Streamlit UI (Microsoft Agent Framework Version)

This demonstrates using Microsoft Agent Framework for RAG, where the agent
orchestrates retrieval and generation automatically.

Key Difference from standard version:
- Standard: We manually retrieve context, construct prompts, and call LLM
- Agent Framework: Agent decides when to retrieve and orchestrates the flow

Run with:
    streamlit run RagChat_AgentFramework/app.py
"""

import streamlit as st
from agent import RAGAgent

# Page configuration
st.set_page_config(
    page_title="AI Course Q&A (Agent Framework)",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Course Q&A Chatbot")
st.caption(
    "Ask me anything about the AI training course! (Powered by Microsoft Agent Framework)")


@st.cache_resource
def initialize_agent():
    """Initialize the RAG Agent."""
    try:
        agent = RAGAgent()
        return agent
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
        st.stop()
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.stop()
    except RuntimeError as e:
        st.error(f"❌ {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error initializing agent: {str(e)}")
        st.stop()


# Initialize the RAG Agent
agent = initialize_agent()

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
                # Ask the agent - it orchestrates everything!
                result = agent.ask(prompt)
                answer = result['answer']

                # Display answer
                st.markdown(answer)

                # Note: Agent Framework handles context internally
                # We don't have direct access to retrieved chunks like before
                st.info("ℹ️ The agent automatically retrieved and used relevant context from the course materials.")

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
    - **Microsoft Agent Framework** for orchestration
    - **Azure AI Foundry** for embeddings & chat
    - **ChromaDB** for vector storage
    - **Streamlit** for the UI

    ### Key Difference from Standard RAG:

    **Standard Approach:**
    - We manually retrieve context
    - We manually construct prompts
    - We manually call the LLM

    **Agent Framework Approach:**
    - Agent decides when to retrieve
    - Agent orchestrates the workflow
    - Agent uses tools autonomously

    The agent has a **retrieval tool** and decides
    when and how to use it!
    """)

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built for AI Training Course - Agent Framework Demo")
