"""
AI Q&A Chatbot - Streamlit UI (Azure AI Foundry + Agent Framework Version)

A chatbot that answers questions about the AI training course content
using RAG (Retrieval Augmented Generation) with Azure AI Foundry and ChromaDB.

Run with:
    streamlit run RagChat/app.py
"""

import streamlit as st
from agent import RAGAgent

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
                # Ask the agent
                result = agent.ask(prompt)
                answer = result['answer']
                context_docs = result['context_docs']

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
