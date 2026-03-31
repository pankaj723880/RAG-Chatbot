"""
Conversation Memory for RAG Chatbot.
Streamlit session + LangChain message history.
"""

import json
from typing import List, Dict, Any
import streamlit as st

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory


class StreamlitChatMemory:
    """
    Production conversation memory for Streamlit RAG chatbot.
    """

    def __init__(self, max_history: int = 10, session_key: str = "chat_history"):
        self.max_history = max_history
        self.session_key = session_key
        self._init_session()

    def _init_session(self):
        """Initialize Streamlit session state."""
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []

    def add_message(self, role: str, content: str, **metadata) -> None:
        """
        Add message to history.
        """

        # 🔥 Fix role mismatch
        role_map = {
            "user": "human",
            "assistant": "ai"
        }
        role = role_map.get(role, role)

        if "timestamp_counter" not in st.session_state:
            st.session_state.timestamp_counter = 0
        st.session_state.timestamp_counter += 1

        message = {
            "role": role,
            "content": content,
            "timestamp": st.session_state.timestamp_counter,
            "metadata": metadata
        }

        # Append and trim
        st.session_state[self.session_key].append(message)
        if len(st.session_state[self.session_key]) > self.max_history:
            st.session_state[self.session_key] = st.session_state[self.session_key][-self.max_history:]

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full chat history."""
        return st.session_state[self.session_key].copy()

    def get_recent(self, n: int = 6) -> List[Dict[str, Any]]:
        """Get recent messages for context."""
        return st.session_state[self.session_key][-n:]

    def to_langchain_messages(self, n: int = 6) -> List[BaseMessage]:
        """Convert to LangChain message format for LLM."""
        recent = self.get_recent(n)
        messages = []

        for msg in recent:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        return messages

    def clear(self) -> None:
        """Clear history."""
        st.session_state[self.session_key] = []

    def export(self) -> str:
        """Export history as JSON."""
        return json.dumps(self.get_history(), indent=2)

    def load(self, history_json: str) -> None:
        """Load history from JSON."""
        try:
            history = json.loads(history_json)
            st.session_state[self.session_key] = history[-self.max_history:]
        except json.JSONDecodeError:
            st.warning("Invalid history JSON")


# 🔥 LangChain Memory Wrapper
class RAGChatMemory(ConversationBufferWindowMemory):
    """LangChain memory wrapper using Streamlit backend."""

    def __init__(self, *args, **kwargs):
        kwargs["k"] = kwargs.get("k", 6)
        super().__init__(*args, **kwargs)
        self.streamlit_memory = StreamlitChatMemory(max_history=12)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save to both LangChain and Streamlit."""
        super().save_context(inputs, outputs)

        human_content = inputs.get("query") or inputs.get("input", "")
        ai_content = outputs.get("response") or outputs.get("output", "")

        self.streamlit_memory.add_message("user", human_content)
        self.streamlit_memory.add_message("assistant", ai_content)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load from Streamlit for LangChain."""
        history = self.streamlit_memory.to_langchain_messages()
        return {"history": history}


# 🔥 Streamlit UI helper
def display_chat_history(memory: StreamlitChatMemory):
    """Display formatted chat history."""
    for msg in memory.get_history():
        with st.chat_message("user" if msg["role"] == "human" else "assistant"):
            st.markdown(msg["content"])

            # ✅ Safe metadata handling
            if msg.get("metadata", {}).get("sources"):
                with st.expander("Sources"):
                    st.markdown(msg["metadata"]["sources"])