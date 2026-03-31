"""
Multi-Context RAG Chatbot - Streamlit App
Final Optimized Version for Chandigarh University Submission
Supports: NEC, Wattmonk, and ANY Custom PDF Uploads
"""

import streamlit as st
import os
import datetime
from utils import get_gemini_api_key, RAG_PROMPT, format_sources, handle_error
from intent_classifier import IntentClassifier
from rag_pipeline import RAGPipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIG & DIRECTORIES ---
PERSIST_DIR = './chroma_db'
DOCS_DIR = './documents'
CUSTOM_DIR = './documents/custom'

for d in [DOCS_DIR, PERSIST_DIR, CUSTOM_DIR]:
    os.makedirs(d, exist_ok=True)

st.set_page_config(
    page_title="Multi-Context RAG Chatbot", 
    page_icon="🤖", 
    layout="wide"
)

# --- STYLING ---
st.markdown("""
<style>
    .stChatMessage { transition: opacity 0.3s ease-in-out; }
    .stMetric { background-color: #f0f2f6; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Multi-Context RAG Chatbot")
st.markdown("**Search NEC, Wattmonk, or ANY uploaded PDF document**")

# --- SESSION STATE INITIALIZATION ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag' not in st.session_state:
    st.session_state.rag = RAGPipeline(PERSIST_DIR)

if 'classifier' not in st.session_state:
    st.session_state.classifier = IntentClassifier()

# --- SIDEBAR: DOCUMENT UPLOAD ---
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=['pdf'], accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Clean filename and save
            filename = uploaded_file.name.replace(" ", "_")
            filepath = os.path.join(CUSTOM_DIR, filename)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ Saved: {filename}")
        
        if st.button("🚀 Process Documents", type="primary"):
            with st.status("Indexing your PDFs...", expanded=True) as status:
                st.session_state.rag.process_custom_documents(CUSTOM_DIR)
                status.update(label="✅ Knowledge base updated!", state="complete")
            st.rerun()

    # Show list of files currently in the system
    all_custom = [f for f in os.listdir(CUSTOM_DIR) if f.endswith('.pdf')]
    if all_custom:
        st.subheader("📋 System PDFs")
        for f in all_custom[-5:]:
            st.caption(f"• {f}")

# --- RETRIEVAL LOGIC ---
@st.cache_data
def cached_retrieve(_intent, _prompt):
    """
    Finds documents based on intent with a fallback to 'custom' collection.
    """
    # 1. Try the identified category (nec, wattmonk, etc.)
    retriever = st.session_state.rag.get_retriever(_intent.lower())
    docs = retriever.invoke(_prompt) if retriever else []
    
    # 2. SMART FALLBACK: If specific category is empty, check the 'custom' collection
    if not docs and _intent.lower() != "custom":
        custom_retriever = st.session_state.rag.get_retriever("custom")
        if custom_retriever:
            docs = custom_retriever.invoke(_prompt)
    
    return docs

# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 View Sources"):
                st.markdown(msg["sources"])

# --- CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Handle Greetings
        if any(greet in prompt.lower() for greet in ["hi", "hello", "hey", "who are you"]):
            response_text = "Hello! I am your RAG assistant. Upload a PDF in the sidebar and I can help you analyze it. What would you like to know?"
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            try:
                # 1. Intent Classification
                intent = st.session_state.classifier.classify_intent(prompt)
                
                # 2. Retrieval with Fallback
                docs = cached_retrieve(intent, prompt)
                sources = format_sources(docs) if docs else "General knowledge."
                
                # 3. Prepare LLM Context
                context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No specific PDF context found."
                
                # 4. Generate Response via Gemini
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    google_api_key=get_gemini_api_key(),
                    temperature=0.1
                )
                
                history_msgs = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" 
                    else AIMessage(content=m["content"]) 
                    for m in st.session_state.messages[-6:]
                ]

                chain = RAG_PROMPT.partial(context=context) | llm
                response = chain.invoke({"history": history_msgs, "query": prompt})
                
                # 5. UI Feedback & Display
                col1, col2 = st.columns([3,1])
                with col2:
                    st.info(f"🔍 **{intent.upper()}**")
                
                with col1:
                    st.markdown(response.content)
                    if docs:
                        with st.expander("📚 Sources"):
                            st.markdown(sources)

                # Save history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.content, 
                    "sources": sources
                })

            except Exception as e:
                st.error(handle_error(e))

# Footer
st.markdown("---")
st.caption("Built by Pankaj ")
