from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Gemini API Key
def get_gemini_api_key():
    """Get Google Gemini API key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env. Get from https://aistudio.google.com/app/apikey")
    return api_key


# RAG Prompt Template
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "history", "query"],
    template="""You are an intelligent assistant.

Use the provided context to answer the question.
If the answer is not in the context, answer from general knowledge.

Context:
{context}

Conversation History:
{history}

Question:
{query}

Answer:"""
)


# Intent Classification Prompt Template
INTENT_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""Classify the following user query into exactly one category: general, nec, wattmonk.

Query: {query}

Respond with only the category name (general, nec, or wattmonk)."""
)


# Format sources
def format_sources(docs):
    """Format document sources with metadata."""
    sources = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        filename = doc.metadata.get("filename", "")
        page = doc.metadata.get("page", "?")
        if source == "custom":
            sources.append(f"{i+1}. Custom ({filename}, p{page})")
        else:
            sources.append(f"{i+1}. {source} (p{page})")
    return "\n".join(sources)


# Error handler
def handle_error(e):
    return f"Error: {str(e)}"

