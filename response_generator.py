"""
RAG Response Generation for Multi-Context Chatbot.
Retrieves context -> Injects into prompt -> Gemini response + source attribution.
Production-ready with fallbacks.
"""

from typing import List, Optional, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage, AIMessage

# Assuming these are local modules you have built
from utils import get_gemini_api_key
from chroma_manager import ChromaManager


def generate_rag_response(
    query: str,
    retriever: Optional[BaseRetriever],
    model: str = "gemini-2.0-flash",
    k: int = 4,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Full RAG response generation pipeline.
    
    Args:
        query: User question
        retriever: Chroma retriever (nec/wattmonk/general=None)
        model: Gemini model
        k: Retrieval count
        temperature: Creativity (0.1 = focused)
    
    Returns:
        {
            'answer': str,
            'sources': str, 
            'context_used': bool,
            'confidence': str
        }
    """
    try:
        # LLM setup
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=get_gemini_api_key(),
            temperature=temperature
        )
        
        # Retrieve context
        context = ""
        sources = []
        
        if retriever:
            docs: List[Document] = retriever.invoke(query)
            
            if docs:
                context = "\n\n".join([doc.page_content for doc in docs[:k]])
                sources = [
                    f"{i+1}. {doc.metadata.get('source', 'Unknown')} (p{doc.metadata.get('page', '?')})"
                    for i, doc in enumerate(docs[:k])
                ]
                source_text = "\n".join(sources)
            else:
                source_text = "No specific sources found."
        else:
            source_text = "General knowledge."
        
        # RAG Prompt Template
        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""
You are a helpful assistant for technical queries.

CONTEXT (use ONLY this information when relevant):
{context}

QUESTION: {query}

Guidelines:
- Answer precisely using context when available
- If context insufficient, use general knowledge
- Be concise but complete
- Cite sources implicitly in response

RESPONSE:
"""
        )
        
        # Chain execution
        chain = prompt_template | llm
        response = chain.invoke({
            "context": context, 
            "query": query
        })
        
        answer = response.content.strip()
        
        # Metadata
        context_used = bool(context and len(context) > 50)
        confidence = "High" if context_used else "Medium"
        
        return {
            "answer": answer,
            "sources": source_text,
            "context_used": context_used,
            "confidence": confidence
        }
        
    except Exception as e:
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}. Please try rephrasing.",
            "sources": "Error - no sources.",
            "context_used": False,
            "confidence": "Low"
        }


def full_rag_pipeline(query: str, collection_name: str) -> Dict[str, Any]:
    """
    Wrapper to handle retriever instantiation and execute the RAG pipeline.
    """
    try:
        # Initialize your custom vector store manager
        chroma_manager = ChromaManager()
        
        # Note: Adjust 'get_retriever' to whatever method ChromaManager actually uses
        retriever = chroma_manager.get_retriever(collection_name) 
    except Exception as e:
        print(f"Warning: Could not load retriever for '{collection_name}'. Falling back to general knowledge. Error: {e}")
        retriever = None

    # Generate response
    return generate_rag_response(query, retriever)


if __name__ == "__main__":
    # Test
    result = full_rag_pipeline("NEC breaker sizing requirements?", "nec")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Confidence:", result["confidence"])