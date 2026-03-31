"""
Advanced RAG Features: Confidence, Follow-ups, Multi-lang, Citations.
Production-ready extensions for your chatbot.
"""

import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from utils import get_gemini_api_key
from intent_classifier import classify_query

# ⚡ Initialize LLM globally to improve performance and reduce latency
try:
    llm_flash = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=get_gemini_api_key(),
        temperature=0.7
    )
except Exception as e:
    print(f"Warning: Failed to initialize LLM globally. {e}")
    llm_flash = None


def calculate_confidence(
    docs_retrieved: int, 
    context_length: int, 
    relevance_score: float = 0.8
) -> Tuple[float, str]:
    """
    Calculate confidence score (0.0-1.0) with label.
    
    Args:
        docs_retrieved: # relevant docs found
        context_length: Total context chars  
        relevance_score: Retrieval similarity (mocked)
    
    Returns:
        (score, label) e.g., (0.92, "High")
    """
    base_score = min(docs_retrieved * 0.25, 0.5)
    context_score = min(context_length / 2000, 0.3)  # Quality heuristic
    final_score = min(base_score + context_score + relevance_score * 0.2, 1.0)
    
    if final_score >= 0.8:
        label = "High"
    elif final_score >= 0.6:
        label = "Medium" 
    else:
        label = "Low"
    
    return final_score, label


def generate_followups(
    query: str, 
    docs: List[Document], 
    num_suggestions: int = 3
) -> List[str]:
    """
    Generate context-aware follow-up questions using LLM.
    """
    try:
        if not llm_flash:
            raise ValueError("LLM not initialized")

        context_snippet = "\n".join([doc.page_content[:200] for doc in docs[:2]])
        
        prompt = f"""
Based on this query and context, suggest {num_suggestions} smart follow-up questions.

Original Query: {query}
Context Preview: {context_snippet}

Suggestions (numbered, concise):
"""
        
        response = llm_flash.invoke(prompt)
        suggestions = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|$)', response.content, re.DOTALL)
        
        return suggestions[:num_suggestions] if suggestions else []
        
    except Exception as e:
        print(f"Follow-up generation failed: {e}")
        # Fallback generic suggestions based on intent
        intent = classify_query(query)
        fallbacks = {
            "nec": ["What are the exact code references?", "Can you show the calculation?", "Related grounding requirements?"],
            "wattmonk": ["What is the full policy?", "Who do I contact?", "Timeline for this process?"],
            "general": ["Can you explain further?", "Any examples?", "What are alternatives?"]
        }
        return fallbacks.get(intent, ["Can you explain further?"])[:num_suggestions]


def detect_language(text: str) -> str:
    """Detect Hindi/English (simple heuristic)."""
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len(text)
    
    if total_chars == 0:
        return "english"
        
    if hindi_chars / total_chars > 0.3:
        return "hindi"
    return "english"


def translate_response(text: str, target_lang: str = "english") -> str:
    """Auto-translate responses to preferred language."""
    if detect_language(text) == target_lang:
        return text
        
    try:
        if not llm_flash:
            raise ValueError("LLM not initialized")
            
        prompt = f"""
Translate this response to natural {target_lang}:

{text}

Keep technical terms unchanged. Translate naturally:
"""
        
        response = llm_flash.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # Fallback to original


def format_citations(docs: List[Document]) -> str:
    """
    Format document citations with page numbers.
    Returns: "1. NEC Code §110.14 (p3), 2. Handbook (p17)"
    """
    if not docs:
        return "No specific citations."
    
    citations = []
    for i, doc in enumerate(docs[:4]):  # Top 4
        meta = doc.metadata
        source = meta.get('source', 'Unknown')
        page = meta.get('page', '?')
        filename = meta.get('filename', '').split('.')[0]
        
        citations.append(
            f"{i+1}. {source} - {filename} (p{page})"
        )
    
    return " | ".join(citations)


def enhance_rag_response(
    base_response: Dict[str, Any], 
    docs: List[Document],
    query: str,
    target_lang: str = "english"
) -> Dict[str, Any]:
    """
    Add advanced features to base RAG response.
    
    Input: {'answer': '...', 'sources': '...'}
    Output: Enhanced with confidence, followups, citations, lang
    """
    # 1. Confidence
    conf_score, conf_label = calculate_confidence(
        docs_retrieved=len(docs), 
        context_length=sum(len(d.page_content) for d in docs)
    )
    
    # 2. Citations  
    citations = format_citations(docs)
    
    # 3. Multi-lang
    answer = translate_response(base_response['answer'], target_lang)
    
    # 4. Follow-ups
    followups = generate_followups(query, docs)
    
    return {
        **base_response,
        "confidence_score": round(conf_score, 2),
        "confidence_label": conf_label,
        "citations": citations,
        "followups": followups,
        "detected_lang": detect_language(query),
        "answer_translated": answer
    }


# Demo
if __name__ == "__main__":
    print("Testing enhancements...")
    
    # 🐛 Fixed: Using LangChain Document objects instead of dicts
    mock_docs = [
        Document(
            page_content="NEC breaker rules require continuous loads to be sized at 125% of the total load...", 
            metadata={"source": "NEC", "page": 3, "filename": "nec_code_2023.pdf"}
        )
    ]
    
    response = {"answer": "Use 125% continuous load rule.", "sources": "NEC docs"}
    
    try:
        enhanced = enhance_rag_response(response, mock_docs, "What are the breaker sizing rules in NEC?")
        print("\n✅ Enhancement Successful!")
        for k, v in enhanced.items():
            print(f"- {k}: {v}")
    except Exception as e:
        print(f"\n❌ Error during enhancement: {e}")