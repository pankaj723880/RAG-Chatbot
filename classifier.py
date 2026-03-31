"""
Intent Classifier for Multi-Context RAG
Classifies queries into 'general', 'nec', or 'wattmonk'
"""

from utils import get_gemini_api_key, INTENT_PROMPT, handle_error
from langchain_google_genai import ChatGoogleGenerativeAI

class IntentClassifier:
    def __init__(self, model: str = 'gemini-2.5-flash'):
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=get_gemini_api_key(), temperature=0.1)
        self.prompt = INTENT_PROMPT
    
    def classify_intent(self, query: str) -> str:
        """
        Classify user query intent.
        Returns: 'general', 'nec', or 'wattmonk'
        """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({'query': query})
            intent = response.content.strip().lower()
            
            # Normalize response
            if 'general' in intent:
                return 'general'
            elif 'nec' in intent:
                return 'nec'
            elif 'wattmonk' in intent:
                return 'wattmonk'
            else:
                return 'general'  # Fallback
        except Exception as e:
            print(f"Classification error: {e}")
            return 'general'

