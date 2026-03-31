"""
Modified Keyword-based Intent Classifier
Supports 'Custom' as a dynamic fallback for any uploaded PDF.
"""

class IntentClassifier:
    """
    Classifies user query into 'nec', 'wattmonk', 'custom', or 'general'.
    """
    def __init__(self):
        self.keywords = {
            "nec": ["nec", "code", "electrical", "breaker", "voltage", "wiring", "ampacity", "grounding"],
            "wattmonk": ["wattmonk", "policy", "handbook", "pto", "hr", "leave", "holiday"],
        }

    def classify_intent(self, query: str) -> str:
        query_lower = query.lower()
        
        # 1. Check for specific context keywords first
        for intent, keys in self.keywords.items():
            if any(key in query_lower for key in keys):
                return intent
        
        # 2. Check if the user is explicitly referring to their own upload
        custom_triggers = ["pdf", "document", "file", "upload", "this doc", "above"]
        if any(trigger in query_lower for trigger in custom_triggers):
            return "custom"
            
        # 3. FALLBACK: Instead of 'general', return 'custom' 
        # This forces the app to check your PDF index for ANY question.
        return "custom"