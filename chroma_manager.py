"""
Modular Chroma Vector DB Manager for Multi-Context RAG.
Uses Gemini embeddings, separate collections (NEC/Wattmonk).
Production-ready with create/load/retrieve functions.
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from utils import get_gemini_api_key


class ChromaManager:
    """
    Production Chroma DB manager:
    - Gemini embeddings
    - NEC/Wattmonk collections
    - Create/load/retrieve
    """
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # Gemini embeddings (high quality)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=get_gemini_api_key()
        )
        
        self.collections: dict[str, Chroma] = {}
    
    def create_db(self, chunks: List[Document]) -> None:
        """
        Create Chroma DB from processed chunks.
        Groups by source metadata (NEC/Wattmonk).
        """
        sources = set(doc.metadata["source"] for doc in chunks)
        
        for source in sources:
            if source not in ["NEC", "Wattmonk"]:
                print(f"⚠️ Skipping unknown source: {source}")
                continue
            
            # Filter chunks by source
            source_chunks = [doc for doc in chunks if doc.metadata["source"] == source]
            
            if not source_chunks:
                continue
            
            print(f"🔄 Creating {source} collection ({len(source_chunks)} chunks)")
            
            # Create collection
            vectorstore = Chroma(
                collection_name=source.lower(),
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir)
            )
            
            # Add documents
            vectorstore.add_documents(source_chunks)
            vectorstore.persist()
            
            self.collections[source.lower()] = vectorstore
            print(f"✅ {source} collection created")
    
    def load_db(self) -> None:
        """Load existing collections (NEC/wattmonk)."""
        possible_sources = ["nec", "wattmonk"]
        
        for source in possible_sources:
            try:
                vectorstore = Chroma(
                    collection_name=source,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_dir)
                )
                self.collections[source] = vectorstore
                print(f"📂 Loaded {source} collection")
            except Exception as e:
                print(f"⚠️ {source} not found: {e}")
    
    def get_retriever(self, source: str, k: int = 4) -> Optional[BaseRetriever]:
        """
        Get retriever for source.
        Args:
            source: 'nec' or 'wattmonk'
            k: Top-k results
        """
        if source not in self.collections:
            print(f"❌ No {source} collection loaded")
            return None
        
        return self.collections[source].as_retriever(
            search_kwargs={"k": k}
        )
    
    def delete_db(self) -> None:
        """Delete all collections (reset)."""
        for coll in self.collections.values():
            coll.delete_collection()
        self.collections.clear()
        print("🗑️ DB reset")


# Standalone usage
def main():
    """Example workflow."""
    from document_processor import process_rag_documents
    
    # 1. Process documents
    chunks = process_rag_documents()
    
    # 2. Create DB
    manager = ChromaManager()
    manager.create_db(chunks)
    
    # 3. Test retriever
    retriever = manager.get_retriever("nec", k=3)
    if retriever:
        docs = retriever.invoke("breaker sizing")
        print(f"Retrieved {len(docs)} NEC docs")


if __name__ == "__main__":
    main()

