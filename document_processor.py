"""
Clean, reusable PDF document processing for RAG Chatbot.
Loads NEC/Wattmonk PDFs, extracts text, chunks, adds metadata.
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def process_rag_documents(
    docs_dir: str = "./documents",
    chunk_size: int = 400,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Process PDF files for RAG:
    - Loads PDFs from directory (NEC/Wattmonk naming convention)
    - Extracts text with PyPDFLoader
    - Splits into 200-500 token chunks
    - Adds metadata: source ("NEC"/"Wattmonk"), page
    
    Args:
        docs_dir: Directory containing PDFs
        chunk_size: Target chunk size (tokens)
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of processed Document chunks with metadata
    
    Example:
        chunks = process_rag_documents()
        # Ready for embedding/storage
    """
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(exist_ok=True)
    
    if not any(docs_dir.glob("*.pdf")):
        print("❌ No PDF files found in", docs_dir)
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\\n\\n", "\\n", ". ", " ", ""]
    )
    
    all_chunks: List[Document] = []
    
    for pdf_path in docs_dir.glob("*.pdf"):
        print(f"📄 Processing: {pdf_path.name}")
        
        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # Determine source from filename
        source = "NEC" if "nec" in pdf_path.name.lower() else "Wattmonk"
        
        # Add metadata to each page
        for i, page in enumerate(pages):
            page.metadata.update({
                "source": source,
                "filename": pdf_path.name,
                "page": i + 1
            })
        
        # Split into chunks (200-500 tokens typical with these settings)
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"   → {len(chunks)} chunks created")
    
    print(f"✅ Total: {len(all_chunks)} chunks processed")
    return all_chunks


# Usage example
if __name__ == "__main__":
    chunks = process_rag_documents()
    print(f"Ready for embedding: {len(chunks)} documents")

