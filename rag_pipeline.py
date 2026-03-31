"""
RAG Pipeline for Multi-Context Chatbot
MODIFIED: Supports any PDF filename
"""

import os
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils import handle_error, get_gemini_api_key

class RAGPipeline:
    def __init__(self, persist_dir: str = './chroma_db'):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=get_gemini_api_key(),
            version="v1",
            task_type="retrieval_document"
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100, # Added slight overlap for better context
            length_function=len
        )

        self.collections = {}

    def _get_vectorstore(self, collection_name: str):
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )

    def process_documents(self, docs_dir: str = './documents'):
        """Processes all PDFs in the directory, regardless of name."""
        try:
            pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
            if not pdf_files:
                print("❌ No PDF files found.")
                return

            for pdf_file in pdf_files:
                filepath = os.path.join(docs_dir, pdf_file)
                filename_lower = pdf_file.lower()
                
                # --- DYNAMIC CLASSIFICATION ---
                # Still check for specific keywords, but don't 'continue' (skip) if they aren't found
                if 'nec' in filename_lower:
                    source = 'nec'
                elif 'wattmonk' in filename_lower:
                    source = 'wattmonk'
                else:
                    source = 'custom' # Any other PDF goes into the 'custom' collection

                print(f"📄 Processing {source.upper()}: {pdf_file}")
                loader = PyPDFLoader(filepath)
                pages = loader.load()

                for i, page in enumerate(pages):
                    page.metadata.update({
                        "source": source,
                        "filename": pdf_file,
                        "page": i + 1
                    })

                chunks = self.splitter.split_documents(pages)

                if source not in self.collections:
                    self.collections[source] = self._get_vectorstore(source)

                self.collections[source].add_documents(chunks)
                
            print("✅ All documents processed successfully.")
        except Exception as e:
            print(handle_error(e))

    def process_custom_documents(self, custom_dir: str = './documents/custom'):
        """Handled by the same logic as process_documents, defaulting to 'custom'."""
        self.process_documents(docs_dir=custom_dir)

    def get_retriever(self, collection: str, k: int = 5):
        if collection not in self.collections:
            try:
                self.collections[collection] = self._get_vectorstore(collection)
            except:
                return None
        return self.collections[collection].as_retriever(search_kwargs={'k': k})

    def load_existing(self):
        for coll in ['nec', 'wattmonk', 'custom']:
            try:
                self.collections[coll] = self._get_vectorstore(coll)
            except:
                pass