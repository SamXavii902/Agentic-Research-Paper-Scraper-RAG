import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

import threading

class RAGStore:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RAGStore, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, persist_directory="data/chroma_db"):
        if self._initialized:
            return
            
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        # Use a good, lightweight local embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize chroma db
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            client_settings=Settings(anonymized_telemetry=False)
        )
        self._initialized = True

    def add_document(self, text, metadata=None):
        """Chunk and add a document to the vector store."""
        if not text:
            return
            
        chunks = self.text_splitter.create_documents([text], metadatas=[metadata] if metadata else None)
        print(f"Adding {len(chunks)} chunks to vector store...")
        self.vector_store.add_documents(chunks)

    def similarity_search(self, query, k=10):
        """Search the vector store using MMR for maximum marginal relevance."""
        # fetch_k retrieves more documents initially, then MMR picks the top k most diverse ones
        return self.vector_store.max_marginal_relevance_search(query, k=k, fetch_k=k*5)

if __name__ == "__main__":
    # Test
    # store = RAGStore()
    pass
