# rag_components/knowledge_base.py

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import json
import os
import pickle
from typing import List, Dict, Any
import re

class KnowledgeBase:
    """
    Manages persistent storage of documents, vector index, and keyword index
    for hybrid retrieval.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.doc_store_path = os.path.join(output_dir, "doc_store.json")
        self.vector_index_path = os.path.join(output_dir, "code_index.faiss")
        self.keyword_index_path = os.path.join(output_dir, "code_index.bm25")
        self.metadata_path = os.path.join(output_dir, "kb_metadata.json")

        self.doc_store: Dict[str, Dict[str, Any]] = {}
        self.vector_index = None
        self.keyword_index = None
        self.metadata = {}
        self.is_ready = False

    def _code_tokenizer(self, text: str) -> List[str]:
        """A sophisticated tokenizer for code that handles snake_case, camelCase, and code patterns."""
        # Handle camelCase: insertBefore -> insert Before
        text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
        
        # Handle snake_case: my_function -> my function
        text = text.replace('_', ' ')
        
        # Handle dot notation: obj.method -> obj method
        text = text.replace('.', ' ')
        
        # Handle common code separators
        text = re.sub(r'[(){}[\];,:]', ' ', text)
        
        # Extract alphanumeric tokens and preserve important symbols
        tokens = re.findall(r'[A-Za-z0-9]+|[<>=!]+|[+\-*/]', text.lower())
        
        # Filter out very short tokens and common code noise
        filtered_tokens = [t for t in tokens if len(t) > 1 or t in ['=', '+', '-', '*', '/', '>', '<']]
        
        return filtered_tokens

    def build(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """Builds and saves all indices from scratch."""
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Building knowledge base with {len(chunks)} chunks...")

        # 1. Build Document Store
        self.doc_store = {chunk['id']: chunk for chunk in chunks}
        with open(self.doc_store_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_store, f, indent=2, ensure_ascii=False)

        # 2. Build Vector (FAISS) Index
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        index_flat = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        
        # Map FAISS's internal sequential IDs to our string chunk_ids
        self.vector_index = faiss.IndexIDMap(index_flat)
        
        # Create numeric IDs from chunk IDs
        chunk_ids_numeric = np.array([hash(chunk['id']) % (2**31) for chunk in chunks], dtype=np.int64)
        
        self.vector_index.add_with_ids(embeddings.astype('float32'), chunk_ids_numeric)
        faiss.write_index(self.vector_index, self.vector_index_path)

        # 3. Build Keyword (BM25) Index
        print("Building BM25 keyword index...")
        tokenized_corpus = [self._code_tokenizer(chunk['content']) for chunk in chunks]
        self.keyword_index = BM25Okapi(tokenized_corpus)
        with open(self.keyword_index_path, 'wb') as f:
            pickle.dump(self.keyword_index, f)

        # 4. Save metadata
        self.metadata = {
            'num_chunks': len(chunks),
            'embedding_dimension': dimension,
            'chunk_id_to_numeric': {chunk['id']: hash(chunk['id']) % (2**31) for chunk in chunks},
            'numeric_to_chunk_id': {hash(chunk['id']) % (2**31): chunk['id'] for chunk in chunks}
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.is_ready = True
        print("✅ Knowledge base built and saved successfully.")

    def load(self):
        """Loads all indices from disk."""
        required_files = [self.doc_store_path, self.vector_index_path, self.keyword_index_path, self.metadata_path]
        
        if not all(os.path.exists(p) for p in required_files):
            print("⚠️ Knowledge base files not found. Please generate them first.")
            self.is_ready = False
            return
        
        try:
            with open(self.doc_store_path, 'r', encoding='utf-8') as f:
                self.doc_store = json.load(f)
                
            self.vector_index = faiss.read_index(self.vector_index_path)
            
            with open(self.keyword_index_path, 'rb') as f:
                self.keyword_index = pickle.load(f)
                
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.is_ready = True
            print(f"✅ Knowledge base loaded successfully ({len(self.doc_store)} chunks).")
            
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            self.is_ready = False

    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """Retrieve a specific chunk by ID."""
        return self.doc_store.get(chunk_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if not self.is_ready:
            return {"status": "not_ready"}
        
        return {
            "status": "ready",
            "total_chunks": len(self.doc_store),
            "embedding_dimension": self.metadata.get('embedding_dimension', 0),
            "index_size_mb": os.path.getsize(self.vector_index_path) / (1024*1024) if os.path.exists(self.vector_index_path) else 0
        }