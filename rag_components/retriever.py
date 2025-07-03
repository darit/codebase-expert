# rag_components/retriever.py

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retrieval system that combines semantic search (FAISS) with 
    keyword search (BM25) for improved code search accuracy.
    """
    def __init__(self, embedding_manager, knowledge_base):
        self.embedding_manager = embedding_manager
        self.kb = knowledge_base
        if not self.kb.is_ready:
            self.kb.load()

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.ones_like(scores) if min_s > 0 else np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion for combining multiple ranked lists.
        More robust than weighted score fusion.
        """
        fused_scores = {}
        
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k + rank + 1)
        
        return fused_scores

    def semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Perform semantic search using FAISS vector index."""
        if not self.kb.is_ready:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_manager.get_embeddings([query])
            
            # Normalize for cosine similarity
            import faiss
            faiss.normalize_L2(query_embedding.astype('float32'))
            
            # Search in FAISS index
            similarities, numeric_ids = self.kb.vector_index.search(query_embedding, top_k)
            
            results = []
            for i, numeric_id in enumerate(numeric_ids[0]):
                if numeric_id != -1:
                    # Convert numeric ID back to chunk ID
                    chunk_id = self.kb.metadata['numeric_to_chunk_id'].get(str(numeric_id))
                    if chunk_id and chunk_id in self.kb.doc_store:
                        similarity_score = float(similarities[0][i])
                        results.append((chunk_id, similarity_score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def keyword_search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Perform keyword search using BM25."""
        if not self.kb.is_ready:
            return []
        
        try:
            # Tokenize query
            tokenized_query = self.kb._code_tokenizer(query)
            if not tokenized_query:
                return []
            
            # Get BM25 scores
            bm25_scores = self.kb.keyword_index.get_scores(tokenized_query)
            
            # Get top results
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            
            results = []
            chunk_ids = list(self.kb.doc_store.keys())
            
            for idx in top_indices:
                if idx < len(chunk_ids) and bm25_scores[idx] > 0:
                    chunk_id = chunk_ids[idx]
                    score = float(bm25_scores[idx])
                    results.append((chunk_id, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    def search(self, query: str, top_k: int = 10, semantic_weight: float = 0.7, 
               use_rrf: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword approaches.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            use_rrf: Whether to use Reciprocal Rank Fusion instead of score fusion
        """
        if not self.kb.is_ready:
            logger.warning("Knowledge base not ready")
            return []

        # Perform both searches
        semantic_results = self.semantic_search(query, top_k * 2)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        if not semantic_results and not keyword_results:
            return []

        if use_rrf:
            # Use Reciprocal Rank Fusion
            semantic_ranked = [result[0] for result in semantic_results]
            keyword_ranked = [result[0] for result in keyword_results]
            
            fused_scores = self._reciprocal_rank_fusion([semantic_ranked, keyword_ranked])
            
            # Sort by fused score
            sorted_ids = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
            
        else:
            # Use weighted score fusion
            keyword_weight = 1.0 - semantic_weight
            
            # Convert to dictionaries for easier lookup
            semantic_dict = dict(semantic_results)
            keyword_dict = dict(keyword_results)
            
            # Get all unique chunk IDs
            all_ids = set(semantic_dict.keys()) | set(keyword_dict.keys())
            
            # Normalize scores
            if semantic_results:
                semantic_scores = np.array(list(semantic_dict.values()))
                norm_semantic_scores = self._normalize_scores(semantic_scores)
                norm_semantic_dict = dict(zip(semantic_dict.keys(), norm_semantic_scores))
            else:
                norm_semantic_dict = {}
                
            if keyword_results:
                keyword_scores = np.array(list(keyword_dict.values()))
                norm_keyword_scores = self._normalize_scores(keyword_scores)
                norm_keyword_dict = dict(zip(keyword_dict.keys(), norm_keyword_scores))
            else:
                norm_keyword_dict = {}
            
            # Combine scores
            combined_scores = {}
            for chunk_id in all_ids:
                s_score = norm_semantic_dict.get(chunk_id, 0)
                k_score = norm_keyword_dict.get(chunk_id, 0)
                combined_scores[chunk_id] = (semantic_weight * s_score) + (keyword_weight * k_score)
            
            # Sort by combined score
            sorted_ids = sorted(combined_scores.keys(), key=lambda k: combined_scores[k], reverse=True)
            fused_scores = combined_scores

        # Retrieve full documents
        final_results = []
        for chunk_id in sorted_ids[:top_k]:
            doc = self.kb.doc_store.get(chunk_id)
            if doc:
                doc_copy = doc.copy()
                doc_copy['relevance_score'] = fused_scores[chunk_id]
                
                # Add search method breakdown for debugging
                doc_copy['search_info'] = {
                    'semantic_score': dict(semantic_results).get(chunk_id, 0),
                    'keyword_score': dict(keyword_results).get(chunk_id, 0),
                    'found_in_semantic': chunk_id in dict(semantic_results),
                    'found_in_keyword': chunk_id in dict(keyword_results)
                }
                
                final_results.append(doc_copy)
        
        return final_results

    def search_similar_chunks(self, chunk_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find chunks similar to a given chunk."""
        chunk = self.kb.get_chunk_by_id(chunk_id)
        if not chunk:
            return []
        
        return self.search(chunk['content'], top_k=top_k)