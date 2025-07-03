# rag_components/embedding_manager.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """
    Manages the embedding model for generating code-aware embeddings.
    Supports both CodeBERT and SentenceTransformers models for memvid compatibility.
    """
    def __init__(self, model_name: str = "microsoft/codebert-base", use_sentence_transformer: bool = False):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_sentence_transformer = use_sentence_transformer
        
        print(f"Loading embedding model '{model_name}' on device '{self.device}'...")
        
        if use_sentence_transformer:
            # For memvid compatibility - use sentence-transformers wrapper
            self.model = SentenceTransformer(model_name)
            self.model.to(self.device)
            self.tokenizer = None
        else:
            # Direct transformers usage for more control
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        
        print("Embedding model loaded successfully.")

    def _mean_pooling(self, model_output, attention_mask):
        """Helper function to pool token embeddings into a single sentence embedding."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generates embeddings for a list of text chunks in batches.
        """
        if self.use_sentence_transformer:
            # Use sentence-transformers encode method
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
            return embeddings
        
        # Use direct transformers approach for more control
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings to unit length
            from torch.nn import functional as F
            normalized_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(normalized_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def get_sentence_transformer_model(self):
        """Returns the sentence transformer model for memvid integration."""
        if self.use_sentence_transformer:
            return self.model
        else:
            # Convert to sentence transformer if needed
            return SentenceTransformer(self.model_name)