# chunkers/base.py

from abc import ABC, abstractmethod
from typing import List

class ChunkingError(Exception):
    """Custom exception for errors during chunking."""
    pass

class BaseChunker(ABC):
    """Abstract base class for a code chunker."""
    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, content: str) -> List[str]:
        """
        Splits code content into semantic chunks.
        
        Should raise ChunkingError on failure.
        """
        pass

def get_node_text(node, content_bytes: bytes) -> str:
    """Helper to get the text content of a tree-sitter node."""
    return content_bytes[node.start_byte:node.end_byte].decode('utf-8')